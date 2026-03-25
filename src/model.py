import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.metrics import AUC, Precision, Recall


class CrossAttention(layers.Layer):
    def __init__(self, num_heads, key_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha  = layers.MultiHeadAttention(num_heads=num_heads,
                                               key_dim=key_dim, dropout=dropout)
        self.norm = layers.LayerNormalization(epsilon=1e-6)
        self.add  = layers.Add()
        self.last_attn_weights = None

    def call(self, x, training=False):
        attn_out, attn_weights = self.mha(
            x, x, x,
            training=training,
            return_attention_scores=True
        )
        self.last_attn_weights = attn_weights
        return self.norm(self.add([x, attn_out]))


def resnet_block(x, filters, kernel_size=7, downsample=False):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(filters, kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.Activation("relu")(layers.Add()([x, shortcut]))


def build_model(cfg):
    inp = keras.Input(shape=(cfg["target_len"], cfg["n_leads"]), name="ecg_input")

    x = layers.Conv1D(32, 15, strides=2, padding="same", use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    for filters, kernel in [(32, 7), (32, 7)]:
        x = resnet_block(x, filters, kernel)
    x = layers.MaxPooling1D(2)(x)
    x = layers.SpatialDropout1D(0.1)(x)

    for filters, kernel in [(64, 7), (64, 7)]:
        x = resnet_block(x, filters, kernel, downsample=(filters != x.shape[-1]))
    x = layers.MaxPooling1D(2)(x)
    x = layers.SpatialDropout1D(0.1)(x)

    for filters, kernel in [(128, 5), (128, 5)]:
        x = resnet_block(x, filters, kernel, downsample=(filters != x.shape[-1]))
    x = layers.MaxPooling1D(2)(x)
    x = layers.SpatialDropout1D(0.1)(x)

    x = resnet_block(x, 128, 3)

    key_dim    = x.shape[-1] // cfg["attn_heads"]
    attn_layer = CrossAttention(cfg["attn_heads"], key_dim, dropout=0.1,
                                name="cross_attention")
    x = attn_layer(x)

    x = layers.GlobalAveragePooling1D()(x)
    for units in cfg["dense_units"]:
        x = layers.Dense(units, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(cfg["dropout_rate"])(x)

    out = layers.Dense(1, activation="sigmoid", name="output")(x)
    model = Model(inp, out, name="BrugadaNet")
    model._attn_layer = attn_layer
    return model


def focal_loss(alpha=0.75, gamma=2.5, label_smoothing=0.02, class_weight_pos=1.0):
    def loss_fn(y_true, y_pred):
        y_true  = tf.cast(y_true, tf.float32)
        y_pred  = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1 - 1e-7)
        w       = y_true * class_weight_pos + (1.0 - y_true)
        y_s     = y_true * (1 - label_smoothing) + label_smoothing / 2.0
        bce     = -y_s * tf.math.log(y_pred) - (1 - y_s) * tf.math.log(1 - y_pred)
        p_t     = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha  + (1 - y_true) * (1 - alpha)
        return tf.reduce_mean(w * alpha_t * tf.pow(1 - p_t, gamma) * bce)
    return loss_fn


def compile_model(model, cfg, class_weight_pos=1.0):
    model.compile(
        optimizer=keras.optimizers.Adam(cfg["lr"]),
        loss=focal_loss(cfg["focal_alpha"], cfg["focal_gamma"],
                        cfg["label_smoothing"], class_weight_pos),
        metrics=["accuracy", Precision(name="precision"),
                 Recall(name="recall"), AUC(name="auc")]
    )
    return model
