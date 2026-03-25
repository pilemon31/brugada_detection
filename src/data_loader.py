import os
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt


LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]


def load_metadata(path):
    df = pd.read_csv(path)
    print(f"[INFO] Metadata: {df.shape[0]} records")
    print(df["brugada"].value_counts())
    return df


def load_ecg_signal(record_path):
    try:
        return wfdb.rdrecord(record_path).p_signal.astype(np.float32)
    except Exception as e:
        print(f"[WARN] {record_path}: {e}")
        return None


def load_dataset(metadata, data_dir):
    signals, labels = [], []
    for _, row in metadata.iterrows():
        pid   = str(row["patient_id"])
        label = int(row["brugada"])
        if label not in [0, 1]:
            continue
        sig = load_ecg_signal(os.path.join(data_dir, pid, pid))
        if sig is not None:
            signals.append(sig)
            labels.append(label)
    print(f"[INFO] Loaded {len(signals)} records.")
    return signals, labels


def bandpass_filter(signal, cfg):
    nyq  = 0.5 * cfg["fs"]
    b, a = butter(cfg["filter_order"],
                  [cfg["lowcut"] / nyq, cfg["highcut"] / nyq], btype="band")
    out = np.zeros_like(signal)
    for l in range(signal.shape[1]):
        out[:, l] = filtfilt(b, a, signal[:, l])
    return out


def pad_or_trim(signal, target_len):
    T = signal.shape[0]
    if T >= target_len:
        return signal[:target_len]
    return np.pad(signal, ((0, target_len - T), (0, 0)))


def zscore_normalize(signal):
    out = np.zeros_like(signal)
    for l in range(signal.shape[1]):
        x = signal[:, l]
        out[:, l] = (x - x.mean()) / (x.std() + 1e-8)
    return out


def preprocess(signals, labels, cfg):
    X = []
    for sig in signals:
        sig = bandpass_filter(sig, cfg)
        sig = pad_or_trim(sig, cfg["target_len"])
        sig = zscore_normalize(sig)
        X.append(sig)
    X = np.stack(X).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"[INFO] X: {X.shape} | Normal: {(y==0).sum()} Brugada: {(y==1).sum()}")
    return X, y


def augment_signal(sig, cfg):
    aug    = sig.copy()
    choice = np.random.randint(0, 5)
    T, L   = aug.shape
    if choice == 0:
        aug += np.random.normal(0, cfg["aug_noise_std"], aug.shape).astype(np.float32)
    elif choice == 1:
        aug *= np.random.uniform(0.85, 1.15, (1, L)).astype(np.float32)
    elif choice == 2:
        shift = np.random.randint(-cfg["aug_time_shift"], cfg["aug_time_shift"])
        aug   = np.roll(aug, shift, axis=0)
    elif choice == 3:
        for l in range(L):
            if np.random.rand() < cfg["aug_lead_drop_p"]:
                aug[:, l] = 0.0
    else:
        t      = np.linspace(0, 2 * np.pi, T)
        wander = 0.05 * np.sin(np.random.uniform(0.1, 0.5) * t
                               + np.random.uniform(0, np.pi))
        aug += wander[:, None].astype(np.float32)
    return aug


def augment_minority(X, y, cfg):
    idx = np.where(y == 1)[0]
    X_aug, y_aug = [X], [y]
    for _ in range(cfg["aug_n_copies"]):
        X_new = np.stack([augment_signal(X[i], cfg) for i in idx])
        X_aug.append(X_new)
        y_aug.append(np.ones(len(idx), dtype=np.int32))
    X_out = np.concatenate(X_aug)
    y_out = np.concatenate(y_aug)
    perm  = np.random.permutation(len(X_out))
    print(f"[INFO] Post-aug | Normal: {(y_out==0).sum()} Brugada: {(y_out==1).sum()}")
    return X_out[perm], y_out[perm]


def mixup_batch(X_batch, y_batch, alpha=0.2):
    lam   = np.random.beta(alpha, alpha)
    idx   = np.random.permutation(len(X_batch))
    X_mix = (lam * X_batch + (1 - lam) * X_batch[idx]).astype(np.float32)
    y_mix = (lam * y_batch + (1 - lam) * y_batch[idx]).astype(np.float32)
    return X_mix, y_mix


def make_mixup_dataset(X, y, batch_size, cfg, alpha=0.2, seed=42):
    import tensorflow as tf
    T, L = X.shape[1], X.shape[2]
    def apply_mixup(xb, yb):
        x_mix, y_mix = tf.py_function(
            func=lambda x, y: mixup_batch(x.numpy(), y.numpy(), alpha),
            inp=[xb, yb], Tout=[tf.float32, tf.float32]
        )
        x_mix.set_shape([None, T, L])
        y_mix.set_shape([None])
        return x_mix, y_mix
    return (
        tf.data.Dataset.from_tensor_slices((X, y.astype(np.float32)))
        .shuffle(len(X), reshuffle_each_iteration=True, seed=seed)
        .batch(batch_size, drop_remainder=False)
        .map(apply_mixup, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
