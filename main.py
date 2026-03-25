import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from src.data_loader import load_metadata, load_dataset, preprocess
from src.train import (
    train_with_cv, ensemble_predict, threshold_sweep,
    evaluate_ensemble, plot_cv_results
)
from src.explain import run_xai_pipeline, xai_summary_stats

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

CONFIG = {
    "data_dir":        os.getenv("DATA_DIR",     "data/raw"),
    "metadata_csv":    os.getenv("METADATA_CSV", "data/raw/metadata.csv"),
    "model_dir":       os.getenv("MODEL_DIR",    "models"),
    "xai_dir":         os.getenv("XAI_DIR",      "data/processed/xai_results"),

    "fs":           100,
    "target_len":   1200,
    "n_leads":      12,
    "lowcut":       0.5,
    "highcut":      40.0,
    "filter_order": 4,

    "test_size":    0.20,
    "n_folds":      5,
    "batch_size":   8,
    "epochs":       150,
    "lr":           3e-4,

    "attn_heads":   4,
    "dense_units":  [64, 32],
    "dropout_rate": 0.5,

    "focal_alpha":     0.75,
    "focal_gamma":     2.5,
    "label_smoothing": 0.02,

    "aug_n_copies":    6,
    "aug_noise_std":   0.02,
    "aug_time_shift":  50,
    "aug_lead_drop_p": 0.10,

    "xai_n_samples":   10,
    "xai_shap_bg":     50,
    "gradcam_layer":   "cross_attention",
}


def main():
    print("\n" + "="*60)
    print("  BrugadaNet v3.3 + XAI | ResNet1D + CrossAttention")
    print("="*60)

    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["xai_dir"],   exist_ok=True)

    metadata            = load_metadata(CONFIG["metadata_csv"])
    raw_signals, labels = load_dataset(metadata, CONFIG["data_dir"])
    X, y                = preprocess(raw_signals, labels, CONFIG)

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], stratify=y, random_state=SEED
    )
    print(f"\n[INFO] Dev:{X_dev.shape} Test:{X_test.shape}")

    fold_models, oof_probs, oof_labels, fold_metrics, _ = \
        train_with_cv(X_dev, y_dev, CONFIG)

    oof_auc = roc_auc_score(oof_labels, oof_probs)
    print(f"\n[INFO] OOF AUC: {oof_auc:.4f}")
    plot_cv_results(oof_probs, oof_labels, fold_metrics, CONFIG["xai_dir"])

    y_pred, y_prob, thr = ensemble_predict(
        fold_models, X_test, fold_metrics, oof_probs, oof_labels
    )

    best_f1_thr = threshold_sweep(y_test, y_prob, thr)
    final_auc   = evaluate_ensemble(
        y_test, y_pred, y_prob, thr, fold_metrics, CONFIG["xai_dir"]
    )

    print(f"\n{'='*60}")
    print(f"  OOF AUC (CV)         : {oof_auc:.4f}")
    print(f"  Test AUC (Ensemble)  : {final_auc:.4f}")
    print(f"  OOF threshold        : {thr:.3f}")
    print(f"  Best-F1 threshold    : {best_f1_thr:.3f}")
    print(f"{'='*60}")

    best_fold_idx = int(np.argmax([fm["auc"] for fm in fold_metrics]))
    best_model    = fold_models[best_fold_idx]
    print(f"\n[XAI] Menggunakan model Fold {best_fold_idx+1} "
          f"(AUC={fold_metrics[best_fold_idx]['auc']:.4f}) untuk XAI")

    run_xai_pipeline(best_model, X_test, y_test, y_prob, CONFIG)
    xai_summary_stats(best_model, X_test, y_test, y_prob, CONFIG)


if __name__ == "__main__":
    main()
