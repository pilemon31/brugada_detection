import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from src.data_loader import augment_minority, make_mixup_dataset
from src.model import build_model, compile_model


SEED = 42


def find_best_threshold(y_true, y_prob, metric="f1", min_prec=0.35):
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    if metric == "f1":
        idx = np.argmax(f1)
    elif metric == "recall":
        mask = prec >= min_prec
        idx  = np.argmax(rec * mask) if mask.any() else np.argmax(rec)
    else:
        idx = np.argmax(prec)
    idx = min(idx, len(thr) - 1)
    return float(thr[idx]), float(prec[idx]), float(rec[idx]), float(f1[idx])


def train_with_cv(X_dev, y_dev, cfg):
    skf         = StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=SEED)
    fold_models = []
    oof_probs   = np.zeros(len(y_dev))
    oof_labels  = np.zeros(len(y_dev))
    fold_metrics, fold_thresholds = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_dev, y_dev)):
        print(f"\n{'='*60}\n  FOLD {fold+1}/{cfg['n_folds']}\n{'='*60}")
        X_tr, X_val = X_dev[tr_idx], X_dev[va_idx]
        y_tr, y_val = y_dev[tr_idx], y_dev[va_idx]

        X_tr, y_tr = augment_minority(X_tr, y_tr, cfg)
        cw_pos     = float((y_tr == 0).sum()) / float((y_tr == 1).sum() + 1e-8)

        model = build_model(cfg)
        model = compile_model(model, cfg, class_weight_pos=cw_pos)

        lr_sched = keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=cfg["lr"], first_decay_steps=20,
            t_mul=1.5, m_mul=0.9, alpha=1e-6
        )
        model.optimizer.learning_rate = lr_sched

        fold_path = os.path.join(cfg["model_dir"], f"fold_{fold+1}.keras")
        callbacks = [
            EarlyStopping("val_auc", patience=25, mode="max",
                          restore_best_weights=True, verbose=1),
            ModelCheckpoint(fold_path, monitor="val_auc",
                            save_best_only=True, mode="max", verbose=0),
        ]
        model.fit(
            make_mixup_dataset(X_tr, y_tr, cfg["batch_size"], cfg, seed=SEED),
            validation_data=(X_val, y_val),
            epochs=cfg["epochs"], callbacks=callbacks, verbose=1
        )

        fold_prob          = model.predict(X_val, verbose=0).squeeze()
        oof_probs[va_idx]  = fold_prob
        oof_labels[va_idx] = y_val

        thr, p, r, f1 = find_best_threshold(y_val, fold_prob, "f1")
        fold_thresholds.append(thr)
        auc = roc_auc_score(y_val, fold_prob)
        fold_metrics.append(dict(fold=fold+1, auc=auc, f1=f1,
                                 recall=r, precision=p, threshold=thr))
        fold_models.append(model)
        print(f"  Fold {fold+1} | AUC:{auc:.4f} F1:{f1:.3f} "
              f"Recall:{r:.3f} Prec:{p:.3f} Thr:{thr:.3f}")

    return fold_models, oof_probs, oof_labels, fold_metrics, fold_thresholds


def ensemble_predict(fold_models, X_test, fold_metrics, oof_probs, oof_labels):
    aucs    = np.array([fm["auc"] for fm in fold_metrics])
    weights = aucs / aucs.sum()

    all_probs = np.stack([m.predict(X_test, verbose=0).squeeze()
                          for m in fold_models])
    y_prob = (all_probs * weights[:, None]).sum(axis=0)

    prec_oof, rec_oof, thr_oof = precision_recall_curve(oof_labels, oof_probs)
    f1_oof  = 2 * prec_oof * rec_oof / (prec_oof + rec_oof + 1e-8)
    best    = min(np.argmax(f1_oof), len(thr_oof) - 1)
    thr     = float(thr_oof[best])
    y_pred  = (y_prob >= thr).astype(int)

    print(f"\n[INFO] Fold weights: "
          + "  ".join([f"F{i+1}={w:.3f}" for i, w in enumerate(weights)]))
    print(f"[INFO] OOF threshold (F1-max): {thr:.3f}")
    return y_pred, y_prob, thr


def threshold_sweep(y_test, y_prob, current_thr):
    print(f"\n{'Thr':>5}  {'Prec':>7}  {'Recall':>8}  {'F1':>7}  "
          f"{'TP':>4}  {'FP':>4}  {'FN':>4}")
    print("  " + "-" * 55)
    best_f1_thr, best_f1 = 0.5, 0.0
    for t in np.arange(0.25, 0.80, 0.05):
        yp = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_test, yp)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            p  = tp / (tp + fp + 1e-8)
            r  = tp / (tp + fn + 1e-8)
            f1 = 2 * p * r / (p + r + 1e-8)
            if f1 > best_f1:
                best_f1, best_f1_thr = f1, t
            note = " <- current" if abs(t - current_thr) < 0.026 else ""
            print(f"{t:>5.2f}  {p:>7.3f}  {r:>8.3f}  {f1:>7.3f}  "
                  f"{int(tp):>4}  {int(fp):>4}  {int(fn):>4}{note}")
    print(f"\n  Best F1 threshold: {best_f1_thr:.2f} (F1={best_f1:.3f})")
    return best_f1_thr


def evaluate_ensemble(y_test, y_pred, y_prob, thr, fold_metrics, xai_dir):
    print(f"\n{'='*60}\n  ENSEMBLE EVALUATION\n{'='*60}")
    for fm in fold_metrics:
        print(f"  Fold {fm['fold']}  AUC:{fm['auc']:.4f}  F1:{fm['f1']:.3f}  "
              f"Recall:{fm['recall']:.3f}  Prec:{fm['precision']:.3f}")
    aucs = [fm["auc"] for fm in fold_metrics]
    print(f"  Mean CV AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

    auc = roc_auc_score(y_test, y_prob)
    ap  = average_precision_score(y_test, y_prob)
    print(f"\nEnsemble Test (thr={thr:.3f}):")
    print(classification_report(y_test, y_pred,
                                target_names=["Normal", "Brugada"]))
    print(f"ROC-AUC: {auc:.4f}  |  PR-AUC: {ap:.4f}")

    prec_t, rec_t, thr_t = precision_recall_curve(y_test, y_prob)
    f1_t     = 2 * prec_t * rec_t / (prec_t + rec_t + 1e-8)
    bi       = min(np.argmax(f1_t), len(thr_t) - 1)
    best_thr = float(thr_t[bi])
    y_best   = (y_prob >= best_thr).astype(int)
    print(f"\nBest-F1 threshold ({best_thr:.2f}):")
    print(classification_report(y_test, y_best,
                                target_names=["Normal", "Brugada"]))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                           display_labels=["Normal","Brugada"]
                           ).plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"CM — OOF threshold (thr={thr:.2f})")

    ConfusionMatrixDisplay(confusion_matrix(y_test, y_best),
                           display_labels=["Normal","Brugada"]
                           ).plot(ax=axes[1], colorbar=False, cmap="Greens")
    axes[1].set_title(f"CM — Best F1 threshold ★ (thr={best_thr:.2f})")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[2].plot(fpr, tpr, color="#e63946", lw=2, label=f"AUC={auc:.4f}")
    axes[2].plot([0,1],[0,1],"k--",lw=1)
    axes[2].set(xlabel="FPR", ylabel="TPR", title="ROC Curve — Ensemble")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.suptitle("BrugadaNet v3.3 — Ensemble Evaluation", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(xai_dir, "evaluation_final.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    return auc


def plot_cv_results(oof_probs, oof_labels, fold_metrics, xai_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(oof_probs[oof_labels==0], bins=30, alpha=0.7,
                 color="#457b9d", label="Normal", density=True)
    axes[0].hist(oof_probs[oof_labels==1], bins=15, alpha=0.7,
                 color="#e63946", label="Brugada", density=True)
    axes[0].set(xlabel="P(Brugada)", ylabel="Density",
                title="OOF Probability Distribution")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    folds  = [f"Fold {fm['fold']}" for fm in fold_metrics]
    aucs   = [fm["auc"] for fm in fold_metrics]
    mean_a = np.mean(aucs)
    colors = ["#457b9d" if a >= mean_a else "#e63946" for a in aucs]
    axes[1].bar(folds, aucs, color=colors, edgecolor="white")
    axes[1].axhline(mean_a, color="black", linestyle="--", lw=1.5,
                    label=f"Mean={mean_a:.4f}")
    axes[1].set(ylim=(0.5, 1.0), ylabel="ROC-AUC", title="Per-Fold AUC")
    axes[1].legend(); axes[1].grid(alpha=0.3, axis="y")
    plt.suptitle("Cross-Validation Summary — BrugadaNet v3.3",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(xai_dir, "cv_results_final.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
