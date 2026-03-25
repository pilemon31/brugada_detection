import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model

LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]


def compute_gradcam_1d(model, signal_input, target_layer_name="cross_attention"):
    grad_model = Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(target_layer_name).output, model.output]
    )
    x = tf.cast(signal_input, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        layer_out, predictions = grad_model(x, training=False)
        pred_score = predictions[:, 0]

    grads           = tape.gradient(pred_score, layer_out)
    pooled_grads    = tf.reduce_mean(grads, axis=-1)[0]
    layer_out_np    = layer_out[0].numpy()
    pooled_grads_np = pooled_grads.numpy()

    heatmap_compressed = np.mean(
        layer_out_np * pooled_grads_np[:, None], axis=-1
    )
    heatmap_compressed = np.maximum(heatmap_compressed, 0)
    if heatmap_compressed.max() > 0:
        heatmap_compressed /= heatmap_compressed.max()

    T_original   = signal_input.shape[1]
    T_compressed = len(heatmap_compressed)
    t_comp = np.linspace(0, 1, T_compressed)
    t_orig = np.linspace(0, 1, T_original)
    heatmap = np.interp(t_orig, t_comp, heatmap_compressed)

    pred = float(predictions[0, 0])
    return heatmap, pred


def plot_gradcam(signal, heatmap, pred, true_label, sample_idx,
                 lead_idx=0, cfg=None, save=True):
    fs        = cfg["fs"]
    t         = np.arange(signal.shape[0]) / fs
    sig       = signal[:, lead_idx]
    label_str = "Brugada" if true_label == 1 else "Normal"
    pred_str  = f"P(Brugada)={pred:.3f}"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(t, sig, color="black", lw=1.2, zorder=3)
    for i in range(len(t) - 1):
        ax1.axvspan(t[i], t[i+1], alpha=0.35 * heatmap[i],
                    color="#e63946", zorder=1)

    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylabel(f"Lead {LEAD_NAMES[lead_idx]} (mV)", fontsize=11)
    ax1.set_title(
        f"GradCAM — Sampel #{sample_idx} | True: {label_str} | {pred_str}",
        fontsize=12, fontweight="bold"
    )
    ax1.grid(alpha=0.3)

    for beat_start in np.arange(0, min(t[-1], 3.0), 0.8):
        segments = [
            (beat_start+0.00, beat_start+0.10, "P",   "#1D9E75", 0.08),
            (beat_start+0.10, beat_start+0.20, "PR",  "#888780", 0.04),
            (beat_start+0.20, beat_start+0.28, "QRS", "#185FA5", 0.08),
            (beat_start+0.28, beat_start+0.36, "ST",  "#E24B4A", 0.12),
            (beat_start+0.36, beat_start+0.60, "T",   "#BA7517", 0.06),
        ]
        for s, e, lbl, col, alpha in segments:
            if e <= t[-1]:
                ax1.axvspan(s, e, ymin=0, ymax=0.08,
                            color=col, alpha=alpha, zorder=2)
                mid = (s + e) / 2
                ax1.text(mid, ax1.get_ylim()[0], lbl,
                         ha="center", va="bottom", fontsize=7, color=col)

    ax2.fill_between(t, heatmap, color="#e63946", alpha=0.7)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Waktu (detik)", fontsize=11)
    ax2.set_ylabel("Skor\nGradCAM", fontsize=10)
    ax2.axhline(0.5, color="gray", lw=0.8, linestyle="--")
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    if save and cfg:
        fname = os.path.join(
            cfg["xai_dir"],
            f"gradcam_s{sample_idx}_{label_str}_lead{LEAD_NAMES[lead_idx]}.png"
        )
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def extract_attention_weights(model, signal_input):
    _ = model(signal_input, training=False)
    attn_layer   = model._attn_layer
    attn_weights = attn_layer.last_attn_weights

    if attn_weights is None:
        raise ValueError("Attention weights tidak tersedia.")

    attn_np      = attn_weights[0].numpy()
    attn_avg     = attn_np.mean(axis=0)
    attn_received = attn_avg.mean(axis=0)
    attn_received = (attn_received - attn_received.min()) / \
                    (attn_received.max() - attn_received.min() + 1e-8)

    T_orig  = signal_input.shape[1]
    T_comp  = len(attn_received)
    t_comp  = np.linspace(0, 1, T_comp)
    t_orig  = np.linspace(0, 1, T_orig)
    return np.interp(t_orig, t_comp, attn_received)


def plot_attention_weights(signal, attn_map, pred, true_label, sample_idx,
                           lead_idx=0, cfg=None, save=True):
    fs        = cfg["fs"]
    t         = np.arange(signal.shape[0]) / fs
    sig       = signal[:, lead_idx]
    label_str = "Brugada" if true_label == 1 else "Normal"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6),
                                    gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(t, sig, color="#1a1a2e", lw=1.2, zorder=3)
    for i in range(len(t) - 1):
        ax1.axvspan(t[i], t[i+1], alpha=0.4 * attn_map[i],
                    color="#457b9d", zorder=1)

    ax1.set_xlim(t[0], t[-1])
    ax1.set_ylabel(f"Lead {LEAD_NAMES[lead_idx]} (mV)", fontsize=11)
    ax1.set_title(
        f"Attention Weights — Sampel #{sample_idx} | "
        f"True: {label_str} | P(Brugada)={pred:.3f}",
        fontsize=12, fontweight="bold"
    )
    ax1.grid(alpha=0.3)

    ax2.fill_between(t, attn_map, color="#457b9d", alpha=0.8)
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Waktu (detik)", fontsize=11)
    ax2.set_ylabel("Attention\nWeight", fontsize=10)
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    if save and cfg:
        fname = os.path.join(
            cfg["xai_dir"],
            f"attention_s{sample_idx}_{label_str}_lead{LEAD_NAMES[lead_idx]}.png"
        )
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def compute_shap_lead_importance(model, X_background, X_explain,
                                  cfg=None, n_samples=None):
    try:
        import shap
    except ImportError:
        print("[WARN] shap tidak terinstal. Jalankan: pip install shap")
        return None, None

    if n_samples is not None:
        X_explain = X_explain[:n_samples]

    print(f"[XAI] Menghitung SHAP dengan {len(X_background)} background "
          f"dan {len(X_explain)} sampel yang dijelaskan...")

    explainer   = shap.DeepExplainer(model, X_background.astype(np.float32))
    shap_values = explainer.shap_values(X_explain.astype(np.float32))

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    lead_importance = np.abs(shap_values).mean(axis=(0, 1))
    print(f"[XAI] SHAP selesai. Shape: {shap_values.shape}")
    return lead_importance, shap_values


def plot_shap_lead_importance(lead_importance, y_explain, cfg=None, save=True):
    colors = ["#e63946" if l.startswith("V") else "#457b9d"
              for l in LEAD_NAMES]
    order  = np.argsort(lead_importance)[::-1]
    labels = [LEAD_NAMES[i] for i in order]
    values = lead_importance[order]
    clr    = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=clr, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Lead ECG", fontsize=12)
    ax.set_ylabel("Mean |SHAP value|", fontsize=12)
    ax.set_title("SHAP Lead Importance — Kontribusi Tiap Lead terhadap Prediksi",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#e63946", label="Precordial (V1–V6)"),
        Patch(facecolor="#457b9d", label="Limb leads (I,II,III,aVR,aVL,aVF)"),
    ], loc="upper right", fontsize=10)

    plt.tight_layout()
    if save and cfg:
        plt.savefig(os.path.join(cfg["xai_dir"], "shap_lead_importance.png"),
                    dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"\n[XAI] Top-3 lead paling penting: "
          f"{labels[0]} ({values[0]:.4f}), "
          f"{labels[1]} ({values[1]:.4f}), "
          f"{labels[2]} ({values[2]:.4f})")


def plot_shap_timeseries(shap_values, signal, sample_idx, true_label,
                          cfg=None, n_leads_show=3, save=True):
    label_str = "Brugada" if true_label == 1 else "Normal"
    fs        = cfg["fs"]
    t         = np.arange(cfg["target_len"]) / fs

    lead_imp  = np.abs(shap_values[sample_idx]).mean(axis=0)
    top_leads = np.argsort(lead_imp)[::-1][:n_leads_show]

    fig, axes = plt.subplots(n_leads_show, 1,
                              figsize=(14, 3 * n_leads_show), sharex=True)
    if n_leads_show == 1:
        axes = [axes]

    for row_idx, lead_idx in enumerate(top_leads):
        ax     = axes[row_idx]
        shap_t = shap_values[sample_idx, :, lead_idx]
        sig    = signal[:, lead_idx]

        ax2 = ax.twinx()
        ax.plot(t, sig, color="black", lw=1, alpha=0.7, zorder=3)
        ax2.fill_between(t, shap_t, 0,
                         where=(shap_t > 0), color="#e63946", alpha=0.4,
                         label="kontribusi positif (→Brugada)")
        ax2.fill_between(t, shap_t, 0,
                         where=(shap_t < 0), color="#457b9d", alpha=0.4,
                         label="kontribusi negatif (→Normal)")
        ax2.axhline(0, color="gray", lw=0.5)

        ax.set_ylabel(f"Lead {LEAD_NAMES[lead_idx]}\n(mV)", fontsize=10)
        ax2.set_ylabel("SHAP", fontsize=9)
        ax.grid(alpha=0.2)
        if row_idx == 0:
            ax.set_title(
                f"SHAP Time Series — Sampel #{sample_idx} | True: {label_str}",
                fontsize=12, fontweight="bold"
            )
        ax2.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Waktu (detik)", fontsize=11)
    plt.tight_layout()
    if save and cfg:
        fname = os.path.join(
            cfg["xai_dir"],
            f"shap_timeseries_s{sample_idx}_{label_str}.png"
        )
        plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


def run_xai_pipeline(model, X_test, y_test, y_prob, cfg):
    print(f"\n{'='*60}")
    print("  XAI PIPELINE — Explainable AI")
    print(f"{'='*60}")

    tp_idx = np.where((y_test == 1) & (y_prob >= 0.40))[0]
    tn_idx = np.where((y_test == 0) & (y_prob <  0.40))[0]
    fn_idx = np.where((y_test == 1) & (y_prob <  0.40))[0]

    n_show   = cfg["xai_n_samples"]
    selected = {
        "TP (Brugada terdeteksi)": tp_idx[:n_show],
        "TN (Normal benar)":       tn_idx[:n_show],
        "FN (Brugada terlewat)":   fn_idx[:min(3, len(fn_idx))],
    }

    print(f"[XAI] Sampel dipilih: "
          f"TP={len(tp_idx)}, TN={len(tn_idx)}, FN={len(fn_idx)}")
    print("\n[XAI] 1/3 — Menghitung GradCAM dan Attention Weights...")

    for group_name, indices in selected.items():
        for i, idx in enumerate(indices):
            sample = X_test[idx:idx+1]
            true_l = int(y_test[idx])

            heatmap, pred = compute_gradcam_1d(
                model, sample, target_layer_name="cross_attention"
            )

            for lead_i in [6, 1]:
                plot_gradcam(
                    signal=sample[0], heatmap=heatmap, pred=pred,
                    true_label=true_l, sample_idx=idx,
                    lead_idx=lead_i, cfg=cfg
                )

            try:
                attn_map = extract_attention_weights(model, sample)
                plot_attention_weights(
                    signal=sample[0], attn_map=attn_map, pred=pred,
                    true_label=true_l, sample_idx=idx,
                    lead_idx=6, cfg=cfg
                )
            except Exception as e:
                print(f"  [WARN] Attention weights gagal untuk sampel {idx}: {e}")

    print("\n[XAI] 2/3 — Menghitung SHAP lead importance...")

    bg_idx = np.concatenate([
        np.where(y_test == 0)[0][:cfg["xai_shap_bg"] // 2],
        np.where(y_test == 1)[0][:cfg["xai_shap_bg"] // 2],
    ])
    X_background = X_test[bg_idx]
    explain_idx  = np.where(y_test == 1)[0]
    X_explain    = X_test[explain_idx]

    lead_imp, shap_values = compute_shap_lead_importance(
        model, X_background, X_explain,
        cfg=cfg, n_samples=min(20, len(X_explain))
    )

    if lead_imp is not None:
        plot_shap_lead_importance(lead_imp, y_test[explain_idx], cfg=cfg)

        print("\n[XAI] 3/3 — Menggambar SHAP time series...")
        for i in range(min(3, len(explain_idx))):
            plot_shap_timeseries(
                shap_values=shap_values,
                signal=X_explain[i],
                sample_idx=i,
                true_label=1,
                cfg=cfg
            )
    else:
        print("[WARN] SHAP dilewati (shap tidak terinstal).")

    print(f"\n[XAI] Semua visualisasi tersimpan di: {cfg['xai_dir']}")
    print("[XAI] File yang dihasilkan:")
    for f in sorted(os.listdir(cfg["xai_dir"])):
        print(f"  {f}")


def xai_summary_stats(model, X_test, y_test, y_prob, cfg):
    segments = {
        "P-wave":  (0,  8),
        "PR":      (8,  16),
        "QRS":     (16, 24),
        "ST":      (24, 32),
        "T-wave":  (32, 60),
    }

    results = {seg: {"Brugada": [], "Normal": []} for seg in segments}

    print("\n[XAI] Menghitung GradCAM segment statistics...")
    for idx in range(min(len(y_test), 40)):
        sample    = X_test[idx:idx+1]
        true_l    = int(y_test[idx])
        label_str = "Brugada" if true_l == 1 else "Normal"

        heatmap, _ = compute_gradcam_1d(model, sample,
                                         target_layer_name="cross_attention")

        for seg_name, (start_frac, end_frac) in segments.items():
            seg_score = heatmap[start_frac:end_frac].mean() if end_frac <= len(heatmap) else 0
            results[seg_name][label_str].append(seg_score)

    seg_names    = list(segments.keys())
    brugada_mean = [np.mean(results[s]["Brugada"]) for s in seg_names]
    normal_mean  = [np.mean(results[s]["Normal"])  for s in seg_names]

    x   = np.arange(len(seg_names))
    wid = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - wid/2, brugada_mean, wid, label="Brugada",
           color="#e63946", alpha=0.85, edgecolor="white")
    ax.bar(x + wid/2, normal_mean,  wid, label="Normal",
           color="#457b9d", alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(seg_names, fontsize=12)
    ax.set_ylabel("Rata-rata GradCAM Score", fontsize=12)
    ax.set_title(
        "GradCAM per Segmen ECG — Brugada vs Normal\n"
        "(lebih tinggi = lebih diperhatikan model)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    st_idx = seg_names.index("ST")
    ax.axvspan(st_idx - 0.5, st_idx + 0.5, color="yellow", alpha=0.15, zorder=0)
    ax.text(st_idx, max(brugada_mean) * 1.05, "← kritis\nBrugada",
            ha="center", fontsize=9, color="#993C1D")

    plt.tight_layout()
    plt.savefig(os.path.join(cfg["xai_dir"], "gradcam_segment_stats.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

    print("\n  Segmen    | Brugada | Normal")
    print("  " + "-" * 35)
    for s, bm, nm in zip(seg_names, brugada_mean, normal_mean):
        print(f"  {s:<10}| {bm:.4f}  | {nm:.4f}")
