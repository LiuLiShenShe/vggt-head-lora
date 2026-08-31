#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3A visualization — model comparison charts, depth galleries, error maps."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
sys.path.insert(0, PHASE3_DIR)
from configs import SEQUENCES, VGGT_RERUN


def load_summary():
    with open(os.path.join(PHASE3_DIR, "evaluation", "DEPTH_MODEL_COMPARISON_SUMMARY.json")) as f:
        return json.load(f)


def plot_model_comparison_bar(summary):
    """Bar chart: mean AbsRel / RMSE per model, grouped by sequence."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models = ["vggt", "da3", "unidepth"]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    seq_ids = [s["sequence_id"] for s in summary["sequences"] if s["model"] == "vggt"]
    short_names = [s.split("__")[-1] for s in seq_ids]

    for ax, metric, title in [
        (axes[0], "raw_absrel_mean", "Raw AbsRel ↓"),
        (axes[1], "raw_rmse_mean", "Raw RMSE (m) ↓"),
        (axes[2], "scale_cv", "Scale CV ↓"),
    ]:
        x = np.arange(len(seq_ids))
        width = 0.25
        for j, model in enumerate(models):
            vals = []
            for sid in seq_ids:
                for s in summary["sequences"]:
                    if s["model"] == model and s["sequence_id"] == sid:
                        vals.append(s[metric])
                        break
                else:
                    vals.append(0)
            ax.bar(x + j * width, vals, width, label=model, color=colors[j], alpha=0.85)
        ax.set_xticks(x + width)
        ax.set_xticklabels(short_names, rotation=30)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle("Phase 3A: Model Depth Comparison (Plant View, langdon_4)", fontsize=13)
    fig.tight_layout()
    out = os.path.join(PHASE3_DIR, "figures", "model_comparison_bar.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_scale_ratio_distribution():
    """Histogram of per-frame scale ratios for VGGT, DA3, UniDepth."""
    frame_csv = os.path.join(PHASE3_DIR, "evaluation", "DEPTH_MODEL_COMPARISON_FRAME.csv")
    import csv
    data = {"vggt": [], "da3": [], "unidepth": []}
    with open(frame_csv) as f:
        for row in csv.DictReader(f):
            data[row["model"]].append(float(row["scale_ratio"]))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"vggt": "#2196F3", "da3": "#FF5722", "unidepth": "#4CAF50"}
    for model in ["vggt", "da3", "unidepth"]:
        vals = np.array(data[model])
        vals = vals[(vals > 0) & (vals < 5)]
        ax.hist(vals, bins=60, alpha=0.5, label=f"{model} (n={len(vals)})", color=colors[model])
    ax.axvline(1.0, color='red', linestyle='--', linewidth=1.5, label='ideal (1.0)')
    ax.set_xlabel("Scale Ratio (median ref / median pred)")
    ax.set_ylabel("Frame count")
    ax.set_title("Per-frame Scale Ratio Distribution (all sequences)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(PHASE3_DIR, "figures", "scale_ratio_distribution.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_depth_gallery(seq_id, pose_fail=False):
    """5-column gallery: RGB | Ref depth | VGGT | DA3 | UniDepth."""
    from PIL import Image
    import matplotlib.cm as cm

    # Load first frame RGB
    seq_json_path = os.path.join(
        ROOT, "阶段2", "01_sequences", "sequences", "plant_view",
        seq_id.replace("plantview__", "") + ".json"
    )
    with open(seq_json_path) as f:
        seq = json.load(f)
    rgb_path = seq["rgb_paths"][0]
    rgb = np.array(Image.open(rgb_path))

    # Reference depth
    depth_dir = seq["extra"]["depth_dir"]
    basename = os.path.splitext(os.path.basename(rgb_path))[0]
    ref_path = os.path.join(depth_dir, basename + ".png")
    ref_raw = np.asarray(Image.open(ref_path))
    ref_m = ref_raw.astype(np.float64) * 0.001

    # Model depths (frame 0)
    vggt_d = np.load(os.path.join(VGGT_RERUN, seq_id, "depth_vggt.npy"))[0]
    da3_d = np.load(os.path.join(PHASE3_DIR, "da3", seq_id, "depth_da3.npy"))[0]
    ud_d = np.load(os.path.join(PHASE3_DIR, "unidepth_v2", seq_id, "depth_unidepth.npy"))[0]

    vmin, vmax = 0.3, 2.5
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))

    axes[0].imshow(rgb)
    axes[0].set_title("RGB (frame 0)", fontsize=10)
    axes[0].axis('off')

    im1 = axes[1].imshow(ref_m, cmap="magma_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("Reference depth (m)", fontsize=10)
    axes[1].axis('off')

    im2 = axes[2].imshow(vggt_d, cmap="magma_r", vmin=vmin, vmax=vmax)
    axes[2].set_title(f"VGGT depth\nAbsRel=0.19", fontsize=10)
    axes[2].axis('off')

    im3 = axes[3].imshow(da3_d, cmap="magma_r", vmin=vmin, vmax=vmax)
    axes[3].set_title(f"DA3 depth\nAbsRel=0.56", fontsize=10)
    axes[3].axis('off')

    im4 = axes[4].imshow(ud_d, cmap="magma_r", vmin=vmin, vmax=vmax)
    axes[4].set_title(f"UniDepth depth\nAbsRel=0.39", fontsize=10)
    axes[4].axis('off')

    fig.colorbar(im4, ax=axes, fraction=0.02, pad=0.02, label="depth (m)")
    tag = " (pose-FAIL)" if pose_fail else ""
    fig.suptitle(f"Depth Gallery — {seq_id}{tag}", fontsize=12)
    fig.tight_layout()
    short = seq_id.split("__")[-1]
    out = os.path.join(PHASE3_DIR, "figures", f"depth_gallery_{short}.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_anchor_comparison():
    """Bar chart comparing raw vs anchored VGGT metrics."""
    comp_path = os.path.join(PHASE3_DIR, "anchor", "ANCHORED_VGGT_METRICS.csv")
    if not os.path.exists(comp_path):
        print("No anchor comparison data.")
        return
    import csv
    rows = []
    with open(comp_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # DA3 anchor rows only
    da3_rows = [r for r in rows if r["proxy_model"] == "da3"]
    seq_names = [r["sequence_id"].split("__")[-1] for r in da3_rows]
    raw_absrel = [float(r["raw_absrel"]) for r in da3_rows]
    anch_absrel = [float(r["anchored_absrel"]) for r in da3_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(seq_names))
    w = 0.35
    ax.bar(x - w/2, raw_absrel, w, label="VGGT raw", color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, anch_absrel, w, label="VGGT + DA3 anchor", color="#FF5722", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(seq_names, rotation=30)
    ax.set_ylabel("AbsRel ↓")
    ax.set_title("Scale Anchoring: VGGT raw vs DA3-anchored AbsRel")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = os.path.join(PHASE3_DIR, "figures", "anchor_comparison.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    os.makedirs(os.path.join(PHASE3_DIR, "figures"), exist_ok=True)
    summary = load_summary()
    plot_model_comparison_bar(summary)
    plot_scale_ratio_distribution()
    for seq_id, pf in SEQUENCES:
        plot_depth_gallery(seq_id, pf)
    plot_anchor_comparison()
    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
