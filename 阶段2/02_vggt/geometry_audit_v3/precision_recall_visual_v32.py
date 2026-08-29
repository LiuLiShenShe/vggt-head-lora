#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3-1: Precision/Recall visual explanation.

For each pose-PASS plant_view sequence (05/13/20-03-24), generate a
single figure showing:
  - reference (green)
  - all VGGT foreground (orange)
  - VGGT points within 10mm of reference (yellow)
  - VGGT points >10mm (red)
  - VGGT points >20mm (dark red)
  - VGGT points >50mm (magenta)

This directly visualizes the "inflation" problem:
  recall ≈ 0.96–1.00 (reference covered)
  precision ≈ 0.37–0.44 @10mm (VGGT points spread beyond reference)

Uses P_fore (aligned VGGT foreground) and Q_fore (reference foreground)
from the v3.1 driver's saved point clouds.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
sys.path.insert(0, ROOT)

from scipy.spatial import cKDTree

REPRESENTATIVES = [
    "plantview__langdon_4__05-03-24",
    "plantview__langdon_4__13-02-24",
    "plantview__langdon_4__20-02-24",
]

def nn_distances(src, dst, k=1):
    t = cKDTree(dst)
    d, _ = t.query(src, k=k, workers=-1)
    return d

def main():
    out_dir = os.path.join(ROOT, "figures_v31")
    os.makedirs(out_dir, exist_ok=True)

    for sid in REPRESENTATIVES:
        npz_pred = os.path.join(ROOT, "per_seq", f"{sid}_pred_aligned.npy")
        if not os.path.exists(npz_pred):
            print(f"SKIP {sid} (pred_aligned.npy missing)")
            continue

        # Load aligned VGGT foreground (P_fore)
        P_fore = np.load(npz_pred).astype(np.float64)
        # Reference foreground (Q_fore): load from geo_v31.json? use ref.npy
        npz_ref = os.path.join(ROOT, "per_seq", f"{sid}_ref.npy")
        Q_fore = np.load(npz_ref).astype(np.float64)

        # Subsample for speed
        SUBS = 300000
        if len(P_fore) > SUBS:
            idx = np.random.default_rng(0).choice(len(P_fore), SUBS, replace=False)
            P_fore = P_fore[idx]
        if len(Q_fore) > SUBS:
            idx = np.random.default_rng(1).choice(len(Q_fore), SUBS, replace=False)
            Q_fore = Q_fore[idx]

        # Distances
        d_p2g = nn_distances(P_fore, Q_fore)  # each VGGT point -> ref
        d_g2p = nn_distances(Q_fore, P_fore)  # each ref point -> VGGT

        # Color VGGT points by their distance to reference
        colors_p = np.zeros(len(P_fore), dtype=int)  # 0=within10, 1=10-20, 2=20-50, 3=>50
        colors_p[d_p2g > 0.010] = 1
        colors_p[d_p2g > 0.020] = 2
        colors_p[d_p2g > 0.050] = 3

        cmap_p = ListedColormap(["#FFD700", "#FF8C00", "#DC143C", "#8B008B"])
        labels_p = ["≤10mm", "10–20mm", "20–50mm", ">50mm"]

        # Subsample further for plotting
        plot_n = min(80000, len(P_fore))
        pi = np.random.default_rng(2).choice(len(P_fore), plot_n, replace=False)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: XY view
        ax = axes[0]
        ax.scatter(Q_fore[::20, 0], Q_fore[::20, 1], s=0.2, c="green", alpha=0.3, label="reference")
        sc = ax.scatter(P_fore[pi, 0], P_fore[pi, 1], s=0.3, c=colors_p[pi], cmap=cmap_p, vmin=0, vmax=3)
        ax.set_aspect("equal"); ax.set_title(f"{sid}\nXY view — VGGT foreground colored by dist-to-ref")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        # Add precision info
        prec_10 = np.mean(d_p2g <= 0.010)
        rec_10 = np.mean(d_g2p <= 0.010)
        ax.text(0.02, 0.98, f"precision@10mm={prec_10:.2f}\nrecall@10mm={rec_10:.2f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="gray"))

        # Right: XZ view
        ax = axes[1]
        ax.scatter(Q_fore[::20, 0], Q_fore[::20, 2], s=0.2, c="green", alpha=0.3, label="reference")
        ax.scatter(P_fore[pi, 0], P_fore[pi, 2], s=0.3, c=colors_p[pi], cmap=cmap_p, vmin=0, vmax=3)
        ax.set_aspect("equal"); ax.set_title(f"{sid}\nXZ view — VGGT foreground colored by dist-to-ref")
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="green", alpha=0.4, label="reference (all)")]
        for i, lab in enumerate(labels_p):
            legend_elements.append(Patch(facecolor=cmap_p(i), label=f"VGGT {lab}"))
        fig.legend(handles=legend_elements, loc="lower center", ncol=5)

        out_p = os.path.join(out_dir, f"{sid}_precision_recall_explanation.png")
        fig.savefig(out_p, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_p} (prec@10mm={prec_10:.3f}, rec@10mm={rec_10:.3f})")

if __name__ == "__main__":
    main()
