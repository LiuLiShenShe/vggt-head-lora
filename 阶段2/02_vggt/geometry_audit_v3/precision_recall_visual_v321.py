#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-3 (v3.2.1): Precision/Recall visual explanation — REAL foreground-only.

修复 v3.2 (precision_recall_visual_v32.py): 旧脚本加载 per_seq/{sid}_pred_aligned.npy,
那是 FULL-SCENE 点云却被标记为 "foreground". 本脚本只加载真实前景:
  - per_seq/{sid}_pred_foreground_aligned.npy  (VGGT 前景, 已相机中心 Sim3 对齐)
  - per_seq/{sid}_reference_foreground.npy     (参考前景)
并计算真实 precision/recall@10/20mm, 标题写入实测值.

含 12-03-24 (pose-FAIL, 但前景+深度仍可用于可视化 — 标注其 pose_gate).

FIGURE_INPUT_MANIFEST.json 记录每个图用的点云路径 + sha256, 供 test_figure_metric_same_array 校验.
"""
import os, sys, json, hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
sys.path.insert(0, ROOT)

from scipy.spatial import cKDTree

REPRESENTATIVES = [
    ("plantview__langdon_4__05-03-24", False),
    ("plantview__langdon_4__12-03-24", True),
    ("plantview__langdon_4__13-02-24", False),
    ("plantview__langdon_4__20-02-24", False),
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nn_distances(src, dst, k=1):
    t = cKDTree(dst)
    d, _ = t.query(src, k=k, workers=-1)
    return d


def main():
    out_dir = os.path.join(ROOT, "figures_v321")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {"figure_inputs": [], "note": "all points are PLANT-FOREGROUND-ONLY (not full-scene)"}

    for sid, pose_fail in REPRESENTATIVES:
        p_pred = os.path.join(ROOT, "per_seq", f"{sid}_pred_foreground_aligned.npy")
        p_ref = os.path.join(ROOT, "per_seq", f"{sid}_reference_foreground.npy")
        if not (os.path.exists(p_pred) and os.path.exists(p_ref)):
            print(f"SKIP {sid} (fg arrays missing)")
            continue

        P_fore = np.load(p_pred).astype(np.float64)
        Q_fore = np.load(p_ref).astype(np.float64)
        print(f"[{sid}] P_fore={len(P_fore)} Q_fore={len(Q_fore)}")

        SUBS = 300000
        if len(P_fore) > SUBS:
            P_fore = P_fore[np.random.default_rng(0).choice(len(P_fore), SUBS, replace=False)]
        if len(Q_fore) > SUBS:
            Q_fore = Q_fore[np.random.default_rng(1).choice(len(Q_fore), SUBS, replace=False)]

        d_p2g = nn_distances(P_fore, Q_fore)
        d_g2p = nn_distances(Q_fore, P_fore)

        # Color by distance-to-ref
        colors_p = np.zeros(len(P_fore), dtype=int)
        colors_p[d_p2g > 0.010] = 1
        colors_p[d_p2g > 0.020] = 2
        colors_p[d_p2g > 0.050] = 3
        cmap_p = ListedColormap(["#FFD700", "#FF8C00", "#DC143C", "#8B008B"])
        labels_p = ["≤10mm", "10–20mm", "20–50mm", ">50mm"]

        plot_n = min(80000, len(P_fore))
        pi = np.random.default_rng(2).choice(len(P_fore), plot_n, replace=False)

        prec_10 = float(np.mean(d_p2g <= 0.010))
        rec_10 = float(np.mean(d_g2p <= 0.010))
        prec_20 = float(np.mean(d_p2g <= 0.020))
        rec_20 = float(np.mean(d_g2p <= 0.020))
        f10 = 2 * prec_10 * rec_10 / (prec_10 + rec_10) if (prec_10 + rec_10) else 0.0

        tag = " (pose-FAIL)" if pose_fail else ""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        ax = axes[0]
        ax.scatter(Q_fore[::20, 0], Q_fore[::20, 1], s=0.2, c="green", alpha=0.3, label="reference")
        ax.scatter(P_fore[pi, 0], P_fore[pi, 1], s=0.3, c=colors_p[pi], cmap=cmap_p, vmin=0, vmax=3)
        ax.set_aspect("equal"); ax.set_title(f"{sid}{tag}\nXY — VGGT foreground colored by dist-to-ref")
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
        ax.text(0.02, 0.98, f"P@10mm={prec_10:.2f}  R@10mm={rec_10:.2f}  F@10mm={f10:.2f}",
                transform=ax.transAxes, va="top", fontsize=10, bbox=dict(boxstyle="round", fc="white", ec="gray"))
        ax = axes[1]
        ax.scatter(Q_fore[::20, 0], Q_fore[::20, 2], s=0.2, c="green", alpha=0.3, label="reference")
        ax.scatter(P_fore[pi, 0], P_fore[pi, 2], s=0.3, c=colors_p[pi], cmap=cmap_p, vmin=0, vmax=3)
        ax.set_aspect("equal"); ax.set_title(f"{sid}{tag}\nXZ — VGGT foreground colored by dist-to-ref")
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        legend_elements = [Patch(facecolor="green", alpha=0.4, label="reference (all)")]
        for i, lab in enumerate(labels_p):
            legend_elements.append(Patch(facecolor=cmap_p(i), label=f"VGGT {lab}"))
        fig.legend(handles=legend_elements, loc="lower center", ncol=5)

        out_p = os.path.join(out_dir, f"{sid}_precision_recall_explanation.png")
        fig.savefig(out_p, dpi=90, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_p} (P@10={prec_10:.3f} R@10={rec_10:.3f} F@10={f10:.3f})")

        manifest["figure_inputs"].append({
            "sequence_id": sid, "pose_fail": pose_fail,
            "pred_foreground_npy": os.path.relpath(p_pred, ROOT),
            "reference_foreground_npy": os.path.relpath(p_ref, ROOT),
            "pred_sha256": sha256_file(p_pred),
            "ref_sha256": sha256_file(p_ref),
            "precision_10mm": round(prec_10, 4), "recall_10mm": round(rec_10, 4),
            "fscore_10mm": round(f10, 4),
            "precision_20mm": round(prec_20, 4), "recall_20mm": round(rec_20, 4),
        })

    json.dump(manifest, open(os.path.join(ROOT, "FIGURE_INPUT_MANIFEST.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\n-> FIGURE_INPUT_MANIFEST.json ({len(manifest['figure_inputs'])} figures)")


if __name__ == "__main__":
    main()
