#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scale anchor pilot — estimate VGGT scale using DA3/UniDepth as proxy (no GT).

Anchor: a_i = median(D_metric[valid] / D_VGGT[valid]) per frame.
Sequence anchor: median of per-frame anchors.
Robust anchor: median after 3σ outlier rejection.
"""
import os, sys, csv, json
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
sys.path.insert(0, PHASE3_DIR)
from configs import SEQUENCES, VGGT_RERUN, DEPTH_SCALE_TO_METER

EVAL_DIR = os.path.join(PHASE3_DIR, "evaluation")


def load_depth(model, seq_id):
    if model == "vggt":
        return np.load(os.path.join(VGGT_RERUN, seq_id, "depth_vggt.npy"))
    elif model == "da3":
        return np.load(os.path.join(PHASE3_DIR, "da3", seq_id, "depth_da3.npy"))
    elif model == "unidepth":
        return np.load(os.path.join(PHASE3_DIR, "unidepth_v2", seq_id, "depth_unidepth.npy"))


def compute_frame_anchors(d_metric, d_vggt):
    """Per-frame anchor: median ratio of valid pixels."""
    S = d_metric.shape[0]
    anchors = np.full(S, np.nan)
    for i in range(S):
        dm = d_metric[i].ravel()
        dv = d_vggt[i].ravel()
        valid = (dm > 0) & (dv > 0) & np.isfinite(dm) & np.isfinite(dv)
        if valid.sum() >= 100:
            anchors[i] = float(np.median(dm[valid] / dv[valid]))
    return anchors


def robust_anchor(anchors):
    """Median with 3σ outlier rejection."""
    valid = anchors[~np.isnan(anchors)]
    if len(valid) == 0:
        return np.nan, 0
    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    mask = np.abs(valid - med) < 3 * max(mad, 1e-10)
    return float(np.median(valid[mask])), int(mask.sum())


def compute_anchored_metrics(d_vggt, anchor_seq, ref_depths, fg_masks, out_h, out_w):
    """Compute metrics for VGGT depth scaled by anchor."""
    from unified_depth_evaluator import resize_nearest, compute_metrics
    S = d_vggt.shape[0]
    raw_absrels, raw_rmses = [], []
    aligned_absrels = []
    scale_ratios = []

    for i in range(S):
        if ref_depths[i] is None or fg_masks[i] is None:
            continue
        # Apply anchor
        d_anchored = d_vggt[i] * anchor_seq
        ref_m = ref_depths[i].astype(np.float64) * DEPTH_SCALE_TO_METER
        ref_resized = resize_nearest(ref_m, out_h, out_w)
        fg_resized = resize_nearest(fg_masks[i], out_h, out_w)
        valid = fg_resized & (ref_resized > 0) & (ref_resized < 65.0) & (d_anchored > 0) & np.isfinite(d_anchored)
        if valid.sum() < 100:
            continue
        d_pred = d_anchored[valid].astype(np.float64)
        d_ref = ref_resized[valid].astype(np.float64)
        raw = compute_metrics(d_pred, d_ref)
        raw_absrels.append(raw["absrel"])
        raw_rmses.append(raw["rmse"])
        # Scale-aligned
        scale = float(np.median(d_ref) / np.median(d_pred)) if np.median(d_pred) > 1e-10 else 1.0
        aligned = compute_metrics(d_pred * scale, d_ref)
        aligned_absrels.append(aligned["absrel"])
        scale_ratios.append(scale)

    if not raw_absrels:
        return None
    return {
        "raw_absrel_mean": float(np.mean(raw_absrels)),
        "raw_rmse_mean": float(np.mean(raw_rmses)),
        "aligned_absrel_mean": float(np.mean(aligned_absrels)),
        "scale_mean": float(np.mean(scale_ratios)),
        "scale_cv": float(np.std(scale_ratios) / np.mean(scale_ratios)) if np.mean(scale_ratios) > 1e-10 else 0,
    }


def main():
    from PIL import Image
    from configs import load_sequence_meta, get_depth_path, get_mask_path

    anchor_rows = []
    anchor_comparison = []

    # Load raw VGGT metrics for comparison
    summary_path = os.path.join(EVAL_DIR, "DEPTH_MODEL_COMPARISON_SUMMARY.json")
    with open(summary_path) as f:
        eval_summary = json.load(f)

    for seq_id, pose_fail in SEQUENCES:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id} (pose_fail={pose_fail})")
        print(f"{'='*60}")

        meta = load_sequence_meta(seq_id)
        rgb_paths = meta["rgb_paths"]
        depth_dir = meta["depth_dir"]
        mask_dir = meta["mask_dir"]

        d_vggt = load_depth("vggt", seq_id)
        out_h, out_w = d_vggt.shape[1], d_vggt.shape[2]

        # Pre-load reference/masks
        ref_depths = []
        fg_masks = []
        for rp in rgb_paths:
            dp = get_depth_path(depth_dir, rp)
            mp = get_mask_path(mask_dir, rp)
            ref_depths.append(np.asarray(Image.open(dp)) if os.path.exists(dp) else None)
            fg_masks.append(np.asarray(Image.open(mp).convert("L")) > 0 if os.path.exists(mp) else None)

        for proxy_model in ["da3", "unidepth"]:
            d_proxy = load_depth(proxy_model, seq_id)

            # Compute frame-level anchors
            # Need to resize both to same grid for ratio
            # VGGT: 518x518, DA3: 504x504, UniDepth: 1080x1080
            # Resize proxy depth to VGGT grid for comparison
            from unified_depth_evaluator import resize_nearest
            d_proxy_resized = np.stack([
                resize_nearest(d_proxy[i], out_h, out_w) for i in range(d_proxy.shape[0])
            ], axis=0)

            frame_anchors = compute_frame_anchors(d_proxy_resized, d_vggt)
            valid_anchors = frame_anchors[~np.isnan(frame_anchors)]

            if len(valid_anchors) == 0:
                print(f"  {proxy_model}: no valid anchors!")
                continue

            seq_anchor = float(np.median(valid_anchors))
            robust_a, n_robust = robust_anchor(frame_anchors)

            print(f"  {proxy_model}: frame_median={np.median(valid_anchors):.4f} "
                  f"seq_anchor={seq_anchor:.4f} robust={robust_a:.4f} "
                  f"(n={len(valid_anchors)}, n_robust={n_robust})")
            print(f"    anchor stats: mean={valid_anchors.mean():.4f} std={valid_anchors.std():.4f} "
                  f"min={valid_anchors.min():.4f} max={valid_anchors.max():.4f}")

            for i in range(len(frame_anchors)):
                if not np.isnan(frame_anchors[i]):
                    anchor_rows.append({
                        "sequence_id": seq_id,
                        "pose_fail": pose_fail,
                        "proxy_model": proxy_model,
                        "frame_idx": i,
                        "anchor_value": float(frame_anchors[i]),
                    })

            # Compute anchored VGGT metrics
            anchored = compute_anchored_metrics(d_vggt, seq_anchor, ref_depths, fg_masks, out_h, out_w)
            if anchored:
                print(f"    anchored AbsRel={anchored['raw_absrel_mean']:.4f} "
                      f"RMSE={anchored['raw_rmse_mean']:.4f} "
                      f"scale_CV={anchored['scale_cv']:.4f}")

                # Find raw VGGT metrics for this sequence
                raw_vggt = None
                for r in eval_summary["sequences"]:
                    if r["model"] == "vggt" and r["sequence_id"] == seq_id:
                        raw_vggt = r
                        break

                anchor_comparison.append({
                    "sequence_id": seq_id,
                    "pose_fail": pose_fail,
                    "proxy_model": proxy_model,
                    "seq_anchor": seq_anchor,
                    "raw_absrel": raw_vggt["raw_absrel_mean"] if raw_vggt else None,
                    "anchored_absrel": anchored["raw_absrel_mean"],
                    "absrel_change_pct": ((anchored["raw_absrel_mean"] - raw_vggt["raw_absrel_mean"]) / raw_vggt["raw_absrel_mean"] * 100) if raw_vggt else None,
                    "raw_rmse": raw_vggt["raw_rmse_mean"] if raw_vggt else None,
                    "anchored_rmse": anchored["raw_rmse_mean"],
                    "raw_scale_cv": raw_vggt["scale_cv"] if raw_vggt else None,
                    "anchored_scale_cv": anchored["scale_cv"],
                })

    # Save anchor values
    anchor_dir = os.path.join(PHASE3_DIR, "anchor")
    os.makedirs(anchor_dir, exist_ok=True)

    anchor_csv = os.path.join(anchor_dir, "SCALE_ANCHOR_VALUES.csv")
    if anchor_rows:
        with open(anchor_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(anchor_rows[0].keys()))
            w.writeheader()
            w.writerows(anchor_rows)
        print(f"\nAnchor CSV: {anchor_csv} ({len(anchor_rows)} rows)")

    comp_csv = os.path.join(anchor_dir, "ANCHORED_VGGT_METRICS.csv")
    if anchor_comparison:
        with open(comp_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(anchor_comparison[0].keys()))
            w.writeheader()
            w.writerows(anchor_comparison)
        print(f"Comparison CSV: {comp_csv}")

    # Summary
    print(f"\n{'='*60}")
    print(f"ANCHOR COMPARISON SUMMARY")
    print(f"{'='*60}")
    for row in anchor_comparison:
        print(f"  {row['sequence_id']} | proxy={row['proxy_model']} | "
              f"anchor={row['seq_anchor']:.4f} | "
              f"AbsRel: {row['raw_absrel']:.4f} → {row['anchored_absrel']:.4f} "
              f"({row['absrel_change_pct']:+.1f}%) | "
              f"RMSE: {row['raw_rmse']:.4f} → {row['anchored_rmse']:.4f} | "
              f"scale_CV: {row['raw_scale_cv']:.4f} → {row['anchored_scale_cv']:.4f}")


if __name__ == "__main__":
    main()
