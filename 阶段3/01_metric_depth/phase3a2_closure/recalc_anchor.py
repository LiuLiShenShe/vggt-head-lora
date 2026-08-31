#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5: DA3 official-metric anchor recalculation.

Computes scale anchors using DA3Metric-official depth (NOT raw network output).
Then evaluates VGGT raw vs anchored (frame-level and sequence-level).

Anchor: a_i = median(D_DA3_metric[valid] / D_VGGT[valid]) per frame
Sequence anchor: median(per_frame_anchors) with 3σ rejection.

No GT used for anchor estimation.
"""
import os, sys, csv, json
import numpy as np
from PIL import Image

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
sys.path.insert(0, PHASE3_DIR)
from configs import SEQUENCES, load_sequence_meta, get_depth_path, get_mask_path, VGGT_RERUN
from unified_depth_evaluator import evaluate_one_frame

DEPTH_SCALE_TO_METER = 0.001
AUDIT_DIR = os.path.join(PHASE3_DIR, "phase3a2_closure")


def robust_anchor(per_frame_anchors):
    """Median with 3σ outlier rejection."""
    anchors = np.array(per_frame_anchors)
    med = np.median(anchors)
    std = np.std(anchors)
    if std < 1e-10:
        return med
    mask = np.abs(anchors - med) < 3 * std
    return float(np.median(anchors[mask]))


def main():
    anchor_rows = []
    comparison_rows = []

    for seq_id, pose_fail in SEQUENCES:
        meta = load_sequence_meta(seq_id)
        rgb_paths = meta["rgb_paths"]
        depth_dir = meta["depth_dir"]
        mask_dir = meta["mask_dir"]
        n_frames = len(rgb_paths)

        # Load models
        vggt_path = os.path.join(VGGT_RERUN, seq_id, "depth_vggt.npy")
        da3_metric_path = os.path.join(PHASE3_DIR, "da3", seq_id, "depth_da3_metric.npy")
        vggt_stack = np.load(vggt_path)
        da3_stack = np.load(da3_metric_path)
        out_h, out_w = vggt_stack.shape[1], vggt_stack.shape[2]

        # Pre-load reference depths and masks
        ref_depths = []
        fg_masks = []
        for rp in rgb_paths:
            dp = get_depth_path(depth_dir, rp)
            ref_depths.append(np.asarray(Image.open(dp)) if os.path.exists(dp) else None)
            mp = get_mask_path(mask_dir, rp)
            fg_masks.append(np.asarray(Image.open(mp).convert("L")) > 0 if os.path.exists(mp) else None)

        # Compute per-frame anchors (DA3_metric / VGGT)
        per_frame_anchors = []
        n_computed = 0

        for i in range(min(n_frames, vggt_stack.shape[0], da3_stack.shape[0])):
            if ref_depths[i] is None or fg_masks[i] is None:
                per_frame_anchors.append(np.nan)
                continue

            # Resize DA3 metric to VGGT resolution
            da3_frame = da3_stack[i]
            if da3_frame.shape != (out_h, out_w):
                da3_resized = np.array(Image.fromarray(da3_frame.astype(np.float32), mode='F').resize(
                    (out_w, out_h), Image.BILINEAR)).astype(np.float64)
            else:
                da3_resized = da3_frame.astype(np.float64)

            # Resize mask
            mask_resized = fg_masks[i]
            if mask_resized.shape != (out_h, out_w):
                mask_pil = Image.fromarray(fg_masks[i].astype(np.uint8) * 255)
                mask_resized = np.array(mask_pil.resize((out_w, out_h), Image.NEAREST)) > 127

            vggt_frame = vggt_stack[i].astype(np.float64)

            # Valid pixels: foreground + both models > 0
            valid = mask_resized & (da3_resized > 0) & (vggt_frame > 0) & np.isfinite(da3_resized) & np.isfinite(vggt_frame)
            if valid.sum() < 100:
                per_frame_anchors.append(np.nan)
                continue

            ratio = da3_resized[valid] / vggt_frame[valid]
            anchor = float(np.median(ratio))
            per_frame_anchors.append(anchor)
            n_computed += 1

        # Sequence anchor
        valid_anchors = [a for a in per_frame_anchors if np.isfinite(a)]
        if not valid_anchors:
            print(f"  {seq_id}: no valid anchors")
            continue

        seq_anchor = robust_anchor(valid_anchors)
        anchor_cv = float(np.std(valid_anchors) / np.mean(valid_anchors))

        anchor_rows.append({
            "seq_id": seq_id,
            "pose_fail": pose_fail,
            "n_computed": n_computed,
            "seq_anchor": seq_anchor,
            "anchor_mean": float(np.mean(valid_anchors)),
            "anchor_median": float(np.median(valid_anchors)),
            "anchor_cv": anchor_cv,
        })
        print(f"\n{seq_id}: anchor={seq_anchor:.4f} (CV={anchor_cv:.4f}, n={n_computed})")

        # Evaluate: VGGT raw vs anchored
        # VGGT raw
        vggt_absrels = []
        vggt_rmses = []
        vggt_scales = []
        vggt_d1s = []

        # VGGT + frame anchor
        vggt_fa_absrels = []
        vggt_fa_rmses = []
        vggt_fa_scales = []
        vggt_fa_d1s = []

        # VGGT + sequence anchor
        vggt_sa_absrels = []
        vggt_sa_rmses = []
        vggt_sa_scales = []
        vggt_sa_d1s = []

        # DA3Metric direct
        da3_absrels = []
        da3_rmses = []
        da3_scales = []
        da3_d1s = []

        for i in range(min(n_frames, vggt_stack.shape[0], da3_stack.shape[0])):
            if ref_depths[i] is None or fg_masks[i] is None:
                continue

            # Resize DA3 to VGGT resolution
            da3_frame = da3_stack[i]
            if da3_frame.shape != (out_h, out_w):
                da3_resized = np.array(Image.fromarray(da3_frame.astype(np.float32), mode='F').resize(
                    (out_w, out_h), Image.BILINEAR)).astype(np.float64)
            else:
                da3_resized = da3_frame.astype(np.float64)

            vggt_frame = vggt_stack[i].astype(np.float64)
            anchor = per_frame_anchors[i]

            # VGGT raw
            r = evaluate_one_frame(vggt_frame, ref_depths[i], fg_masks[i], out_h, out_w)
            if r:
                vggt_absrels.append(r["raw"]["absrel"])
                vggt_rmses.append(r["raw"]["rmse"])
                vggt_scales.append(r["scale_ratio"])
                vggt_d1s.append(r["raw"]["delta1"])

            # DA3Metric direct
            r = evaluate_one_frame(da3_resized, ref_depths[i], fg_masks[i], out_h, out_w)
            if r:
                da3_absrels.append(r["raw"]["absrel"])
                da3_rmses.append(r["raw"]["rmse"])
                da3_scales.append(r["scale_ratio"])
                da3_d1s.append(r["raw"]["delta1"])

            # VGGT + frame anchor (if valid)
            if np.isfinite(anchor):
                vggt_anchored = vggt_frame * anchor
                r = evaluate_one_frame(vggt_anchored, ref_depths[i], fg_masks[i], out_h, out_w)
                if r:
                    vggt_fa_absrels.append(r["raw"]["absrel"])
                    vggt_fa_rmses.append(r["raw"]["rmse"])
                    vggt_fa_scales.append(r["scale_ratio"])
                    vggt_fa_d1s.append(r["raw"]["delta1"])

            # VGGT + sequence anchor
            vggt_seq_anchored = vggt_frame * seq_anchor
            r = evaluate_one_frame(vggt_seq_anchored, ref_depths[i], fg_masks[i], out_h, out_w)
            if r:
                vggt_sa_absrels.append(r["raw"]["absrel"])
                vggt_sa_rmses.append(r["raw"]["rmse"])
                vggt_sa_scales.append(r["scale_ratio"])
                vggt_sa_d1s.append(r["raw"]["delta1"])

        def _summary(absrels, rmse, scales, d1s):
            if not absrels:
                return {}
            s = np.array(scales)
            return {
                "abs_rel_mean": float(np.mean(absrels)),
                "rmse_mean": float(np.mean(rmse)),
                "scale_mean": float(s.mean()),
                "scale_cv": float(s.std() / s.mean()) if s.mean() > 1e-10 else 0,
                "delta1_mean": float(np.mean(d1s)),
                "n": len(absrels),
            }

        models_results = {
            "vggt_raw": _summary(vggt_absrels, vggt_rmses, vggt_scales, vggt_d1s),
            "da3_metric_direct": _summary(da3_absrels, da3_rmses, da3_scales, da3_d1s),
            "vggt_frame_anchor": _summary(vggt_fa_absrels, vggt_fa_rmses, vggt_fa_scales, vggt_fa_d1s),
            "vggt_seq_anchor": _summary(vggt_sa_absrels, vggt_sa_rmses, vggt_sa_scales, vggt_sa_d1s),
        }

        for mname, ms in models_results.items():
            if not ms:
                continue
            comparison_rows.append({
                "seq_id": seq_id,
                "pose_fail": pose_fail,
                "model": mname,
                "seq_anchor": seq_anchor if "anchor" in mname else None,
                **ms,
            })
            print(f"  {mname:25s}: AbsRel={ms['abs_rel_mean']:.4f} RMSE={ms['rmse_mean']:.4f} "
                  f"scale={ms['scale_mean']:.4f} δ1={ms['delta1_mean']:.4f}")

    # Save anchor CSV
    os.makedirs(AUDIT_DIR, exist_ok=True)
    anchor_csv = os.path.join(AUDIT_DIR, "DA3_METRIC_ANCHOR_VALUES.csv")
    if anchor_rows:
        with open(anchor_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(anchor_rows[0].keys()))
            w.writeheader()
            w.writerows(anchor_rows)
        print(f"\nAnchor CSV: {anchor_csv}")

    # Save comparison CSV
    comp_csv = os.path.join(AUDIT_DIR, "DA3_METRIC_ANCHOR_COMPARISON.csv")
    if comparison_rows:
        with open(comp_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            w.writeheader()
            w.writerows(comparison_rows)
        print(f"Comparison CSV: {comp_csv}")

    # Pose-PASS summary
    pass_comp = [r for r in comparison_rows if not r["pose_fail"]]
    print("\n=== Pose-PASS Anchor Summary ===")
    for model in ["vggt_raw", "da3_metric_direct", "vggt_frame_anchor", "vggt_seq_anchor"]:
        m = [r for r in pass_comp if r["model"] == model]
        if m:
            mean_abs = np.mean([r["abs_rel_mean"] for r in m])
            mean_rmse = np.mean([r["rmse_mean"] for r in m])
            mean_scale = np.mean([r["scale_mean"] for r in m])
            print(f"  {model:25s}: AbsRel={mean_abs:.4f} RMSE={mean_rmse:.4f} scale={mean_scale:.4f}")

    # Decision: does anchor help or hurt VGGT?
    vggt_raw_pass = [r for r in pass_comp if r["model"] == "vggt_raw"]
    vggt_frame_pass = [r for r in pass_comp if r["model"] == "vggt_frame_anchor"]
    vggt_seq_pass = [r for r in pass_comp if r["model"] == "vggt_seq_anchor"]

    if vggt_raw_pass and vggt_frame_pass:
        raw_abs = np.mean([r["abs_rel_mean"] for r in vggt_raw_pass])
        frame_abs = np.mean([r["abs_rel_mean"] for r in vggt_frame_pass])
        seq_abs = np.mean([r["abs_rel_mean"] for r in vggt_seq_pass]) if vggt_seq_pass else float('nan')
        print(f"\n=== ANCHOR DECISION ===")
        print(f"  VGGT raw AbsRel:     {raw_abs:.4f}")
        print(f"  VGGT+frame anchor:   {frame_abs:.4f}  ({'HELP' if frame_abs < raw_abs else 'WORSE'})")
        print(f"  VGGT+seq anchor:     {seq_abs:.4f}  ({'HELP' if seq_abs < raw_abs else 'WORSE'})")

        if frame_abs >= raw_abs and seq_abs >= raw_abs:
            print(f"  → MSAM = NOT_JUSTIFIED (anchors worsen VGGT)")
        else:
            print(f"  → MSAM = HOLD (anchors may help, needs further analysis)")


if __name__ == "__main__":
    main()
