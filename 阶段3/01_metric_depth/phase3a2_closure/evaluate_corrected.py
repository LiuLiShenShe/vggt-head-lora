#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3+4: Evaluate all models including DA3-official and UniDepth calK pilot.

Reuses the existing evaluator's path construction and metric computation.
Produces CORRECTED_COMPARISON.csv.
"""
import os, sys, csv
import numpy as np
from PIL import Image

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
sys.path.insert(0, PHASE3_DIR)
from configs import SEQUENCES, load_sequence_meta, get_depth_path, get_mask_path, VGGT_RERUN
from unified_depth_evaluator import evaluate_one_frame, resize_nearest

DEPTH_SCALE_TO_METER = 0.001
AUDIT_DIR = os.path.join(PHASE3_DIR, "phase3a2_closure")

# Pilot frame indices (matches run_unidepth_calibrated_k.py)
SAMPLE_INTERVAL = 16
MAX_FRAMES = 20


def load_model_data(seq_id, model):
    """Load predicted depth array for a model."""
    if model == "vggt":
        path = os.path.join(VGGT_RERUN, seq_id, "depth_vggt.npy")
    elif model == "da3_metric":
        path = os.path.join(PHASE3_DIR, "da3", seq_id, "depth_da3_metric.npy")
    elif model == "unidepth_auto":
        path = os.path.join(PHASE3_DIR, "unidepth_v2", seq_id, "depth_unidepth.npy")
    elif model == "unidepth_calK":
        path = os.path.join(PHASE3_DIR, "unidepth_v2", seq_id, "depth_unidepth_calK_pilot.npy")
    else:
        raise ValueError(f"Unknown model: {model}")
    return np.load(path)


def main():
    rows = []

    for seq_id, pose_fail in SEQUENCES:
        meta = load_sequence_meta(seq_id)
        rgb_paths = meta["rgb_paths"]
        depth_dir = meta["depth_dir"]
        mask_dir = meta["mask_dir"]
        n_frames = len(rgb_paths)

        # Pre-load reference depths and masks
        ref_depths = []
        fg_masks = []
        for rp in rgb_paths:
            dp = get_depth_path(depth_dir, rp)
            ref_depths.append(np.asarray(Image.open(dp)) if os.path.exists(dp) else None)
            mp = get_mask_path(mask_dir, rp)
            fg_masks.append(
                np.asarray(Image.open(mp).convert("L")) > 0 if os.path.exists(mp) else None
            )

        # Pilot frame indices
        pilot_indices = list(range(0, min(n_frames, MAX_FRAMES * SAMPLE_INTERVAL), SAMPLE_INTERVAL))[:MAX_FRAMES]

        models_to_eval = ["vggt", "da3_metric", "unidepth_auto", "unidepth_calK"]
        print(f"\n{seq_id}: {n_frames} frames, {len(pilot_indices)} pilot frames")

        for model in models_to_eval:
            try:
                pred_stack = load_model_data(seq_id, model)
            except FileNotFoundError as e:
                print(f"  SKIP {model}: {e}")
                continue

            is_pilot = (model == "unidepth_calK")
            frame_indices = pilot_indices if is_pilot else range(min(n_frames, pred_stack.shape[0]))
            out_h, out_w = pred_stack.shape[1], pred_stack.shape[2]

            per_frame_absrel = []
            per_frame_rmse = []
            per_frame_delta1 = []
            per_frame_scale = []
            n_valid_frames = 0

            for idx in frame_indices:
                if idx >= pred_stack.shape[0]:
                    continue
                if ref_depths[idx] is None or fg_masks[idx] is None:
                    continue

                result = evaluate_one_frame(pred_stack[idx], ref_depths[idx], fg_masks[idx], out_h, out_w)
                if result is None:
                    continue

                n_valid_frames += 1
                per_frame_absrel.append(result["raw"]["absrel"])
                per_frame_rmse.append(result["raw"]["rmse"])
                per_frame_delta1.append(result["raw"]["delta1"])
                per_frame_scale.append(result["scale_ratio"])

            if n_valid_frames == 0:
                print(f"  {model}: no valid frames")
                continue

            row = {
                "seq_id": seq_id,
                "pose_fail": pose_fail,
                "model": model,
                "n_frames": n_valid_frames,
                "abs_rel_mean": float(np.mean(per_frame_absrel)),
                "abs_rel_median": float(np.median(per_frame_absrel)),
                "rmse_mean": float(np.mean(per_frame_rmse)),
                "rmse_median": float(np.median(per_frame_rmse)),
                "delta1_mean": float(np.mean(per_frame_delta1)),
                "scale_mean": float(np.mean(per_frame_scale)),
                "scale_median": float(np.median(per_frame_scale)),
                "scale_cv": float(np.std(per_frame_scale) / np.mean(per_frame_scale)),
            }
            rows.append(row)
            print(f"  {model:20s}: AbsRel={row['abs_rel_mean']:.4f} RMSE={row['rmse_mean']:.4f} "
                  f"δ1={row['delta1_mean']:.4f} scale={row['scale_mean']:.4f} "
                  f"{'[PILOT]' if is_pilot else ''}")

    # Save CSV
    csv_path = os.path.join(AUDIT_DIR, "CORRECTED_COMPARISON.csv")
    os.makedirs(AUDIT_DIR, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved: {csv_path} ({len(rows)} rows)")

    # Summary across pose-PASS
    pass_rows = [r for r in rows if not r["pose_fail"]]
    print("\n=== Pose-PASS Summary ===")
    for model in ["vggt", "da3_metric", "unidepth_auto", "unidepth_calK"]:
        m = [r for r in pass_rows if r["model"] == model]
        if m:
            mean_abs = np.mean([r["abs_rel_mean"] for r in m])
            mean_rmse = np.mean([r["rmse_mean"] for r in m])
            mean_scale = np.mean([r["scale_mean"] for r in m])
            mean_cv = np.mean([r["scale_cv"] for r in m])
            n = sum(r["n_frames"] for r in m)
            print(f"  {model:20s}: AbsRel={mean_abs:.4f} RMSE={mean_rmse:.4f} "
                  f"scale={mean_scale:.4f} CV={mean_cv:.4f} (n={n})")


if __name__ == "__main__":
    main()
