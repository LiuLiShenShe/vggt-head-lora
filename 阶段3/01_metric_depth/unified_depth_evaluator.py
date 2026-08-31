#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified depth evaluator — compares VGGT / DA3 / UniDepth vs reference depth.

Per-frame: RAW + SCALE-ALIGNED metrics (foreground only).
Per-sequence: mean / median / P90 / scale stats.
Summary: cross-model comparison.

Usage:
  python unified_depth_evaluator.py
"""
import os, sys, json, csv
import numpy as np
from PIL import Image
from collections import defaultdict

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
sys.path.insert(0, PHASE3_DIR)
from configs import SEQUENCES, load_sequence_meta, get_depth_path, get_mask_path, VGGT_RERUN

DEPTH_SCALE_TO_METER = 0.001


def resize_nearest(arr, out_h, out_w):
    """Resize 2D array to (out_h, out_w) via nearest neighbor (PIL)."""
    if arr.shape == (out_h, out_w):
        return arr
    # Use PIL nearest for consistency with depth_audit_v3.py
    if arr.dtype == np.bool_ or arr.dtype == bool:
        pil = Image.fromarray(arr.astype(np.uint8) * 255)
        resized = pil.resize((out_w, out_h), Image.NEAREST)
        return np.asarray(resized) > 127
    else:
        # For depth: convert to float32 for PIL, use NEAREST
        pil = Image.fromarray(arr.astype(np.float32), mode='F')
        resized = pil.resize((out_w, out_h), Image.NEAREST)
        return np.asarray(resized, dtype=np.float64)


def compute_metrics(d_pred_valid, d_ref_valid):
    """Compute depth metrics on valid pixels. Returns dict."""
    diff = d_pred_valid - d_ref_valid
    ratio = d_pred_valid / np.maximum(d_ref_valid, 1e-10)
    ratio_inv = d_ref_valid / np.maximum(d_pred_valid, 1e-10)
    worse = np.maximum(ratio, ratio_inv)
    return {
        "absrel": float(np.mean(np.abs(diff) / np.maximum(d_ref_valid, 1e-10))),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "logrmse": float(np.sqrt(np.mean(
            (np.log(np.maximum(d_pred_valid, 1e-6)) - np.log(np.maximum(d_ref_valid, 1e-6))) ** 2))),
        "delta1": float((worse < 1.1).mean()),
        "delta2": float((worse < 1.25).mean()),
        "delta3": float((worse < 1.5625).mean()),
    }


def evaluate_one_frame(pred_depth_2d, ref_depth_raw, fg_mask_raw, out_h, out_w):
    """Evaluate single frame.

    pred_depth_2d: (out_h, out_w) predicted depth in meters
    ref_depth_raw: (720,720) uint16 raw reference depth
    fg_mask_raw: (1080,1080) bool foreground mask
    out_h, out_w: target resolution (model's output size)
    """
    # Resize reference depth to pred resolution
    ref_m = ref_depth_raw.astype(np.float64) * DEPTH_SCALE_TO_METER
    ref_resized = resize_nearest(ref_m, out_h, out_w)

    # Resize foreground mask to pred resolution
    fg_resized = resize_nearest(fg_mask_raw, out_h, out_w)

    # Valid mask
    valid = fg_resized & (ref_resized > 0) & (ref_resized < 65.0) & (pred_depth_2d > 0) & np.isfinite(pred_depth_2d)

    n_valid = int(valid.sum())
    if n_valid < 100:
        return None  # Not enough valid pixels

    d_pred = pred_depth_2d[valid].astype(np.float64)
    d_ref = ref_resized[valid].astype(np.float64)

    # RAW metrics
    raw = compute_metrics(d_pred, d_ref)

    # Scale ratio (median)
    scale_ratio = float(np.median(d_ref) / np.median(d_pred)) if np.median(d_pred) > 1e-10 else 1.0

    # Aligned metrics
    d_pred_aligned = d_pred * scale_ratio
    aligned = compute_metrics(d_pred_aligned, d_ref)

    return {
        "n_valid": n_valid,
        "raw": raw,
        "aligned": aligned,
        "scale_ratio": scale_ratio,
    }


def load_model_depths(seq_id, model):
    """Load predicted depth for a sequence and model."""
    if model == "vggt":
        path = os.path.join(VGGT_RERUN, seq_id, "depth_vggt.npy")
    elif model == "da3":
        path = os.path.join(PHASE3_DIR, "da3", seq_id, "depth_da3.npy")
    elif model == "unidepth":
        path = os.path.join(PHASE3_DIR, "unidepth_v2", seq_id, "depth_unidepth.npy")
    else:
        raise ValueError(f"Unknown model: {model}")
    return np.load(path)


def evaluate_all():
    frame_rows = []
    seq_stats = []

    for seq_id, pose_fail in SEQUENCES:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id} (pose_fail={pose_fail})")
        print(f"{'='*60}")

        meta = load_sequence_meta(seq_id)
        rgb_paths = meta["rgb_paths"]
        depth_dir = meta["depth_dir"]
        mask_dir = meta["mask_dir"]
        n_frames = len(rgb_paths)

        # Pre-load reference depths and masks for all frames
        ref_depths = []
        fg_masks = []
        for i, rp in enumerate(rgb_paths):
            # Reference depth (720x720 uint16)
            dp = get_depth_path(depth_dir, rp)
            if os.path.exists(dp):
                ref_depths.append(np.asarray(Image.open(dp)))
            else:
                ref_depths.append(None)

            # Foreground mask (1080x1080 binary)
            mp = get_mask_path(mask_dir, rp)
            if os.path.exists(mp):
                fg_masks.append(np.asarray(Image.open(mp).convert("L")) > 0)
            else:
                fg_masks.append(None)

        for model in ["vggt", "da3", "unidepth"]:
            print(f"\n  Model: {model}")
            try:
                pred_stack = load_model_depths(seq_id, model)
            except FileNotFoundError:
                print(f"    SKIP — depth file not found")
                continue

            out_h, out_w = pred_stack.shape[1], pred_stack.shape[2]
            print(f"    pred shape: {pred_stack.shape}, out=({out_h},{out_w})")

            per_seq_raw_absrel = []
            per_seq_raw_rmse = []
            per_seq_raw_mae = []
            per_seq_raw_delta1 = []
            per_seq_aligned_absrel = []
            per_seq_scale_ratios = []
            n_valid_frames = 0

            for i in range(n_frames):
                if ref_depths[i] is None or fg_masks[i] is None:
                    continue
                pred_2d = pred_stack[i]
                result = evaluate_one_frame(pred_2d, ref_depths[i], fg_masks[i], out_h, out_w)
                if result is None:
                    continue

                n_valid_frames += 1
                per_seq_raw_absrel.append(result["raw"]["absrel"])
                per_seq_raw_rmse.append(result["raw"]["rmse"])
                per_seq_raw_mae.append(result["raw"]["mae"])
                per_seq_raw_delta1.append(result["raw"]["delta1"])
                per_seq_aligned_absrel.append(result["aligned"]["absrel"])
                per_seq_scale_ratios.append(result["scale_ratio"])

                frame_rows.append({
                    "model": model,
                    "sequence_id": seq_id,
                    "pose_fail": pose_fail,
                    "frame_idx": i,
                    "n_valid_pixels": result["n_valid"],
                    "raw_absrel": result["raw"]["absrel"],
                    "raw_mae": result["raw"]["mae"],
                    "raw_rmse": result["raw"]["rmse"],
                    "raw_logrmse": result["raw"]["logrmse"],
                    "raw_delta1": result["raw"]["delta1"],
                    "raw_delta2": result["raw"]["delta2"],
                    "raw_delta3": result["raw"]["delta3"],
                    "scale_ratio": result["scale_ratio"],
                    "aligned_absrel": result["aligned"]["absrel"],
                    "aligned_mae": result["aligned"]["mae"],
                    "aligned_rmse": result["aligned"]["rmse"],
                    "aligned_logrmse": result["aligned"]["logrmse"],
                    "aligned_delta1": result["aligned"]["delta1"],
                    "aligned_delta2": result["aligned"]["delta2"],
                    "aligned_delta3": result["aligned"]["delta3"],
                })

            if n_valid_frames == 0:
                print(f"    No valid frames!")
                continue

            # Sequence summary
            def _stats(arr):
                a = np.array(arr)
                return {
                    "mean": float(a.mean()),
                    "median": float(np.median(a)),
                    "std": float(a.std()),
                    "p90": float(np.percentile(a, 90)),
                }

            scale_arr = np.array(per_seq_scale_ratios)
            scale_mean = float(scale_arr.mean())
            scale_std = float(scale_arr.std())
            scale_cv = float(scale_std / scale_mean) if scale_mean > 1e-10 else 0.0

            seq_row = {
                "model": model,
                "sequence_id": seq_id,
                "pose_fail": pose_fail,
                "n_frames": n_frames,
                "n_valid_frames": n_valid_frames,
                "raw_absrel_mean": _stats(per_seq_raw_absrel)["mean"],
                "raw_absrel_median": _stats(per_seq_raw_absrel)["median"],
                "raw_absrel_std": _stats(per_seq_raw_absrel)["std"],
                "raw_absrel_p90": _stats(per_seq_raw_absrel)["p90"],
                "raw_rmse_mean": _stats(per_seq_raw_rmse)["mean"],
                "raw_rmse_median": _stats(per_seq_raw_rmse)["median"],
                "raw_rmse_std": _stats(per_seq_raw_rmse)["std"],
                "raw_mae_mean": _stats(per_seq_raw_mae)["mean"],
                "raw_delta1_mean": _stats(per_seq_raw_delta1)["mean"],
                "aligned_absrel_mean": _stats(per_seq_aligned_absrel)["mean"],
                "scale_mean": scale_mean,
                "scale_median": float(np.median(scale_arr)),
                "scale_std": scale_std,
                "scale_cv": scale_cv,
            }
            seq_stats.append(seq_row)
            print(f"    {n_valid_frames}/{n_frames} valid frames")
            print(f"    raw AbsRel: mean={seq_row['raw_absrel_mean']:.4f} median={seq_row['raw_absrel_median']:.4f} "
                  f"std={seq_row['raw_absrel_std']:.4f}")
            print(f"    raw RMSE: mean={seq_row['raw_rmse_mean']:.4f}  aligned AbsRel: {seq_row['aligned_absrel_mean']:.4f}")
            print(f"    scale: mean={scale_mean:.4f} std={scale_std:.4f} CV={scale_cv:.4f}")

    # Write frame CSV
    eval_dir = os.path.join(PHASE3_DIR, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    frame_csv = os.path.join(eval_dir, "DEPTH_MODEL_COMPARISON_FRAME.csv")
    if frame_rows:
        fieldnames = list(frame_rows[0].keys())
        with open(frame_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(frame_rows)
        print(f"\nFrame CSV: {frame_csv} ({len(frame_rows)} rows)")

    # Write sequence CSV
    seq_csv = os.path.join(eval_dir, "DEPTH_MODEL_COMPARISON_SEQ.csv")
    if seq_stats:
        fieldnames = list(seq_stats[0].keys())
        with open(seq_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(seq_stats)
        print(f"Seq CSV: {seq_csv} ({len(seq_stats)} rows)")

    # Write summary JSON
    summary = {"sequences": seq_stats, "n_frame_rows": len(frame_rows)}
    # Model-level summary: aggregate across all sequences (excluding pose-FAIL)
    for model in ["vggt", "da3", "unidepth"]:
        model_rows = [r for r in seq_stats if r["model"] == model]
        model_rows_pass = [r for r in model_rows if not r["pose_fail"]]
        if model_rows_pass:
            summary[f"{model}_pose_pass_mean_absrel"] = float(np.mean([r["raw_absrel_mean"] for r in model_rows_pass]))
            summary[f"{model}_pose_pass_mean_rmse"] = float(np.mean([r["raw_rmse_mean"] for r in model_rows_pass]))
            summary[f"{model}_pose_pass_mean_scale_cv"] = float(np.mean([r["scale_cv"] for r in model_rows_pass]))
            summary[f"{model}_pose_pass_mean_scale"] = float(np.mean([r["scale_mean"] for r in model_rows_pass]))

    summary_path = os.path.join(eval_dir, "DEPTH_MODEL_COMPARISON_SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary: {summary_path}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON (pose-PASS only)")
    print(f"{'='*80}")
    for model in ["vggt", "da3", "unidepth"]:
        k = f"{model}_pose_pass_mean_absrel"
        if k in summary:
            print(f"  {model:10s}  AbsRel={summary[f'{model}_pose_pass_mean_absrel']:.4f}  "
                  f"RMSE={summary[f'{model}_pose_pass_mean_rmse']:.4f}  "
                  f"scale_CV={summary[f'{model}_pose_pass_mean_scale_cv']:.4f}  "
                  f"scale={summary[f'{model}_pose_pass_mean_scale']:.4f}")

    return frame_rows, seq_stats, summary


if __name__ == "__main__":
    evaluate_all()
