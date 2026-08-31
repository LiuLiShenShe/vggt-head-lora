#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3A.1 — Full corrected model comparison.

Applies:
- DA3 + focal/300 calibration (using calibrated intrinsics)
- UniDepth + K correction (using calibrated intrinsics)
- Compares all variants vs VGGT vs reference
"""
import os, sys, csv, json
import numpy as np
from PIL import Image

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
AUDIT_DIR = os.path.join(PHASE3_DIR, "phase3a1_audit")
sys.path.insert(0, PHASE3_DIR)
from configs import SEQUENCES, load_sequence_meta, get_depth_path, get_mask_path, VGGT_RERUN

DEPTH_SCALE_TO_METER = 0.001

# Calibrated intrinsics
CALIBRATED_FX = 1371.82
CALIBRATED_FY = 1370.79
IMG_SIZE = 1080
DA3_RES = 504


def resize_nearest(arr, out_h, out_w):
    if arr.shape == (out_h, out_w):
        return arr
    if arr.dtype == np.bool_ or arr.dtype == bool:
        pil = Image.fromarray(arr.astype(np.uint8) * 255)
        return np.asarray(pil.resize((out_w, out_h), Image.NEAREST)) > 127
    else:
        pil = Image.fromarray(arr.astype(np.float32), mode='F')
        return np.asarray(pil.resize((out_w, out_h), Image.NEAREST), dtype=np.float64)


def compute_metrics(d_pred_valid, d_ref_valid):
    diff = d_pred_valid - d_ref_valid
    ratio = d_pred_valid / np.maximum(d_ref_valid, 1e-10)
    ratio_inv = d_ref_valid / np.maximum(d_pred_valid, 1e-10)
    worse = np.maximum(ratio, ratio_inv)
    return {
        "absrel": float(np.mean(np.abs(diff) / np.maximum(d_ref_valid, 1e-10))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "delta1": float((worse < 1.1).mean()),
        "delta2": float((worse < 1.25).mean()),
    }


def evaluate_frame(pred_2d, ref_raw, fg_raw, out_h, out_w):
    ref_m = ref_raw.astype(np.float64) * DEPTH_SCALE_TO_METER
    ref_r = resize_nearest(ref_m, out_h, out_w)
    fg_r = resize_nearest(fg_raw, out_h, out_w)
    valid = fg_r & (ref_r > 0) & (ref_r < 65.0) & (pred_2d > 0) & np.isfinite(pred_2d)
    if int(valid.sum()) < 100:
        return None
    dp = pred_2d[valid].astype(np.float64)
    dr = ref_r[valid].astype(np.float64)
    raw = compute_metrics(dp, dr)
    scale = float(np.median(dr) / np.median(dp)) if np.median(dp) > 1e-10 else 1.0
    aligned = compute_metrics(dp * scale, dr)
    return {"raw": raw, "aligned": aligned, "scale": scale, "n_valid": int(valid.sum())}


def main():
    # Compute DA3 focal correction factor
    fx_net = CALIBRATED_FX * DA3_RES / IMG_SIZE
    fy_net = CALIBRATED_FY * DA3_RES / IMG_SIZE
    focal_cal_net = (fx_net + fy_net) / 2
    da3_correction = focal_cal_net / 300.0  # ~2.133

    print(f"DA3 correction factor: {da3_correction:.4f}")
    print(f"  (focal_cal_network={focal_cal_net:.2f} / 300)")

    # Variants to evaluate
    variants = ["vggt", "da3_raw", "da3_calibrated", "unidepth_raw", "unidepth_k_corrected"]

    frame_rows = []
    seq_rows = []

    for seq_id, pose_fail in SEQUENCES:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id} (pose_fail={pose_fail})")
        print(f"{'='*60}")

        meta = load_sequence_meta(seq_id)
        rgb_paths = meta["rgb_paths"]
        depth_dir = meta["depth_dir"]
        mask_dir = meta["mask_dir"]
        n_frames = len(rgb_paths)

        # Load reference depths and masks
        ref_depths, fg_masks = [], []
        for rp in rgb_paths:
            dp = get_depth_path(depth_dir, rp)
            mp = get_mask_path(mask_dir, rp)
            ref_depths.append(np.asarray(Image.open(dp)) if os.path.exists(dp) else None)
            fg_masks.append(np.asarray(Image.open(mp).convert("L")) > 0 if os.path.exists(mp) else None)

        # Load model depths
        vggt_path = os.path.join(VGGT_RERUN, seq_id, "depth_vggt.npy")
        da3_path = os.path.join(PHASE3_DIR, "da3", seq_id, "depth_da3.npy")
        ud_path = os.path.join(PHASE3_DIR, "unidepth_v2", seq_id, "depth_unidepth.npy")
        ud_intr_path = os.path.join(PHASE3_DIR, "unidepth_v2", seq_id, "intrinsics_unidepth.npy")

        d_vggt = np.load(vggt_path) if os.path.exists(vggt_path) else None
        d_da3 = np.load(da3_path) if os.path.exists(da3_path) else None
        d_ud = np.load(ud_path) if os.path.exists(ud_path) else None
        ud_intr = np.load(ud_intr_path) if os.path.exists(ud_intr_path) else None

        # Build variant depths
        variant_depths = {}
        if d_vggt is not None:
            variant_depths["vggt"] = d_vggt
        if d_da3 is not None:
            variant_depths["da3_raw"] = d_da3
            variant_depths["da3_calibrated"] = d_da3 * da3_correction
        if d_ud is not None:
            variant_depths["unidepth_raw"] = d_ud
            if ud_intr is not None:
                # K correction: scale by predicted_fx / calibrated_fx (average over frames)
                k_fx_ratios = ud_intr[:, 0, 0] / CALIBRATED_FX
                mean_k_ratio = float(np.mean(k_fx_ratios))
                variant_depths["unidepth_k_corrected"] = d_ud * mean_k_ratio
                print(f"  UniDepth K correction: mean(pred_fx/cal_fx)={mean_k_ratio:.4f}")

        # Evaluate each variant
        for variant in variants:
            if variant not in variant_depths:
                continue
            dv = variant_depths[variant]
            out_h, out_w = dv.shape[1], dv.shape[2]

            absrels, rmss, aligned_absrels, scales = [], [], [], []

            for i in range(n_frames):
                if ref_depths[i] is None or fg_masks[i] is None:
                    continue
                result = evaluate_frame(dv[i], ref_depths[i], fg_masks[i], out_h, out_w)
                if result is None:
                    continue
                absrels.append(result["raw"]["absrel"])
                rmss.append(result["raw"]["rmse"])
                aligned_absrels.append(result["aligned"]["absrel"])
                scales.append(result["scale"])

            if not absrels:
                continue

            scale_arr = np.array(scales)
            scale_mean = float(scale_arr.mean())
            scale_cv = float(scale_arr.std() / scale_mean) if scale_mean > 1e-10 else 0.0

            seq_row = {
                "model": variant,
                "sequence_id": seq_id,
                "pose_fail": pose_fail,
                "n_frames": n_frames,
                "n_valid": len(absrels),
                "raw_absrel": float(np.mean(absrels)),
                "raw_rmse": float(np.mean(rmss)),
                "aligned_absrel": float(np.mean(aligned_absrels)),
                "scale_mean": scale_mean,
                "scale_cv": scale_cv,
            }
            seq_rows.append(seq_row)

            print(f"  {variant:30s}  AbsRel={seq_row['raw_absrel']:.4f}  "
                  f"RMSE={seq_row['raw_rmse']:.4f}  "
                  f"aligned={seq_row['aligned_absrel']:.4f}  "
                  f"scale={scale_mean:.4f}  CV={scale_cv:.4f}")

    # Save CSV
    csv_path = os.path.join(AUDIT_DIR, "CORRECTED_MODEL_COMPARISON.csv")
    if seq_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(seq_rows[0].keys()))
            w.writeheader()
            w.writerows(seq_rows)
        print(f"\nSaved: {csv_path} ({len(seq_rows)} rows)")

    # Summary table
    print(f"\n{'='*80}")
    print("CORRECTED MODEL COMPARISON (pose-PASS sequences)")
    print(f"{'='*80}")
    print(f"{'Model':30s}  {'raw AbsRel':>10s}  {'RMSE':>8s}  {'aligned':>8s}  {'scale':>8s}  {'CV':>6s}")
    print("-" * 80)

    pass_rows = [r for r in seq_rows if not r["pose_fail"]]
    for variant in variants:
        v_rows = [r for r in pass_rows if r["model"] == variant]
        if v_rows:
            mean_absrel = float(np.mean([r["raw_absrel"] for r in v_rows]))
            mean_rmse = float(np.mean([r["raw_rmse"] for r in v_rows]))
            mean_aligned = float(np.mean([r["aligned_absrel"] for r in v_rows]))
            mean_scale = float(np.mean([r["scale_mean"] for r in v_rows]))
            mean_cv = float(np.mean([r["scale_cv"] for r in v_rows]))
            print(f"{variant:30s}  {mean_absrel:10.4f}  {mean_rmse:8.4f}  {mean_aligned:8.4f}  {mean_scale:8.4f}  {mean_cv:6.4f}")

    # Save summary
    summary = {
        "da3_correction_factor": float(da3_correction),
        "focal_cal_network": float(focal_cal_net),
        "variants_compared": variants,
        "seq_comparison": seq_rows,
    }
    summary_path = os.path.join(AUDIT_DIR, "CORRECTED_SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
