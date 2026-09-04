#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.2 Step 1: Center-scale baseline for all pairwise window overlaps.

For each valid pair (A, B) with shared frames, estimate s from camera centers
with Q fixed (from Phase 3C.1). This establishes the baseline scale chain.

Also computes Step 2 dense scale estimates using point_map for comparison.

Usage:
    python 02_dense_pairwise_scale/pairwise_scale_estimates.py [--seq SEQ ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
GAUGE_DIR = os.path.join(PHASE3C, "05_global_stitching_v31")
ALIGN_DIR = os.path.join(PHASE3C, "04_window_alignment_v31")
OUT_DIR = os.path.join(PHASE3C, "11_scale_sync_v32", "02_dense_pairwise_scale")

sys.path.insert(0, os.path.join(PHASE3C, "05_global_stitching_v31"))
from run_gauge_stitching import (
    w2c_centers, c2w_rotations, find_overlap_frames,
    compute_pairwise_overlap_q, estimate_scale_translation_fixed_Q,
    compose_transform, apply_transform
)

os.makedirs(OUT_DIR, exist_ok=True)


def estimate_dense_scale_for_pair(pm_A, pm_B, Q_AB, mask=None):
    """Estimate scale from dense point_map correspondences.

    Args:
        pm_A: (n_frames, 518, 518, 3) point_map from window A
        pm_B: (n_frames, 518, 518, 3) point_map from window B
        Q_AB: (3, 3) rotation from A gauge to B gauge
        mask: (518, 518) optional boolean foreground mask

    Returns:
        s_dense: scalar scale estimate
        per_frame_scales: list of per-frame scale estimates
        confidence: 1 / (MAD + eps)
    """
    n_frames = pm_A.shape[0]
    per_frame_scales = []

    for i in range(n_frames):
        X_A = pm_A[i].reshape(-1, 3)  # (518*518, 3)
        Y_B_raw = pm_B[i].reshape(-1, 3)

        # Rotate B points into A's gauge
        Y_B = (Q_AB @ Y_B_raw.T).T  # (N, 3)

        # Valid mask: both finite and optionally foreground
        valid = np.isfinite(X_A).all(axis=1) & np.isfinite(Y_B).all(axis=1)
        if mask is not None:
            valid &= mask.ravel()

        # Range filter: points too close or too far are unreliable
        norms_A = np.linalg.norm(X_A, axis=1)
        norms_B = np.linalg.norm(Y_B, axis=1)
        valid &= (norms_A > 0.05) & (norms_A < 15.0)
        valid &= (norms_B > 0.05) & (norms_B < 15.0)

        if valid.sum() < 100:
            continue

        X_v = X_A[valid]
        Y_v = Y_B[valid]

        # Per-pixel scale: s_p = dot(X, Y) / dot(Y, Y)
        s_p = np.sum(X_v * Y_v, axis=1) / np.maximum(np.sum(Y_v * Y_v, axis=1), 1e-10)

        # Robust: trimmed median (remove top/bottom 10%)
        s_sorted = np.sort(s_p)
        trim = max(1, len(s_sorted) // 10)
        s_trimmed = s_sorted[trim:-trim]
        if len(s_trimmed) > 10:
            per_frame_scales.append(float(np.median(s_trimmed)))

    if len(per_frame_scales) == 0:
        return None, [], 0.0

    # Aggregate across frames
    log_scales = np.log(np.array(per_frame_scales))
    s_dense = float(np.exp(np.median(log_scales)))
    mad = float(np.median(np.abs(log_scales - np.median(log_scales))))
    confidence = 1.0 / (mad + 1e-6)

    return s_dense, per_frame_scales, confidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()

    # Find sequences with gauge stitching results
    gauge_files = sorted(glob.glob(os.path.join(GAUGE_DIR, "*_GAUGE_GLOBAL_CAMERAS.npz")))
    if args.seq:
        gauge_files = [g for g in gauge_files if any(seq in g for seq in args.seq)]

    all_pair_rows = []
    all_frame_rows = []

    for gauge_path in gauge_files:
        seq_id = os.path.basename(gauge_path).replace("_GAUGE_GLOBAL_CAMERAS.npz", "")
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")

        # Load windows
        seq_window_dir = os.path.join(WINDOW_DIR, seq_id)
        window_files = sorted(glob.glob(os.path.join(seq_window_dir, "window_*.npz")))
        if len(window_files) < 2:
            print(f"  SKIP: only {len(window_files)} windows")
            continue

        windows = []
        for wf in window_files:
            data = np.load(wf)
            windows.append({
                "ext_w2c": data["ext_w2c_vggt"],
                "frame_idx": data["frame_idx"],
                "point_map": data["point_map"],
            })

        # Load pairwise Q from Phase 3C.1 alignment CSV
        align_csv = os.path.join(ALIGN_DIR, f"{seq_id}_PAIRWISE_GAUGE_ALIGNMENT.csv")
        pair_results = compute_pairwise_overlap_q(windows, seq_id)
        valid_pairs = [pr for pr in pair_results if "Q_star" in pr]

        for pr in valid_pairs:
            a, b = pr["window_a"], pr["window_b"]
            Q_star = pr["Q_star"]

            # Center scale
            C_A = w2c_centers(windows[a]["ext_w2c"])
            C_B = w2c_centers(windows[b]["ext_w2c"])
            inlier_frames = [f for f, m in zip(pr["frame_indices"], pr["inlier_mask"]) if m]
            frame_to_A = {int(f): idx for idx, f in enumerate(windows[a]["frame_idx"])}
            frame_to_B = {int(f): idx for idx, f in enumerate(windows[b]["frame_idx"])}

            if len(inlier_frames) < 2:
                continue

            C_A_ov = np.array([C_A[frame_to_A[f]] for f in inlier_frames])
            C_B_ov = np.array([C_B[frame_to_B[f]] for f in inlier_frames])
            s_center, t_center, rmse_center = estimate_scale_translation_fixed_Q(C_A_ov, C_B_ov, Q_star)

            # Dense scale from point_map
            # Get shared frame indices in each window's local indexing
            shared_local_a = [frame_to_A[f] for f in inlier_frames]
            shared_local_b = [frame_to_B[f] for f in inlier_frames]

            pm_A_shared = windows[a]["point_map"][shared_local_a]  # (n_shared, 518, 518, 3)
            pm_B_shared = windows[b]["point_map"][shared_local_b]

            s_dense, per_frame_scales, confidence = estimate_dense_scale_for_pair(
                pm_A_shared, pm_B_shared, Q_star)

            # Per-frame scale details
            for fi, (lf_a, lf_b, fidx) in enumerate(zip(shared_local_a, shared_local_b, inlier_frames)):
                all_frame_rows.append({
                    "sequence_id": seq_id,
                    "pair": f"W{a}-W{b}",
                    "frame_in_A": int(lf_a),
                    "frame_in_B": int(lf_b),
                    "global_frame": int(fidx),
                    "s_dense_per_frame": per_frame_scales[fi] if fi < len(per_frame_scales) else "",
                })

            row = {
                "sequence_id": seq_id,
                "window_a": a,
                "window_b": b,
                "n_overlap": pr["n_overlap"],
                "n_inliers": pr["n_inliers"],
                "s_center": float(s_center),
                "rmse_center": float(rmse_center),
                "s_dense": s_dense if s_dense is not None else "",
                "dense_confidence": confidence if s_dense is not None else "",
                "n_dense_frames": len(per_frame_scales),
                "s_ratio": float(s_dense / s_center) if s_dense is not None and s_center > 0 else "",
            }
            all_pair_rows.append(row)
            dense_str = f"s_dense={s_dense:.4f}" if s_dense is not None else "DENSE_NA"
            print(f"  W{a}→W{b}: s_center={s_center:.4f} {dense_str} "
                  f"overlap={pr['n_overlap']} inliers={pr['n_inliers']}")

    # Save pair-level CSV
    if all_pair_rows:
        csv_path = os.path.join(OUT_DIR, "PAIRWISE_SCALE_ESTIMATES.csv")
        fields = list(all_pair_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_pair_rows)
        print(f"\nSaved: {csv_path} ({len(all_pair_rows)} pairs)")

    # Save frame-level CSV
    if all_frame_rows:
        csv_path = os.path.join(OUT_DIR, "SHARED_FRAME_SCALE_DISTRIBUTION.csv")
        fields = list(all_frame_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_frame_rows)
        print(f"Saved: {csv_path} ({len(all_frame_rows)} frames)")

    # Summary: scale chain drift
    print(f"\n{'='*60}")
    print("SCALE CHAIN SUMMARY")
    print(f"{'='*60}")
    for gauge_path in gauge_files:
        seq_id = os.path.basename(gauge_path).replace("_GAUGE_GLOBAL_CAMERAS.npz", "")
        seq_pairs = [r for r in all_pair_rows if r["sequence_id"] == seq_id]
        if not seq_pairs:
            continue

        # Compose center scales
        s_chain = 1.0
        s_dense_chain = 1.0
        for r in seq_pairs:
            s_chain *= r["s_center"]
            if r["s_dense"] != "":
                s_dense_chain *= float(r["s_dense"])

        s_center_vals = [r["s_center"] for r in seq_pairs]
        s_dense_vals = [float(r["s_dense"]) for r in seq_pairs if r["s_dense"] != ""]

        print(f"\n  {seq_id}:")
        print(f"    Pairs: {len(seq_pairs)}")
        print(f"    Center scale chain: {s_chain:.4f} (drift: {abs(s_chain-1)*100:.1f}%)")
        print(f"    Center range: [{min(s_center_vals):.4f}, {max(s_center_vals):.4f}]")
        if s_dense_vals:
            print(f"    Dense scale chain: {s_dense_chain:.4f} (drift: {abs(s_dense_chain-1)*100:.1f}%)")
            print(f"    Dense range: [{min(s_dense_vals):.4f}, {max(s_dense_vals):.4f}]")
            cv_c = np.std(s_center_vals) / np.mean(s_center_vals) if len(s_center_vals) > 1 else 0
            cv_d = np.std(s_dense_vals) / np.mean(s_dense_vals) if len(s_dense_vals) > 1 else 0
            print(f"    CV center={cv_c:.4f} dense={cv_d:.4f} → {'dense BETTER' if cv_d < cv_c else 'center BETTER'}")


if __name__ == "__main__":
    main()
