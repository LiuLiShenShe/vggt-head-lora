#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.1: Gauge-Aware Orientation Stitching.

Uses overlap camera orientations to recover cross-window gauge rotation Q,
then estimates scale + translation from camera centers with Q fixed.

Key insight: VGGT windows have different local gauges (first-camera-defined).
Overlap orientations provide GT-free rotation correspondences:
  Q_i = R_c2w_A_i @ R_c2w_B_i^T  for each overlap frame i

Method B (ORIENTATION_Q_PLUS_CENTER_ST):
1. Compute Q_i from overlap orientations
2. Robust SO(3) mean → Q_star
3. Fix Q_star, estimate s,t from centers: C_A = s Q_star C_B + t

Usage:
    python 05_global_stitching_v31/run_gauge_stitching.py [--seq SEQUENCE_ID ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
OUT_DIR = os.path.join(PHASE3C, "05_global_stitching_v31")
ALIGN_DIR = os.path.join(PHASE3C, "04_window_alignment_v31")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ALIGN_DIR, exist_ok=True)


# ─── Utility functions ───

def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def w2c_centers(ext_w2c_3x4):
    R = ext_w2c_3x4[:, :3, :3]
    t = ext_w2c_3x4[:, :3, 3]
    return np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)


def c2w_rotations(ext_w2c_3x4):
    """Extract c2w rotation matrices from w2c extrinsics."""
    return ext_w2c_3x4[:, :3, :3].transpose(0, 2, 1)


def find_overlap_frames(frames_a, frames_b):
    set_a = set(frames_a.tolist()) if isinstance(frames_a, np.ndarray) else set(frames_a)
    set_b = set(frames_b.tolist()) if isinstance(frames_b, np.ndarray) else set(frames_b)
    return sorted(set_a & set_b)


# ─── SO(3) robust averaging ───

def so3_robust_mean(rotations, outlier_threshold_mad=3.0):
    """Compute robust SO(3) mean using quaternion averaging with MAD outlier rejection.

    Args:
        rotations: (n, 3, 3) rotation matrices
        outlier_threshold_mad: MAD multiplier for outlier rejection

    Returns:
        R_mean: (3, 3) mean rotation
        inlier_mask: (n,) boolean mask
        stats: dict with residual info
    """
    n = len(rotations)
    if n == 0:
        return np.eye(3), np.array([], dtype=bool), {}
    if n == 1:
        return rotations[0], np.array([True]), {"median": 0, "p90": 0, "max": 0}

    # Sign canonicalization: flip quaternions to same hemisphere
    quats = Rotation.from_matrix(rotations).as_quat()  # (n, 4) xyzw
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[0]) < 0:
            quats[i] = -quats[i]

    # Use median quaternion as initial mean (more robust than mean)
    median_idx = len(quats) // 2
    R_mean = Rotation.from_quat(quats[median_idx]).as_matrix()

    # Compute geodesic distances from initial mean
    geodesics = np.array([rot_angle_deg(R_mean @ R.T) for R in rotations])

    # MAD-based outlier rejection
    median_geo = np.median(geodesics)
    mad = np.median(np.abs(geodesics - median_geo))

    # If MAD is very small (< 0.1 deg), rotations are nearly identical — skip rejection
    if mad < 0.1:
        inlier_mask = np.ones(n, dtype=bool)
        n_outliers = 0
    else:
        threshold = median_geo + outlier_threshold_mad * mad * 1.4826
        inlier_mask = geodesics < threshold
        n_outliers = n - inlier_mask.sum()

        # Safety: if ALL rejected, keep all (initial mean is best available)
        if inlier_mask.sum() == 0:
            inlier_mask = np.ones(n, dtype=bool)
            n_outliers = 0

    # Recompute mean on inliers via quaternion averaging
    inlier_quats = quats[inlier_mask]
    mean_quat_f = inlier_quats.mean(axis=0)
    mean_quat_f /= np.linalg.norm(mean_quat_f)
    R_mean = Rotation.from_quat(mean_quat_f).as_matrix()

    # Final residuals
    geodesics = np.array([rot_angle_deg(R_mean @ R.T) for R in rotations])

    stats = {
        "median": float(np.median(geodesics)),
        "p90": float(np.percentile(geodesics, 90)),
        "max": float(np.max(geodesics)),
        "n_outliers": int(n_outliers),
    }
    return R_mean, inlier_mask, stats


# ─── Pairwise overlap Q computation ───

def compute_pairwise_overlap_q(windows, seq_id):
    """Compute Q_i = R_c2w_A @ R_c2w_B^T for each overlap frame pair.

    Returns list of dicts with per-pair info.
    """
    results = []
    for i in range(len(windows) - 1):
        overlap = find_overlap_frames(windows[i]["frame_idx"], windows[i+1]["frame_idx"])
        if len(overlap) < 2:
            results.append({
                "sequence": seq_id, "window_a": i, "window_b": i + 1,
                "n_overlap": len(overlap), "status": "INSUFFICIENT_OVERLAP",
            })
            continue

        R_c2w_A = c2w_rotations(windows[i]["ext_w2c"])
        R_c2w_B = c2w_rotations(windows[i+1]["ext_w2c"])

        frame_to_A = {int(f): idx for idx, f in enumerate(windows[i]["frame_idx"])}
        frame_to_B = {int(f): idx for idx, f in enumerate(windows[i+1]["frame_idx"])}

        Q_list = []
        frame_indices = []
        for f in overlap:
            idx_A = frame_to_A[f]
            idx_B = frame_to_B[f]
            Q_i = R_c2w_A[idx_A] @ R_c2w_B[idx_B].T
            Q_list.append(Q_i)
            frame_indices.append(int(f))

        Q_array = np.array(Q_list)

        # Robust mean
        Q_star, inlier_mask, stats = so3_robust_mean(Q_array)

        results.append({
            "sequence": seq_id,
            "window_a": i,
            "window_b": i + 1,
            "n_overlap": len(overlap),
            "n_inliers": int(inlier_mask.sum()),
            "Q_dispersion_median": stats["median"],
            "Q_dispersion_p90": stats["p90"],
            "Q_dispersion_max": stats["max"],
            "n_outliers": stats["n_outliers"],
            "Q_star": Q_star,
            "Q_list": Q_list,
            "inlier_mask": inlier_mask,
            "frame_indices": frame_indices,
        })

    return results


# ─── Fixed-Q scale+translation estimation ───

def estimate_scale_translation_fixed_Q(C_A, C_B, Q):
    """Estimate s, t such that C_A ≈ s * Q @ C_B + t.

    Closed-form least squares.
    """
    # Transform B centers into A's orientation
    X = (Q @ C_B.T).T  # (n, 3)

    X_mean = X.mean(axis=0)
    C_mean = C_A.mean(axis=0)
    X_c = X - X_mean
    C_c = C_A - C_mean

    # Scale
    s = np.sum(C_c * X_c) / max(np.sum(X_c ** 2), 1e-10)

    # Translation
    t = C_mean - s * X_mean

    # Residual
    C_pred = s * X + t
    rmse = float(np.sqrt(np.mean(np.linalg.norm(C_pred - C_A, axis=1) ** 2)))

    return s, t, rmse


# ─── Sim(3) composition ───

def compose_transform(T_prev, S):
    """Compose transforms: T_new = T_prev ∘ S.

    T maps from local gauge to global gauge.
    S maps from (k+1) gauge to k gauge.
    """
    s_prev, Q_prev, t_prev = T_prev
    s_S, Q_S, t_S = S
    s_new = s_prev * s_S
    Q_new = Q_prev @ Q_S
    t_new = s_prev * (Q_prev @ t_S) + t_prev
    return (s_new, Q_new, t_new)


def apply_transform(ext_w2c, s, Q, t):
    """Apply gauge transform (s, Q, t) to all cameras in a window.

    C_global = s * Q @ C_local + t
    R_c2w_global = Q @ R_c2w_local
    """
    R_w2c = ext_w2c[:, :3, :3]
    t_w2c = ext_w2c[:, :3, 3]

    # c2w rotation
    R_c2w = R_w2c.transpose(0, 2, 1)

    # Camera centers in local frame
    centers_local = np.einsum("sij,sj->si", R_c2w, -t_w2c)

    # Transform centers
    centers_global = s * (Q @ centers_local.T).T + t

    # Transform rotations: R_c2w_global = Q @ R_c2w_local
    R_c2w_global = np.einsum("ij,sjk->sik", Q, R_c2w)

    # Convert back to w2c
    R_w2c_global = R_c2w_global.transpose(0, 2, 1)
    t_w2c_global = -np.einsum("sij,sj->si", R_w2c_global, centers_global)

    new_ext = np.zeros_like(ext_w2c)
    new_ext[:, :3, :3] = R_w2c_global
    new_ext[:, :3, 3] = t_w2c_global
    if ext_w2c.ndim == 3 and ext_w2c.shape[1] == 4 and ext_w2c.shape[2] == 4:
        new_ext[:, 3, 3] = 1
    return new_ext


# ─── Main stitching pipeline ───

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()

    if args.seq:
        seq_dirs = [os.path.join(WINDOW_DIR, s) for s in args.seq]
    else:
        seq_dirs = sorted([os.path.join(WINDOW_DIR, d)
                          for d in os.listdir(WINDOW_DIR)
                          if os.path.isdir(os.path.join(WINDOW_DIR, d))])

    all_pair_results = []

    for seq_dir in seq_dirs:
        if not os.path.isdir(seq_dir):
            continue
        seq_id = os.path.basename(seq_dir)
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")
        print(f"{'='*60}")

        window_files = sorted(glob.glob(os.path.join(seq_dir, "window_*.npz")))
        if len(window_files) < 2:
            print(f"  SKIP: only {len(window_files)} window(s)")
            continue

        windows = []
        for wf in window_files:
            data = np.load(wf)
            windows.append({
                "ext_w2c": data["ext_w2c_vggt"],
                "frame_idx": data["frame_idx"],
            })

        # ─── Step 1: Pairwise overlap Q computation ───
        pair_results = compute_pairwise_overlap_q(windows, seq_id)

        print("\n  Pairwise Q dispersion:")
        for pr in pair_results:
            if "Q_star" not in pr:
                print(f"    W{pr['window_a']:2d}->W{pr['window_b']:2d}: {pr['status']}")
                continue
            print(f"    W{pr['window_a']:2d}->W{pr['window_b']:2d}: "
                  f"n_overlap={pr['n_overlap']} "
                  f"n_inliers={pr['n_inliers']} "
                  f"Q_disp_med={pr['Q_dispersion_median']:.2f}° "
                  f"Q_disp_p90={pr['Q_dispersion_p90']:.2f}° "
                  f"Q_disp_max={pr['Q_dispersion_max']:.2f}°")
            all_pair_results.append(pr)

        # ─── Step 2: Check Q consistency across all pairs ───
        valid_pairs = [pr for pr in pair_results if "Q_star" in pr]
        if not valid_pairs:
            print("  No valid pairwise alignments")
            continue

        all_Q_disp_medians = [pr["Q_dispersion_median"] for pr in valid_pairs]
        mean_Q_disp = np.mean(all_Q_disp_medians)
        max_Q_disp = np.max(all_Q_disp_medians)

        print(f"\n  Overall Q dispersion: mean={mean_Q_disp:.2f}° max={max_Q_disp:.2f}°")

        if max_Q_disp < 10.0:
            consistency = "HIGH"
        elif max_Q_disp < 20.0:
            consistency = "MODERATE"
        else:
            consistency = "LOW"
        print(f"  Overlap orientation consistency: {consistency}")

        # ─── Step 3: Build global transforms via chaining ───
        # Window 0 is the reference gauge (identity)
        global_transforms = [(1.0, np.eye(3), np.zeros(3))]

        for i, pr in enumerate(valid_pairs):
            if "Q_star" not in pr:
                global_transforms.append(None)
                continue

            Q_star = pr["Q_star"]
            C_A = w2c_centers(windows[pr["window_a"]]["ext_w2c"])
            C_B = w2c_centers(windows[pr["window_b"]]["ext_w2c"])

            # Overlap frames
            overlap = find_overlap_frames(
                windows[pr["window_a"]]["frame_idx"],
                windows[pr["window_b"]]["frame_idx"]
            )
            frame_to_A = {int(f): idx for idx, f in enumerate(windows[pr["window_a"]]["frame_idx"])}
            frame_to_B = {int(f): idx for idx, f in enumerate(windows[pr["window_b"]]["frame_idx"])}

            # Use inlier overlap frames for center alignment
            inlier_frames = [f for f, m in zip(pr["frame_indices"], pr["inlier_mask"]) if m]
            if len(inlier_frames) < 2:
                global_transforms.append(None)
                continue

            C_A_overlap = np.array([C_A[frame_to_A[f]] for f in inlier_frames])
            C_B_overlap = np.array([C_B[frame_to_B[f]] for f in inlier_frames])

            # Fixed-Q scale+translation estimation
            s_AB, t_AB, center_rmse = estimate_scale_translation_fixed_Q(
                C_A_overlap, C_B_overlap, Q_star
            )

            # S_{B→A}: maps from B's gauge to A's gauge
            S_BtoA = (s_AB, Q_star, t_AB)

            # Compose: G_{B} = G_{A} ∘ S_{B→A}
            T_prev = global_transforms[i]
            if T_prev is None:
                global_transforms.append(None)
                continue

            G_new = compose_transform(T_prev, S_BtoA)
            global_transforms.append(G_new)

            # Orientation overlap residual
            R_c2w_A = c2w_rotations(windows[pr["window_a"]]["ext_w2c"])
            R_c2w_B = c2w_rotations(windows[pr["window_b"]]["ext_w2c"])
            orient_errs = []
            for fi in pr["frame_indices"]:
                if not pr["inlier_mask"][pr["frame_indices"].index(fi)]:
                    continue
                idx_A = frame_to_A[fi]
                idx_B = frame_to_B[fi]
                Q_actual = R_c2w_A[idx_A] @ R_c2w_B[idx_B].T
                orient_errs.append(rot_angle_deg(Q_actual @ Q_star.T))

            print(f"  W{pr['window_a']:2d}->W{pr['window_b']:2d}: "
                  f"s={s_AB:.4f} center_rmse={center_rmse:.4f} "
                  f"orient_res_med={np.median(orient_errs):.2f}° "
                  f"center_rmse_before={pr.get('center_rmse_before', 'N/A')}")

            # Save pair result
            pr["scale"] = float(s_AB)
            pr["translation_norm"] = float(np.linalg.norm(t_AB))
            pr["center_rmse_after"] = center_rmse
            pr["orientation_residual_median"] = float(np.median(orient_errs))
            pr["orientation_residual_p90"] = float(np.percentile(orient_errs, 90)) if orient_errs else 0
            pr["status"] = "OK"

        # ─── Step 3b: Global rotation optimization ───
        # Sequential chaining accumulates rotation error over long chains.
        # Solve for global rotations R_0..R_N that minimize:
        #   sum_k ||R_k^T @ R_{k+1} - Q_k||_F^2
        # where Q_k is the pairwise gauge rotation from window k to k+1.
        #
        # This distributes error evenly instead of accumulating it.

        n_wins = len(windows)
        # Extract pairwise Q measurements and initial rotations from chain
        pairwise_Q = [None] * (n_wins - 1)
        for pr in valid_pairs:
            if "Q_star" in pr:
                pairwise_Q[pr["window_a"]] = pr["Q_star"]

        # Initial rotations from sequential chain
        R_chain = [gt[1] for gt in global_transforms if gt is not None]
        if len(R_chain) < n_wins:
            # Some windows failed chaining; fill with identity
            R_chain_full = [np.eye(3)] * n_wins
            idx = 0
            for i in range(n_wins):
                if global_transforms[i] is not None:
                    R_chain_full[i] = global_transforms[i][1]
            R_chain = R_chain_full

        # Gauss-Newton optimization on SO(3) — pin window 0
        R_global = [R.copy() for R in R_chain]
        R_global[0] = R_chain[0].copy()  # pin reference
        for iteration in range(50):
            max_update = 0
            for k in range(n_wins - 1):
                if pairwise_Q[k] is None:
                    continue
                Q_k = pairwise_Q[k]
                R_rel = R_global[k].T @ R_global[k+1]
                E = Q_k.T @ R_rel
                log_E = 0.5 * (E - E.T)
                err_vec = np.array([log_E[2,1], log_E[0,2], log_E[1,0]])
                err_norm = np.linalg.norm(err_vec)
                if err_norm < 1e-10:
                    continue
                correction = 0.5 * err_vec
                delta = np.eye(3) + np.array([
                    [0, -correction[2], correction[1]],
                    [correction[2], 0, -correction[0]],
                    [-correction[1], correction[0], 0]
                ])
                if k == 0:
                    # Only update R_global[1], keep R_global[0] pinned
                    R_global[k+1] = R_global[k+1] @ delta
                else:
                    R_global[k] = R_global[k] @ delta.T
                    R_global[k+1] = R_global[k+1] @ delta
                max_update = max(max_update, err_norm)

            if max_update < 1e-8:
                break

        # Compute rotation improvement
        def total_chain_error(R_list, Q_list):
            total = 0
            for k in range(len(R_list) - 1):
                if Q_list[k] is None:
                    continue
                R_rel = R_list[k].T @ R_list[k+1]
                E = Q_list[k].T @ R_rel
                total += rot_angle_deg(E)
            return total

        err_before = total_chain_error(R_chain, pairwise_Q)
        err_after = total_chain_error(R_global, pairwise_Q)
        print(f"\n  Global rotation optimization: chain_error {err_before:.2f}° -> {err_after:.2f}°")

        # Update global_transforms with optimized rotations
        for i in range(n_wins):
            if global_transforms[i] is not None:
                s, _, t = global_transforms[i]
                global_transforms[i] = (s, R_global[i], t)

        # ─── Step 3c: Chain Sim(3) transforms ───
        # Each pairwise Sim(3) S_{B→A} = (s_AB, Q_star, t_AB) maps B's gauge to A's.
        # Global transform G_k = G_0 ∘ S_{0→1} ∘ ... ∘ S_{(k-1)→k}
        # G_0 = (1, I, 0) — window 0 is the reference gauge.

        n_wins = len(windows)
        global_transforms = [(1.0, R_global[0], np.zeros(3))]

        for i, pr in enumerate(valid_pairs):
            if "Q_star" not in pr:
                global_transforms.append(None)
                continue

            a, b_win = pr["window_a"], pr["window_b"]
            Q_pair = R_global[a].T @ R_global[b_win]
            C_A = w2c_centers(windows[a]["ext_w2c"])
            C_B = w2c_centers(windows[b_win]["ext_w2c"])
            inlier_frames = [f for f, m in zip(pr["frame_indices"], pr["inlier_mask"]) if m]
            if len(inlier_frames) < 2:
                global_transforms.append(None)
                continue
            frame_to_A = {int(f): idx for idx, f in enumerate(windows[a]["frame_idx"])}
            frame_to_B = {int(f): idx for idx, f in enumerate(windows[b_win]["frame_idx"])}
            C_A_ov = np.array([C_A[frame_to_A[f]] for f in inlier_frames])
            C_B_ov = np.array([C_B[frame_to_B[f]] for f in inlier_frames])

            s_AB, t_AB, center_rmse = estimate_scale_translation_fixed_Q(C_A_ov, C_B_ov, Q_pair)
            S_BtoA = (s_AB, Q_pair, t_AB)

            T_prev = global_transforms[i]
            if T_prev is None:
                global_transforms.append(None)
                continue

            G_new = compose_transform(T_prev, S_BtoA)
            global_transforms.append(G_new)

            pr["scale"] = float(s_AB)
            pr["translation_norm"] = float(np.linalg.norm(t_AB))
            pr["center_rmse_after"] = center_rmse
            pr["status"] = "OK"

        # Print scale stats
        scales = [gt[0] for gt in global_transforms if gt is not None]
        print(f"\n  Composed scales: min={min(scales):.4f} max={max(scales):.4f} "
              f"cv={np.std(scales)/max(np.mean(scales),1e-10):.4f}")

        # ─── Step 4: Apply global transforms ───
        all_original_frames = {}
        all_frame_sources = {}
        all_frame_local_pos = {}

        for wid, w in enumerate(windows):
            T = global_transforms[wid]
            if T is None:
                continue
            s, Q, t = T
            ext_global = apply_transform(w["ext_w2c"], s, Q, t)

            frames = w["frame_idx"]
            n = len(frames)
            for i in range(n):
                fid = int(frames[i])
                local_pos = i
                if fid in all_original_frames:
                    prev_wid = all_frame_sources[fid]
                    prev_local = all_frame_local_pos[fid]
                    prev_n = len(windows[prev_wid]["frame_idx"])
                    prev_dist = min(prev_local, prev_n - 1 - prev_local)
                    curr_dist = min(local_pos, n - 1 - local_pos)
                    if curr_dist > prev_dist:
                        all_original_frames[fid] = ext_global[i]
                        all_frame_sources[fid] = wid
                        all_frame_local_pos[fid] = local_pos
                else:
                    all_original_frames[fid] = ext_global[i]
                    all_frame_sources[fid] = wid
                    all_frame_local_pos[fid] = local_pos

        sorted_frames = sorted(all_original_frames.keys())
        n_unique = len(sorted_frames)
        S_total = max(sorted_frames) + 1 if sorted_frames else 0
        coverage = n_unique / S_total if S_total > 0 else 0

        global_ext_w2c = np.array([all_original_frames[f] for f in sorted_frames])
        original_frame_indices = np.array(sorted_frames)
        window_sources = np.array([all_frame_sources[f] for f in sorted_frames])

        # Fusion count
        from collections import Counter
        frame_counts = Counter()
        for wid, w in enumerate(windows):
            for fid in w["frame_idx"]:
                frame_counts[int(fid)] += 1
        fusion_count = np.array([frame_counts[f] for f in sorted_frames])

        # Scale drift stats
        scales = [pr.get("scale", 1.0) for pr in valid_pairs if "scale" in pr]
        scale_stats = {
            "mean": float(np.mean(scales)) if scales else 1.0,
            "std": float(np.std(scales)) if scales else 0.0,
            "cv": float(np.std(scales) / np.mean(scales)) if scales and np.mean(scales) > 0 else 0.0,
        }

        # Save
        out_path = os.path.join(OUT_DIR, f"{seq_id}_GAUGE_GLOBAL_CAMERAS.npz")
        np.savez_compressed(out_path,
            original_frame_index=original_frame_indices,
            global_extrinsic=global_ext_w2c,
            window_sources=window_sources,
            fusion_count=fusion_count,
        )

        # Save manifest
        manifest = {
            "sequence_id": seq_id,
            "method": "gauge_aware_orientation_center",
            "n_windows": len(windows),
            "n_unique_frames": n_unique,
            "coverage_ratio": coverage,
            "scale_stats": scale_stats,
            "mean_Q_dispersion": float(mean_Q_disp),
            "max_Q_dispersion": float(max_Q_disp),
            "overlap_orientation_consistency": consistency,
            "n_pairwise_alignments": len(valid_pairs),
        }
        manifest_path = os.path.join(OUT_DIR, f"{seq_id}_GAUGE_MANIFEST.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Save pair results CSV
        csv_path = os.path.join(ALIGN_DIR, f"{seq_id}_PAIRWISE_GAUGE_ALIGNMENT.csv")
        csv_fields = [
            "sequence", "window_a", "window_b", "n_overlap", "n_inliers",
            "Q_dispersion_median", "Q_dispersion_p90", "Q_dispersion_max",
            "scale", "translation_norm", "center_rmse_after",
            "orientation_residual_median", "orientation_residual_p90", "status",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for pr in valid_pairs:
                writer.writerow(pr)

        print(f"\n  Unique frames: {n_unique}/{S_total} ({coverage:.1%})")
        print(f"  Scale drift: mean={scale_stats['mean']:.4f} CV={scale_stats['cv']:.4f}")
        print(f"  Saved: {out_path}")

    # Save overall pair results
    if all_pair_results:
        csv_path = os.path.join(ALIGN_DIR, "ALL_PAIRWISE_Q_DISPERSION.csv")
        csv_fields = [
            "sequence", "window_a", "window_b", "n_overlap", "n_inliers",
            "Q_dispersion_median", "Q_dispersion_p90", "Q_dispersion_max", "status",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
            writer.writeheader()
            for pr in all_pair_results:
                writer.writerow({k: v for k, v in pr.items() if k in csv_fields})
        print(f"\nSaved all pair results: {csv_path}")


if __name__ == "__main__":
    main()
