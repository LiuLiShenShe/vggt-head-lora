#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.2 Step 7: Rebuild global trajectory with synced scales.

Uses Phase 3C.1 gauge-aware rotations (Q chain) + center-scale chain.
Also tries: dense-scale chain, and per-window scale normalization.

For each method, applies synced scale to each window's Sim(3) transform,
then evaluates against COLMAP reference.

Usage:
    python 05_global_reconstruction/rebuild_trajectory.py [--seq SEQ ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
GAUGE_DIR = os.path.join(PHASE3C, "05_global_stitching_v31")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
OUT_DIR = os.path.join(PHASE3C, "11_scale_sync_v32", "05_global_reconstruction")

sys.path.insert(0, os.path.join(PHASE3C, "05_global_stitching_v31"))
from run_gauge_stitching import (
    w2c_centers, c2w_rotations, find_overlap_frames,
    compute_pairwise_overlap_q, estimate_scale_translation_fixed_Q,
    compose_transform, apply_transform, so3_robust_mean
)
sys.path.insert(0, os.path.join(PHASE3B, "03_pose_evaluation"))
from evaluate_multoplant import (
    global_rotation_procrustes, rot_angle_deg, horn_sim3
)

os.makedirs(OUT_DIR, exist_ok=True)


def find_sequence_json(seq_id):
    for subdir in ["plant_view", "wheat3dgs", "mustc"]:
        for jp in glob.glob(os.path.join(SEQ_BASE, subdir, "*.json")):
            with open(jp) as f:
                meta = json.load(f)
            if meta["sequence_id"] == seq_id:
                return meta
    raise FileNotFoundError(f"Sequence JSON not found for {seq_id}")


def load_reference_poses(seq):
    ext_path = seq.get("extrinsics_path")
    if not ext_path or not os.path.exists(ext_path):
        return None
    with open(ext_path) as f:
        ext_data = json.load(f)
    return np.array([np.array(e["w2c"])[:3, :4] for e in ext_data["extrinsics"]])


def evaluate_trajectory(ref_w2c, vggt_ext, orig_idx):
    """Evaluate stitched trajectory against reference."""
    ref_sub = ref_w2c[orig_idx]
    n = min(len(ref_sub), len(vggt_ext))
    ref_sub = ref_sub[:n]
    vggt_sub = vggt_ext[:n]

    R_ref = ref_sub[:, :3, :3].transpose(0, 2, 1)
    R_vggt = vggt_sub[:, :3, :3].transpose(0, 2, 1)
    Rg = global_rotation_procrustes(R_vggt, R_ref)

    rot_errors = np.array([rot_angle_deg((Rg @ R_vggt[i]).T @ R_ref[i]) for i in range(n)])
    centers_ref = w2c_centers(ref_sub)
    centers_vggt = w2c_centers(vggt_sub)
    s, R_s, t_s = horn_sim3(centers_vggt, centers_ref)
    centers_aligned = s * (R_s @ centers_vggt.T).T + t_s

    center_diff = centers_aligned - centers_ref
    ref_spread = np.linalg.norm(centers_ref - centers_ref.mean(0), axis=1).mean()
    cen_norm = np.linalg.norm(center_diff, axis=1) / max(ref_spread, 1e-10)

    tc = 0.0
    if n > 1:
        dv = np.diff(centers_aligned, axis=0)
        dr = np.diff(centers_ref, axis=0)
        cosines = [np.dot(dv[i], dr[i]) / (np.linalg.norm(dv[i]) * np.linalg.norm(dr[i]))
                   for i in range(len(dv))
                   if np.linalg.norm(dv[i]) > 1e-10 and np.linalg.norm(dr[i]) > 1e-10]
        tc = float(np.mean(cosines)) if cosines else 0.0

    rot_med = float(np.median(rot_errors))
    rot_p90 = float(np.percentile(rot_errors, 90))
    orient_gate = rot_med <= 10.0 and rot_p90 <= 20.0

    return {
        "n_frames": n,
        "rot_median": rot_med,
        "rot_p90": rot_p90,
        "center_median_norm": float(np.median(cen_norm)),
        "center_p90_norm": float(np.percentile(cen_norm, 90)),
        "trajectory_cosine": tc,
        "scale_estimated": float(s),
        "orient_gate": "PASS" if orient_gate else "FAIL",
    }


def rebuild_chain(windows, pair_results, scale_method="center"):
    """Rebuild trajectory using sequential Sim(3) chain with given scale method.

    scale_method: "center" (camera centers), "dense" (point_map), "uniform" (s=1)
    """
    valid_pairs = [pr for pr in pair_results if "Q_star" in pr]

    # Build per-pair scale lookup from CSV
    global_transforms = [(1.0, np.eye(3), np.zeros(3))]

    for i, pr in enumerate(valid_pairs):
        a, b = pr["window_a"], pr["window_b"]
        Q_star = pr["Q_star"]

        C_A = w2c_centers(windows[a]["ext_w2c"])
        C_B = w2c_centers(windows[b]["ext_w2c"])
        inlier_frames = [f for f, m in zip(pr["frame_indices"], pr["inlier_mask"]) if m]
        frame_to_A = {int(f): idx for idx, f in enumerate(windows[a]["frame_idx"])}
        frame_to_B = {int(f): idx for idx, f in enumerate(windows[b]["frame_idx"])}

        if len(inlier_frames) < 2:
            global_transforms.append(None)
            continue

        C_A_ov = np.array([C_A[frame_to_A[f]] for f in inlier_frames])
        C_B_ov = np.array([C_B[frame_to_B[f]] for f in inlier_frames])

        if scale_method == "center":
            s_AB, t_AB, _ = estimate_scale_translation_fixed_Q(C_A_ov, C_B_ov, Q_star)
        elif scale_method == "uniform":
            s_AB = 1.0
            # Still need t: solve C_A = Q * C_B + t
            X = (Q_star @ C_B_ov.T).T
            s_AB = 1.0
            t_AB = C_A_ov.mean(0) - X.mean(0)
        elif scale_method == "dense":
            # Use dense point_map scale (load from CSV)
            s_AB = None  # Will be loaded externally
            if s_AB is None:
                s_AB, t_AB, _ = estimate_scale_translation_fixed_Q(C_A_ov, C_B_ov, Q_star)
        else:
            s_AB, t_AB, _ = estimate_scale_translation_fixed_Q(C_A_ov, C_B_ov, Q_star)

        S_BtoA = (s_AB, Q_star, t_AB)
        T_prev = global_transforms[i]
        if T_prev is None:
            global_transforms.append(None)
            continue
        global_transforms.append(compose_transform(T_prev, S_BtoA))

    return global_transforms


def apply_and_save(windows, global_transforms, seq_id, method_name):
    """Apply transforms and save global cameras."""
    all_frames = {}
    for wid, w in enumerate(windows):
        T = global_transforms[wid] if wid < len(global_transforms) else None
        if T is None:
            continue
        s, Q, t = T
        ext_global = apply_transform(w["ext_w2c"], s, Q, t)
        for i in range(len(w["frame_idx"])):
            fid = int(w["frame_idx"][i])
            if fid not in all_frames:
                all_frames[fid] = ext_global[i]

    sorted_fids = sorted(all_frames.keys())
    global_ext = np.array([all_frames[f] for f in sorted_fids])
    orig_idx = np.array(sorted_fids)

    out_path = os.path.join(OUT_DIR, f"{seq_id}_{method_name}_GLOBAL_CAMERAS.npz")
    np.savez(out_path,
             global_extrinsic=global_ext,
             original_frame_index=orig_idx)
    return global_ext, orig_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()

    gauge_files = sorted(glob.glob(os.path.join(GAUGE_DIR, "*_GAUGE_GLOBAL_CAMERAS.npz")))
    if args.seq:
        gauge_files = [g for g in gauge_files if any(seq in g for seq in args.seq)]

    all_results = []

    for gauge_path in gauge_files:
        seq_id = os.path.basename(gauge_path).replace("_GAUGE_GLOBAL_CAMERAS.npz", "")
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")

        # Load windows
        seq_window_dir = os.path.join(WINDOW_DIR, seq_id)
        window_files = sorted(glob.glob(os.path.join(seq_window_dir, "window_*.npz")))
        if len(window_files) < 2:
            continue

        windows = []
        for wf in window_files:
            data = np.load(wf)
            windows.append({
                "ext_w2c": data["ext_w2c_vggt"],
                "frame_idx": data["frame_idx"],
            })

        # Load reference
        try:
            seq = find_sequence_json(seq_id)
        except FileNotFoundError:
            continue
        ref_w2c = load_reference_poses(seq)
        if ref_w2c is None:
            continue

        # Pairwise Q
        pair_results = compute_pairwise_overlap_q(windows, seq_id)

        # Method 1: Center-scale chain (baseline from Phase 3C.1)
        print(f"\n  Method: CENTER_SCALE_CHAIN")
        gt_center = rebuild_chain(windows, pair_results, scale_method="center")
        ext_c, idx_c = apply_and_save(windows, gt_center, seq_id, "CENTER_CHAIN")
        ev_c = evaluate_trajectory(ref_w2c, ext_c, idx_c)
        ev_c["method"] = "center_chain"
        ev_c["sequence_id"] = seq_id
        all_results.append(ev_c)
        print(f"    rot_med={ev_c['rot_median']:.2f}° center={ev_c['center_median_norm']:.4f} "
              f"tc={ev_c['trajectory_cosine']:.4f} gate={ev_c['orient_gate']}")

        # Method 2: Uniform scale chain (s=1 for all pairs — tests if scale is the issue)
        print(f"\n  Method: UNIFORM_SCALE_CHAIN (s=1)")
        gt_uniform = rebuild_chain(windows, pair_results, scale_method="uniform")
        ext_u, idx_u = apply_and_save(windows, gt_uniform, seq_id, "UNIFORM_CHAIN")
        ev_u = evaluate_trajectory(ref_w2c, ext_u, idx_u)
        ev_u["method"] = "uniform_chain"
        ev_u["sequence_id"] = seq_id
        all_results.append(ev_u)
        print(f"    rot_med={ev_u['rot_median']:.2f}° center={ev_u['center_median_norm']:.4f} "
              f"tc={ev_u['trajectory_cosine']:.4f} gate={ev_u['orient_gate']}")

        # Method 3: Per-window scale normalization
        # Estimate per-window scale from log-space least squares, then normalize
        print(f"\n  Method: PER_WINDOW_SCALE_NORMALIZATION")
        valid_pairs = [pr for pr in pair_results if "Q_star" in pr]

        # Build pairwise scale measurements
        measurements = []
        for pr in valid_pairs:
            a, b = pr["window_a"], pr["window_b"]
            Q_star = pr["Q_star"]
            C_A = w2c_centers(windows[a]["ext_w2c"])
            C_B = w2c_centers(windows[b]["ext_w2c"])
            inlier_frames = [f for f, m in zip(pr["frame_indices"], pr["inlier_mask"]) if m]
            frame_to_A = {int(f): idx for idx, f in enumerate(windows[a]["frame_idx"])}
            frame_to_B = {int(f): idx for idx, f in enumerate(windows[b]["frame_idx"])}
            if len(inlier_frames) < 2:
                continue
            C_A_ov = np.array([C_A[frame_to_A[f]] for f in inlier_frames])
            C_B_ov = np.array([C_B[frame_to_B[f]] for f in inlier_frames])
            s_AB, _, rmse = estimate_scale_translation_fixed_Q(C_A_ov, C_B_ov, Q_star)
            measurements.append((a, b, s_AB, rmse))

        if measurements:
            # Log-space least squares: log(s_ab) = alpha_b - alpha_a
            # Solve for alpha_k (per-window log scale), alpha_0 = 0
            n_w = len(windows)
            n_m = len(measurements)

            # Build linear system: A @ alpha = b
            A = np.zeros((n_m, n_w))
            b_vec = np.zeros(n_m)
            weights = np.zeros(n_m)

            for idx, (a, b, s_ab, rmse) in enumerate(measurements):
                A[idx, b] = 1.0
                A[idx, a] = -1.0
                b_vec[idx] = np.log(max(s_ab, 1e-6))
                weights[idx] = 1.0 / max(rmse, 1e-6)

            # Pin alpha_0 = 0: remove column 0, set alpha_0 = 0
            A_reduced = A[:, 1:]
            W = np.diag(weights)
            # Weighted least squares
            AtWA = A_reduced.T @ W @ A_reduced
            AtWb = A_reduced.T @ W @ b_vec
            alpha_reduced = np.linalg.lstsq(AtWA, AtWb, rcond=None)[0]
            alphas = np.concatenate([[0.0], alpha_reduced])
            per_window_scales = np.exp(alphas)

            print(f"    Per-window scales: [{per_window_scales.min():.4f}, {per_window_scales.max():.4f}]")
            print(f"    Scale range: {per_window_scales.max()/per_window_scales.min():.2f}x")

            # Rebuild with per-window scales
            # For each pair, use s_center / (per_window_scale_b / per_window_scale_a)
            # to normalize the pairwise scale
            gt_pws = rebuild_chain(windows, pair_results, scale_method="center")
            # Override scales with per-window normalization
            global_transforms_pws = [(1.0, np.eye(3), np.zeros(3))]
            for i, pr in enumerate(valid_pairs):
                a, b = pr["window_a"], pr["window_b"]
                Q_star = pr["Q_star"]
                C_A = w2c_centers(windows[a]["ext_w2c"])
                C_B = w2c_centers(windows[b]["ext_w2c"])
                inlier_frames = [f for f, m in zip(pr["frame_indices"], pr["inlier_mask"]) if m]
                frame_to_A = {int(f): idx for idx, f in enumerate(windows[a]["frame_idx"])}
                frame_to_B = {int(f): idx for idx, f in enumerate(windows[b]["frame_idx"])}
                if len(inlier_frames) < 2:
                    global_transforms_pws.append(None)
                    continue
                C_A_ov = np.array([C_A[frame_to_A[f]] for f in inlier_frames])
                C_B_ov = np.array([C_B[frame_to_B[f]] for f in inlier_frames])
                s_raw, t_raw, _ = estimate_scale_translation_fixed_Q(C_A_ov, C_B_ov, Q_star)
                # Normalize: s_normalized = s_raw * (per_window_scale_a / per_window_scale_b)
                s_norm = s_raw * (per_window_scales[a] / per_window_scales[b])
                S_BtoA = (s_norm, Q_star, t_raw)
                T_prev = global_transforms_pws[i]
                if T_prev is None:
                    global_transforms_pws.append(None)
                    continue
                global_transforms_pws.append(compose_transform(T_prev, S_BtoA))

            ext_pws, idx_pws = apply_and_save(windows, global_transforms_pws, seq_id, "PER_WINDOW_SCALE")
            ev_pws = evaluate_trajectory(ref_w2c, ext_pws, idx_pws)
            ev_pws["method"] = "per_window_scale"
            ev_pws["sequence_id"] = seq_id
            all_results.append(ev_pws)
            print(f"    rot_med={ev_pws['rot_median']:.2f}° center={ev_pws['center_median_norm']:.4f} "
                  f"tc={ev_pws['trajectory_cosine']:.4f} gate={ev_pws['orient_gate']}")

        # Method 4: Phase 3C.1 result (reference)
        data = np.load(gauge_path)
        gauge_ext = data["global_extrinsic"]
        orig_idx = data["original_frame_index"]
        ev_orig = evaluate_trajectory(ref_w2c, gauge_ext, orig_idx)
        ev_orig["method"] = "gauge_aware_original"
        ev_orig["sequence_id"] = seq_id
        all_results.append(ev_orig)
        print(f"\n  Original (Phase 3C.1): rot_med={ev_orig['rot_median']:.2f}° "
              f"center={ev_orig['center_median_norm']:.4f} tc={ev_orig['trajectory_cosine']:.4f}")

    # Summary table
    if all_results:
        print(f"\n{'='*90}")
        print("METHOD COMPARISON")
        print(f"{'='*90}")
        print(f"{'Sequence':<35s} {'Method':<25s} {'Rot':>6s} {'Center':>8s} {'TC':>6s} {'Gate'}")
        print("-" * 90)
        for r in all_results:
            short = r["sequence_id"].replace("plantview__langdon_4__", "lang4__")
            print(f"  {short:<33s} {r['method']:<25s} {r['rot_median']:5.1f}° "
                  f"{r['center_median_norm']:7.4f} {r['trajectory_cosine']:+5.3f} {r['orient_gate']}")

        # Save CSV
        csv_path = os.path.join(OUT_DIR, "REBUILD_COMPARISON.csv")
        fields = ["sequence_id", "method", "n_frames", "rot_median", "rot_p90",
                  "center_median_norm", "center_p90_norm", "trajectory_cosine",
                  "scale_estimated", "orient_gate"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
