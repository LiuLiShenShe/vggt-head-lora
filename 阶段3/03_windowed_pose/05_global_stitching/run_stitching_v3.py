#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C P1-v3: Global Trajectory Stitching with correct Sim(3) application.

VGGT windows are in different coordinate frames (overlap RMSE ~0.7-0.9m).
Alignment IS needed. Bug in v1: applied Sim(3) rotation to w2c R incorrectly.

Correct approach:
1. For each window, extract camera centers in that window's local frame
2. Compute Sim(3) S_{k→k+1} mapping overlap centers from frame_k to frame_{k+1}
3. Invert to get S_{k+1→k}: mapping from frame_{k+1} to frame_k
4. Propagate: for window k+1 cameras, transform centers and rotations

Key fix: R_global = S_R @ R_local (not R_local @ S_R)

Usage:
    python 05_global_stitching/run_stitching_v3.py [--seq SEQUENCE_ID ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
OUT_DIR = os.path.join(PHASE3C, "05_global_stitching")


def w2c_centers(ext_w2c_3x4):
    R = ext_w2c_3x4[:, :3, :3]
    t = ext_w2c_3x4[:, :3, 3]
    return np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)


def find_overlap_frames(frames_a, frames_b):
    set_a = set(frames_a.tolist()) if isinstance(frames_a, np.ndarray) else set(frames_a)
    set_b = set(frames_b.tolist()) if isinstance(frames_b, np.ndarray) else set(frames_b)
    return sorted(set_a & set_b)


def umeyama_sim3(src, dst):
    """Compute Sim(3): dst ≈ s * R @ src + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / max((sc ** 2).sum() / len(src), 1e-10)
    t = mu_d - s * R @ mu_s
    return s, R, t


def compute_alignment(windows):
    """Compute pairwise Sim(3) alignments between consecutive windows.
    Returns list of (s, R, t) where S maps frame_k+1 centers INTO frame_k.
    """
    alignments = []
    for i in range(len(windows) - 1):
        overlap = find_overlap_frames(windows[i]["frame_idx"], windows[i+1]["frame_idx"])
        if len(overlap) < 2:
            alignments.append(None)
            continue

        centers_a = w2c_centers(windows[i]["ext_w2c"])
        centers_b = w2c_centers(windows[i+1]["ext_w2c"])

        frame_to_a = {int(f): idx for idx, f in enumerate(windows[i]["frame_idx"])}
        frame_to_b = {int(f): idx for idx, f in enumerate(windows[i+1]["frame_idx"])}

        pts_a = np.array([centers_a[frame_to_a[f]] for f in overlap])
        pts_b = np.array([centers_b[frame_to_b[f]] for f in overlap])

        # S_{a→b}: maps frame_a centers to frame_b
        s_ab, R_ab, t_ab = umeyama_sim3(pts_a, pts_b)

        # Invert to get S_{b→a}: maps frame_b centers to frame_a
        R_ba = R_ab.T
        s_ba = 1.0 / s_ab
        t_ba = -s_ba * R_ba @ t_ab

        # Verify
        pts_b_in_a = s_ba * (R_ba @ pts_b.T).T + t_ba
        rmse = float(np.sqrt(np.mean(np.linalg.norm(pts_b_in_a - pts_a, axis=1)**2)))

        alignments.append((s_ba, R_ba, t_ba, rmse))
    return alignments


def apply_sim3_to_window(ext_w2c, s, R, t):
    """Apply Sim(3) transform to all cameras in a window.

    Transform maps from window's local frame to global frame.
    For each camera:
      center_global = s * R @ center_local + t
      R_c2w_global = R @ R_c2w_local
    """
    n = ext_w2c.shape[0]
    R_w2c = ext_w2c[:, :3, :3]
    t_w2c = ext_w2c[:, :3, 3]

    # c2w rotation
    R_c2w = R_w2c.transpose(0, 2, 1)
    # Camera centers in local frame
    centers_local = np.einsum("sij,sj->si", R_c2w, -t_w2c)

    # Transform centers to global
    centers_global = s * (R @ centers_local.T).T + t

    # Transform rotations: R_c2w_global = S_R @ R_c2w_local
    R_c2w_global = np.einsum("ij,sjk->sik", R, R_c2w)

    # Convert back to w2c
    R_w2c_global = R_c2w_global.transpose(0, 2, 1)
    t_w2c_global = -np.einsum("sij,sj->si", R_w2c_global, centers_global)

    new_ext = np.zeros_like(ext_w2c)
    new_ext[:, :3, :3] = R_w2c_global
    new_ext[:, :3, 3] = t_w2c_global
    if ext_w2c.ndim == 3 and ext_w2c.shape[1] == 4 and ext_w2c.shape[2] == 4:
        new_ext[:, 3, 3] = 1
    return new_ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.seq:
        seq_dirs = [os.path.join(WINDOW_DIR, s) for s in args.seq]
    else:
        seq_dirs = sorted([os.path.join(WINDOW_DIR, d)
                          for d in os.listdir(WINDOW_DIR)
                          if os.path.isdir(os.path.join(WINDOW_DIR, d))])

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

        # Compute alignments
        alignments = compute_alignment(windows)

        # Propagate: Window 0 is global reference (identity transform)
        # T_i maps from window_i's frame to global (window_0's frame)
        global_transforms = [(1.0, np.eye(3), np.zeros(3))]  # (s, R, t) for W0 = identity

        for i, align in enumerate(alignments):
            if align is None:
                print(f"  W{i}->W{i+1}: FAILED (insufficient overlap)")
                global_transforms.append(None)
                continue

            s_ba, R_ba, t_ba, rmse = align

            # T_{i+1} = T_i ∘ S_{i+1→i}
            # But we need to compose correctly:
            # Window i+1 cameras are in frame_{i+1}
            # S_{i+1→i} maps frame_{i+1} → frame_i
            # T_i maps frame_i → global
            # So T_{i+1} = compose(T_i, S_{i+1→i})

            T_prev = global_transforms[i]
            s_prev, R_prev, t_prev = T_prev

            # Compose: s_new = s_prev * s_ba
            # R_new = R_prev @ R_ba
            # t_new = s_prev * R_prev @ t_ba + t_prev
            s_new = s_prev * s_ba
            R_new = R_prev @ R_ba
            t_new = s_prev * (R_prev @ t_ba) + t_prev

            global_transforms.append((s_new, R_new, t_new))
            print(f"  W{i:2d}->W{i+1:2d}: RMSE_before={rmse:.4f} scale={s_ba:.4f}")

        # Apply global transforms to all windows
        all_original_frames = {}
        all_frame_sources = {}
        all_frame_local_pos = {}

        for wid, w in enumerate(windows):
            T = global_transforms[wid]
            if T is None:
                continue
            s, R, t = T
            ext_global = apply_sim3_to_window(w["ext_w2c"], s, R, t)

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

        from collections import Counter
        frame_counts = Counter()
        for wid, w in enumerate(windows):
            for fid in w["frame_idx"]:
                frame_counts[int(fid)] += 1
        fusion_count = np.array([frame_counts[f] for f in sorted_frames])

        # Scale drift stats
        scales = []
        for align in alignments:
            if align is not None:
                scales.append(align[0])  # s_ba
        scale_stats = {
            "mean": float(np.mean(scales)) if scales else 1.0,
            "std": float(np.std(scales)) if scales else 0.0,
            "cv": float(np.std(scales) / np.mean(scales)) if scales and np.mean(scales) > 0 else 0.0,
            "max_jump": float(np.max(np.abs(np.diff(scales)))) if len(scales) > 1 else 0,
        }

        out_path = os.path.join(OUT_DIR, f"{seq_id}_WINDOWED_GLOBAL_CAMERAS.npz")
        np.savez_compressed(out_path,
            original_frame_index=original_frame_indices,
            global_extrinsic=global_ext_w2c,
            window_sources=window_sources,
            fusion_count=fusion_count,
        )

        manifest = {
            "sequence_id": seq_id,
            "n_windows": len(windows),
            "n_unique_frames": n_unique,
            "coverage_ratio": coverage,
            "scale_stats": scale_stats,
            "fusion_count_mean": float(np.mean(fusion_count)),
            "fusion_count_max": int(np.max(fusion_count)),
            "alignment_method": "umeyama_sim3_center",
        }
        manifest_path = os.path.join(OUT_DIR, f"{seq_id}_STITCHING_MANIFEST.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  Unique frames: {n_unique}/{S_total} ({coverage:.1%})")
        print(f"  Scale drift: mean={scale_stats['mean']:.4f} CV={scale_stats['cv']:.4f} "
              f"max_jump={scale_stats['max_jump']:.4f}")


if __name__ == "__main__":
    main()
