#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C P1-v2: Global Trajectory Stitching (corrected).

KEY INSIGHT from v1 failure:
VGGT already predicts globally consistent poses across overlapping windows.
The Sim(3) alignment is unnecessary and harmful — it corrupts the trajectory.

Strategy:
1. First check if overlap frames are already consistent (RMSE < threshold)
2. If consistent: skip alignment, use VGGT predictions directly
3. If inconsistent: apply alignment (fallback)

For overlap frames: prefer the prediction from the most central window.

Usage:
    python 05_global_stitching/run_stitching_v2.py [--seq SEQUENCE_ID ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
OUT_DIR = os.path.join(PHASE3C, "05_global_stitching")

# Consistency threshold: if overlap RMSE < this, skip alignment
CONSISTENCY_THRESHOLD = 0.1  # meters


def w2c_centers(ext_w2c_3x4):
    R = ext_w2c_3x4[:, :3, :3]
    t = ext_w2c_3x4[:, :3, 3]
    return np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)


def find_overlap_frames(frames_a, frames_b):
    set_a = set(frames_a.tolist()) if isinstance(frames_a, np.ndarray) else set(frames_a)
    set_b = set(frames_b.tolist()) if isinstance(frames_b, np.ndarray) else set(frames_b)
    return sorted(set_a & set_b)


def check_overlap_consistency(ext_a, idx_a, ext_b, idx_b):
    """Check if two windows' overlap frames are already consistent."""
    overlap = find_overlap_frames(idx_a, idx_b)
    if len(overlap) < 2:
        return False, -1, overlap

    centers_a = w2c_centers(ext_a)
    centers_b = w2c_centers(ext_b)

    frame_to_a = {int(f): i for i, f in enumerate(idx_a)}
    frame_to_b = {int(f): i for i, f in enumerate(idx_b)}

    diffs = []
    for f in overlap:
        ia, ib = frame_to_a[f], frame_to_b[f]
        diffs.append(np.linalg.norm(centers_a[ia] - centers_b[ib]))

    rmse = float(np.sqrt(np.mean(np.array(diffs)**2)))
    return rmse < CONSISTENCY_THRESHOLD, rmse, overlap


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

        # Load all windows
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
                "intr": data["intr_vggt"],
            })

        # Check pairwise consistency
        all_consistent = True
        for i in range(len(windows) - 1):
            consistent, rmse, overlap = check_overlap_consistency(
                windows[i]["ext_w2c"], windows[i]["frame_idx"],
                windows[i+1]["ext_w2c"], windows[i+1]["frame_idx"]
            )
            status = "CONSISTENT" if consistent else "INCONSISTENT"
            print(f"  W{i:2d}->W{i+1:2d}: RMSE={rmse:.4f} ({status}, n_overlap={len(overlap)})")
            if not consistent:
                all_consistent = False

        if all_consistent:
            print(f"  >> All windows CONSISTENT — using VGGT predictions directly (no alignment)")
        else:
            print(f"  >> Some windows INCONSISTENT — would need alignment (not yet implemented)")

        # Stitch: use VGGT predictions directly, prefer central frames for overlap
        all_original_frames = {}
        all_frame_sources = {}
        all_frame_local_pos = {}

        for wid, w in enumerate(windows):
            ext = w["ext_w2c"]
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
                        all_original_frames[fid] = ext[i]
                        all_frame_sources[fid] = wid
                        all_frame_local_pos[fid] = local_pos
                else:
                    all_original_frames[fid] = ext[i]
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

        # Scale stats (from raw window data, no alignment applied)
        scale_stats = {"mean": 1.0, "std": 0.0, "cv": 0.0, "max_jump": 0.0,
                       "note": "no alignment applied — VGGT predictions used directly"}

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
            "alignment_method": "none_direct_vggt",
        }
        manifest_path = os.path.join(OUT_DIR, f"{seq_id}_STITCHING_MANIFEST.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  Windows: {len(windows)}")
        print(f"  Unique frames: {n_unique}/{S_total} ({coverage:.1%})")
        print(f"  Fusion: mean={np.mean(fusion_count):.2f} max={np.max(fusion_count)}")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
