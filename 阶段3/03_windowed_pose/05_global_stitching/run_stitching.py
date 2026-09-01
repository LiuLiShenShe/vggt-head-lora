#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C P1: Global Trajectory Stitching.

Propagates window alignment transforms to produce globally consistent cameras.
Window 0 is the global reference frame.

Sequential propagation:
  T_global_{k+1} = T_global_k ∘ S_{k→k+1}

For overlap frames: uses the prediction from the most central window
(prefer window where frame is farther from boundary).

Usage:
    python 05_global_stitching/run_stitching.py [--seq SEQUENCE_ID ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
ALIGN_DIR = os.path.join(PHASE3C, "04_window_alignment")
OUT_DIR = os.path.join(PHASE3C, "05_global_stitching")


def apply_transform(ext_w2c, s, R, t):
    """Apply Sim(3) transform T to camera extrinsics.
    T maps from window-A frame to window-B frame.
    New camera: T ∘ original camera.
    """
    # Convert w2c to c2w
    R_w2c = ext_w2c[:, :3, :3]
    t_w2c = ext_w2c[:, :3, 3]
    R_c2w = R_w2c.transpose(0, 2, 1)
    centers = np.einsum("sij,sj->si", R_c2w, -t_w2c)

    # Apply Sim(3): new_center = s * R @ center + t
    new_centers = s * (R @ centers.T).T + t

    # For rotation: R_new_c2w = R_align @ R_c2w (approximately)
    # Then convert back to w2c
    new_R_c2w = R @ R_c2w  # (n, 3, 3)
    new_R_w2c = new_R_c2w.transpose(0, 2, 1)
    new_t_w2c = -np.einsum("sij,sj->si", new_R_w2c, new_centers)

    new_ext = np.zeros_like(ext_w2c)
    new_ext[:, :3, :3] = new_R_w2c
    new_ext[:, :3, 3] = new_t_w2c
    if ext_w2c.shape[-1] == 4 and ext_w2c.shape[-2] == 4:
        new_ext[:, 3, 3] = 1
    return new_ext


def load_alignment_manifest(seq_id):
    """Load alignment results for a sequence."""
    path = os.path.join(ALIGN_DIR, f"{seq_id}_ALIGNMENT_MANIFEST.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_window_data(seq_dir, window_id):
    """Load window npz."""
    path = os.path.join(seq_dir, f"window_{window_id:03d}.npz")
    return np.load(path)


def choose_source_window(frame_idx, all_windows, window_id):
    """For overlap frames, choose the window where frame is most central."""
    best_wid = window_id
    best_dist = -1
    for wid, w_frames in all_windows.items():
        if frame_idx in w_frames:
            local_pos = list(w_frames).index(frame_idx)
            dist_to_boundary = min(local_pos, len(w_frames) - 1 - local_pos)
            if dist_to_boundary > best_dist:
                best_dist = dist_to_boundary
                best_wid = wid
    return best_wid


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

        # Load alignment manifest
        align_manifest = load_alignment_manifest(seq_id)
        if not align_manifest or not align_manifest.get("alignments"):
            print(f"  SKIP: no alignment manifest")
            continue

        # Load all windows
        window_files = sorted(glob.glob(os.path.join(seq_dir, "window_*.npz")))
        if not window_files:
            print(f"  SKIP: no window outputs")
            continue

        n_windows = len(window_files)
        windows_data = {}
        window_frame_map = {}  # window_id -> set of frame indices
        for wf in window_files:
            wid = int(os.path.basename(wf).replace("window_", "").replace(".npz", ""))
            data = np.load(wf)
            windows_data[wid] = data
            window_frame_map[wid] = set(data["frame_idx"].tolist())

        # Build sequential transforms
        # Window 0 is global reference (T_global_0 = identity)
        global_transforms = {0: np.eye(4)}
        alignment_ok = True

        for align in align_manifest["alignments"]:
            wa = align["window_a"]
            wb = align["window_b"]
            if align["status"] != "OK":
                print(f"  W{wa}->W{wb}: SKIPPED ({align['status']})")
                alignment_ok = False
                continue

            s = align["s"]
            R = np.array(align["R"])
            t = np.array(align["t"])

            # Compose: T_global_{wb} = T_global_{wa} ∘ S_{wa→wb}
            T_wa = global_transforms[wa]
            # S as 4x4 matrix
            S_mat = np.eye(4)
            S_mat[:3, :3] = s * R
            S_mat[:3, 3] = t
            global_transforms[wb] = T_wa @ S_mat

        if not alignment_ok:
            print(f"  WARNING: some alignments failed")

        # Stitch all windows to global frame
        all_original_frames = {}  # original_frame_idx -> global_ext_w2c
        all_frame_sources = {}    # original_frame_idx -> source_window_id
        all_frame_local_pos = {}  # original_frame_idx -> position in source window

        for wid in sorted(windows_data.keys()):
            data = windows_data[wid]
            ext_w2c = data["ext_w2c_vggt"]  # (n, 3, 4)
            frame_idx = data["frame_idx"]

            T_global = global_transforms.get(wid, np.eye(4))
            global_ext = apply_transform(ext_w2c, 1, T_global[:3, :3], T_global[:3, 3])

            n = len(frame_idx)
            for i in range(n):
                fid = int(frame_idx[i])
                local_pos = i
                window_len = n
                # Prefer the window where frame is more central
                if fid in all_original_frames:
                    prev_wid = all_frame_sources[fid]
                    prev_local = all_frame_local_pos[fid]
                    prev_window_len = len(windows_data[prev_wid]["frame_idx"])
                    prev_dist = min(prev_local, prev_window_len - 1 - prev_local)
                    curr_dist = min(local_pos, window_len - 1 - local_pos)
                    if curr_dist > prev_dist:
                        all_original_frames[fid] = global_ext[i]
                        all_frame_sources[fid] = wid
                        all_frame_local_pos[fid] = local_pos
                else:
                    all_original_frames[fid] = global_ext[i]
                    all_frame_sources[fid] = wid
                    all_frame_local_pos[fid] = local_pos

        # Sort by original frame index
        sorted_frames = sorted(all_original_frames.keys())
        n_unique = len(sorted_frames)
        S_total = max(sorted_frames) + 1 if sorted_frames else 0
        coverage = n_unique / S_total if S_total > 0 else 0

        # Build output arrays
        global_ext_w2c = np.array([all_original_frames[f] for f in sorted_frames])
        original_frame_indices = np.array(sorted_frames)
        window_sources = np.array([all_frame_sources[f] for f in sorted_frames])
        fusion_count = np.ones(n_unique, dtype=int)  # Default: each frame from 1 window

        # Count fusion for overlap frames
        from collections import Counter
        frame_counts = Counter()
        for wid in sorted(windows_data.keys()):
            for fid in windows_data[wid]["frame_idx"]:
                frame_counts[int(fid)] += 1
        for i, fid in enumerate(sorted_frames):
            fusion_count[i] = frame_counts[fid]

        # Compute scale drift statistics
        scales = [a["alignment_scale"] for a in align_manifest["alignments"]
                  if a["status"] == "OK"]
        scale_stats = {
            "mean": float(np.mean(scales)) if scales else -1,
            "std": float(np.std(scales)) if scales else -1,
            "cv": float(np.std(scales) / np.mean(scales)) if scales and np.mean(scales) > 0 else -1,
            "max_jump": float(np.max(np.abs(np.diff(scales)))) if len(scales) > 1 else 0,
        }

        # Save
        out_path = os.path.join(OUT_DIR, f"{seq_id}_WINDOWED_GLOBAL_CAMERAS.npz")
        np.savez_compressed(out_path,
            original_frame_index=original_frame_indices,
            global_extrinsic=global_ext_w2c,  # (n_unique, 3, 4)
            window_sources=window_sources,
            fusion_count=fusion_count,
        )

        manifest = {
            "sequence_id": seq_id,
            "n_windows": n_windows,
            "n_unique_frames": n_unique,
            "coverage_ratio": coverage,
            "scale_stats": scale_stats,
            "fusion_count_mean": float(np.mean(fusion_count)),
            "fusion_count_max": int(np.max(fusion_count)),
        }
        manifest_path = os.path.join(OUT_DIR, f"{seq_id}_STITCHING_MANIFEST.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  Windows: {n_windows}")
        print(f"  Unique frames: {n_unique}/{S_total} ({coverage:.1%})")
        print(f"  Scale drift: mean={scale_stats['mean']:.4f} CV={scale_stats['cv']:.4f} "
              f"max_jump={scale_stats['max_jump']:.4f}")
        print(f"  Fusion: mean={np.mean(fusion_count):.2f} max={np.max(fusion_count)}")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
