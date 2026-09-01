#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leakage test: verify that randomizing reference poses does NOT change stitching output."""
import numpy as np
import os, sys, json, glob

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
sys.path.insert(0, os.path.join(PHASE3C, "05_global_stitching_v31"))
from run_gauge_stitching import (
    compute_pairwise_overlap_q, w2c_centers, compose_transform,
    apply_transform, find_overlap_frames, estimate_scale_translation_fixed_Q
)
from scipy.spatial.transform import Rotation


def run_stitching_for_seq(seq_id):
    """Run gauge-aware stitching, return global cameras."""
    window_dir = os.path.join(PHASE3C, "03_window_inference", "window_outputs", seq_id)
    window_files = sorted(glob.glob(os.path.join(window_dir, "window_*.npz")))
    windows = []
    for wf in window_files:
        data = np.load(wf)
        windows.append({"ext_w2c": data["ext_w2c_vggt"], "frame_idx": data["frame_idx"]})

    pair_results = compute_pairwise_overlap_q(windows, seq_id)
    valid_pairs = [pr for pr in pair_results if "Q_star" in pr]

    global_transforms = [(1.0, np.eye(3), np.zeros(3))]
    for i, pr in enumerate(valid_pairs):
        Q_star = pr["Q_star"]
        C_A = w2c_centers(windows[pr["window_a"]]["ext_w2c"])
        C_B = w2c_centers(windows[pr["window_b"]]["ext_w2c"])
        overlap = find_overlap_frames(
            windows[pr["window_a"]]["frame_idx"],
            windows[pr["window_b"]]["frame_idx"]
        )
        frame_to_A = {int(f): idx for idx, f in enumerate(windows[pr["window_a"]]["frame_idx"])}
        frame_to_B = {int(f): idx for idx, f in enumerate(windows[pr["window_b"]]["frame_idx"])}
        inlier_frames = [f for f, m in zip(pr["frame_indices"], pr["inlier_mask"]) if m]
        if len(inlier_frames) < 2:
            global_transforms.append(None)
            continue
        C_A_overlap = np.array([C_A[frame_to_A[f]] for f in inlier_frames])
        C_B_overlap = np.array([C_B[frame_to_B[f]] for f in inlier_frames])
        s_AB, t_AB, _ = estimate_scale_translation_fixed_Q(C_A_overlap, C_B_overlap, Q_star)
        S_BtoA = (s_AB, Q_star, t_AB)
        T_prev = global_transforms[i]
        if T_prev is None:
            global_transforms.append(None)
            continue
        global_transforms.append(compose_transform(T_prev, S_BtoA))

    # Apply transforms
    all_frames = {}
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
            if fid not in all_frames:
                all_frames[fid] = ext_global[i]

    sorted_fids = sorted(all_frames.keys())
    global_ext = np.array([all_frames[f] for f in sorted_fids])
    return global_ext, np.array(sorted_fids)


def test_leakage():
    """Randomize reference poses; stitching output must NOT change."""
    seq_id = "plantview__langdon_4__05-03-24"

    # Run stitching (original)
    ext_orig, idx_orig = run_stitching_for_seq(seq_id)

    # Randomize reference poses (load, apply random transform, save)
    seq = None
    for subdir in ["plant_view", "wheat3dgs", "mustc"]:
        for jp in glob.glob(os.path.join(SEQ_BASE, subdir, "*.json")):
            with open(jp) as f:
                meta = json.load(f)
            if meta["sequence_id"] == seq_id:
                seq = meta
                break
        if seq:
            break

    ext_path = seq["extrinsics_path"]
    with open(ext_path) as f:
        ext_data = json.load(f)

    # Save original
    with open(ext_path + ".bak", "w") as f:
        json.dump(ext_data, f)

    # Apply random rotation + translation to all reference extrinsics
    R_rand = Rotation.random().as_matrix()
    t_rand = np.random.randn(3) * 100

    for e in ext_data["extrinsics"]:
        w2c = np.array(e["w2c"])
        R_w2c = w2c[:3, :3]
        t_w2c = w2c[:3, 3]
        # Random gauge: x_rand = R_rand @ x_orig + t_rand
        # R_w2c_rand = R_w2c @ R_rand.T
        # t_w2c_rand = R_w2c @ (-R_rand.T @ t_rand) + t_w2c  ... actually simpler:
        # Just change the stored extrinsics to something random
        R_new = Rotation.random().as_matrix()
        t_new = np.random.randn(3)
        e["w2c"] = (np.eye(4)).tolist()
        e["w2c"][:3] = np.hstack([R_new, t_new[:, None]]).tolist()

    with open(ext_path, "w") as f:
        json.dump(ext_data, f)

    # Run stitching again (should be identical since reference is not used)
    ext_random, idx_random = run_stitching_for_seq(seq_id)

    # Restore original
    with open(ext_path + ".bak") as f:
        orig_data = json.load(f)
    with open(ext_path, "w") as f:
        json.dump(orig_data, f)
    os.remove(ext_path + ".bak")

    # Compare
    assert np.array_equal(idx_orig, idx_random), "Frame indices differ!"
    assert np.allclose(ext_orig, ext_random, atol=1e-10), \
        f"Max diff: {np.max(np.abs(ext_orig - ext_random))}"

    print("PASS: test_leakage — reference pose randomization does NOT change stitching output")


if __name__ == "__main__":
    test_leakage()
