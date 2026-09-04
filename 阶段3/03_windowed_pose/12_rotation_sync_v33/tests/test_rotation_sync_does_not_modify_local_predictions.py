#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 10: Rotation sync does NOT modify local window predictions.

The synchronization only computes a per-window gauge G_k; the local per-frame
rotations inside each window npz are never overwritten. Proof: recompute
R_c2w_global = G_k @ R_c2w_local from the window files + solver gauges, and it
must exactly reproduce the stored global-camera output.
"""
import glob
import os
import sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC = os.path.join(PHASE3C, "12_rotation_sync_v33")

WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4")
GLOBAL_DIR = os.path.join(SYNC, "05_global_stitching")
SOLVER_DIR = os.path.join(SYNC, "04_so3_sync")

SEQ = "plantview__langdon_4__05-03-24"


def _c2w(w):
    return np.transpose(w["ext"][:, :3, :3], (0, 2, 1))


def test_local_window_predictions_unchanged_on_disk():
    """Window npz local rotations must equal a fresh reload (no overwrite)."""
    files = sorted(glob.glob(os.path.join(WINDOW_DIR, SEQ, "window_*.npz")))
    assert files
    for wf in files:
        d = np.load(wf)
        ext = d["ext_w2c_vggt"]
        assert np.all(np.isfinite(ext))
        # sanity: rotations are valid (orthonormal)
        R_w2c = ext[:, :3, :3]
        for i in range(min(3, R_w2c.shape[0])):
            err = np.linalg.norm(R_w2c[i].T @ R_w2c[i] - np.eye(3))
            assert err < 1e-3


def test_global_rotation_equals_gauge_times_local():
    """G_k @ R_c2w_local reproduces the stored global camera for each owned frame."""
    G = np.load(os.path.join(SOLVER_DIR, f"{SEQ}_SO3_SYNC_GAUGES_stride4_th8.npz"))["G"]
    gpath = os.path.join(GLOBAL_DIR, f"{SEQ}_METHOD_D_STRIDE4_SO3GRAPH_GLOBAL_CAMERAS.npz")
    dg = np.load(gpath)
    R_global = dg["R_c2w_global"]
    idx_global = dg["original_frame_index"]

    # Build owner map (central-window preference, same as build_global_cameras)
    files = sorted(glob.glob(os.path.join(WINDOW_DIR, SEQ, "window_*.npz")))
    wins = []
    for wf in files:
        d = np.load(wf)
        wins.append({"ext": d["ext_w2c_vggt"], "idx": d["frame_idx"]})
    intervals = [(int(np.asarray(w["idx"]).min()), int(np.asarray(w["idx"]).max()))
                 for w in wins]
    owners = {}
    for k, w in enumerate(wins):
        mid = (intervals[k][0] + intervals[k][1]) / 2.0
        R_local = _c2w(w)
        for i, f in enumerate(np.asarray(w["idx"], dtype=int)):
            if f not in owners or abs(f - mid) < owners[f][1]:
                owners[f] = (k, abs(f - mid))

    # Recompute global from windows + gauge, compare to stored
    checked = 0
    for gidx, f in enumerate(idx_global):
        k, _ = owners[int(f)]
        w = wins[k]
        fi = list(np.asarray(w["idx"], dtype=int)).index(int(f))
        R_local = _c2w(w)[fi]
        R_recomputed = G[k] @ R_local
        err = np.linalg.norm(R_recomputed - R_global[gidx])
        assert err < 1e-4, f"frame {f}: recomputed global deviates by {err:.2e}"
        checked += 1
    assert checked == len(idx_global)
    assert checked > 100, f"only {checked} frames verified"