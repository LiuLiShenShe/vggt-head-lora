#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 5: Q orientation convention — Q_ij,f = R_c2w_i,f @ R_c2w_j,f^T (both c2w).

Verifies against the real stride-4 windows: for shared frame f in windows (i,j),
recompute the per-frame Q and compare to the edge Q (robust mean) in the CSV.
"""
import csv
import glob
import os
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC = os.path.join(PHASE3C, "12_rotation_sync_v33")

EDGES_CSV = os.path.join(SYNC, "02_rotation_edges", "ROTATION_GRAPH_EDGES.csv")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4")

SEQ = "plantview__langdon_4__05-03-24"


def rot_angle_deg(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def _load_windows():
    files = sorted(glob.glob(os.path.join(WINDOW_DIR, SEQ, "window_*.npz")))
    wins = []
    for f in files:
        d = np.load(f)
        wins.append({"ext": d["ext_w2c_vggt"], "idx": d["frame_idx"]})
    return wins


def _c2w(w):
    """ext_w2c_vggt is (16,3,4) w2c → c2w rotations (16,3,3)."""
    R_w2c = w["ext"][:, :3, :3]
    return np.transpose(R_w2c, (0, 2, 1))


def _edge_q_from_csv(seq):
    with open(EDGES_CSV) as f:
        for r in csv.DictReader(f):
            if r["sequence"] == seq:
                Q = np.array([[float(r[f"q{rr}{cc}"]) for cc in range(3)] for rr in range(3)])
                return int(r["window_i"]), int(r["window_j"]), Q, int(r["n_overlap"])
    raise AssertionError("no edges found")


def test_q_equals_rc2w_i_at_rc2w_j_t():
    wins = _load_windows()
    i, j, Q_csv, n_overlap = _edge_q_from_csv(SEQ)
    fi = {int(f): k for k, f in enumerate(np.asarray(wins[i]["idx"], dtype=int))}
    fj = {int(f): k for k, f in enumerate(np.asarray(wins[j]["idx"], dtype=int))}
    overlap = sorted(set(fi) & set(fj))
    assert len(overlap) >= n_overlap * 0.5

    # Robust mean of per-frame Q's → compare to CSV edge Q
    Q_list = []
    for f in overlap:
        R_i = _c2w(wins[i])[fi[f]]
        R_j = _c2w(wins[j])[fj[f]]
        Q_f = R_i @ R_j.T
        Q_list.append(Q_f)
    Q_mean = Rotation.from_matrix(np.mean(np.array(Q_list), axis=0)).as_matrix()
    # CSV Q (which is a robust mean) must be within ~1° of the naive mean here
    err = rot_angle_deg(Q_mean.T @ Q_csv)
    assert err < 2.0, f"CSV edge Q deviates {err:.2f}° from recomputed mean"


def test_q_is_not_identity():
    """Edge Q must be a real relative rotation, not the identity (validates it carries info)."""
    i, j, Q_csv, _ = _edge_q_from_csv(SEQ)
    assert rot_angle_deg(Q_csv) > 0.5, "edge Q near-identity — suspect bogus edge"
    assert abs(1 - np.linalg.det(Q_csv)) < 1e-4, "Q not a rotation matrix"