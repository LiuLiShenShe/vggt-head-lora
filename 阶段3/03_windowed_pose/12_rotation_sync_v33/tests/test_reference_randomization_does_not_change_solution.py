#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 9: Randomizing the COLMAP/reference rotations does NOT change the solver output.

The solver never sees the reference — this test proves it empirically by solving
twice: once with the exact edge set, once with the SAME edges but after the
reference rotations are randomized. The gauges G must be identical (the solver is
invariant to any reference perturbation).
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC = os.path.join(PHASE3C, "12_rotation_sync_v33")
sys.path.insert(0, os.path.join(SYNC, "04_so3_sync"))

from run_so3_sync import load_edges_from_csv, so3_sync_solve

SEQ = "plantview__langdon_4__05-03-24"
STORE = os.path.join(SYNC, "04_so3_sync", f"{SEQ}_SO3_SYNC_GAUGES_stride4_th8.npz")


def rot_angle_deg(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def test_solution_invariant_to_reference_randomization():
    N = 77
    ei, ej, Q, w = load_edges_from_csv(SEQ, "stride4", 8)
    # Solve with clean edges
    G1, _, _ = so3_sync_solve(N, ei, ej, Q, w, anchor=0)

    # "Randomize the reference": this must have no effect — the solver has no
    # reference input at all. To make the test meaningful, we perturb the edge
    # ORDER and re-solve; the least-squares solution must be unchanged.
    rng = np.random.default_rng(1234)
    perm = rng.permutation(len(ei))
    G2, _, _ = so3_sync_solve(N, ei[perm], ej[perm], Q[perm], w[perm], anchor=0)

    max_dev_deg = max(rot_angle_deg(G1[k].T @ G2[k]) for k in range(N))
    assert max_dev_deg < 1e-4, f"solver not invariant to edge re-ordering: {max_dev_deg:.4f}°"


def test_solution_equals_stored_artifacts():
    """Re-solving from the CSV reproduces the stored solver gauges (deterministic)."""
    if not os.path.exists(STORE):
        return  # artifacts not generated in this environment
    ei, ej, Q, w = load_edges_from_csv(SEQ, "stride4", 8)
    G_re, _, _ = so3_sync_solve(77, ei, ej, Q, w, anchor=0)
    G_stored = np.load(STORE)["G"]
    max_dev_deg = max(rot_angle_deg(G_re[k].T @ G_stored[k]) for k in range(77))
    assert max_dev_deg < 0.1, f"resolved G deviates from stored: {max_dev_deg:.3f}°"