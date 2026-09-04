#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 7: A spanning-tree-only graph (cycle_rank=0) CANNOT remove long-chain drift —
the solver on a tree reproduces the chain error, validating that redundancy is the
mechanism (this is the scientific negative control)."""
import os
import sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
TESTS_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33", "tests")
sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, os.path.join(PHASE3C, "12_rotation_sync_v33", "04_so3_sync"))

from run_so3_sync import so3_sync_solve
from test_so3_sync_solver import make_synthetic_graph, chain_compose_error


def rot_angle_deg(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def test_tree_graph_cannot_remove_drift():
    N = 60
    G_true, ei, ej, Qs, w, _ = make_synthetic_graph(
        N=N, noise_deg=2.0, outlier_fraction=0.0, seed=99, chain_only=True)

    G_tree, _, _ = so3_sync_solve(N, ei, ej, Qs, w, anchor=0)
    tree_errs = np.array([rot_angle_deg(G_true[k].T @ G_tree[k]) for k in range(N)])
    tree_med = float(np.median(tree_errs[1:]))

    G_chain = chain_compose_error(N, ei, ej, Qs, anchor=0)
    chain_errs = np.array([rot_angle_deg(G_true[k].T @ G_chain[k]) for k in range(N)])
    chain_med = float(np.median(chain_errs[1:]))

    # Tree solver = chain reparameterization → error must be comparable (both large)
    assert tree_med > 5.0, f"tree graph should FAIL to remove drift, got {tree_med:.2f}°"
    assert tree_med / max(chain_med, 1e-9) > 0.5, \
        f"tree should be as bad as chain: tree={tree_med:.2f} chain={chain_med:.2f}"


def test_tree_solver_equals_chain_when_no_redundancy():
    """At cycle_rank=0 the least-squares solution should not beat sequential chaining."""
    N = 40
    G_true, ei, ej, Qs, w, _ = make_synthetic_graph(
        N=N, noise_deg=3.0, outlier_fraction=0.0, seed=5, chain_only=True)
    G_tree, _, _ = so3_sync_solve(N, ei, ej, Qs, w, anchor=0)
    G_chain = chain_compose_error(N, ei, ej, Qs, anchor=0)
    for k in range(1, N):
        diff = rot_angle_deg(G_tree[k].T @ G_chain[k])
        assert diff < 3.0, f"node {k}: tree vs chain differ by {diff:.2f}°"