#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic validation of the SO(3) graph synchronization solver.

Tests:
1. Zero-noise exact recovery (cycle graph)
2. 1-3° noise recovery: graph sync << chain error
3. 5% outlier robustness with Huber
4. Spanning-tree (cycle_rank=0) cannot remove drift

Run:
    cd 阶段3/03_windowed_pose && pytest 12_rotation_sync_v33/tests/test_so3_sync_solver.py -v
"""
import os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
sys.path.insert(0, os.path.join(PHASE3C, "12_rotation_sync_v33", "04_so3_sync"))
from run_so3_sync import so3_sync_solve, mst_init

from scipy.spatial.transform import Rotation


def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def make_synthetic_graph(N=50, noise_deg=2.0, outlier_fraction=0.05, seed=1234,
                         chain_only=False):
    """Generate a synthetic SO(3) averaging problem matching stride-4 structure.

    Chain-only: edges (k, k+1) only → cycle_rank = 0.
    Full: banded edges (k, k+1) and (k, k+2) → redundant cycles.
    """
    rng = np.random.default_rng(seed)

    # Ground-truth gauge rotations (random)
    G_true = np.array([Rotation.random(random_state=rng).as_matrix() for _ in range(N)])
    G_true[0] = np.eye(3)

    # Edges mimicking stride-4 (hop ≤ 2)
    edges = []
    for k in range(N):
        for hop in [1, 2]:
            if chain_only and hop != 1:
                continue
            if k + hop < N:
                edges.append((k, k + hop))

    # Build Q observations with noise + outliers
    Q_obs = {}
    n_out = 0
    for (i, j) in edges:
        Q_true_ij = G_true[i].T @ G_true[j]
        if rng.random() < outlier_fraction:
            # Outlier: random rotation (no relationship to ground truth)
            Q_obs[(i, j)] = Rotation.random(random_state=rng).as_matrix()
            n_out += 1
        else:
            noise = Rotation.from_rotvec(
                np.radians(noise_deg) * rng.normal(size=3)
            ).as_matrix()
            Q_obs[(i, j)] = noise @ Q_true_ij

    # Edge lists for solver (i<j ordering, mirrored for Q orientation)
    edges_i = np.array([min(a, b) for (a, b) in edges])
    edges_j = np.array([max(a, b) for (a, b) in edges])
    Qs = np.array([Q_obs[(a, b)] for (a, b) in edges])
    # weight: 1.0 for all (no n_overlap info in synthetic)
    weights = np.ones(len(edges))

    return G_true, edges_i, edges_j, Qs, weights, n_out


def chain_compose_error(N, edges_i, edges_j, Qs, anchor=0):
    """Sequential chain error: compose adjacent edges from anchor, compare to GT."""
    G = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
    # Traverse adjacency order (edges sorted by i)
    for k in range(anchor, N - 1):
        idx = np.where((edges_i == k) & (edges_j == k + 1))[0]
        if len(idx):
            G[k + 1] = G[k] @ Qs[idx[0]]
        else:
            G[k + 1] = G[k]
    return G


def test_zero_noise_exact_recovery():
    """Zero noise → solver recovers G within 1e-3 deg (up to right-global-gauge)."""
    N = 30
    G_true, ei, ej, Qs, w, n_out = make_synthetic_graph(
        N=N, noise_deg=0.0, outlier_fraction=0.0, seed=10)
    G, stats, per_edge = so3_sync_solve(N, ei, ej, Qs, w, anchor=0)
    # Compare G to G_true up to global gauge (both anchor at G_0=I; align via node 0)
    # Since G_true[0]=I and our G[0]=I, direct comparison valid
    errs = [rot_angle_deg(G_true[k].T @ G[k]) for k in range(N)]
    assert max(errs) < 1e-3, f"zero-noise recovery failed, max_err={max(errs):.4f} deg"


def test_noise_recovery_sgraph_beats_chain():
    """1-3° noise: graph sync << chain error (redundant graph matters)."""
    N = 50
    G_true, ei, ej, Qs, w, n_out = make_synthetic_graph(
        N=N, noise_deg=2.0, outlier_fraction=0.0, seed=42)

    # Graph sync
    G_graph, stats, per_edge = so3_sync_solve(N, ei, ej, Qs, w, anchor=0)
    # Direct comparison (anchor G_0=I both sides)
    graph_errs = [rot_angle_deg(G_true[k].T @ G_graph[k]) for k in range(N)]
    graph_med = float(np.median(graph_errs[1:]))  # skip anchor

    # Chain: compose only adjacent edges
    G_chain = chain_compose_error(N, ei, ej, Qs, anchor=0)
    chain_errs = [rot_angle_deg(G_true[k].T @ G_chain[k]) for k in range(N)]
    chain_med = float(np.median(chain_errs[1:]))

    print(f"  graph_med_err={graph_med:.3f}°  chain_med_err={chain_med:.3f}°")
    assert graph_med < chain_med * 0.5, \
        f"graph should beat chain: graph={graph_med:.3f} chain={chain_med:.3f}"
    assert graph_med < 5.0, f"graph error too high: {graph_med:.3f}°"


def test_outlier_robustness_huber():
    """5% outliers: graph sync survives (error still small)."""
    N = 40
    G_true, ei, ej, Qs, w, n_out = make_synthetic_graph(
        N=N, noise_deg=1.0, outlier_fraction=0.05, seed=7)
    assert n_out > 0, "no outliers generated"

    G_graph, stats, per_edge = so3_sync_solve(N, ei, ej, Qs, w, anchor=0)
    graph_errs = [rot_angle_deg(G_true[k].T @ G_graph[k]) for k in range(N)]
    graph_med = float(np.median(graph_errs[1:]))
    graph_max = float(np.max(graph_errs[1:]))

    print(f"  outliers={n_out}/{len(ei)} graph_med={graph_med:.3f}° max={graph_max:.3f}°")
    assert graph_med < 8.0, f"outlier robustness degraded: med={graph_med:.3f}°"


def test_tree_graph_cannot_remove_drift():
    """Spanning-tree-only (cycle_rank=0) → solver CANNOT remove long-chain drift."""
    N = 60
    G_true, ei, ej, Qs, w, n_out = make_synthetic_graph(
        N=N, noise_deg=2.0, outlier_fraction=0.0, seed=99, chain_only=True)

    G_tree, stats, per_edge = so3_sync_solve(N, ei, ej, Qs, w, anchor=0)
    tree_errs = [rot_angle_deg(G_true[k].T @ G_tree[k]) for k in range(N)]
    tree_med = float(np.median(tree_errs[1:]))

    # Chain should accumulate; tree solver with cycle_rank=0 cannot fix it
    G_chain = chain_compose_error(N, ei, ej, Qs, anchor=0)
    chain_errs = [rot_angle_deg(G_true[k].T @ G_chain[k]) for k in range(N)]
    chain_med = float(np.median(chain_errs[1:]))

    print(f"  tree_med={tree_med:.3f}° chain_med={chain_med:.3f}°")
    # Tree solver = chain reparameterization; error should be comparable (both large)
    assert tree_med > 5.0, \
        f"tree graph should FAIL to remove drift, got tree_med={tree_med:.3f}°"


if __name__ == "__main__":
    test_zero_noise_exact_recovery()
    print("test_zero_noise_exact_recovery PASS")
    test_noise_recovery_sgraph_beats_chain()
    print("test_noise_recovery_sgraph_beats_chain PASS")
    test_outlier_robustness_huber()
    print("test_outlier_robustness_huber PASS")
    test_tree_graph_cannot_remove_drift()
    print("test_tree_graph_cannot_remove_drift PASS")