#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Method B: second-order gauge-trajectory regularized SO(3) sync.

Extends the Phase 3C.3 solver with a second-difference (ACCELERATION) penalty
on the gauge trajectory:

    min_G  Σ_e w_e ρ_huber(‖Log(Q_eᵀ G_aᵀ G_b)‖²)
         + λ Σ_k ‖Log(G_{k-1}ᵀ G_k) − Log(G_kᵀ G_{k+1})‖²

The second term penalizes CHANGE in adjacent rotation rates. Key property
(which this experiment is designed to VERIFY, not assume):

  * A constant-rate coherent bias has ZERO second derivative — the regularizer
    is symmetric about it, so the dominant coherent mode passes UNHURT
    (first-order Gaussian smoothing has a linear-ramp null space; this term
    shares that null space for exact ramps).
  * It DOES suppress the incoherent (random-walk) 24% component and any
    acceleration in the bias.

Anchor: G_0 = I (same as Phase 3C.3). Parameterization: rotvec per free node.
Solver: scipy least_squares trf, Huber f_scale = 0.05 rad (FROZEN, same as
Phase 3C.3 protocol) to isolate the effect of λ.

Input: edge arrays + Q from the Phase 3C.3 pipeline (reads master edge CSV,
reusing run_so3_sync.load_edges_from_csv and mst_init for reproducibility).

Outputs (in 02_regularization_methods/):
  REGULARIZED_SOLVER_SUMMARY.json  — per (scenario, λ) cost/residual stats

Usage:
    python 02_regularization_methods/regularized_solver.py \
        --edge-npz <npz with i,j,Q,weight> --n-windows 77 --lambda 0.1
"""
import argparse, json, os
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
METHODS_DIR = os.path.join(PHASE3C, "14_regularization_ceiling_v34", "02_regularization_methods")
os.makedirs(METHODS_DIR, exist_ok=True)

HUBER_F_SCALE_RAD = 0.05   # FROZEN (Phase 3C.3 protocol)
MAX_NFEV = 800


def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def log_matrix(R):
    return Rotation.from_matrix(R).as_rotvec()


def mst_init(n_nodes, edges_i, edges_j, Qs, weights, anchor=0):
    """Max-weight-tree init (copied semantics from run_so3_sync.mst_init)."""
    import networkx as nx
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    for a, b, w in zip(edges_i, edges_j, weights):
        g.add_edge(int(a), int(b), weight=float(w))
    G = np.zeros((n_nodes, 3, 3))
    G[anchor] = np.eye(3)
    if g.number_of_edges() == 0:
        for k in range(n_nodes):
            G[k] = np.eye(3)
        return G
    tree = nx.maximum_spanning_tree(g, weight="weight")
    for k in range(n_nodes):
        if k not in tree:
            tree.add_node(k)
    q_lookup = {}
    for a, b, q in zip(edges_i, edges_j, Qs):
        q_lookup[(int(a), int(b))] = q
    visited = {anchor}
    queue = [anchor]
    while queue:
        node = queue.pop(0)
        for nb in tree.neighbors(node):
            if nb in visited:
                continue
            a, b = (node, nb) if node < nb else (nb, node)
            q = q_lookup.get((a, b), np.eye(3))
            G[nb] = G[node] @ (q if node < nb else q.T)
            visited.add(nb)
            queue.append(nb)
    return G


def regularized_so3_sync(n_nodes, edges_i, edges_j, Qs, weights,
                         lam, anchor=0, init_G=None, use_huber=True):
    """Solve Method B: graph sync + λ × second-difference trajectory penalty.

    Returns (G, stats, per_edge_deg).
    """
    if init_G is None:
        init_G = mst_init(n_nodes, edges_i, edges_j, Qs, weights, anchor)
    free_nodes = [k for k in range(n_nodes) if k != anchor]
    idx_map = {k: v for v, k in enumerate(free_nodes)}

    x0 = np.concatenate([Rotation.from_matrix(init_G[k]).as_rotvec() for k in free_nodes])

    def residuals(x):
        G = np.broadcast_to(np.eye(3), (n_nodes, 3, 3)).copy()
        for k in free_nodes:
            G[k] = Rotation.from_rotvec(x[idx_map[k] * 3:(idx_map[k] + 1) * 3]).as_matrix()
        # edge residuals
        res = np.zeros(len(edges_i) * 3)
        for e in range(len(edges_i)):
            a, b = edges_i[e], edges_j[e]
            E = Qs[e].T @ (G[a].T @ G[b])
            res[e * 3:(e + 1) * 3] = np.sqrt(weights[e]) * log_matrix(E)
        # second-difference penalty (acceleration)
        acc = []
        for k in range(1, n_nodes - 1):
            v_prev = log_matrix(G[k - 1].T @ G[k])
            v_next = log_matrix(G[k].T @ G[k + 1])
            acc.append(v_next - v_prev)
        acc = np.array(acc).reshape(-1) if acc else np.zeros(0)
        # Huber applied separately on data vs penalty (penalty is squared)
        return np.concatenate([res, np.sqrt(lam) * acc])

    if use_huber:
        result = least_squares(residuals, x0, loss="huber", f_scale=HUBER_F_SCALE_RAD,
                               method="trf", max_nfev=MAX_NFEV,
                               xtol=1e-10, ftol=1e-10, gtol=1e-10)
    else:
        result = least_squares(residuals, x0, loss="linear", method="trf",
                               max_nfev=MAX_NFEV, xtol=1e-10, ftol=1e-10, gtol=1e-10)

    G = np.broadcast_to(np.eye(3), (n_nodes, 3, 3)).copy()
    for k in free_nodes:
        G[k] = Rotation.from_rotvec(result.x[idx_map[k] * 3:(idx_map[k] + 1) * 3]).as_matrix()

    per_edge_deg = np.zeros(len(edges_i))
    for e in range(len(edges_i)):
        a, b = edges_i[e], edges_j[e]
        E = Qs[e].T @ (G[a].T @ G[b])
        per_edge_deg[e] = rot_angle_deg(E)

    acc_deg = []
    for k in range(1, n_nodes - 1):
        v_prev = log_matrix(G[k - 1].T @ G[k])
        v_next = log_matrix(G[k].T @ G[k + 1])
        acc_deg.append(np.degrees(np.linalg.norm(v_next - v_prev)))
    acc_deg = np.array(acc_deg)

    stats = {
        "n_windows": n_nodes,
        "n_edges": len(edges_i),
        "lambda": lam,
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": result.nfev,
        "edge_res_mean_deg": float(np.mean(per_edge_deg)),
        "edge_res_median_deg": float(np.median(per_edge_deg)),
        "edge_res_p90_deg": float(np.percentile(per_edge_deg, 90)),
        "edge_res_max_deg": float(np.max(per_edge_deg)),
        "accel_median_deg": float(np.median(acc_deg)) if len(acc_deg) else 0.0,
        "accel_p90_deg": float(np.percentile(acc_deg, 90)) if len(acc_deg) else 0.0,
    }
    return G, stats, per_edge_deg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-npz", required=True)
    ap.add_argument("--n-windows", type=int, required=True)
    ap.add_argument("--lambda", type=float, dest="lam", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.edge_npz)
    G_out, stats, _ = regularized_so3_sync(
        args.n_windows, d["i"].astype(int), d["j"].astype(int), d["Q"],
        d["weight"], lam=args.lam)

    out_dir = METHODS_DIR
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez(args.out, G=G_out, lam=args.lam, method="B_regularized_solver",
                 edge_res_median_deg=stats["edge_res_median_deg"])
        print(f"lambda={args.lam}: bridge res_med={stats['edge_res_median_deg']:.3f}° "
              f"accel_med={stats['accel_median_deg']:.3f}° -> {args.out}")
    else:
        for k, v in stats.items():
            print(f"  {k} = {v}")