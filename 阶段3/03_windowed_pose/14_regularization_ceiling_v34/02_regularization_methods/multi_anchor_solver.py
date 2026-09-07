#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Method D: multi-anchor gauge solver (diagnostic only).

Anchors additional windows (e.g. midpoint and/or endpoint) to the identity,
beyond the standard G_0 = I. This is NOT an external anchor — it reuses the
SAME Q-edge information with different fixed points; it is a DIAGNOSTIC that
quantifies how anchoring strategy redistributes drift along the trajectory.

Interpretation:
  * With K anchors, the solver cannot rotate the sub-trajectory between anchors
    arbitrarily; coherent drift is compressed into the free segments.
  * If drift is a constant-rate global bias, anchoring the endpoint to I forces
    the endpoint error to 0 but injects bias into the middle (the drift has to
    go somewhere).
  * This reveals whether the drift is a NET rotation of the whole trajectory
    (single global gauge freedom — a single anchor already captures it under
    Procrustes alignment, and per-window errors stay small) vs an INTERNAL
    trajectory deformation (multi-anchor changes per-window error at
    fixed alignment scale).

NOT DEPLOYABLE: requires the assumption that midpoint/endpoint gauges should
equal I, which is only true for closed-ish trajectories. Marked DIAGNOSTIC_ONLY.

Input/Output: same conventions as Methods B/C (edge npz + n_windows).

Usage:
    python 02_regularization_methods/multi_anchor_solver.py \
        --edge-npz <npz> --n-windows 77 --anchors 0,38,76 --out <npz>
"""
import argparse, json, os, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
METHODS_DIR = os.path.join(PHASE3C, "14_regularization_ceiling_v34", "02_regularization_methods")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33", "04_so3_sync")
os.makedirs(METHODS_DIR, exist_ok=True)

# reuse the baseline solver's MST warm start (avoids the identity-init Huber
# plateau: with ~25°/step accumulated rotation every residual is flat at
# f_scale=0.05 rad and gradients vanish → least_squares crawls for minutes)
sys.path.insert(0, SYNC_DIR)
from run_so3_sync import mst_init

HUBER_F_SCALE_RAD = 0.05
MAX_NFEV = 800


def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def log_matrix(R):
    return Rotation.from_matrix(R).as_rotvec()


def multi_anchor_solve(n_nodes, edges_i, edges_j, Qs, weights, anchors,
                       max_nfev=MAX_NFEV):
    """Free nodes = all except anchors; anchored nodes pinned to I.

    The chain of anchors splits the trajectory; drift accumulates within each
    free segment between anchors.
    """
    anchors = sorted(int(a) for a in anchors)
    free_nodes = sorted(set(range(n_nodes)) - set(anchors))
    idx_map = {k: v for v, k in enumerate(free_nodes)}

    # warm start: propagate from anchor 0 along the max-weight tree (same init
    # as the baseline solver). Anchored nodes' values are ignored; only free
    # nodes' rotvecs are used as the starting point.
    G_init_full = mst_init(n_nodes, edges_i, edges_j, Qs, weights, anchor=0)
    x0 = (np.concatenate([log_matrix(G_init_full[k]) for k in free_nodes])
          if free_nodes else np.zeros(0))

    def residuals(x):
        G = np.broadcast_to(np.eye(3), (n_nodes, 3, 3)).copy()
        if free_nodes:
            G[free_nodes] = Rotation.from_rotvec(x.reshape(-1, 3)).as_matrix()
        # vectorized edge residuals: res = sqrt(w) * Log(Q^T G_a^T G_b)
        Ga = G[edges_i]                      # (E,3,3)
        Gb = G[edges_j]                      # (E,3,3)
        E = Qs.transpose(0, 2, 1) @ (Ga.transpose(0, 2, 1) @ Gb)   # (E,3,3)
        rv = Rotation.from_matrix(E).as_rotvec()                   # (E,3)
        return (np.sqrt(weights)[:, None] * rv).reshape(-1)

    result = least_squares(residuals, x0, loss="huber", f_scale=HUBER_F_SCALE_RAD,
                           method="trf", max_nfev=max_nfev,
                           xtol=1e-10, ftol=1e-10, gtol=1e-10)

    G = np.broadcast_to(np.eye(3), (n_nodes, 3, 3)).copy()
    if free_nodes:
        G[free_nodes] = Rotation.from_rotvec(result.x.reshape(-1, 3)).as_matrix()

    per_edge_deg = np.zeros(len(edges_i))
    Ga = G[edges_i]
    Gb = G[edges_j]
    E = Qs.transpose(0, 2, 1) @ (Ga.transpose(0, 2, 1) @ Gb)
    for e in range(len(edges_i)):
        per_edge_deg[e] = rot_angle_deg(E[e])

    stats = {
        "n_anchors": len(anchors),
        "anchors": anchors,
        "n_free": len(free_nodes),
        "edge_res_median_deg": float(np.median(per_edge_deg)),
        "edge_res_p90_deg": float(np.percentile(per_edge_deg, 90)),
        "edge_res_max_deg": float(np.max(per_edge_deg)),
        "success": bool(result.success),
        "cost": float(result.cost),
    }
    return G, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-npz", required=True)
    ap.add_argument("--n-windows", type=int, required=True)
    ap.add_argument("--anchors", required=True, help="comma list, e.g. 0,38,76")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    anchors = [int(a) for a in args.anchors.split(",")]
    d = np.load(args.edge_npz)
    G, stats = multi_anchor_solve(args.n_windows, d["i"].astype(int),
                                  d["j"].astype(int), d["Q"], d["weight"], anchors)
    stats["diagnostic_note"] = "DIAGNOSTIC_ONLY — anchors to I are NOT deployable constraints"
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez(args.out, G=G, anchors=anchors, method="D_multi_anchor",
                 edge_res_median_deg=stats["edge_res_median_deg"],
                 diagnostic_note=stats["diagnostic_note"])
        print(f"anchors={anchors}: res_med={stats['edge_res_median_deg']:.3f}° -> {args.out}")
    else:
        print(json.dumps(stats, indent=2))