#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3 Step 4: Robust SO(3) graph synchronization solver.

Solves:
    min_G  Σ_(i,j)  w_ij * ρ_huber( || Log_SO3(Q_ij^T G_i^T G_j) ||² )
    subject to G_0 = I (anchor)

- Manifold residual via scipy Rotation.from_matrix(E).as_rotvec() (exact Log map)
- Parametrization: each free window k has a rotvec δ_k, G_k = Exp(δ_k)
- Init: maximum-spanning-tree by edge confidence (w high = prefer), then scipy refine
- No GT: input is ONLY window npz + shared-frame indices / edge Q's
- Huber f_scale FROZEN (no tuning): 0.05 rad ≈ 2.9° (protocol range 2-5°)

Reads edges directly from the master edge CSVs:
  stride4 → ROTATION_GRAPH_EDGES.csv            (superset incl. hop-3; filter by n_overlap)
  stride8 → ROTATION_GRAPH_EDGES_STRIDE8.csv    (pure chain, negative control)

Outputs:
  <seq>_SO3_SYNC_GAUGES.npz        — G_k (N,3,3) for the chosen stride/threshold
  SO3_SYNC_CONVERGENCE.csv         — per-sequence init/final residual stats
  SO3_METHOD_EXEC_SUMMARY.json     — params

Usage:
    python 04_so3_sync/run_so3_sync.py [--seq ...] [--threshold 8] [--dataset stride4|stride8]
"""
import argparse, csv, json, os, sys
import numpy as np
import networkx as nx
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
EDGES_DIR = os.path.join(SYNC_DIR, "02_rotation_edges")
OUT_DIR = os.path.join(SYNC_DIR, "04_so3_sync")
os.makedirs(OUT_DIR, exist_ok=True)

HUBER_F_SCALE_RAD = 0.05  # ≈ 2.9 degrees — protocol frozen (range 2-5 deg), NO tuning
MAX_NFEV = 500

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)

EDGE_CSV = {
    "stride4": "ROTATION_GRAPH_EDGES.csv",
    "stride8": "ROTATION_GRAPH_EDGES_STRIDE8.csv",
}


def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def load_edges_from_csv(seq_id, dataset, threshold):
    """Load (i, j, Q, weight) for a sequence from the master edge CSV."""
    path = os.path.join(EDGES_DIR, EDGE_CSV[dataset])
    if not os.path.exists(path):
        return None, None, None, None
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["sequence"] == seq_id and int(r["n_overlap"]) >= threshold]
    if not rows:
        return None, None, None, None
    i = np.array([int(r["window_i"]) for r in rows])
    j = np.array([int(r["window_j"]) for r in rows])
    w = np.array([float(r["weight"]) for r in rows])
    Q = np.array([[[float(r[f"q{rr}{cc}"]) for cc in range(3)] for rr in range(3)] for r in rows])
    return i, j, Q, w


def window_count(seq_id, dataset):
    """Real window count from manifest."""
    if dataset == "stride4":
        man = os.path.join(SYNC_DIR, "01_stride4_inference", "STRIDE4_WINDOW_MANIFEST.json")
        if os.path.exists(man):
            with open(man) as f:
                d = json.load(f)
            if seq_id in d:
                return d[seq_id]["n_windows"]
    man8 = os.path.join(PHASE3C, "03_window_inference", "window_outputs",
                        seq_id, "WINDOW_RUN_MANIFEST.json")
    if os.path.exists(man8):
        with open(man8) as f:
            return json.load(f)["n_windows"]
    return None


def mst_init(n_nodes, edges_i, edges_j, Qs, weights, anchor=0):
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    for a, b, w in zip(edges_i, edges_j, weights):
        g.add_edge(a, b, weight=float(w))

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

    # adjacency lookup: dict[(a,b)] -> (Q, orientation)
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
            if node < nb:
                G[nb] = G[node] @ q
            else:
                G[nb] = G[node] @ q.T
            visited.add(nb)
            queue.append(nb)
    return G


def so3_sync_solve(n_nodes, edges_i, edges_j, Qs, weights, anchor=0, init_G=None):
    if init_G is None:
        init_G = mst_init(n_nodes, edges_i, edges_j, Qs, weights, anchor)

    free_nodes = [k for k in range(n_nodes) if k != anchor]
    idx_map = {k: v for v, k in enumerate(free_nodes)}

    x0 = np.concatenate([Rotation.from_matrix(init_G[k]).as_rotvec() for k in free_nodes])

    def residuals(x):
        G = np.broadcast_to(np.eye(3), (n_nodes, 3, 3)).copy()
        for k in free_nodes:
            G[k] = Rotation.from_rotvec(x[idx_map[k] * 3:(idx_map[k] + 1) * 3]).as_matrix()
        res = np.zeros(len(edges_i) * 3)
        for e in range(len(edges_i)):
            a, b = edges_i[e], edges_j[e]
            E = Qs[e].T @ (G[a].T @ G[b])
            res[e * 3:(e + 1) * 3] = np.sqrt(weights[e]) * Rotation.from_matrix(E).as_rotvec()
        return res

    result = least_squares(
        residuals, x0,
        loss="huber", f_scale=HUBER_F_SCALE_RAD,
        method="trf", max_nfev=MAX_NFEV, xtol=1e-10, ftol=1e-10, gtol=1e-10,
    )

    G = np.broadcast_to(np.eye(3), (n_nodes, 3, 3)).copy()
    for k in free_nodes:
        G[k] = Rotation.from_rotvec(result.x[idx_map[k] * 3:(idx_map[k] + 1) * 3]).as_matrix()

    per_edge_deg = np.zeros(len(edges_i))
    for e in range(len(edges_i)):
        a, b = edges_i[e], edges_j[e]
        E = Qs[e].T @ (G[a].T @ G[b])
        per_edge_deg[e] = rot_angle_deg(E)

    stats = {
        "n_windows": n_nodes,
        "n_edges": len(edges_i),
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": result.nfev,
        "chi2": float(np.sum(result.fun**2)),
        "final_residual_mean_deg": float(np.mean(per_edge_deg)),
        "final_residual_median_deg": float(np.median(per_edge_deg)),
        "final_residual_p90_deg": float(np.percentile(per_edge_deg, 90)),
        "final_residual_max_deg": float(np.max(per_edge_deg)),
    }
    return G, stats, per_edge_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    ap.add_argument("--threshold", type=int, default=8,
                    help="min_overlap for edge inclusion (8 headline, 4 diagnostic)")
    ap.add_argument("--dataset", choices=["stride4", "stride8"], default="stride4")
    args = ap.parse_args()

    sequences = args.seq if args.seq else SEQUENCES
    conv_rows = []

    for seq_id in sequences:
        print(f"\n=== {seq_id} [{args.dataset}, threshold={args.threshold}] ===")
        ei, ej, qs, w = load_edges_from_csv(seq_id, args.dataset, args.threshold)
        if ei is None or len(ei) == 0:
            print("  no edges (inference/graph missing)")
            continue
        n_nodes = window_count(seq_id, args.dataset)
        if n_nodes is None:
            print("  window count unavailable — skip")
            continue

        G_sync, stats, per_edge_deg = so3_sync_solve(n_nodes, ei, ej, qs, w, anchor=0)

        tag = f"{args.dataset}_th{args.threshold}"
        np.savez(os.path.join(OUT_DIR, f"{seq_id}_SO3_SYNC_GAUGES_{tag}.npz"),
                 G=G_sync, edges_i=ei, edges_j=ej, Q=qs, weight=w,
                 anchor_window=0, huber_f_scale_rad=HUBER_F_SCALE_RAD,
                 dataset=args.dataset, min_overlap_threshold=args.threshold,
                 final_res_mean_deg=stats["final_residual_mean_deg"])

        row = {"sequence": seq_id, "dataset": args.dataset,
               "threshold": args.threshold, **stats}
        conv_rows.append(row)
        print(f"  → V={n_nodes} E={len(ei)} res_mean={stats['final_residual_mean_deg']:.3f}° "
              f"res_p90={stats['final_residual_p90_deg']:.3f}° cost={stats['cost']:.4f}")

    if conv_rows:
        csv_path = os.path.join(OUT_DIR, f"SO3_SYNC_CONVERGENCE_{args.dataset}_th{args.threshold}.csv")
        fields = ["sequence", "dataset", "threshold", "n_windows", "n_edges", "success",
                  "cost", "final_residual_mean_deg", "final_residual_median_deg",
                  "final_residual_p90_deg", "final_residual_max_deg"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(conv_rows)
        print(f"\nSaved: {csv_path}")

    summary = {
        "huber_f_scale_rad": HUBER_F_SCALE_RAD,
        "loss": "huber",
        "anchor_window": 0,
        "max_nfev": MAX_NFEV,
        "method": "scipy.optimize.least_squares trf",
        "residual": "Rotation.as_rotvec (exact Log map)",
        "init": "max-confidence spanning tree then refine all edges",
        "gt_used_in_solver": False,
    }
    with open(os.path.join(OUT_DIR, "SO3_METHOD_EXEC_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()