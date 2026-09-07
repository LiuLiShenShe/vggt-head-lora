#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Step 2d: Edge reweighting (temporal + bias detection).

Temporal attenuation: w'_e = w_e · (1 − α·d_e/d_max) — tests whether drift is
more trustworthy near the anchor.

Bias detection: after a baseline sync, residuals r_e = Log(Q_e^T G_i^T G_j);
estimate coherent direction û = normalize(Σ r_e) and coherency
γ = ‖Σ r_e‖/Σ‖r_e‖. Downweight edges most aligned with û:
w'_e = w_e / (1 + β·⟨r̂_e, û⟩₊).

Hypothesis under test: if γ ≈ 1 at baseline, the residuals are fully consistent
with the drift (CASE C) — reweighting cannot move the solution (that is itself
the finding). If reweighting DOES reduce drift, the drift is partially
incoherent — an important positive finding.

Outputs: REGULARIZED_GAUGES_<seq>_rw_t{α}.npz and _rw_b{β}.npz

Usage:
    python 02_regularization_methods/edge_reweighting.py [--seq ...]
"""
import argparse, json, os, sys
import numpy as np
import networkx as nx

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
SOLVER_DIR = os.path.join(SYNC_DIR, "04_so3_sync")
OUT_DIR = os.path.join(PHASE3C, "14_regularization_ceiling_v34", "02_regularization_methods")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, SOLVER_DIR)
from run_so3_sync import load_edges_from_csv, so3_sync_solve, rot_angle_deg, window_count
from scipy.spatial.transform import Rotation

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
BETAS = [0.0, 5, 20, 100]


# --------------------------------------------------------------------------
# Reweighting
# --------------------------------------------------------------------------
def reweight_by_anchor_distance(ei, ej, w, alpha):
    """w'_e = w_e · (1 − α·d_e/d_max), d_e = j−i (window distance)."""
    d = np.asarray(ej, float) - np.asarray(ei, float)
    dmax = d.max()
    if dmax <= 0:
        return w.copy()
    scale = np.clip(1.0 - alpha * d / dmax, 0.0, None)
    return w * scale


def residual_coherent_direction(ei, ej, Q, w, G):
    """Per-edge residual rotvecs → coherent direction û + coherency γ."""
    r = np.zeros((len(ei), 3))
    for e in range(len(ei)):
        a, b = int(ei[e]), int(ej[e])
        E = Q[e].T @ (G[a].T @ G[b])
        r[e] = Rotation.from_matrix(E).as_rotvec()
    total = np.linalg.norm(r.sum(axis=0))
    scalar = np.linalg.norm(r, axis=1).sum()
    gamma = float(total / scalar) if scalar > 0 else 0.0
    if total > 1e-12:
        uhat = r.sum(axis=0) / total
    else:
        uhat = np.zeros(3)
    return uhat, gamma, r


def reweight_by_bias_alignment(ei, ej, w, r, uhat, beta):
    """w'_e = w_e / (1 + β·⟨r̂_e, û⟩₊) — downweight edges aligned with the drift."""
    mag = np.linalg.norm(r, axis=1)
    rhat = r / np.maximum(mag[:, None], 1e-12)
    proj = rhat @ uhat
    pos = np.clip(proj, 0.0, None)
    return w / (1.0 + beta * pos)


def run_reweighted_sync(n_nodes, ei, ej, Q, w_custom, anchor=0):
    """Solve with custom weights; connectivity gate via networkx."""
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    for a, b, ww in zip(ei, ej, w_custom):
        g.add_edge(int(a), int(b), weight=float(ww))
    if not nx.is_connected(g):
        return None, {"error": "disconnected graph"}
    return so3_sync_solve(n_nodes, ei, ej, Q, w_custom, anchor=anchor)


def temporal_position_experiment(seq_id, n_nodes, alphas=ALPHAS):
    """Sweep α, return per-α median gauge change vs baseline (α=0)."""
    ei, ej, Q, w = load_edges_from_csv(seq_id, "stride4", 8)
    if ei is None or len(ei) == 0:
        return []
    G_base, stats, _ = so3_sync_solve(n_nodes, ei, ej, Q, w, anchor=0)
    out = []
    for alpha in alphas:
        w2 = reweight_by_anchor_distance(ei, ej, w, alpha)
        res = run_reweighted_sync(n_nodes, ei, ej, Q, w2)
        if res[0] is None:
            out.append({"alpha": alpha, "error": "disconnected"})
            continue
        G2 = res[0]
        n = min(len(G_base), len(G2))
        chg = float(np.median([rot_angle_deg(G_base[k].T @ G2[k]) for k in range(n)]))
        out.append({"alpha": alpha, "median_gauge_change_deg": round(chg, 4)})
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()
    sequences = args.seq if args.seq else list(SEQUENCES)

    summary = []

    for seq_id in sequences:
        n_nodes = window_count(seq_id, "stride4")
        if n_nodes is None:
            print(f"  SKIP {seq_id}: no window count")
            continue
        ei, ej, Q, w = load_edges_from_csv(seq_id, "stride4", 8)
        if ei is None or len(ei) == 0:
            print(f"  SKIP {seq_id}: no edges")
            continue
        print(f"=== {seq_id} (N={n_nodes}, E={len(ei)}) ===")

        # baseline
        G_base, stats_base, per_edge = so3_sync_solve(n_nodes, ei, ej, Q, w, anchor=0)
        uhat, gamma, r = residual_coherent_direction(ei, ej, Q, w, G_base)
        print(f"  baseline residual coherency γ={gamma:.3f} dir={uhat.round(3)}")

        # temporal attenuation
        for alpha in ALPHAS:
            w2 = reweight_by_anchor_distance(ei, ej, w, alpha)
            res = run_reweighted_sync(n_nodes, ei, ej, Q, w2)
            if res[0] is None:
                print(f"  α={alpha}: disconnected")
                continue
            G2 = res[0]
            n = min(len(G_base), len(G2))
            chg = float(np.median([rot_angle_deg(G_base[k].T @ G2[k]) for k in range(n)]))
            print(f"  temporal α={alpha}: gauge change med {chg:.4f}°")
            npz = os.path.join(OUT_DIR, f"REGULARIZED_GAUGES_{seq_id}_rw_t{alpha}.npz")
            np.savez(npz, G=G2, method="edge_reweight_temporal", alpha=alpha,
                     anchor_window=0, input_gauge="so3_sync_stride4_th8")
            summary.append({"sequence": seq_id, "method": "temporal", "alpha": alpha,
                            "median_gauge_change_deg": round(chg, 4)})

        # bias alignment
        for beta in BETAS:
            w2 = reweight_by_bias_alignment(ei, ej, w, r, uhat, beta)
            res = run_reweighted_sync(n_nodes, ei, ej, Q, w2)
            if res[0] is None:
                print(f"  β={beta}: disconnected")
                continue
            G2 = res[0]
            n = min(len(G_base), len(G2))
            chg = float(np.median([rot_angle_deg(G_base[k].T @ G2[k]) for k in range(n)]))
            print(f"  bias β={beta}: gauge change med {chg:.4f}°")
            npz = os.path.join(OUT_DIR, f"REGULARIZED_GAUGES_{seq_id}_rw_b{beta}.npz")
            np.savez(npz, G=G2, method="edge_reweight_bias", beta=beta,
                     anchor_window=0, input_gauge="so3_sync_stride4_th8",
                     residual_gamma=gamma)
            summary.append({"sequence": seq_id, "method": "bias", "beta": beta,
                            "median_gauge_change_deg": round(chg, 4),
                            "residual_gamma": round(gamma, 4)})

    summary_path = os.path.join(OUT_DIR, "METHOD_SUMMARY.json")
    prev = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            prev = json.load(f)
    prev["edge_reweighting"] = {"gt_used": False, "runs": summary}
    with open(summary_path, "w") as f:
        json.dump(prev, f, indent=2)
    print(f"\nUpdated METHOD_SUMMARY.json (edge_reweighting: {len(summary)} runs)")


if __name__ == "__main__":
    main()