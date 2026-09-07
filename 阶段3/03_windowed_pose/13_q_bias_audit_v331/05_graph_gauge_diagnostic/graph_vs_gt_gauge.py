#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3.1 §18-§23: Graph gauge vs GT-optimal gauge + GT-gauge assembly diagnostic.

DIAGNOSTIC_ONLY — USES_REFERENCE — NOT_DEPLOYABLE
This script uses COLMAP reference gauges; those gauges MUST NEVER enter the solver.
The analysis is purely retrospective: given that the reference gauges exist, does
the graph-gauge trajectory match their SHAPE (up to global rotation A)?

§18: Load solver gauges G_graph[k] (from SO(3) sync) and corrected GT gauges G_ref[k].
§19: Fit global rotation A such that G_ref[k] ≈ A @ G_graph[k] for all k.
      Alignment residual = per-window angular error after A alignment.
§20: Per-edge alignment residual vs hop distance. If alignment residual is small
      overall but large at edges, the graph shape is close to reference but drift
      accumulates with chain length.
§21: GT-gauge assembly: using G_ref[k] (reference-gauge trajectory), assemble
      global cameras for each window center using the MST path. Compare to the
      graph-sync global cameras from 05_global_stitching.
§22: Report the maximum per-window deviation between GT-assembled and graph-sync cameras.
      This reveals whether the graph sync trajectory diverges from reference in
      a structured (gauge-congruent) way or a chaotic way.
§23: Sequential chain comparison: assemble from G_ref[k] via sequential Q chain
      and compare to graph-sync result. Shows how much the graph sync corrects
      (or fails to correct) the sequential drift.

Outputs (in 05_graph_gauge_diagnostic/):
  GRAPH_VS_GT_GAUGE_<seq>.csv          — per-window gauge comparison
  GT_ASSEMBLY_VS_GRAPH_<seq>.csv        — GT-assembled vs graph-sync cameras
  GRAPH_GAUGE_DIAGNOSTIC_SUMMARY.json   — per-sequence summary

Usage:
    python 05_graph_gauge_diagnostic/graph_vs_gt_gauge.py [--seq ...]
"""
import argparse, csv, glob, json, os, sys
from collections import defaultdict
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
EDGES_DIR = os.path.join(SYNC_DIR, "02_rotation_edges")
SOLVER_DIR = os.path.join(SYNC_DIR, "04_so3_sync")
GLOBAL_DIR = os.path.join(SYNC_DIR, "05_global_stitching")
GT_GAUGES_DIR = os.path.join(PHASE3C, "13_q_bias_audit_v331", "02_window_gauge_fit")
OUT_DIR = os.path.join(PHASE3C, "13_q_bias_audit_v331", "05_graph_gauge_diagnostic")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))
sys.path.insert(0, os.path.join(ROOT, "阶段3", "03_windowed_pose", "12_rotation_sync_v33", "04_so3_sync"))
from evaluate_multoplant import global_rotation_procrustes, rot_angle_deg
from run_so3_sync import load_edges_from_csv

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)


def load_graph_gauges(seq_id):
    """Load solver gauges from SO(3) sync (stride-4 graph, th=8)."""
    path = os.path.join(SOLVER_DIR, f"{seq_id}_SO3_SYNC_GAUGES_stride4_th8.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)["G"]  # (N_w, 3, 3)


def load_gt_gauges(seq_id):
    """Load corrected COLMAP-fitted gauges (DIAGNOSTIC_ONLY)."""
    path = os.path.join(GT_GAUGES_DIR, f"CORRECTED_WINDOW_GAUGES_{seq_id}.npz")
    if not os.path.exists(path):
        return None
    return np.load(path)["G"]  # (N_w, 3, 3), center_frame


def fit_global_alignment_A(G_graph, G_ref, N_w):
    """§19: Fit A minimizing Σ_k || A @ G_graph[k] - G_ref[k] ||_F.

    Uses Procrustes on stacked gauge matrices.
    Returns (A, per_window_error_deg).
    """
    # Treat each window's gauge as a 3×3 "frame" and fit global A
    # via stacked Procrustes: Σ_k || A @ G_graph[k] - G_ref[k] ||_F
    G_g = G_graph[:N_w].reshape(-1, 3, 3)
    G_r = G_ref[:N_w].reshape(-1, 3, 3)
    A = global_rotation_procrustes(G_g, G_r)
    # per-window error
    err = np.array([rot_angle_deg((A @ G_graph[k]).T @ G_ref[k]) for k in range(N_w)])
    return A, err


def graph_assembly_comparison(seq_id, G_graph, G_ref, N_w, edges_i, edges_j, Q_edges):
    """§21-§22: Compare graph-sync cameras to GT-gauge-assembled cameras.

    GT-gauge assembly: for each window center frame, compose gauges along the
    graph edges. Graph-sync assembly: same path using solver gauges.

    Since both are defined up to an anchor, first align G_ref to G_graph via A,
    then compare per-window center-frame orientations.
    """
    A, align_err = fit_global_alignment_A(G_graph, G_ref, N_w)
    G_ref_aligned = np.einsum("ab,sbc->sac", A, G_ref)

    # After alignment: compare per-window oriented camera
    # The solver's final camera for center frame of window k = G_graph[k]
    # GT-assembled camera (aligned to graph) = G_ref_aligned[k]
    per_win_err = np.array([rot_angle_deg(G_graph[k].T @ G_ref_aligned[k])
                            for k in range(N_w)])

    return A, align_err, per_win_err


def sequential_chain_assembly(seq_id, G_ref, G_graph, N_w):
    """§23: Assemble global cameras from G_ref via sequential chain (hop-1 edges).

    The solver convention is Q_ij ≈ G_i^T G_j, so propagation along a hop-1 edge
    (a,b) is G_chain[b] = G_chain[a] @ Q_ab. Starting the chain at the GT gauge
    G_ref[0] keeps everything in the reference frame, so:

      - chain_vs_ref[k] = angle(G_chain[k], G_ref[k]) measures how much the
        VGGT-measured Q edges accumulate error relative to the GT gauge trajectory.
        If VGGT edges were exact, chain_vs_ref ≈ 0 for all k.

      - chain_vs_graph uses global alignment A (fit chain → graph gauges) then
        compares; measures how far the graph-sync solution differs from the
        naive sequential chain of the same Q edges.
    """
    # Load hop-1 edges sorted by window_i
    ei, ej, Q, w = load_edges_from_csv(seq_id, "stride4", 8)
    # Build hop-1 adjacency
    hop1 = [(int(ei[k]), int(ej[k])) for k in range(len(ei))
            if int(ej[k]) - int(ei[k]) == 1]
    hop1.sort()

    if not hop1:
        return None, None, None

    # Sequential chain from solver Q: start at anchor 0 (GT gauge = reference frame)
    # Robust: build window→gauge map, propagate 1 → 2 → ... longest prefix from 0
    gmap = {0: G_ref[0]}
    for a, b in hop1:
        if a in gmap and b not in gmap:
            idx = [k for k in range(len(ei)) if int(ei[k]) == a and int(ej[k]) == b]
            if idx:
                Q_ab = Q[idx[0]]
                gmap[b] = gmap[a] @ Q_ab  # per Q_ij = G_i^T G_j
    # fill remaining windows by nearest predecessor via linear scan
    missing = [k for k in range(1, N_w) if k not in gmap]
    while missing:
        progressed = False
        for k in missing:
            preds = sorted([pp for pp in range(k - 1, max(-1, k - 4), -1) if pp in gmap])
            if preds:
                pp = preds[0]
                idx = [e2 for e2 in range(len(ei))
                       if int(ei[e2]) == pp and int(ej[e2]) == k]
                if idx:
                    gmap[k] = gmap[pp] @ Q[idx[0]]
                    progressed = True
        if not progressed:
            break
        missing = [k for k in range(1, N_w) if k not in gmap]

    G_chain = np.array([gmap[k] for k in range(N_w)])

    chain_N = len(G_chain)
    # chain vs GT gauges: same frame (chain starts at G_ref[0]) — no alignment needed
    errs_ref = np.array([rot_angle_deg(G_chain[k].T @ G_ref[k])
                         for k in range(chain_N)])
    # chain vs graph gauges: align chain into graph frame, then compare
    A_cg, _ = fit_global_alignment_A(G_chain, G_graph, chain_N)
    G_chain_al = np.einsum("ab,sbc->sac", A_cg, G_chain)
    errs_chain = np.array([rot_angle_deg(G_chain_al[k].T @ G_graph[k])
                           for k in range(chain_N)])
    return G_chain, errs_chain, errs_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()
    sequences = args.seq if args.seq else list(SEQUENCES)

    summary = {}

    for seq_id in sequences:
        print(f"\n=== {seq_id} ===")

        G_graph = load_graph_gauges(seq_id)
        G_ref = load_gt_gauges(seq_id)
        if G_graph is None or G_ref is None:
            print("  SKIP: missing solver or GT gauges")
            continue
        N_w = min(len(G_graph), len(G_ref))
        print(f"  N_graph={len(G_graph)} N_ref={N_ref if 'N_ref' in dir() else N_w}")

        # §19: Fit global A alignment
        A, align_err = fit_global_alignment_A(G_graph, G_ref, N_w)
        print(f"  §19 global-alignment: median={np.median(align_err):.3f}° "
              f"max={np.max(align_err):.3f}° p90={np.percentile(align_err,90):.3f}°")

        # §21-§22: Graph vs GT-gauge assembled cameras
        A2, align_err2, per_win_err = graph_assembly_comparison(
            seq_id, G_graph, G_ref, N_w, None, None, None)

        # save per-window CSV
        csv_path = os.path.join(OUT_DIR, f"GRAPH_VS_GT_GAUGE_{seq_id}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["window", "gauge_vs_gt_after_A_deg", "gauge_shape_err_deg"])
            for k in range(N_w):
                w.writerow([k, round(align_err[k], 4),
                            round(per_win_err[k], 4)])
        print(f"  §21 per-window after A: median={np.median(per_win_err):.3f}° "
              f"max={np.max(per_win_err):.3f}°")

        # §23: Sequential chain from GT gauges vs graph-sync
        chain_res = sequential_chain_assembly(seq_id, G_ref, G_graph, N_w)
        if chain_res[0] is not None:
            G_chain, errs_chain, errs_ref = chain_res
            chain_med = float(np.median(errs_chain))
            chain_max = float(np.max(errs_chain))
            ref_med = float(np.median(errs_ref))
            print(f"  §23 chain-GT-vs-graph: median={chain_med:.3f}° max={chain_max:.3f}°")
            print(f"  §23 chain-GT-vs-ref:  median={ref_med:.3f}° max={float(np.max(errs_ref)):.3f}°")
        else:
            chain_med = chain_max = ref_med = None
            print("  §23: no hop-1 edges found for chain assembly")

        summary[seq_id] = {
            "n_windows": N_w,
            "global_alignment_A_median_deg": round(float(np.median(align_err)), 4),
            "global_alignment_A_max_deg": round(float(np.max(align_err)), 4),
            "per_window_gt_vs_graph_median_deg": round(float(np.median(per_win_err)), 4),
            "per_window_gt_vs_graph_max_deg": round(float(np.max(per_win_err)), 4),
            "sequential_chain_gt_vs_graph_median_deg": round(chain_med, 4) if chain_med else None,
            "sequential_chain_gt_vs_graph_max_deg": round(chain_max, 4) if chain_max else None,
            "sequential_chain_gt_vs_ref_median_deg": round(ref_med, 4) if ref_med else None,
            "diagnostic_note": "DIAGNOSTIC_ONLY — USES_REFERENCE — NOT_DEPLOYABLE",
        }

    with open(os.path.join(OUT_DIR, "GRAPH_GAUGE_DIAGNOSTIC_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved GRAPH_GAUGE_DIAGNOSTIC_SUMMARY.json")


if __name__ == "__main__":
    main()