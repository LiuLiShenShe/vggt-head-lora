#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Method C: lower edge threshold (hop-3 edges included).

The Phase 3C.3 pipeline computes ALL real-overlap edges for hop 1/2/3 and
stores them in ROTATION_GRAPH_EDGES.csv, but the headline solver applies a
min_overlap threshold of 8, which drops the 74 hop-3 edges (n_overlap = 4).
This method re-runs the graph sync at threshold 4, adding those existing
hop-3 edges (cycle_rank 75 → 149) as extra constraints — NO new edges are
built, only existing ones are re-used at a lower threshold.

Questions this method answers:
  * Does the extra multi-hop redundancy help average out the coherent drift?
    (hypothesis: NO for the coherent component, since hop-3 edges carry
     3× the constant-rate bias; YES only for the incoherent 24% component)
  * What is the actual trade-off: more constraint cycles vs lower overlap
    (noisier Q edges)?

Edge weights are loaded from the CSV verbatim (they already down-weight by
n_overlap/12 and Q dispersion), so hop-3 edges enter with weight ~0.33×.

Outputs (in 02_regularization_methods/):
  LOWER_THRESHOLD_SUMMARY.json — per (scenario, threshold) solve stats

Usage:
    python 02_regularization_methods/lower_threshold_solver.py \
        --edge-npz <npz with i,j,Q,weight> --n-windows 77 --threshold 4
"""
import argparse, json, os
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
METHODS_DIR = os.path.join(PHASE3C, "14_regularization_ceiling_v34", "02_regularization_methods")
os.makedirs(METHODS_DIR, exist_ok=True)

# reuse the baseline solver + MST init from the Phase 3C.3 module
import sys
sys.path.insert(0, os.path.join(PHASE3C, "12_rotation_sync_v33", "04_so3_sync"))
from run_so3_sync import so3_sync_solve


def solve_lower_threshold(n_windows, i, j, Q, w, threshold, out=None):
    """Graph sync with the given edge set (threshold already applied upstream)."""
    G, stats, per_edge = so3_sync_solve(n_windows, i, j, Q, w, anchor=0)
    stats["threshold"] = threshold
    stats["n_edges_used"] = len(i)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez(out, G=G, threshold=threshold, method="C_lower_threshold",
                 edge_res_median_deg=stats["final_residual_median_deg"])
    return G, stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-npz", required=True)
    ap.add_argument("--n-windows", type=int, required=True)
    ap.add_argument("--threshold", type=int, required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.edge_npz)
    G, stats = solve_lower_threshold(args.n_windows, d["i"].astype(int),
                                     d["j"].astype(int), d["Q"], d["weight"],
                                     args.threshold, out=args.out)
    print(f"threshold={args.threshold}: E={len(d['i'])} "
          f"res_med={stats['final_residual_median_deg']:.3f}° "
          f"-> {args.out}")