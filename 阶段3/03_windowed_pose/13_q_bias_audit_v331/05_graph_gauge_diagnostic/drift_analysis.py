#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3.1 §20-§23: Trajectory drift and temporal-coherence analysis.

Mechanism test: per-step Q errors along the 76-hop chain are temporally coherent
(co-directed), producing near-linear trajectory drift far exceeding random walk.

For each sequence:
  1. Build the sequential chain G_chain[k] = Q_01 Q_12 ... Q_{k-1,k} (hop-1 edges)
  2. Load corrected GT gauges G_ref[k]
  3. Anchored trajectory drift: angle(G_chain[k] vs G_ref[0]^T G_ref[k])
     (drift at window 0 = 0 by construction; grows monotonically if errors coherent)
  4. Per-step error rotvec rv_k = log( Q_err_{k,k+1} ) where
     Q_err = (G_chain[k]^T G_chain[k+1])^T (G_ref[k]^T G_ref[k+1])
  5. Temporal coherence ratio C = |Σ_k rv_k| / Σ_k |rv_k|
     C ≈ 1 → errors fully co-directed (linear accumulation)
     C ≈ 0 → errors random (sqrt(N) accumulation)
  6. Random-walk expectation: sqrt(N) × median per-step error, vs observed drift

Outputs (in 05_graph_gauge_diagnostic/):
  DRIFT_TRAJECTORY_<seq>.csv        — per-window accumulated drift
  DRIFT_COHERENCE_SUMMARY.json      — per-sequence coherence statistics

DIAGNOSTIC_ONLY — uses COLMAP reference gauges for comparison only.

Usage:
    python 05_graph_gauge_diagnostic/drift_analysis.py [--seq ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
AUDIT_DIR = os.path.join(PHASE3C, "13_q_bias_audit_v331")
GT_GAUGES_DIR = os.path.join(AUDIT_DIR, "02_window_gauge_fit")
OUT_DIR = os.path.join(AUDIT_DIR, "05_graph_gauge_diagnostic")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(SYNC_DIR, "04_so3_sync"))
sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))
from run_so3_sync import load_edges_from_csv
from evaluate_multoplant import rot_angle_deg

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)


def build_chain(seq_id, G_ref, N_w):
    """Build the sequential chain of gauges from hop-1 measured Q edges.

    G_chain[0] = G_ref[0]  (anchor in the reference frame)
    G_chain[k] = G_chain[k-1] @ Q_{k-1,k}  (per Q_ij = G_i^T G_j)
    """
    ei, ej, Q, w = load_edges_from_csv(seq_id, "stride4", 8)
    if ei is None:
        return None
    hop1 = sorted([(int(ei[k]), int(ej[k])) for k in range(len(ei))
                   if int(ej[k]) - int(ei[k]) == 1])
    if not hop1:
        return None
    # index Q by (i,j)
    q_by_pair = {}
    for k in range(len(ei)):
        q_by_pair[(int(ei[k]), int(ej[k]))] = Q[k]
    gmap = {0: G_ref[0]}
    for a, b in hop1:
        if a in gmap and b not in gmap and (a, b) in q_by_pair:
            gmap[b] = gmap[a] @ q_by_pair[(a, b)]
    # fill missing windows via nearest predecessor (robust)
    for k in range(1, N_w):
        if k not in gmap:
            for pp in range(k - 1, max(-1, k - 4), -1):
                if pp in gmap and (pp, k) in q_by_pair:
                    gmap[k] = gmap[pp] @ q_by_pair[(pp, k)]
                    break
    G_chain = np.array([gmap[k] for k in range(N_w)])
    return G_chain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()
    sequences = args.seq if args.seq else list(SEQUENCES)

    summary = {}

    for seq_id in sequences:
        print(f"\n=== {seq_id} ===")
        gt_path = os.path.join(GT_GAUGES_DIR, f"CORRECTED_WINDOW_GAUGES_{seq_id}.npz")
        if not os.path.exists(gt_path):
            print("  SKIP: no corrected GT gauges")
            continue
        G_ref = np.load(gt_path)["G"]
        N_w = len(G_ref)

        G_chain = build_chain(seq_id, G_ref, N_w)
        if G_chain is None:
            print("  SKIP: no hop-1 edges")
            continue

        # ---- Anchored trajectory drift ----
        # Anchor: G_chain[0] = G_ref[0] (reference frame); drift[k] = angle(G_chain[k], G_ref[k])
        drift = np.array([rot_angle_deg(G_chain[k].T @ G_ref[k]) for k in range(N_w)])

        # ---- Per-step error rotvecs ----
        step_rv = []
        for k in range(N_w - 1):
            Q_chain = G_chain[k].T @ G_chain[k + 1]   # measured Q (hop-1)
            Q_ref = G_ref[k].T @ G_ref[k + 1]          # GT-implied Q
            Q_err = Q_chain.T @ Q_ref
            r = Rotation.from_matrix(Q_err)
            step_rv.append(r.as_rotvec())  # axis * angle (radians)
        step_rv = np.array(step_rv)
        step_angles_deg = np.degrees(np.linalg.norm(step_rv, axis=1))
        vecsum_deg = np.degrees(np.linalg.norm(step_rv.sum(axis=0)))
        scalar_sum_deg = float(np.sum(step_angles_deg))
        coherence = float(vecsum_deg / scalar_sum_deg) if scalar_sum_deg > 0 else 0.0
        rand_walk_est = float(np.sqrt(len(step_angles_deg)) * np.median(step_angles_deg))

        print(f"  chain: N={N_w} steps={len(step_angles_deg)}")
        print(f"  per-step: median={np.median(step_angles_deg):.3f}° "
              f"mean={np.mean(step_angles_deg):.3f}° max={np.max(step_angles_deg):.3f}°")
        print(f"  drift: final(w{N_w-1})={drift[-1]:.1f}° max={drift.max():.1f}°")
        print(f"  coherence C={coherence:.2f} | random-walk est={rand_walk_est:.1f}°")
        print(f"  amplification = {drift.max()/max(rand_walk_est,1e-9):.1f}x over random walk")

        # ---- Save drift trajectory CSV ----
        drift_csv = os.path.join(OUT_DIR, f"DRIFT_TRAJECTORY_{seq_id}.csv")
        with open(drift_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["window", "accumulated_drift_deg"])
            for k in range(N_w):
                w.writerow([k, round(float(drift[k]), 3)])

        summary[seq_id] = {
            "n_windows": N_w,
            "n_chain_steps": len(step_angles_deg),
            "per_step_error_median_deg": round(float(np.median(step_angles_deg)), 4),
            "per_step_error_mean_deg": round(float(np.mean(step_angles_deg)), 4),
            "per_step_error_max_deg": round(float(np.max(step_angles_deg)), 4),
            "final_drift_deg": round(float(drift[-1]), 2),
            "max_drift_deg": round(float(drift.max()), 2),
            "temporal_coherence_ratio": round(coherence, 3),
            "random_walk_expectation_deg": round(rand_walk_est, 2),
            "drift_amplification_vs_random_walk": round(float(drift.max() / max(rand_walk_est, 1e-9)), 2),
            "scalar_sum_per_step_deg": round(scalar_sum_deg, 1),
            "vector_sum_deg": round(vecsum_deg, 1),
        }

    with open(os.path.join(OUT_DIR, "DRIFT_COHERENCE_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved DRIFT_COHERENCE_SUMMARY.json")


if __name__ == "__main__":
    main()
