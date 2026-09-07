#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Step 4: Real-data application of the regularization methods.

Applies the four regularization methods (A gaussian smoothing, B 2nd-order
penalty, C lower-threshold re-solve, D multi-anchor diagnostic) to the REAL
solver gauges and evaluates against COLMAP reference rotations using the SAME
frame-level convention as Phase 3C.3 (evaluate_rotation_sync.py):

  * per-frame global cameras  R_c2w_global[f] = G_reg[owner] @ R_c2w_local[f]
    (central-window ownership, build_global_cameras.py)
  * single global rotation Procrustes vs COLMAP reference rotations
  * metrics: rot_median, rot_p90, rot_max, pose PASS/FAIL gate (<=10/<=20)

The per-window gauge-vs-GT (corrected gauges) view is retained as a DIAGNOSTIC
track (drift_median/drift_final), NOT the headline: Phase 3C.3's D baseline of
44-53° langdon / 3.3° wheat / 0.60° mustc is per-FRAME, and short sequences
(2 windows) make per-window Procrustes degenerate.

Controls: wheat/mustc must stay below their Phase 3C.3 baseline + tolerance.

GT constraint: COLMAP reference rotations used ONLY for evaluation, never in a
solver. Corrected gauges used ONLY for the diagnostic drift track.

Usage:
    python 04_real_data_application/apply_regularization_to_real.py \
        [--seq ...] [--all] [--run-d]
"""
import argparse, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33", "04_so3_sync")
EDGES_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33", "02_rotation_edges")
GLOBAL_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33", "05_global_stitching")
RTN_EVAL_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33", "06_evaluation")
GT_DIR = os.path.join(PHASE3C, "13_q_bias_audit_v331", "02_window_gauge_fit")
STRIDE4_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4")
METH_DIR = os.path.join(PHASE3C, "14_regularization_ceiling_v34", "02_regularization_methods")
OUT_DIR = os.path.join(PHASE3C, "14_regularization_ceiling_v34", "04_real_data_application")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PHASE3C, "05_global_stitching_v31"))
from run_gauge_stitching import c2w_rotations

sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))
from evaluate_multoplant import global_rotation_procrustes, rot_angle_deg

sys.path.insert(0, RTN_EVAL_DIR)
from evaluate_rotation_sync import find_sequence_json, load_reference_poses, evaluate_orientation_only

sys.path.insert(0, METH_DIR)
from smoothing_gaussian import run_gaussian_smoothing, run_second_order_smoothing

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)

GAUSS_SIGMAS = [0.5, 1, 2, 3, 5, 8, 12, 16]
SECOND_ORDER_LAMS = [0.1, 1, 10, 100]

# Phase 3C.3 D baseline (per-frame rot_median, from ROTATION_SYNC_METHOD_COMPARISON.csv)
#   langdon 05-03 44.92° | 12-03 49.34° | 15-04 52.02° | 19-03 45.02°
#   wheat 461 3.28° | wheat 467 3.23° | mustc 0.60°
# Control gate: regularized rot_median must stay <= baseline + 1.0° and PASS.
CONTROL_TOL_DEG = 1.0


def baseline_from_phase3c():
    """Lock the Phase 3C.3 D-stride4 rot_median per seq (evaluation baseline)."""
    csv_path = os.path.join(RTN_EVAL_DIR, "ROTATION_SYNC_METHOD_COMPARISON.csv")
    base = {}
    with open(csv_path) as f:
        import csv
        for r in csv.DictReader(f):
            if r["method"] == "D_STRIDE4_SO3GRAPH":
                base[r["sequence"]] = float(r["rot_median"])
    return base


def load_windows(seq_id):
    """List window dicts {ext_w2c (n,3,4), frame_idx (n,)} from stride-4 outputs."""
    seq_dir = os.path.join(STRIDE4_DIR, seq_id)
    files = sorted(glob.glob(os.path.join(seq_dir, "window_*.npz")))
    windows = []
    for wf in files:
        d = np.load(wf)
        windows.append({"ext_w2c": d["ext_w2c_vggt"], "frame_idx": d["frame_idx"]})
    return windows


def build_global_rotations(windows, G):
    """R_c2w_global[f] = G_k @ R_c2w_local[f], central-window ownership.

    Mirrors build_global_cameras.py build_global_rotations.
    """
    intervals = []
    for w in windows:
        idx = np.asarray(w["frame_idx"], dtype=int)
        intervals.append((int(idx.min()), int(idx.max())))
    owners = {}
    for k, w in enumerate(windows):
        mid = (intervals[k][0] + intervals[k][1]) / 2.0
        for i, f in enumerate(np.asarray(w["frame_idx"], dtype=int)):
            if f not in owners or abs(f - mid) < owners[f][1]:
                owners[f] = (k, abs(f - mid))
    final_R = {}
    for k, w in enumerate(windows):
        R_local = c2w_rotations(w["ext_w2c"])
        for i, f in enumerate(np.asarray(w["frame_idx"], dtype=int)):
            if owners.get(f, (None,))[0] == k:
                final_R[f] = G[k] @ R_local[i]
    sorted_f = sorted(final_R.keys())
    return np.array([final_R[f] for f in sorted_f]), np.array(sorted_f)


def load_gt_gauges(seq_id):
    p = os.path.join(GT_DIR, f"CORRECTED_WINDOW_GAUGES_{seq_id}.npz")
    return np.load(p)["G"] if os.path.exists(p) else None


def window_drift_diagnostics(G_est, G_ref):
    """Per-window gauge drift vs corrected gauges (DIAGNOSTIC only)."""
    if G_ref is None:
        return {"drift_median": None, "drift_final": None, "note": "no corrected gauges"}
    n = min(len(G_est), len(G_ref))
    drift = np.array([rot_angle_deg(G_est[k].T @ G_ref[k]) for k in range(n)])
    return {"drift_median": round(float(np.median(drift)), 2),
            "drift_final": round(float(drift[-1]), 2)}


def evaluate_seq(seq_id, G_reg, windows, ref_w2c):
    """Frame-level evaluation for one gauges set."""
    R_pred, idx = build_global_rotations(windows, G_reg)
    idx_int = np.asarray(idx, dtype=int)
    ref_sub = ref_w2c[idx_int]
    res = evaluate_orientation_only(R_pred, ref_sub)
    return {"n_frames": res["n_frames"], "rot_median": round(res["rot_median"], 2),
            "rot_p90": round(res["rot_p90"], 2), "rot_max": round(res["rot_max"], 2),
            "pose_gate": res["pose_gate"]}


def apply_to_sequence(seq_id, baseline_d, run_d=False):
    """Apply A/B/C/D to one real sequence; evaluate per-frame vs COLMAP."""
    G_sync_path = os.path.join(SYNC_DIR, f"{seq_id}_SO3_SYNC_GAUGES_stride4_th8.npz")
    if not os.path.exists(G_sync_path):
        return None
    G_sync = np.load(G_sync_path)["G"]
    windows = load_windows(seq_id)
    n = len(G_sync)
    try:
        seq = find_sequence_json(seq_id)
        ref_w2c = load_reference_poses(seq)
    except Exception as e:
        print(f"  SKIP {seq_id}: no COLMAP reference ({e})")
        return None
    if ref_w2c is None:
        return None

    G_ref = load_gt_gauges(seq_id)

    base_phase3c = baseline_d.get(seq_id)
    out = {"sequence": seq_id, "n_windows": n,
           "baseline_phase3c_D_rot_median": base_phase3c,
           "n_frames_ref": len(ref_w2c)}
    methods = []
    b = evaluate_seq(seq_id, G_sync, windows, ref_w2c)
    methods.append({"method": "baseline_sync", "param": 0, **b,
                    **window_drift_diagnostics(G_sync, G_ref),
                    "delta_vs_phase3c_D": round(b["rot_median"] - base_phase3c, 2)
                    if base_phase3c else None})

    for s in GAUSS_SIGMAS:
        G_out = run_gaussian_smoothing(G_sync, s)
        m = evaluate_seq(seq_id, G_out, windows, ref_w2c)
        methods.append({"method": "A_gaussian", "param": s, **m,
                        **window_drift_diagnostics(G_out, G_ref)})

    for lam in SECOND_ORDER_LAMS:
        G_out = run_second_order_smoothing(G_sync, lam)
        m = evaluate_seq(seq_id, G_out, windows, ref_w2c)
        methods.append({"method": "B_second_order", "param": lam, **m,
                        **window_drift_diagnostics(G_out, G_ref)})

    # C: lower-threshold re-solve (full EDGE_QS th>=4 set incl hop-3)
    edge_npz = os.path.join(EDGES_DIR, f"{seq_id}_EDGE_QS.npz")
    if os.path.exists(edge_npz):
        d = np.load(edge_npz)
        sys.path.insert(0, SYNC_DIR)
        from run_so3_sync import so3_sync_solve
        G_c, stats_c, _ = so3_sync_solve(n, d["i"].astype(int), d["j"].astype(int),
                                         d["Q"], d["weight"], anchor=0)
        m = evaluate_seq(seq_id, G_c, windows, ref_w2c)
        methods.append({"method": "C_lower_threshold", "param": 4, **m,
                        "n_edges": int(len(d["i"])),
                        **window_drift_diagnostics(G_c, G_ref)})

    if run_d:
        from multi_anchor_solver import multi_anchor_solve
        d = np.load(edge_npz)
        for anchors in ([0], [0, n // 2], [0, n - 1], [0, n // 2, n - 1]):
            G_d, stats_d = multi_anchor_solve(
                n, d["i"].astype(int), d["j"].astype(int), d["Q"], d["weight"], anchors)
            m = evaluate_seq(seq_id, G_d, windows, ref_w2c)
            methods.append({"method": "D_multi_anchor",
                            "param": "-".join(map(str, anchors)), **m,
                            **window_drift_diagnostics(G_d, G_ref)})

    out["methods"] = methods
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--run-d", action="store_true",
                    help="also run multi-anchor diagnostic (slow, NOT deployable)")
    args = ap.parse_args()
    seqs = args.seq if args.seq else (list(SEQUENCES) if args.all else [SEQUENCES[0]])
    baseline_d = baseline_from_phase3c()

    results = []
    for seq_id in seqs:
        r = apply_to_sequence(seq_id, baseline_d, run_d=args.run_d)
        if r is None:
            continue
        results.append(r)
        b = r["methods"][0]
        best = min((m for m in r["methods"][1:]
                    if m.get("rot_median") is not None), key=lambda m: m["rot_median"], default=None)
        print(f"=== {seq_id} (N={r['n_windows']}, frames={b['n_frames']}) ===")
        print(f"  baseline sync: rot_med={b['rot_median']}° p90={b['rot_p90']}° "
              f"gate={b['pose_gate']} (Phase3C D={r['baseline_phase3c_D_rot_median']}°)")
        if best:
            print(f"  best regularized: {best['method']}@{best['param']} "
                  f"rot_med={best['rot_median']}° ({'improves' if best['rot_median'] < b['rot_median'] else 'does NOT improve'})")
        for m in r["methods"][1:]:
            if m.get("rot_median") is None:
                continue
            if m["method"] in ("C_lower_threshold", "D_multi_anchor") or \
               m["param"] == GAUSS_SIGMAS[0] or m["param"] == SECOND_ORDER_LAMS[0]:
                print(f"    {m['method']:<20} {str(m['param']):>4}: rot_med={m['rot_median']}° "
                      f"drift_final={m.get('drift_final')}°")

    with open(os.path.join(OUT_DIR, "REGULARIZATION_REAL_RESULT.json"), "w") as f:
        json.dump({
            "constraint": "COLMAP reference used ONLY for evaluation, never in a solver",
            "metric": "per-frame rot_median after single global Procrustes (Phase 3C.3 convention)",
            "control_gate": f"rot_median <= phase3c_D_baseline + {CONTROL_TOL_DEG} deg AND pose_gate=PASS",
            "phase3c_D_baseline_rot_median": baseline_d,
            "results": results}, f, indent=2)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'REGULARIZATION_REAL_RESULT.json')}")


if __name__ == "__main__":
    main()