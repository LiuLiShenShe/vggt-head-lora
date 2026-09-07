#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Step 4: Real-data application — transfer regularization to real sequences.

For each of the 7 real sequences, load the Phase 3C.3 solver gauges (stride4_th8,
= baseline Method D), apply ONE transferred configuration per regularization
method (configs come from the synthetic ceiling Pareto knee, NOT a per-seq
re-sweep — anti-overfit guard §5.3 of the plan), assemble global camera
rotations, and evaluate.

Two evaluation views (both GT-for-evaluation-only; GT never enters a method):
  1. COLMAP view  — assembled R_c2w_global vs COLMAP reference, single global
     rotation Procrustes, orientation-only. Same convention as Phase 3C.3
     method-comparison table; baseline D values reproduce the headline
     (langdon 44.9–52.0°, wheat ~3.3°, mustc 0.6°).
  2. GT-gauge view — window-level gauges G_reg vs CORRECTED_WINDOW_GAUGES
     (trusted GT, eval only), single global rotation Procrustes. Measures the
     trajectory deformation (retained drift + signal distortion) directly.

Controls gate: wheat_461/467, mustc — Δrot_median ≤ +1.0° vs baseline D and the
pose gate must stay PASS. 1.0° tolerance is deliberately tight because controls
are short (6 / 2 windows) where any smoothing is pure distortion.

Outputs:
  REGULARIZED_GAUGES_<seq>_xfer_<method>.npz   (G_out, transferred config)
  REGULARIZATION_RESULT.json                   (per seq × method metrics, controls, verdict)

Usage:
    python 04_real_data_application/apply_regularization.py [--seq ...] [--force]
"""
import argparse, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
SOLVER_DIR = os.path.join(SYNC_DIR, "04_so3_sync")
GT_GAUGES_DIR = os.path.join(PHASE3C, "13_q_bias_audit_v331", "02_window_gauge_fit")
WIN_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4")
PHASE34 = os.path.join(PHASE3C, "14_regularization_ceiling_v34")
METH_DIR = os.path.join(PHASE34, "02_regularization_methods")
SYNTH_DIR = os.path.join(PHASE34, "03_synthetic_evaluation")
OUT_DIR = os.path.join(PHASE34, "04_real_data_application")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))
from evaluate_multoplant import global_rotation_procrustes, rot_angle_deg
sys.path.insert(0, os.path.join(PHASE3C, "05_global_stitching_v31"))
from run_gauge_stitching import c2w_rotations
sys.path.insert(0, METH_DIR)
from smoothing_gaussian import run_gaussian_smoothing, run_second_order_smoothing
from so3_gp_smoother import run_so3_gp

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)
CONTROL_SEQUENCES = ["wheat3dgs__plot_461", "wheat3dgs__plot_467",
                     "mustc__plot198__230613__ugv__pos00"]

# Decision bands for the real langdon rot_median (COLMAP view), same convention
# as the plan §6 (baseline D ≈ 45–52°, so ≥44° means "no gain").
GREEN_MAX_DEG = 20.0
YELLOW_MAX_DEG = 44.0

# Transferred configurations. These are the synthetic Pareto-knee operating
# points (method + relative param in window units). Fallback defaults used ONLY
# when the synthetic ceiling result is unavailable; never re-swept on real data.
DEFAULT_TRANSFERS = {
    "gaussian": 3.0,        # σ (windows) — increment-domain Gaussian low-pass
    "second_order": 10.0,   # λ — acceleration penalty probe (constant-rate bias survives)
    "gp": 10.0,             # ℓ (windows) — SO(3) GP, increments mode
    "gp_trend": 10.0,       # ℓ — local-trend GP (trend from first 5 windows)
    "multi_hop": 4,         # threshold — th≥4 subset of existing CSV (hop-3 included)
    "edge_temporal": 0.5,   # α — temporal attenuation w·(1−α·d/dmax)
    "edge_bias": 20.0,      # β — bias-alignment downweighting
}
CONTROL_TOLERANCE_DEG = 1.0
POSE_GATE_MEDIAN_DEG = 10.0
POSE_GATE_P90_DEG = 20.0


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def load_solver_gauges(seq_id):
    """Baseline D gauges (stride4 th8) — the input to every regularizer."""
    p = os.path.join(SOLVER_DIR, f"{seq_id}_SO3_SYNC_GAUGES_stride4_th8.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["G"]


def load_gt_gauges(seq_id):
    """Trusted corrected GT gauges (EVALUATION ONLY — never inside a method)."""
    p = os.path.join(GT_GAUGES_DIR, f"CORRECTED_WINDOW_GAUGES_{seq_id}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["G"]


def load_windows(seq_id):
    seq_dir = os.path.join(WIN_DIR, seq_id)
    import glob
    files = sorted(glob.glob(os.path.join(seq_dir, "window_*.npz")))
    windows = []
    for wf in files:
        d = np.load(wf)
        windows.append({"ext_w2c": d["ext_w2c_vggt"], "frame_idx": d["frame_idx"]})
    return windows


def find_sequence_json(seq_id):
    for subdir in ["plant_view", "wheat3dgs", "mustc"]:
        for jp in __import__("glob").glob(os.path.join(SEQ_BASE, subdir, "*.json")):
            with open(jp) as f:
                meta = json.load(f)
            if meta.get("sequence_id") == seq_id:
                return meta
    raise FileNotFoundError(f"Sequence JSON not found for {seq_id}")


def load_reference_poses(seq):
    ext_path = seq.get("extrinsics_path")
    if not ext_path or not os.path.exists(ext_path):
        return None
    with open(ext_path) as f:
        ext_data = json.load(f)
    ref_exts = ext_data.get("extrinsics", [])
    return np.array([np.array(e["w2c"])[:3, :4] for e in ref_exts])


# --------------------------------------------------------------------------
# Transferred config selection (from synthetic ceiling result)
# --------------------------------------------------------------------------
def select_transferred_configs():
    """Pareto-knee params from the synthetic ceiling result, else documented defaults.

    Reads 03_synthetic_evaluation/SYNTHETIC_CEILING_RESULT.json (best_per_method
    = {method: {param, rot_median_median, decision}}). Method names there match
    the real-data method keys.
    """
    cands = [os.path.join(SYNTH_DIR, "SYNTHETIC_CEILING_RESULT.json")]
    for c in cands:
        if not os.path.exists(c):
            continue
        with open(c) as f:
            res = json.load(f)
        best = res.get("best_per_method") or {}
        if not best:
            break
        configs = {}
        for method, default in DEFAULT_TRANSFERS.items():
            b = best.get(method)
            if b and "param" in b and b.get("param") is not None:
                configs[method] = b["param"]
            else:
                configs[method] = default
        return configs, os.path.basename(c)
    return dict(DEFAULT_TRANSFERS), "DEFAULT_TRANSFERS (synthetic result not found)"


# --------------------------------------------------------------------------
# Method application (1 config per method — no per-seq re-sweep)
# --------------------------------------------------------------------------
def _load_regularized_npz(seq_id, tag):
    p = os.path.join(METH_DIR, f"REGULARIZED_GAUGES_{seq_id}_{tag}.npz")
    if not os.path.exists(p):
        return None
    return np.load(p)["G"]


def apply_transferred_method(seq_id, method, param):
    """Return (G_out, note). G_out has same anchor as G_in (window 0 preserved)."""
    G_in = load_solver_gauges(seq_id)
    if G_in is None:
        return None, "no solver gauges"
    n = len(G_in)

    if method == "gaussian":
        return run_gaussian_smoothing(G_in, float(param)), "increment Gaussian σ=%s" % param
    if method == "second_order":
        return run_second_order_smoothing(G_in, float(param)), "2nd-order λ=%s" % param
    if method == "gp":
        return run_so3_gp(G_in, float(param), mode="increments"), "SO(3) GP ℓ=%s" % param
    if method == "gp_trend":
        return run_so3_gp(G_in, float(param), mode="local_trend"), "local-trend GP ℓ=%s" % param
    if method == "multi_hop":
        G_m = _load_regularized_npz(seq_id, "mhop_th4")
        if G_m is not None and len(G_m) == n:
            return G_m, "th≥4 re-solve (hop-3 from existing CSV)"
        return G_in.copy(), "no hop-3 edges for this sequence (th4==th8)"
    if method == "edge_temporal":
        G_r = _load_regularized_npz(seq_id, f"rw_t{param}")
        if G_r is not None and len(G_r) == n:
            return G_r, "temporal attenuation α=%s" % param
        return G_in.copy(), "reweighting output missing (α=%s)" % param
    if method == "edge_bias":
        G_r = _load_regularized_npz(seq_id, f"rw_b{param}")
        if G_r is not None and len(G_r) == n:
            return G_r, "bias-alignment β=%s" % param
        return G_in.copy(), "reweighting output missing (β=%s)" % param
    raise ValueError(f"unknown method {method}")


def save_regularized(seq_id, method, G_out, note, param):
    npz = os.path.join(OUT_DIR, f"REGULARIZED_GAUGES_{seq_id}_xfer_{method}.npz")
    np.savez(npz, G=G_out, method=method, param=param, note=note,
             anchor_window=0, input_gauge="so3_sync_stride4_th8",
             n_windows=len(G_out), source="transferred-from-synthetic-knee")
    return npz


# --------------------------------------------------------------------------
# Global camera assembly + evaluation
# --------------------------------------------------------------------------
def assemble_global_cameras(seq_id, G_reg):
    """Frame → closest-window-midpoint ownership; R_c2w_global = G_owner @ R_local.

    Identical mapping to Phase 3C.3 build_global_cameras so rot_median is
    directly comparable to the baseline D row of the method-comparison table.
    """
    windows = load_windows(seq_id)
    if len(windows) != len(G_reg):
        return None, None
    intervals = []
    for w in windows:
        idx = np.asarray(w["frame_idx"], dtype=int)
        intervals.append((int(idx.min()), int(idx.max())))

    owners = {}
    for k, w in enumerate(windows):
        mid = (intervals[k][0] + intervals[k][1]) / 2.0
        R_local = c2w_rotations(w["ext_w2c"])
        for i, f in enumerate(np.asarray(w["frame_idx"], dtype=int)):
            if f not in owners:
                owners[f] = (k, abs(f - mid))
            else:
                if abs(f - mid) < owners[f][1]:
                    owners[f] = (k, abs(f - mid))

    final_R = {}
    for k, w in enumerate(windows):
        R_local = c2w_rotations(w["ext_w2c"])
        for i, f in enumerate(np.asarray(w["frame_idx"], dtype=int)):
            if owners.get(f, (None,))[0] == k:
                final_R[f] = G_reg[k] @ R_local[i]

    sorted_f = sorted(final_R.keys())
    R_global = np.array([final_R[f] for f in sorted_f])
    idx = np.array(sorted_f)
    return R_global, idx


def evaluate_vs_reference(seq_id, R_c2w_global, frame_idx):
    """COLMAP view: orientation-only rot_median/p90 + pose gate (Phase 3C.3 convention)."""
    try:
        seq = find_sequence_json(seq_id)
        ref_w2c = load_reference_poses(seq)
    except FileNotFoundError:
        return None
    if ref_w2c is None:
        return None
    idx_int = np.asarray(frame_idx, dtype=int)
    ref_sub = ref_w2c[idx_int]
    n = min(len(R_c2w_global), len(ref_sub))
    R_ref_c2w = ref_sub[:n, :3, :3].transpose(0, 2, 1)
    Rg = global_rotation_procrustes(R_c2w_global[:n], R_ref_c2w)
    errs = np.array([
        rot_angle_deg((Rg @ R_c2w_global[i]).T @ R_ref_c2w[i]) for i in range(n)
    ])
    gate = float(np.median(errs)) <= POSE_GATE_MEDIAN_DEG and \
        float(np.percentile(errs, 90)) <= POSE_GATE_P90_DEG
    return {
        "n_frames": n,
        "rot_median": float(np.median(errs)),
        "rot_p90": float(np.percentile(errs, 90)),
        "rot_mean": float(np.mean(errs)),
        "rot_max": float(np.max(errs)),
        "pose_gate": "PASS" if gate else "FAIL",
    }


def evaluate_vs_gt_gauge(seq_id, G_reg):
    """GT-gauge view: single global Procrustes of G_reg onto corrected GT gauges.

    Window-level trajectory deformation = retained coherent drift + smoothing
    distortion. GT used ONLY here for scoring.

    The anchored drift component is computed after re-anchoring G_reg so that
    G_reg[0] = G_gt[0] (the same normalization drift_analysis.py applies via
    G_chain[0]=G_ref[0]); otherwise a constant global offset between the
    solver anchor (identity) and the corrected GT anchor dominates and the
    drift trajectory is meaningless.
    """
    G_gt = load_gt_gauges(seq_id)
    if G_gt is None:
        return None
    n = min(len(G_reg), len(G_gt))
    Rg = global_rotation_procrustes(G_reg[:n], G_gt[:n])
    errs = np.array([rot_angle_deg((Rg @ G_reg[i]).T @ G_gt[i]) for i in range(n)])
    # re-anchor to GT window-0 frame: G_n = G_reg @ (G_reg[0]^T G_gt[0]) → G_n[0]=G_gt[0]
    G_n = G_reg[:n] @ (G_reg[0].T @ G_gt[0])
    drift_anchored = np.array([rot_angle_deg(G_n[i].T @ G_gt[i]) for i in range(n)])
    return {
        "n_windows": n,
        "gt_rot_median": float(np.median(errs)),
        "gt_rot_p90": float(np.percentile(errs, 90)),
        "gt_rot_max": float(np.max(errs)),
        "gt_drift_final": float(drift_anchored[-1]),
        "gt_drift_max": float(np.max(drift_anchored)),
    }


# --------------------------------------------------------------------------
# Controls gate
# --------------------------------------------------------------------------
def control_regression_check(rows, tolerance_deg=CONTROL_TOLERANCE_DEG):
    """Per-control Δrot_median vs baseline D must stay ≤ +tolerance and gate PASS."""
    base = {r["sequence"]: r["rot_median"] for r in rows if r["is_baseline"]}
    out = {"tolerance_deg": tolerance_deg, "per_control": {}, "pass": True}
    for seq in CONTROL_SEQUENCES:
        if seq not in base or base[seq] is None:
            continue
        entries = []
        for r in rows:
            if r["sequence"] != seq or r["is_baseline"]:
                continue
            delta = None
            if r["rot_median"] is not None:
                delta = r["rot_median"] - base[seq]
            entries.append({"method": r["method"], "rot_median": r["rot_median"],
                            "delta_vs_baseline_deg": round(delta, 4) if delta is not None else None,
                            "pose_gate": r["pose_gate"]})
        ok = all(
            e["rot_median"] is not None
            and e["delta_vs_baseline_deg"] is not None
            and e["delta_vs_baseline_deg"] <= tolerance_deg
            and e["pose_gate"] == "PASS"
            for e in entries
        )
        out["per_control"][seq] = {"baseline_rot_median_deg": round(base[seq], 4),
                                   "methods": entries, "pass": ok}
        out["pass"] = out["pass"] and ok
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()
    sequences = args.seq if args.seq else list(SEQUENCES)

    configs, config_src = select_transferred_configs()
    print(f"Transferred configs from {config_src}:")
    for m, p in configs.items():
        print(f"  {m:14s} {p}")

    rows = []

    for seq_id in sequences:
        G_in = load_solver_gauges(seq_id)
        if G_in is None:
            print(f"  SKIP {seq_id}: no solver gauges")
            continue
        G_gt = load_gt_gauges(seq_id)
        print(f"\n=== {seq_id} (N={len(G_in)}) ===")

        def _eval_row(method, G, note, param=None):
            R_glob, fidx = assemble_global_cameras(seq_id, G)
            ref = evaluate_vs_reference(seq_id, R_glob, fidx) if R_glob is not None else None
            gt = evaluate_vs_gt_gauge(seq_id, G)
            row = {
                "sequence": seq_id, "method": method, "param": param, "note": note,
                "is_baseline": method == "baseline_D",
                "rot_median": ref and ref["rot_median"],
                "rot_p90": ref and ref["rot_p90"],
                "pose_gate": ref and ref["pose_gate"],
                "n_frames": ref and ref["n_frames"],
                "gt_rot_median": gt and gt["gt_rot_median"],
                "gt_drift_final": gt and gt["gt_drift_final"],
                "gt_drift_max": gt and gt["gt_drift_max"],
            }
            rows.append(row)
            if ref:
                print(f"  {method:14s} rot_med={ref['rot_median']:6.2f}° "
                      f"p90={ref['rot_p90']:6.2f}° gate={ref['pose_gate']} "
                      f"| gt_med={gt['gt_rot_median']:6.2f}° drift_final={gt['gt_drift_final']:6.1f}°")
            return row

        # baseline D (reference)
        _eval_row("baseline_D", G_in, "stride4 th8 solver (unmodified)")

        for method in configs:
            param = configs[method]
            G_out, note = apply_transferred_method(seq_id, method, param)
            if G_out is None:
                print(f"  {method:14s} SKIP ({note})")
                continue
            save_regularized(seq_id, method, G_out, note, param)
            _eval_row(method, G_out, note, param)

    # drop None-only eval cells to keep JSON clean
    for r in rows:
        for k in ["rot_median", "rot_p90", "pose_gate", "n_frames",
                  "gt_rot_median", "gt_drift_final", "gt_drift_max"]:
            if r[k] is None:
                r[k] = None

    # controls gate
    controls = control_regression_check(rows)

    # per-method langdon aggregate (COLMAP view)
    agg = {}
    for r in rows:
        if r["sequence"].startswith("plantview__langdon_4__") and r["rot_median"] is not None:
            agg.setdefault(r["method"], []).append(r["rot_median"])
    langdon = {m: {"median_deg": round(float(np.median(v)), 2),
                   "mean_deg": round(float(np.mean(v)), 2)}
               for m, v in agg.items()}
    base_l = langdon.get("baseline_D", {}).get("median_deg")
    langdon_best = None
    for m, v in langdon.items():
        if m == "baseline_D":
            continue
        med = v["median_deg"]
        if langdon_best is None or med < langdon_best["median_deg"]:
            langdon_best = {"method": m, **v}
    if base_l is not None and langdon_best and langdon_best["median_deg"] < base_l:
        verdict = "GREEN" if langdon_best["median_deg"] < GREEN_MAX_DEG \
            else ("YELLOW" if langdon_best["median_deg"] < YELLOW_MAX_DEG else "RED")
    else:
        verdict = "RED"
    if not controls["pass"]:
        verdict = "RED"

    result = {
        "config_source": config_src,
        "transferred_configs": configs,
        "constraints": {
            "no_vggt_reinference": True, "no_new_long_range_edges": True,
            "no_imu_gravity_anchor_lora_msam": True,
            "gt_used_in_solver": False,
            "gt_used_for_evaluation_only": True,
            "phase3c3_untouched": True,
            "one_config_per_method_per_sequence": True,
        },
        "decision_bands_deg": {"GREEN_lt": GREEN_MAX_DEG, "YELLOW_lt": YELLOW_MAX_DEG,
                               "RED_ge": YELLOW_MAX_DEG},
        "baseline_D_langdon_median_deg": base_l,
        "best_regularized_langdon": langdon_best,
        "langdon_rot_median_by_method": langdon,
        "verdict": verdict,
        "controls": controls,
        "rows": rows,
    }

    out_path = os.path.join(OUT_DIR, "REGULARIZATION_RESULT.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"verdict={verdict} | best={langdon_best} | controls_pass={controls['pass']}")


if __name__ == "__main__":
    main()
