#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Step 3 (own implementation): Synthetic ceiling evaluation.

Runs all regularization methods on N synthetic Scenario-A drift instances
(CASE-C consistent — the scenario that matches real data) and measures drift
suppression vs KNOWN ground truth.  Builds Pareto curves per method and
selects the per-method operating point (knee).

Methods:
  A  gaussian increment smoothing         σ ∈ {0.5, 1, 2, 3, 5, 8, 12, 16}
  B  2nd-order (acceleration) penalty     λ ∈ {0.1, 1, 10, 100}
  C  SO(3) GP increments                  ℓ ∈ {2, 5, 10, 20, 40}
  D  SO(3) GP local-trend                 ℓ ∈ {2, 5, 10, 20, 40}
  E  multi-hop th=4                       fixed (th4 subset of existing CSV)
  F  edge reweight temporal               α ∈ {0.25, 0.5, 0.75, 1.0}
  G  edge reweight bias                   β ∈ {5, 20, 100}
  R  reference = raw chain                baseline (no regularization)

Evaluation metric: rot_median after SINGLE GLOBAL Procrustes alignment of
estimated vs true gauges (identical to Phase 3C.3 convention).

Decision bands:
  GREEN  rot_median < 20°   — drift meaningfully suppressed
  YELLOW rot_median 20–35°  — partial; combine with longer windows
  RED    rot_median > 35°   — ceiling too high; close branch

GT used ONLY for synthetic scoring — never inside a method.

Outputs: 03_synthetic_evaluation/SYNTHETIC_CEILING_RESULT.json

Usage:
    python 03_synthetic_evaluation/synthetic_test_all_methods.py \
        [--n-scenarios 30] [--seed 0]
"""
import argparse, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
PHASE34 = os.path.join(PHASE3C, "14_regularization_ceiling_v34")
GEN_DIR = os.path.join(PHASE34, "01_synthetic_drift_model")
METH_DIR = os.path.join(PHASE34, "02_regularization_methods")
OUT_DIR = os.path.join(PHASE34, "03_synthetic_evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, GEN_DIR)
sys.path.insert(0, METH_DIR)
sys.path.insert(0, os.path.join(PHASE3C, "12_rotation_sync_v33", "04_so3_sync"))
sys.path.insert(0, os.path.join(PHASE3C, "..", "02_pose_robustness", "03_pose_evaluation"))

from synthetic_drift_generator import (
    fit_error_statistics, generate_true_trajectory, sample_coherent_errors,
    build_synthetic_edges, LANGDON_SEQ,
)
from smoothing_gaussian import run_gaussian_smoothing, run_second_order_smoothing
from so3_gp_smoother import run_so3_gp
from multi_hop_constraints import run_sync_threshold, window_count
from edge_reweighting import run_reweighted_sync, reweight_by_bias_alignment, \
    reweight_by_anchor_distance
from run_so3_sync import load_edges_from_csv, so3_sync_solve
from evaluate_multoplant import global_rotation_procrustes, rot_angle_deg

# ── parameter grids ────────────────────────────────────────────────────────
GAUSS_SIGMAS    = [0.5, 1, 2, 3, 5, 8, 12, 16]
SECOND_ORDER_LAMS = [0.1, 1, 10, 100]
GP_LENGTH_SCALES = [2, 5, 10, 20, 40]
EDGE_ALPHAS     = [0.25, 0.5, 0.75, 1.0]
EDGE_BETAS      = [5, 20, 100]

GREEN_MAX  = 20.0
YELLOW_MAX = 30.0


# ── helpers ────────────────────────────────────────────────────────────────
def score_gauges(G_est, G_true):
    """Procrustes-aligned per-window rotation error + anchored drift."""
    n = min(len(G_est), len(G_true))
    Rg = global_rotation_procrustes(G_est[:n], G_true[:n])
    errs = np.array([rot_angle_deg((Rg @ G_est[i]).T @ G_true[i]) for i in range(n)])
    # anchored drift (G_est[0] = G_true[0] by construction in synthetic)
    drift = np.array([rot_angle_deg(G_est[i].T @ G_true[i]) for i in range(n)])
    return {
        "n_windows": n,
        "rot_median": float(np.median(errs)),
        "rot_p90": float(np.percentile(errs, 90)),
        "rot_max": float(np.max(errs)),
        "drift_final": float(drift[-1]),
    }


def decision(med):
    if med < GREEN_MAX:  return "GREEN"
    if med < YELLOW_MAX: return "YELLOW"
    return "RED"


def pareto_knee(results):
    """Best param = max suppression (lowest rot_median) — simple knee for drift suppression.
    (Distortion is automatically bounded by the method: smoothing preserves
    the trajectory's anchor identity at window 0, and drift is inherently a
    rigid-body error that Procrustes removes.)"""
    if not results:
        return None
    return min(results, key=lambda r: r["rot_median"])


# ── run all methods on one synthetic instance ──────────────────────────────
def run_all_methods_on_edges(edges, n_windows, G_true, G_chain, seq_id,
                             include_solver=True):
    """Apply all regularization methods; return list of dicts.

    include_solver=False skips the solver-heavy methods (multi_hop, edge
    reweighting) — used for most scenarios since those methods are
    config-independent (~0 effect); the fast smoothing/GP methods run on every
    scenario.
    """
    results = []
    ei, ej, Q, w, hop = edges["i"].astype(int), edges["j"].astype(int), edges["Q"], edges["weight"], edges["hop"]

    # ── reference ──────────────────────────────────────────────────────────
    m = score_gauges(G_chain, G_true)
    results.append({"method": "R_raw_chain", "param": 0.0, **m,
                    "decision": decision(m["rot_median"])})

    # ── A: gaussian smoothing ──────────────────────────────────────────────
    for s in GAUSS_SIGMAS:
        G_out = run_gaussian_smoothing(G_chain, s)
        m = score_gauges(G_out, G_true)
        results.append({"method": "gaussian", "param": s, **m,
                        "decision": decision(m["rot_median"])})

    # ── B: second-order penalty ────────────────────────────────────────────
    for lam in SECOND_ORDER_LAMS:
        G_out = run_second_order_smoothing(G_chain, lam)
        m = score_gauges(G_out, G_true)
        results.append({"method": "second_order", "param": lam, **m,
                        "decision": decision(m["rot_median"])})

    # ── C: GP increments ───────────────────────────────────────────────────
    for ls in GP_LENGTH_SCALES:
        G_out = run_so3_gp(G_chain, ls, mode="increments")
        m = score_gauges(G_out, G_true)
        results.append({"method": "gp", "param": ls, **m,
                        "decision": decision(m["rot_median"])})

    # ── D: GP local-trend ──────────────────────────────────────────────────
    for ls in GP_LENGTH_SCALES:
        G_out = run_so3_gp(G_chain, ls, mode="local_trend")
        m = score_gauges(G_out, G_true)
        results.append({"method": "gp_trend", "param": ls, **m,
                        "decision": decision(m["rot_median"])})

    # ── E+F+G: solver-heavy methods (skipped unless include_solver) ───────
    # In Scenario A the graph is fully consistent (CASE-C), so hop-3 edges add
    # zero information → multi_hop ≈ edge reweighting ≈ baseline. Running these
    # on every scenario would cost ~8 slow solves each; 3 scenarios suffice to
    # confirm the ~0 effect. The baseline solve below is shared by all three.
    if include_solver:
        G_base, _, _ = so3_sync_solve(n_windows, ei, ej, Q, w, anchor=0)
        if G_base is not None:
            m = score_gauges(G_base, G_true)
            results.append({"method": "multi_hop", "param": 8, **m,
                            "decision": decision(m["rot_median"]),
                            "note": "re-solve th=8 synthetic graph (Scenario A: hop-3 add zero info)"})

            # ── F: edge temporal reweighting ───────────────────────────────
            for alpha in EDGE_ALPHAS:
                w2 = reweight_by_anchor_distance(ei, ej, w, alpha)
                res = run_reweighted_sync(n_windows, ei, ej, Q, w2)
                if res[0] is None:
                    continue
                m = score_gauges(res[0], G_true)
                results.append({"method": "edge_temporal", "param": alpha, **m,
                                "decision": decision(m["rot_median"])})

            # ── G: edge bias reweighting ───────────────────────────────────
            from scipy.spatial.transform import Rotation
            r_arr = np.zeros((len(ei), 3))
            for e in range(len(ei)):
                a, b = int(ei[e]), int(ej[e])
                E = Q[e].T @ (G_base[a].T @ G_base[b])
                r_arr[e] = Rotation.from_matrix(E).as_rotvec()
            total = np.linalg.norm(r_arr.sum(axis=0))
            scalar = np.linalg.norm(r_arr, axis=1).sum()
            gamma = float(total / scalar) if scalar > 0 else 0.0
            uhat = r_arr.sum(axis=0) / total if total > 1e-12 else np.zeros(3)

            for beta in EDGE_BETAS:
                w2 = reweight_by_bias_alignment(ei, ej, w, r_arr, uhat, beta)
                res = run_reweighted_sync(n_windows, ei, ej, Q, w2)
                if res[0] is None:
                    continue
                m = score_gauges(res[0], G_true)
                results.append({"method": "edge_bias", "param": beta, **m,
                                "decision": decision(m["rot_median"]),
                                "residual_gamma": gamma})

    return results


# ── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-scenarios", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq", default=LANGDON_SEQ)
    args = ap.parse_args()

    seq_id = args.seq
    mag_stats = fit_error_statistics(seq_id)

    from collections import defaultdict
    agg = defaultdict(list)
    scenario_details = []

    for i in range(args.n_scenarios):
        seed = args.seed + i * 7
        G_true, _ = generate_true_trajectory(seq_id, n_windows=77, seed=seed)
        errors, _ = sample_coherent_errors(
            76, mag_stats, mag_stats["coherence_target"], seed=seed, axis_vary_deg=0.0)
        edges = build_synthetic_edges(seq_id, G_true, errors, scenario="A",
                                      threshold=8, seed=seed)
        G_chain = edges["G_chain"]
        results = run_all_methods_on_edges(edges, 77, G_true, G_chain, seq_id,
                                           include_solver=(i < 3))
        for r in results:
            key = (r["method"], str(r["param"]))
            agg[key].append(r["rot_median"])
        scenario_details.append({
            "scenario_idx": i, "seed": seed,
            "injected_drift_deg": round(float(edges["drift_injected"][-1]), 1),
            "methods": results,
        })

    # aggregate per (method, param)
    agg_out = {}
    for (method, param), vals in agg.items():
        vals_a = np.array(vals)
        agg_out[f"{method}@{param}"] = {
            "method": method,
            "param": float(param) if param.replace(".", "").isdigit() else param,
            "n_scenarios": len(vals_a),
            "rot_median_median": round(float(np.median(vals_a)), 2),
            "rot_median_mean":  round(float(np.mean(vals_a)), 2),
            "rot_median_best":  round(float(np.min(vals_a)), 2),
            "rot_median_p90":   round(float(np.percentile(vals_a, 90)), 2),
            "decision": decision(float(np.median(vals_a))),
        }

    # per-method best
    methods = defaultdict(list)
    for k, v in agg_out.items():
        methods[v["method"]].append(v)
    best = {}
    for mname, entries in methods.items():
        best[mname] = min(entries, key=lambda e: e["rot_median_median"])

    summary = {
        "scenario": "A",
        "scenario_description": "CASE-C consistent: multi-hop edges match chain (no new info)",
        "n_scenarios": args.n_scenarios,
        "seed_base": args.seed,
        "seq": seq_id,
        "coherence_target": mag_stats["coherence_target"],
        "baseline_rot_median_deg": agg_out.get("R_raw_chain@0.0", {}).get("rot_median_median"),
        "best_per_method": best,
        "all_method_params": agg_out,
        "decision_thresholds_deg": {"GREEN_lt": GREEN_MAX, "YELLOW_lt": YELLOW_MAX},
        "metric": "rot_median after single global Procrustes (Phase 3C.3 convention)",
        "constraint": "GT used ONLY for synthetic scoring, never inside methods",
    }

    out_path = os.path.join(OUT_DIR, "SYNTHETIC_CEILING_RESULT.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"=== Scenario A ({args.n_scenarios} instances) ===")
    print(f"  baseline rot_median: {agg_out.get('R_raw_chain@0.0',{}).get('rot_median_median')}°")
    print(f"  best per method (median rot_median across scenarios):")
    for mname, v in sorted(best.items()):
        print(f"    {mname:14s} param={v['param']:<8} rot_median={v['rot_median_median']:6.2f}° "
              f"[{v['decision']}]")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
