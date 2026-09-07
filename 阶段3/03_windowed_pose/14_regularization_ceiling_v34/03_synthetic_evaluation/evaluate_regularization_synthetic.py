#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Step 3: Synthetic ceiling evaluation.

Runs all regularization methods on N synthetic drift scenarios and measures
drift suppression vs KNOWN ground truth.

Metric (same convention as Phase 3C.3 evaluation):
  rot_median = median per-window angle after SINGLE GLOBAL rotation Procrustes
  alignment of G_est to G_true. This removes the global gauge (window-0 anchor)
  and isolates INTERNAL trajectory deformation.

Methods + sweeps:
  A  gaussian increment smoothing         σ ∈ {0.5,1,2,3,5,8,12,16}
  B  2nd-order (acceleration) penalty     λ ∈ {0.1,1,10,100}
  C  lower threshold ≥4 (hop-3 edges)     re-solve on full edge set
  D  multi-anchor {0,mid,end}             diagnostic anchor redistribution
  R  reference = raw chain                no regularization (baseline)

Decision (reported per scenario and aggregate):
  GREEN  rot_median < 20°   — regularization suppresses drift meaningfully
  YELLOW rot_median 20–35°  — partial, worth combining
  RED    rot_median > 35°   — ceiling too high, close branch

GT used ONLY for scoring the synthetic testbed (never inside a method).

Usage:
    python 03_synthetic_evaluation/evaluate_regularization_synthetic.py \
        [--scenario A|B|C] [--n-scenarios 30] [--seed 0]
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
from synthetic_drift_generator import (
    fit_error_statistics, generate_true_trajectory, sample_coherent_errors,
    build_synthetic_edges, realized_stats, rot_angle_deg,
    LANGDON_SEQ,
)
from smoothing_gaussian import (
    run_gaussian_smoothing, run_second_order_smoothing,
)
from lower_threshold_solver import solve_lower_threshold

sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))
from evaluate_multoplant import global_rotation_procrustes

GAUSS_SIGMAS = [0.5, 1, 2, 3, 5, 8, 12, 16]
SECOND_ORDER_LAMS = [0.1, 1, 10, 100]

GREEN_MAX_DEG = 20.0
YELLOW_MAX_DEG = 35.0


def score_gauges(G_est, G_true):
    """Single global Procrustes + per-window rot error (degrees)."""
    n = min(len(G_est), len(G_true))
    Rg = global_rotation_procrustes(G_est[:n], G_true[:n])
    errs = np.array([
        rot_angle_deg((Rg @ G_est[i]).T @ G_true[i]) for i in range(n)
    ])
    drift_anchored = np.array([
        rot_angle_deg(G_est[i].T @ G_true[i]) for i in range(n)
    ])
    return {
        "n_windows": n,
        "rot_median": float(np.median(errs)),
        "rot_p90": float(np.percentile(errs, 90)),
        "rot_max": float(np.max(errs)),
        "drift_median": float(np.median(drift_anchored)),
        "drift_final": float(drift_anchored[-1]),
    }


def decision_from_median(m):
    if m < GREEN_MAX_DEG:
        return "GREEN"
    if m < YELLOW_MAX_DEG:
        return "YELLOW"
    return "RED"


def run_all_methods(edges, n_windows, G_true, G_chain, run_d=True):
    """Apply all methods + sweeps; return list of {method, param, metrics, decision}."""
    results = []

    # Reference: raw chain (baseline, no regularization)
    m = score_gauges(G_chain, G_true)
    results.append({"method": "R_raw_chain", "param": 0.0, **m,
                    "decision": decision_from_median(m["rot_median"])})

    # A: gaussian increment smoothing
    for s in GAUSS_SIGMAS:
        G_out = run_gaussian_smoothing(G_chain, s)
        m = score_gauges(G_out, G_true)
        results.append({"method": "A_gaussian", "param": s, **m,
                        "decision": decision_from_median(m["rot_median"])})

    # B: 2nd-order acceleration penalty
    for lam in SECOND_ORDER_LAMS:
        G_out = run_second_order_smoothing(G_chain, lam)
        m = score_gauges(G_out, G_true)
        results.append({"method": "B_second_order", "param": lam, **m,
                        "decision": decision_from_median(m["rot_median"])})

    # C: lower threshold — full edge set (hop-1/2/3) re-solve
    G_out, stats = solve_lower_threshold(n_windows, edges["i"].astype(int),
                                         edges["j"].astype(int), edges["Q"],
                                         edges["weight"], threshold=4)
    m = score_gauges(G_out, G_true)
    results.append({"method": "C_lower_threshold", "param": 4, **m,
                    "decision": decision_from_median(m["rot_median"])})

    # D: multi-anchor diagnostic (0, mid, end)
    if run_d:
        from multi_anchor_solver import multi_anchor_solve
        for anchors in ([0], [0, n_windows // 2], [0, n_windows - 1],
                        [0, n_windows // 2, n_windows - 1]):
            G_out, stats_d = multi_anchor_solve(
                n_windows, edges["i"].astype(int), edges["j"].astype(int),
                edges["Q"], edges["weight"], anchors)
            m = score_gauges(G_out, G_true)
            results.append({"method": "D_multi_anchor", "param": "-".join(map(str, anchors)),
                            **m, "decision": decision_from_median(m["rot_median"])})

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["A", "B", "C"], default="A")
    ap.add_argument("--n-scenarios", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=int, default=4,
                    help="edge-set threshold: 4 = full 225-edge set incl hop-3 "
                         "(Method C/D), 8 = 151-edge set (Methods A/B use only G_chain)")
    ap.add_argument("--n-d-scenarios", type=int, default=6,
                    help="how many scenarios run the slow multi-anchor diagnostic "
                         "(Method D, ~50s/scenario); must be <= n-scenarios")
    args = ap.parse_args()

    seq_id = LANGDON_SEQ
    mag_stats = fit_error_statistics(seq_id)
    axis_vary = 2.0 if args.scenario == "C" else 0.0

    # collect per-method-per-param medians across scenarios
    from collections import defaultdict
    agg = defaultdict(list)

    scenario_results = []
    for i in range(args.n_scenarios):
        seed = args.seed + i * 7
        G_true, traj_meta = generate_true_trajectory(seq_id, n_windows=77, seed=seed)
        errors, err_meta = sample_coherent_errors(
            76, mag_stats, mag_stats["coherence_target"], seed=seed,
            axis_vary_deg=axis_vary)
        edges = build_synthetic_edges(seq_id, G_true, errors, scenario=args.scenario,
                                      threshold=args.threshold, seed=seed)
        run_d = i < args.n_d_scenarios
        results = run_all_methods(edges, 77, G_true, edges["G_chain"], run_d=run_d)
        for r in results:
            key = (r["method"], str(r["param"]))
            agg[key].append(r["rot_median"])
        scenario_results.append({
            "scenario_idx": i, "seed": seed,
            "injected_drift_deg": round(float(edges["drift_injected"][-1]), 1),
            "methods": results,
        })
        if (i + 1) % 5 == 0 or i == args.n_scenarios - 1:
            r0 = results[0]
            print(f"  scenario {i+1}/{args.n_scenarios} seed={seed} "
                  f"drift={float(edges['drift_injected'][-1]):.1f}° "
                  f"R={r0['rot_median']:.1f}°", flush=True)

    # aggregate
    agg_out = {}
    best = {}
    for (method, param), vals in agg.items():
        vals = np.array(vals)
        entry = {
            "method": method, "param": float(param) if param.replace(".", "").isdigit() else param,
            "n_scenarios": len(vals),
            "rot_median_median": round(float(np.median(vals)), 2),
            "rot_median_mean": round(float(np.mean(vals)), 2),
            "rot_median_best": round(float(np.min(vals)), 2),
            "rot_median_p90": round(float(np.percentile(vals, 90)), 2),
            "decision": decision_from_median(float(np.median(vals))),
        }
        agg_out[f"{method}@{param}"] = entry
        # best per method
        bm = best.get(method)
        if bm is None or entry["rot_median_median"] < bm["rot_median_median"]:
            best[method] = entry

    # baseline reference
    ref_med = agg_out.get("R_raw_chain@0.0", {}).get("rot_median_median")

    summary = {
        "scenario": args.scenario,
        "n_scenarios": args.n_scenarios,
        "n_d_scenarios": args.n_d_scenarios,
        "threshold": args.threshold,
        "seed_base": args.seed,
        "seq": seq_id,
        "coherence_target": mag_stats["coherence_target"],
        "baseline_raw_chain_rot_median_deg": ref_med,
        "best_per_method": {k: v for k, v in sorted(best.items())},
        "all_method_params": agg_out,
        "decision_thresholds_deg": {"GREEN_lt": GREEN_MAX_DEG, "YELLOW_lt": YELLOW_MAX_DEG},
        "metric": "rot_median after single global Procrustes (Phase 3C.3 convention)",
        "constraint": "GT used ONLY for synthetic scoring, never inside methods",
        "scenario_results": scenario_results,
    }

    out_path = os.path.join(OUT_DIR, f"SYNTHETIC_CEILING_RESULT_scen{args.scenario}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"=== Scenario {args.scenario} (n={args.n_scenarios}) ===")
    print(f"  baseline raw chain rot_median: {ref_med}°")
    print(f"  best per method (rot_median_median):")
    for k, v in sorted(best.items()):
        print(f"    {k:24s} param={v['param']:<8} {v['rot_median_median']:6.2f}° "
              f"[{v['decision']}]")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()