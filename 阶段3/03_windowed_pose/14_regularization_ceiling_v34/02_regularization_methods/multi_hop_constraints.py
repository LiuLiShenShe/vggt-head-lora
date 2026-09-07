#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.4 Step 2c: Multi-hop constraint enforcement.

Uses ONLY the existing ROTATION_GRAPH_EDGES.csv (hop-3 edges exist at th≥4 —
74/langdon). Three sub-experiments:

  M1 — lower threshold: rerun the UNCHANGED run_so3_sync.so3_sync_solve with
       threshold=4 (225 edges incl. hop-3) vs th=8 (151). Measures whether the
       existing hop-3 edges measurably change the solution.

  M2 — cycle-consistency diagnostic (the real test): per triangle of
       hop-1/hop-2/hop-3 edges, residual r_tri = ‖Log(Q_ij·(Q_ik Q_kj)ᵀ)‖.
       If ≪ accumulated drift (real: 0.37° ≪ 144°), the redundancy carries NO
       corrective information → multi-hop is provably not a lever.

  M3 — transitivity-augmented graph: add explicit composed-path Q_ik·Q_kj as
       constraints alongside direct Q (existing pairs only). Scenario B only.

Outputs:
  REGULARIZED_GAUGES_<seq>_mhop_th4.npz  (M1)
  M2: per-triangle residual CSVs + stats (real + synthetic)
  REGULARIZED_GAUGES_<seq>_trans.npz     (M3)

Usage:
    python 02_regularization_methods/multi_hop_constraints.py [--seq ...]
        [--synthetic-scen A|B|--no-synthetic]
"""
import argparse, csv, json, os, sys
import numpy as np
from collections import defaultdict

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
SOLVER_DIR = os.path.join(SYNC_DIR, "04_so3_sync")
EDGES_DIR = os.path.join(SYNC_DIR, "02_rotation_edges")
SYNTH_DIR = os.path.join(PHASE3C, "14_regularization_ceiling_v34", "01_synthetic_drift_model")
OUT_DIR = os.path.join(PHASE3C, "14_regularization_ceiling_v34", "02_regularization_methods")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, SOLVER_DIR)
from run_so3_sync import load_edges_from_csv, so3_sync_solve, rot_angle_deg, window_count

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)


# --------------------------------------------------------------------------
# M2: triangle residuals
# --------------------------------------------------------------------------
def triangle_residuals(ei, ej, Q, hop):
    """Per-triangle cycle error (deg) for all triangles (i<j<k) with direct edges.

    For each pair of edges sharing an intermediate window, the residual is
    r = ‖Log(Q_ij (Q_ik Q_kj)ᵀ)‖, i.e. how much the triangle closes.
    Only triangles whose THREE legs are direct edges are used.
    """
    q_by_pair = {}
    hop_by_pair = {}
    for a, b, q, h in zip(ei, ej, Q, hop):
        a, b = int(a), int(b)
        q_by_pair[(a, b)] = q
        hop_by_pair[(a, b)] = int(h)

    res = []
    # enumerate triples (i < j < k) where all legs are direct edges
    pairs = sorted(q_by_pair.keys())
    nodes = sorted(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    for a in nodes:
        # neighbors b>a and c>b via direct edges
        for b in nodes:
            if b <= a or (a, b) not in q_by_pair:
                continue
            for c in nodes:
                if c <= b or (b, c) not in q_by_pair or (a, c) not in q_by_pair:
                    continue
                Q_ac = q_by_pair[(a, c)]
                Q_ab = q_by_pair[(a, b)]
                Q_bc = q_by_pair[(b, c)]
                E = Q_ac.T @ (Q_ab @ Q_bc)
                r = rot_angle_deg(E)
                res.append({"i": a, "j": b, "k": c,
                            "h_ac": hop_by_pair[(a, c)], "h_ab": hop_by_pair[(a, b)],
                            "h_bc": hop_by_pair[(b, c)], "residual_deg": round(r, 4)})
    if not res:
        return np.array([]), res
    return np.array([r["residual_deg"] for r in res]), res


def hop_noise_analysis(seq_id):
    """Direct-vs-composed Q per hop (real data): how independent are multihop edges."""
    ei, ej, Q, w = load_edges_from_csv(seq_id, "stride4", 8)
    if ei is None:
        return {}
    q_by_pair = {}
    for a, b, q in zip(ei, ej, Q):
        q_by_pair[(int(a), int(b))] = q
    hops = defaultdict(list)
    for a, b, q in zip(ei, ej, Q):
        a, b = int(a), int(b)
        h = b - a
        if h == 1:
            continue
        # composed via hop-1 path a→a+1→...→b
        comp = np.eye(3)
        ok = True
        for k in range(a, b):
            if (k, k + 1) not in q_by_pair:
                ok = False
                break
            comp = comp @ q_by_pair[(k, k + 1)]
        if not ok:
            continue
        hops[h].append(rot_angle_deg(comp @ q.T))
    return {int(h): {"n": len(v),
                     "median_deg": round(float(np.median(v)), 4),
                     "mean_deg": round(float(np.mean(v)), 4),
                     "p90_deg": round(float(np.percentile(v, 90)), 4)}
            for h, v in hops.items() if v}


# --------------------------------------------------------------------------
# M1: lower-threshold sync (hop-3 included)
# --------------------------------------------------------------------------
def run_sync_threshold(n_nodes, seq_id, threshold):
    """Rerun the unchanged solver with the given overlap threshold."""
    ei, ej, qs, w = load_edges_from_csv(seq_id, "stride4", threshold)
    if ei is None or len(ei) == 0:
        return None, None, None, None
    return so3_sync_solve(n_nodes, ei, ej, qs, w, anchor=0)


# --------------------------------------------------------------------------
# M3: transitivity-augmented graph
# --------------------------------------------------------------------------
def run_transitivity_augmented(n_nodes, ei, ej, Q, weight):
    """Add composed-path Q_ik·Q_kj as explicit constraints (existing pairs only)."""
    q_by_pair = {}
    for a, b, q in zip(ei, ej, Q):
        q_by_pair[(int(a), int(b))] = q
    extra_i, extra_j, extra_Q, extra_w = [], [], [], []
    pairs = sorted(q_by_pair.keys())
    nodes = sorted(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    for a in nodes:
        for b in nodes:
            if b <= a or (a, b) not in q_by_pair:
                continue
            for c in nodes:
                if c <= b or (b, c) not in q_by_pair or (a, c) not in q_by_pair:
                    continue
                if b - a != 1 or c - b != 1:   # only hop-1 pairs (avoid double counting)
                    continue
                comp = q_by_pair[(a, b)] @ q_by_pair[(b, c)]
                extra_i.append(a); extra_j.append(c)
                extra_Q.append(comp)
                extra_w.append(1.0)
    if not extra_i:
        return so3_sync_solve(n_nodes, ei, ej, Q, weight, anchor=0), [], 0
    ei2 = np.concatenate([ei, np.array(extra_i)])
    ej2 = np.concatenate([ej, np.array(extra_j)])
    Q2 = np.concatenate([Q, np.array(extra_Q)])
    w2 = np.concatenate([weight, np.array(extra_w)])
    G, stats, per_edge = so3_sync_solve(n_nodes, ei2, ej2, Q2, w2, anchor=0)
    return (G, stats, per_edge), list(zip(extra_i, extra_j)), len(extra_i)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _load_synth(seq_id, scenario, seed=0):
    p = os.path.join(SYNTH_DIR, f"SYNTHETIC_{seq_id}_scen{scenario}_th8_seed{seed}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return {"i": d["edges_i"], "j": d["edges_j"], "Q": d["Q"],
            "weight": d["weight"], "hop": d["hop"], "G_chain": d["G_chain"],
            "G_true": d["G_true"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    ap.add_argument("--synthetic-scen", choices=["A", "B"], default=None,
                    help="run M2 triangle diagnostic on synthetic scenario")
    args = ap.parse_args()
    sequences = args.seq if args.seq else list(SEQUENCES)

    summary = {"M1": [], "M2": {}, "M3": []}

    for seq_id in sequences:
        n_nodes = window_count(seq_id, "stride4")
        if n_nodes is None:
            print(f"  SKIP {seq_id}: no window count")
            continue
        print(f"=== {seq_id} (N={n_nodes}) ===")

        # --- M1: threshold sweep ---
        G_th8, stats8, _ = run_sync_threshold(n_nodes, seq_id, 8)
        G_th4, stats4, _ = run_sync_threshold(n_nodes, seq_id, 4)
        diff = None
        if G_th8 is not None and G_th4 is not None:
            n = min(len(G_th8), len(G_th4))
            diff = float(np.median([rot_angle_deg(G_th8[k].T @ G_th4[k])
                                    for k in range(n)]))
            print(f"  M1 th8→th4: median gauge change = {diff:.4f}° "
                  f"(E: {stats8['n_edges']} → {stats4['n_edges']})")
            npz = os.path.join(OUT_DIR, f"REGULARIZED_GAUGES_{seq_id}_mhop_th4.npz")
            np.savez(npz, G=G_th4, method="multi_hop", threshold=4,
                     anchor_window=0, input_gauge="so3_sync_stride4_th4")
        summary["M1"].append({"sequence": seq_id, "th8_edges": stats8 and stats8["n_edges"],
                              "th4_edges": stats4 and stats4["n_edges"],
                              "median_gauge_change_deg": diff})

        # --- M2: real triangle residuals ---
        ei, ej, Q, w = load_edges_from_csv(seq_id, "stride4", 4)
        if ei is not None and len(ei):
            res_arr, res_rows = triangle_residuals(ei, ej, Q, [int(b - a) for a, b in zip(ei, ej)])
            if len(res_arr):
                summary["M2"][seq_id] = {
                    "n_triangles": len(res_arr),
                    "median_deg": round(float(np.median(res_arr)), 4),
                    "mean_deg": round(float(np.mean(res_arr)), 4),
                    "p90_deg": round(float(np.percentile(res_arr, 90)), 4),
                    "max_deg": round(float(np.max(res_arr)), 4),
                }
                print(f"  M2 triangles: n={len(res_arr)} "
                      f"med={np.median(res_arr):.3f}° p90={np.percentile(res_arr,90):.3f}°")
            csv_path = os.path.join(OUT_DIR, f"TRIANGLE_RESIDUAL_{seq_id}.csv")
            with open(csv_path, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=["i", "j", "k", "h_ac", "h_ab", "h_bc",
                                                   "residual_deg"])
                wr.writeheader()
                wr.writerows(res_rows)

        # --- hop noise (real) ---
        hn = hop_noise_analysis(seq_id)
        if hn:
            print(f"  hop-noise (direct vs composed): {hn}")

        # --- M3: transitivity-augmented (real) ---
        ei8, ej8, Q8, w8 = load_edges_from_csv(seq_id, "stride4", 8)
        if ei8 is not None and len(ei8):
            (G_tr, st_tr, _), extras, n_extra = run_transitivity_augmented(
                n_nodes, ei8, ej8, Q8, w8)
            med = float(np.median([rot_angle_deg(G_th8[k].T @ G_tr[k])
                                   for k in range(min(len(G_th8), len(G_tr)))]))
            print(f"  M3 transitivity: +{n_extra} constraints → gauge change "
                  f"median {med:.4f}°")
            npz = os.path.join(OUT_DIR, f"REGULARIZED_GAUGES_{seq_id}_trans.npz")
            np.savez(npz, G=G_tr, method="transitivity_augmented", n_extra=n_extra,
                     anchor_window=0, input_gauge="so3_sync_stride4_th8")
            summary["M3"].append({"sequence": seq_id, "n_extra": n_extra,
                                  "median_gauge_change_deg": round(med, 4)})

    # --- synthetic M2 (optional) ---
    if args.synthetic_scen:
        synth_seq = sequences[0]  # langdon reference
        s = _load_synth(synth_seq, args.synthetic_scen)
        if s is not None:
            res_arr, res_rows = triangle_residuals(s["i"], s["j"], s["Q"], s["hop"])
            print(f"\n=== Synthetic {synth_seq} scenario {args.synthetic_scen} M2 ===")
            print(f"  triangles: n={len(res_arr)} med={np.median(res_arr):.4f}° "
                  f"p90={np.percentile(res_arr,90):.4f}°")
            summary["M2"]["__synthetic__%s" % args.synthetic_scen] = {
                "scenario": args.synthetic_scen,
                "n_triangles": len(res_arr),
                "median_deg": round(float(np.median(res_arr)), 4),
                "p90_deg": round(float(np.percentile(res_arr, 90)), 4),
            }
            with open(os.path.join(OUT_DIR, f"TRIANGLE_RESIDUAL_SYNTHETIC_scen{args.synthetic_scen}.csv"),
                      "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=["i", "j", "k", "h_ac", "h_ab", "h_bc",
                                                   "residual_deg"])
                wr.writeheader()
                wr.writerows(res_rows)

    summary_path = os.path.join(OUT_DIR, "METHOD_SUMMARY.json")
    prev = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            prev = json.load(f)
    prev["multi_hop"] = {"gt_used": False, **summary}
    with open(summary_path, "w") as f:
        json.dump(prev, f, indent=2)
    print(f"\nUpdated METHOD_SUMMARY.json (multi_hop)")


if __name__ == "__main__":
    main()