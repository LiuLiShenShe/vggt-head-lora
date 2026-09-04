#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3 Step 8: Visualization figures for rotation synchronization.

Figures (per langdon sequence, plus a combined summary):
  Fig 1  ROTATION_DRIFT_{seq}.png      — per-frame orientation error over frame index
                                       for stride8 chain (A), stride4 chain (C), stride4 graph (D)
  Fig 2  ROTATION_GRAPH_{seq}.png      — rotation graph: nodes=windows, edges=overlap,
                                       edge color = weight, highlight hop
  Fig 3  CYCLE_RESIDUAL_DIST_{seq}.png — cycle residual distribution histogram
  Fig 4  METHOD_COMPARISON_BAR.png     — rot_median by method, all sequences
  Fig 5  EDGE_BIAS_BY_HOP.png          — edge Q-vs-COLMAP bias by window distance (hop)

Outputs to 08_visualization/. Requires matplotlib.

Usage:
    python 08_visualization/make_figures.py [--seq ...]
"""
import argparse, csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
EVAL_DIR = os.path.join(SYNC_DIR, "06_evaluation")
EDGES_DIR = os.path.join(SYNC_DIR, "02_rotation_edges")
OUT_DIR = os.path.join(SYNC_DIR, "08_visualization")
os.makedirs(OUT_DIR, exist_ok=True)

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)

STYLE = {
    "A_STRIDE8_CHAIN":    ("#888888", "-",  "stride8 chain (A)"),
    "C_STRIDE4_CHAIN":    ("#e0a030", "--", "stride4 chain (C)"),
    "D_STRIDE4_SO3GRAPH": ("#2050d0", "-",  "stride4 SO(3) graph (D)"),
}


def load_per_frame():
    path = os.path.join(EVAL_DIR, "ROTATION_SYNC_PER_FRAME_ERRORS.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out.setdefault((r["sequence"], r["method"]), []).append(
                (int(r["frame_idx"]), float(r["rot_error_deg"])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def load_comparison():
    path = os.path.join(EVAL_DIR, "ROTATION_SYNC_METHOD_COMPARISON.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out.setdefault(r["sequence"], {})[r["method"]] = {
                "rot_median": float(r["rot_median"]),
                "rot_p90": float(r["rot_p90"]),
                "pose_gate": r["pose_gate"],
            }
    return out


def load_cycle():
    path = os.path.join(SYNC_DIR, "07_diagnostics", "ROTATION_CYCLE_RESIDUALS.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["sequence"]] = {
                "n": int(r["n_triangles"]),
                "med": float(r["cycle_err_median_deg"]),
                "p90": float(r["cycle_err_p90_deg"]),
            }
    return out


def load_edge_bias():
    path = os.path.join(SYNC_DIR, "07_diagnostics", "EDGE_ERROR_BY_WINDOW_DISTANCE.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out.setdefault(r["sequence"], {})[int(r["window_distance"])] = {
                "median": float(r["q_vs_ref_median_deg"]),
                "p90": float(r["q_vs_ref_p90_deg"]),
            }
    return out


def fig1_drift(seq, per_frame, out_dir):
    plt.figure(figsize=(11, 4.5))
    for method, (color, ls, label) in STYLE.items():
        arr = per_frame.get((seq, method))
        if arr is None:
            continue
        plt.plot(arr[:, 0], arr[:, 1], color=color, ls=ls, lw=1.6, label=label, alpha=0.9)
    plt.axhline(10, color="green", ls=":", lw=1, label="gate 10°")
    plt.xlabel("frame index"); plt.ylabel("orientation error (deg, after global Procrustes)")
    plt.title(f"Per-frame orientation drift — {seq.split('__')[-1]}")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, f"ROTATION_DRIFT_{seq.split('__')[-1]}.png")
    plt.savefig(p, dpi=150); plt.close()
    return p


def fig2_graph(seq, out_dir):
    import networkx as nx
    # Load stride-4 edges for this sequence
    path = os.path.join(EDGES_DIR, "ROTATION_GRAPH_EDGES.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r["sequence"] == seq and int(r["n_overlap"]) >= 8]
    if not rows:
        return None
    g = nx.Graph()
    for r in rows:
        g.add_edge(int(r["window_i"]), int(r["window_j"]),
                   weight=float(r["weight"]), hop=int(r["window_distance"]))
    plt.figure(figsize=(10, 6))
    pos = {n: (n, 0) for n in g.nodes()}  # linear layout: node = window order
    weights = [g[u][v]["weight"] for u, v in g.edges()]
    wmin, wmax = min(weights), max(weights)
    colors = [plt.cm.viridis((w - wmin) / max(wmax - wmin, 1e-9)) for w in weights]
    hops = [g[u][v]["hop"] for u, v in g.edges()]
    for (u, v), c, hop in zip(g.edges(), colors, hops):
        yoff = 0.05 * hop
        plt.plot([pos[u][0], pos[v][0]], [pos[u][1] + yoff, pos[v][1] + yoff],
                 color=c, alpha=0.6, lw=1.2)
    nx.draw_networkx_nodes(g, pos, node_size=40, node_color="white",
                           edgecolors="black", linewidths=0.6)
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=plt.Normalize(vmin=wmin, vmax=wmax))
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label="edge weight")
    plt.title(f"Rotation graph (stride-4, min_overlap=8) — {seq.split('__')[-1]}: "
              f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    plt.xlabel("window index")
    plt.tight_layout()
    p = os.path.join(out_dir, f"ROTATION_GRAPH_{seq.split('__')[-1]}.png")
    plt.savefig(p, dpi=150); plt.close()
    return p


def fig3_cycle(seq, cycle_info, out_dir):
    if seq not in cycle_info:
        return None
    info = cycle_info[seq]
    if info["n"] == 0:
        return None
    # Recompute distribution from csv triangle data would need full list; use histogram approx
    plt.figure(figsize=(7, 4))
    # sample a triangle-normal-like distribution centered at med with p90
    med, p90 = info["med"], info["p90"]
    # Plot a simple marker instead of fabricated histogram
    plt.bar(["median", "p90"], [med, p90], color=["#2050d0", "#e0a030"])
    plt.ylim(0, max(p90 * 1.2, 1))
    plt.ylabel("cycle residual (deg)")
    plt.title(f"Triangle cycle consistency — {seq.split('__')[-1]} (n={info['n']})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, f"CYCLE_RESIDUAL_{seq.split('__')[-1]}.png")
    plt.savefig(p, dpi=150); plt.close()
    return p


def fig4_comparison(comp, out_dir):
    methods = ["A_STRIDE8_CHAIN", "B_STRIDE8_SO3GRAPH", "C_STRIDE4_CHAIN", "D_STRIDE4_SO3GRAPH"]
    labels = ["A", "B", "C", "D"]
    short = [s.split("__")[-1].replace("pos00", "mustc").replace("plot_", "wheat ") for s in SEQUENCES]
    n = len(SEQUENCES)
    x = np.arange(n)
    width = 0.2
    plt.figure(figsize=(12, 5))
    for mi, (method, lab) in enumerate(zip(methods, labels)):
        vals = []
        for seq in SEQUENCES:
            r = comp.get(seq, {}).get(method)
            vals.append(r["rot_median"] if r else np.nan)
        colors = ["#2050d0" if comp.get(seq, {}).get(method, {}).get("pose_gate") == "PASS"
                  else "#c03030" for seq in SEQUENCES]
        plt.bar(x + (mi - 1.5) * width, vals, width, label=lab, color=colors, alpha=0.85)
    plt.axhline(10, color="green", ls=":", lw=1)
    plt.xticks(x, short, rotation=20, ha="right")
    plt.ylabel("rot_median (deg)")
    plt.title("Rotation median by method — red=FAIL, blue=PASS (gate 10°)")
    plt.legend(); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "METHOD_COMPARISON_BAR.png")
    plt.savefig(p, dpi=150); plt.close()
    return p


def fig5_edge_bias(edge_bias, out_dir):
    hops = [1, 2, 3]
    plt.figure(figsize=(9, 4.5))
    for hop in hops:
        meds = [edge_bias.get(seq, {}).get(hop, {}).get("median")
                for seq in SEQUENCES if edge_bias.get(seq, {}).get(hop)]
        if not meds:
            continue
        short = [s.split("__")[-1].replace("pos00", "mustc").replace("plot_", "wheat ")
                 for s in SEQUENCES if edge_bias.get(s, {}).get(hop)]
        plt.plot(range(len(meds)), meds, "o-", label=f"window distance {hop}", lw=1.5)
        # x ticks: actual seq names
    plt.ylabel("edge Q-vs-COLMAP bias (deg, median)")
    plt.title("Systematic Q bias by window distance")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "EDGE_BIAS_BY_HOP.png")
    plt.savefig(p, dpi=150); plt.close()
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()
    sequences = [s for s in SEQUENCES if not args.seq or s in args.seq]

    per_frame = load_per_frame()
    comp = load_comparison()
    cycle = load_cycle()
    edge_bias = load_edge_bias()

    fig4_comparison(comp, OUT_DIR)
    fig5_edge_bias(edge_bias, OUT_DIR)

    for seq in sequences:
        print(f"=== {seq} ===")
        for f in [fig1_drift(seq, per_frame, OUT_DIR),
                  fig2_graph(seq, OUT_DIR),
                  fig3_cycle(seq, cycle, OUT_DIR)]:
            if f:
                print(f"  saved {os.path.basename(f)}")
    print(f"\nSaved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()