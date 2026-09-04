#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3 Step 3: Graph structure analysis for the rotation edge graph.

Computes REAL graph properties from ROTATION_GRAPH_EDGES.csv using networkx:
  n_nodes, n_edges, n_components, cycle_rank (= E - V + C), mean_degree,
  max_degree, is_connected, is_tree. Also per-threshold (8 / 4) analysis.

Window count (node count) comes from the STRIDE4_WINDOW_MANIFEST.json — the
source of truth for REAL generated windows (never hardcoded).

Acceptance gate for each langdon sequence: connected AND cycle_rank > 0.

Outputs:
  ROTATION_GRAPH_SUMMARY.json

Usage:
    python 03_graph_analysis/analyze_graph.py [--seq ...]
"""
import argparse, csv, json, os, sys
import networkx as nx

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
EDGES_DIR = os.path.join(SYNC_DIR, "02_rotation_edges")
OUT_DIR = os.path.join(SYNC_DIR, "03_graph_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)


def load_window_counts(stride4_manifest_path):
    """Read REAL window counts per sequence from the stride-4 manifest."""
    counts = {}
    if os.path.exists(stride4_manifest_path):
        with open(stride4_manifest_path) as f:
            man = json.load(f)
        for seq_id, meta in man.items():
            counts[seq_id] = meta["n_windows"]
    return counts


def analyze(n_nodes, edges, seq_id, label, min_overlap):
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    for e in edges:
        g.add_edge(int(e["window_i"]), int(e["window_j"]), weight=float(e["weight"]))

    n_edges = g.number_of_edges()
    n_components = nx.number_connected_components(g)
    cycle_rank = n_edges - n_nodes + n_components
    degrees = dict(g.degree())
    mean_degree = sum(degrees.values()) / max(n_nodes, 1)
    max_degree = max(degrees.values()) if n_nodes else 0
    is_connected = nx.is_connected(g)
    is_tree = (n_components == 1 and n_edges == n_nodes - 1)

    hop_counts = {}
    for e in edges:
        d = int(e.get("window_distance", int(e["window_j"]) - int(e["window_i"])))
        hop_counts[f"hop_{d}"] = hop_counts.get(f"hop_{d}", 0) + 1

    result = {
        "sequence_id": seq_id,
        "label": label,
        "min_overlap": min_overlap,
        "n_windows": n_nodes,
        "n_edges": n_edges,
        "n_components": n_components,
        "cycle_rank": max(0, cycle_rank),
        "mean_degree": round(mean_degree, 4),
        "max_degree": max_degree,
        "is_connected": bool(is_connected),
        "is_tree": bool(is_tree),
        "acceptance_gate": bool(is_connected and cycle_rank > 0),
        "hop_distribution": hop_counts,
    }
    print(f"  {seq_id} [{label}, min_overlap={min_overlap}]: "
          f"V={n_nodes} E={n_edges} C={n_components} cycle_rank={result['cycle_rank']} "
          f"connected={is_connected} gate={result['acceptance_gate']}")
    return result


def load_edge_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()

    sequences = args.seq if args.seq else SEQUENCES
    manifest_path = os.path.join(SYNC_DIR, "01_stride4_inference", "STRIDE4_WINDOW_MANIFEST.json")
    window_counts = load_window_counts(manifest_path)
    if not window_counts:
        print("WARNING: stride-4 manifest not found; window counts unavailable.")
    results = []

    # stride-4 master superset (min_overlap=4 includes hop-3); filter by n_overlap column
    s4_path = os.path.join(EDGES_DIR, "ROTATION_GRAPH_EDGES.csv")
    print("=== STRIDE-4 GRAPH (headline min_overlap=8, diagnostic min_overlap=4) ===")
    if os.path.exists(s4_path):
        all_rows = load_edge_rows(s4_path)
        for seq_id in sequences:
            seq_rows_all = [r for r in all_rows if r["sequence"] == seq_id]
            if not seq_rows_all:
                print(f"  {seq_id}: no stride-4 edges (inference pending?)")
                continue
            n_nodes = window_counts.get(seq_id, 0)
            if n_nodes == 0:
                print(f"  {seq_id}: window count unavailable from manifest — skip")
                continue
            # headline: n_overlap >= 8
            seq_rows8 = [r for r in seq_rows_all if int(r["n_overlap"]) >= 8]
            results.append(analyze(n_nodes, seq_rows8, seq_id, "STRIDE4", 8))
            # diagnostic: n_overlap >= 4 (adds hop-3)
            seq_rows4 = [r for r in seq_rows_all if int(r["n_overlap"]) >= 4]
            if len(seq_rows4) != len(seq_rows8):
                results.append(analyze(n_nodes, seq_rows4, seq_id, "STRIDE4_DIAG", 4))

    # stride-8 baseline (min_overlap=8)
    s8_path = os.path.join(EDGES_DIR, "ROTATION_GRAPH_EDGES_STRIDE8.csv")
    print("\n=== STRIDE-8 GRAPH (baseline) ===")
    if os.path.exists(s8_path):
        all_rows = load_edge_rows(s8_path)
        for seq_id in sequences:
            seq_rows = [r for r in all_rows if r["sequence"] == seq_id]
            if not seq_rows:
                continue
            n_nodes = len({r["window_i"] for r in seq_rows} | {r["window_j"] for r in seq_rows})
            # better: use stride-8 window count from existing per-seq manifest
            s8man = os.path.join(PHASE3C, "03_window_inference", "window_outputs",
                                 seq_id, "WINDOW_RUN_MANIFEST.json")
            if os.path.exists(s8man):
                with open(s8man) as f:
                    n_nodes = json.load(f)["n_windows"]
            results.append(analyze(n_nodes, seq_rows, seq_id, "STRIDE8", 8))

    # Save
    out_path = os.path.join(OUT_DIR, "ROTATION_GRAPH_SUMMARY.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path} ({len(results)} entries)")


if __name__ == "__main__":
    main()