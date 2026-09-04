#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.2 Step 4: Analyze scale graph redundancy.

For stride=8 windows, determine if the graph is a pure chain or has cycles.
Cycle rank > 0 enables true global optimization (drift correction).

Also computes what stride-4 would give.
"""
import csv, json, os, glob
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
OUT_DIR = os.path.join(PHASE3C, "11_scale_sync_v32", "03_scale_graph")
os.makedirs(OUT_DIR, exist_ok=True)


def analyze_graph(windows, seq_id, stride):
    """Analyze the overlap graph for a given stride.

    Returns graph properties: nodes, edges, cycle rank, etc.
    """
    n_windows = len(windows)
    nodes = list(range(n_windows))

    # Build edges: pairs with shared frames
    edges = []
    for i in range(n_windows):
        for j in range(i + 1, n_windows):
            frames_i = set(windows[i]["frame_idx"].tolist())
            frames_j = set(windows[j]["frame_idx"].tolist())
            shared = frames_i & frames_j
            if len(shared) >= 2:
                edges.append((i, j, len(shared)))

    # Graph properties
    n_nodes = len(nodes)
    n_edges = len(edges)
    # For a connected graph: cycle_rank = n_edges - n_nodes + 1
    # (first Betti number)
    cycle_rank = max(0, n_edges - n_nodes + 1)

    # Degree distribution
    degree = {n: 0 for n in nodes}
    for i, j, _ in edges:
        degree[i] += 1
        degree[j] += 1
    mean_degree = np.mean(list(degree.values()))

    # Is it a pure chain? (each node has degree ≤ 2, exactly 2 nodes have degree 1)
    is_chain = all(d <= 2 for d in degree.values()) and sum(d == 1 for d in degree.values()) == 2

    # Check first/last overlap (circular trajectory)
    first_last_shared = len(set(windows[0]["frame_idx"].tolist()) & set(windows[-1]["frame_idx"].tolist()))

    # Check skip connections (W_k ↔ W_{k+2})
    skip_edges = [(i, j, s) for i, j, s in edges if j - i > 1]

    return {
        "sequence_id": seq_id,
        "stride": stride,
        "n_windows": n_nodes,
        "n_edges": n_edges,
        "cycle_rank": cycle_rank,
        "is_chain": is_chain,
        "mean_degree": float(mean_degree),
        "first_last_overlap": first_last_shared,
        "n_skip_edges": len(skip_edges),
        "edges": [(i, j, s) for i, j, s in edges],
        "degree": degree,
    }


def main():
    # Analyze stride=8 (current) for all sequences
    seq_dirs = sorted(glob.glob(os.path.join(WINDOW_DIR, "*")))

    all_results = []

    for seq_dir in seq_dirs:
        if not os.path.isdir(seq_dir):
            continue
        seq_id = os.path.basename(seq_dir)
        window_files = sorted(glob.glob(os.path.join(seq_dir, "window_*.npz")))
        if len(window_files) < 2:
            continue

        windows = []
        for wf in window_files:
            data = np.load(wf)
            windows.append({"frame_idx": data["frame_idx"]})

        result = analyze_graph(windows, seq_id, stride=8)
        all_results.append(result)

        print(f"\n{seq_id}:")
        print(f"  Windows: {result['n_windows']}")
        print(f"  Edges: {result['n_edges']}")
        print(f"  Cycle rank: {result['cycle_rank']}")
        print(f"  Is chain: {result['is_chain']}")
        print(f"  Mean degree: {result['mean_degree']:.2f}")
        print(f"  First-last overlap: {result['first_last_overlap']} frames")
        print(f"  Skip edges (|i-j|>1): {result['n_skip_edges']}")

        if result['is_chain']:
            print(f"  → PURE CHAIN: global optimization = pairwise replication")
        elif result['cycle_rank'] > 0:
            print(f"  → HAS CYCLES: global optimization can correct drift")

    # Save summary
    summary = []
    for r in all_results:
        summary.append({
            "sequence_id": r["sequence_id"],
            "stride": r["stride"],
            "n_windows": r["n_windows"],
            "n_edges": r["n_edges"],
            "cycle_rank": r["cycle_rank"],
            "is_chain": r["is_chain"],
            "mean_degree": r["mean_degree"],
            "first_last_overlap": r["first_last_overlap"],
            "n_skip_edges": r["n_skip_edges"],
        })

    csv_path = os.path.join(OUT_DIR, "SCALE_GRAPH_SUMMARY.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"\nSaved: {csv_path}")

    # Also save full edge lists as JSON
    json_path = os.path.join(OUT_DIR, "SCALE_GRAPH_EDGES.json")
    with open(json_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "edges"} for r in all_results], f, indent=2)

    # What stride-4 would give (theoretical analysis for langdon_4)
    print(f"\n{'='*60}")
    print("STRIDE-4 THEORETICAL ANALYSIS (langdon_4 sequences)")
    print(f"{'='*60}")
    for r in all_results:
        if "langdon_4" not in r["sequence_id"]:
            continue
        n = r["n_windows"]
        # Stride-4: W_k connects to W_{k+1}, W_{k+2}, W_{k+3}
        # with 16-frame windows and stride 4, each pair shares 12 or 8 or 4 frames
        edges_s4 = 0
        for i in range(n):
            for j in range(i + 1, min(i + 4, n)):
                # Estimate shared frames: window size 16, stride 4
                # W_k covers frames [k*4, k*4+15]
                # Shared = 16 - (j-i)*4
                shared = 16 - (j - i) * 4
                if shared >= 2:
                    edges_s4 += 1
        cycle_rank_s4 = max(0, edges_s4 - n + 1)
        print(f"\n  {r['sequence_id']}:")
        print(f"    Stride-4 edges (theoretical): {edges_s4}")
        print(f"    Stride-4 cycle rank: {cycle_rank_s4}")
        print(f"    → {'HAS CYCLES' if cycle_rank_s4 > 0 else 'STILL CHAIN'}")


if __name__ == "__main__":
    main()
