#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3 Step 2: Build rotation edge graph from real shared-frame overlaps.

For ANY window pair (i, j) with |frame_idx_i ∩ frame_idx_j| >= min_overlap:
  - Per-shared-frame Q:  Q_ij,f = R_c2w_i,f @ R_c2w_j,f^T   (both c2w, frozen formula)
  - Edge Q: robust SO(3) mean via so3_robust_mean() (quat avg + MAD rejection)
  - Edge confidence:     w = min(n_overlap/12, 1) * 1/(1 + median_disp_deg)

Outputs:
  ROTATION_GRAPH_EDGES.csv            — stride-4 headline (min_overlap=8) + diagnostic (4)
  ROTATION_GRAPH_EDGES_STRIDE8.csv    — stride-8 baseline (min_overlap=8, pure chain)

Usage:
    python 02_rotation_edges/build_rotation_edges.py [--seq ...] [--min-overlap 8]
"""
import argparse, csv, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
STRIDE8_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
STRIDE4_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
OUT_DIR = os.path.join(SYNC_DIR, "02_rotation_edges")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PHASE3C, "05_global_stitching_v31"))
from run_gauge_stitching import (
    c2w_rotations, find_overlap_frames, so3_robust_mean, rot_angle_deg
)

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)


def load_windows(window_dir, seq_id):
    """Load all window npz for a sequence into list of dicts."""
    seq_dir = os.path.join(window_dir, seq_id)
    files = sorted(glob.glob(os.path.join(seq_dir, "window_*.npz")))
    if not files:
        return None
    windows = []
    for wf in files:
        data = np.load(wf)
        windows.append({
            "ext_w2c": data["ext_w2c_vggt"],
            "frame_idx": data["frame_idx"],
        })
    return windows


def build_edges(windows, seq_id, min_overlap, include_all_hops=True):
    """Build all real-overlap edges for a sequence.

    Returns list of edge dicts.
    """
    n = len(windows)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if not include_all_hops and j != i + 1:
                continue
            overlap = find_overlap_frames(windows[i]["frame_idx"], windows[j]["frame_idx"])
            if len(overlap) < min_overlap:
                continue

            R_c2w_A = c2w_rotations(windows[i]["ext_w2c"])
            R_c2w_B = c2w_rotations(windows[j]["ext_w2c"])
            frame_to_A = {int(f): idx for idx, f in enumerate(windows[i]["frame_idx"])}
            frame_to_B = {int(f): idx for idx, f in enumerate(windows[j]["frame_idx"])}

            Q_list = []
            for f in overlap:
                Q_i = R_c2w_A[frame_to_A[f]] @ R_c2w_B[frame_to_B[f]].T
                Q_list.append(Q_i)
            Q_array = np.array(Q_list)

            Q_star, inlier_mask, stats = so3_robust_mean(Q_array)

            n_overlap = len(overlap)
            n_inliers = int(inlier_mask.sum())
            disp_med = stats["median"]
            disp_p90 = stats["p90"]
            disp_max = stats["max"]

            w_overlap = min(n_overlap / 12.0, 1.0)
            w_disp = 1.0 / (1.0 + disp_med)
            weight = w_overlap * w_disp

            edges.append({
                "sequence": seq_id,
                "window_i": i,
                "window_j": j,
                "window_distance": j - i,
                "n_overlap": n_overlap,
                "n_inliers": n_inliers,
                "Q_disp_med_deg": round(disp_med, 4),
                "Q_disp_p90_deg": round(disp_p90, 4),
                "Q_disp_max_deg": round(disp_max, 4),
                "weight": round(weight, 6),
                "Q_star": Q_star,
                "edge_status": "OK",
            })
    return edges


def write_edges_csv(edges, path):
    """Write edges to CSV with serialized Q matrix."""
    fields = ["sequence", "window_i", "window_j", "window_distance",
              "n_overlap", "n_inliers", "Q_disp_med_deg", "Q_disp_p90_deg",
              "Q_disp_max_deg", "weight", "edge_status",
              "q00", "q01", "q02", "q10", "q11", "q12", "q20", "q21", "q22"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for e in edges:
            row = {k: v for k, v in e.items() if k != "Q_star"}
            for r in range(3):
                for c in range(3):
                    row[f"q{r}{c}"] = round(float(e["Q_star"][r, c]), 6)
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    ap.add_argument("--min-overlap", type=int, default=8, help="Headline threshold")
    args = ap.parse_args()

    sequences = args.seq if args.seq else SEQUENCES
    all_edges_s4 = []
    all_edges_s8 = []

    print("=== STRIDE-4 WINDOWS (headline) ===")
    for seq_id in sequences:
        windows = load_windows(STRIDE4_DIR, seq_id)
        if windows is None:
            print(f"  {seq_id}: stride-4 windows NOT FOUND (inference pending?)")
            continue
        edges = build_edges(windows, seq_id, args.min_overlap, include_all_hops=True)
        all_edges_s4.extend(edges)
        n_adj = sum(1 for e in edges if e["window_distance"] == 1)
        n_hop2 = sum(1 for e in edges if e["window_distance"] == 2)
        n_hop3 = sum(1 for e in edges if e["window_distance"] == 3)
        print(f"  {seq_id}: {len(windows)} windows, {len(edges)} edges "
              f"(hop1={n_adj}, hop2={n_hop2}, hop3={n_hop3})")

    print("\n=== STRIDE-8 WINDOWS (baseline chain) ===")
    for seq_id in sequences:
        windows = load_windows(STRIDE8_DIR, seq_id)
        if windows is None:
            print(f"  {seq_id}: stride-8 windows NOT FOUND")
            continue
        edges = build_edges(windows, seq_id, 8, include_all_hops=True)
        all_edges_s8.extend(edges)
        n_adj = sum(1 for e in edges if e["window_distance"] == 1)
        print(f"  {seq_id}: {len(windows)} windows, {len(edges)} edges "
              f"(hop1={n_adj}, others={len(edges)-n_adj})")

    if all_edges_s4:
        p = os.path.join(OUT_DIR, "ROTATION_GRAPH_EDGES.csv")
        write_edges_csv(all_edges_s4, p)
        print(f"\nSaved: {p} ({len(all_edges_s4)} edges)")
    if all_edges_s8:
        p = os.path.join(OUT_DIR, "ROTATION_GRAPH_EDGES_STRIDE8.csv")
        write_edges_csv(all_edges_s8, p)
        print(f"Saved: {p} ({len(all_edges_s8)} edges)")

    # Also write Q matrix (3x3) into npz for the solver to consume directly
    if all_edges_s4:
        seq_map = {}
        for e in all_edges_s4:
            seq_map.setdefault(e["sequence"], []).append(e)
        for seq_id, es in seq_map.items():
            np.savez(os.path.join(OUT_DIR, f"{seq_id}_EDGE_QS.npz"),
                     i=np.array([e["window_i"] for e in es]),
                     j=np.array([e["window_j"] for e in es]),
                     Q=np.array([e["Q_star"] for e in es]),
                     weight=np.array([e["weight"] for e in es]),
                     disp_med=np.array([e["Q_disp_med_deg"] for e in es]),
                     n_overlap=np.array([e["n_overlap"] for e in es]))
        print(f"Saved per-sequence edge Q npz: {OUT_DIR}/<seq>_EDGE_QS.npz")


if __name__ == "__main__":
    main()