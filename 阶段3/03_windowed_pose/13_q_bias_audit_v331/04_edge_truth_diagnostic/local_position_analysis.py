#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3.1 §15-§17, §24-§25: Local-position and central-owner analysis.

Motivation: if VGGT rotations are predicted from a 16-frame context, boundary
frames see fewer neighboring frames than center frames. The edge Q_ij is
computed from shared frames between adjacent windows. The final global cameras
use CENTER frames (positions 7-8 of each window), but edges are computed using
SHARED frames (boundary regions). This mismatch may cause EDGE_CONSTRAINT_REGION
≠ FINAL_CAMERA_REGION.

For each window k and frame position p ∈ {0,...,15}:
  1. Per-frame fit residual (from corrected gauge fit — gauge-dependent but diagnostic)
  2. Within-window relative rotation error at various separations
  3. Frame-by-frame rotation error vs reference (gauge-aligned)

Cross-window (§17, §24-§25): for each image index appearing in multiple windows,
  4. Fit the gauge G_k for each window containing that frame
  5. Compute R_vggt^T @ R_ref after gauge alignment for each window
  6. Report dispersion across windows: if VGGT orientation is context-invariant
     (i.e. prediction of the same image doesn't depend on which 16-frame context
     it appears in), this dispersion should be ~0.

Outputs (in 04_edge_truth_diagnostic/):
  LOCAL_POSITION_ANALYSIS_<seq>.csv    — per-frame position error stats
  CENTRAL_OWNER_ANALYSIS_<seq>.csv     — edge vs final-camera frame overlap
  CROSS_WINDOW_CONTEXT_<seq>.csv       — same-image multi-window dispersion
  LOCAL_POSITION_SUMMARY.json          — per-sequence summary

DIAGNOSTIC_ONLY: uses COLMAP reference for evaluation only; no solver modification.

Usage:
    python 04_edge_truth_diagnostic/local_position_analysis.py [--seq ...]
"""
import argparse, csv, glob, json, os, sys
from collections import defaultdict
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
EDGES_DIR = os.path.join(SYNC_DIR, "02_rotation_edges")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4")
OUT_DIR = os.path.join(PHASE3C, "13_q_bias_audit_v331", "04_edge_truth_diagnostic")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))
sys.path.insert(0, os.path.join(ROOT, "阶段3", "03_windowed_pose", "12_rotation_sync_v33", "06_evaluation"))
from evaluate_multoplant import global_rotation_procrustes, rot_angle_deg
from evaluate_rotation_sync import find_sequence_json, load_reference_poses

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)

THRESHOLD = 8


def load_window_data(seq_id, window_idx):
    path = os.path.join(WINDOW_DIR, seq_id, f"window_{window_idx:03d}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    ext = d["ext_w2c_vggt"]
    R_c2w = ext[:, :3, :3].transpose(0, 2, 1)
    return R_c2w, d["frame_idx"].astype(int)


def local_position_analysis(seq_id, R_ref_seq):
    """§15: Per-frame-position error statistics across all windows.

    For each window position p ∈ [0, n_frames):
      - per-frame rotation error after gauge alignment
    """
    win_files = sorted(glob.glob(os.path.join(WINDOW_DIR, seq_id, "window_*.npz")))
    n_w = len(win_files)
    # collect per-position errors
    pos_errors = defaultdict(list)

    for k in range(n_w):
        data = load_window_data(seq_id, k)
        if data is None:
            continue
        R_v, fidx = data
        n_fr = len(fidx)
        R_ref = R_ref_seq[fidx]

        # fit gauge for this window
        Gk = global_rotation_procrustes(R_v, R_ref)
        R_v_aligned = np.einsum("ab,sbc->sac", Gk, R_v)

        for p in range(n_fr):
            err = rot_angle_deg(R_v_aligned[p].T @ R_ref[p])
            pos_errors[p].append(err)

    # save per-position stats
    rows = []
    for p in sorted(pos_errors.keys()):
        errs = np.array(pos_errors[p])
        rows.append({
            "position": p,
            "n": len(errs),
            "median": float(np.median(errs)),
            "p90": float(np.percentile(errs, 90)),
            "mean": float(np.mean(errs)),
            "max": float(np.max(errs)),
        })

    csv_path = os.path.join(OUT_DIR, f"LOCAL_POSITION_ANALYSIS_{seq_id}.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return rows, pos_errors


def central_owner_analysis(seq_id):
    """§16: Characterize edge-frame vs center-frame overlap.

    For each edge (i,j):
      - edge_frames = shared frames between windows i and j
      - center_frames_i = frames at positions [7,8] of window i (used for global cameras)
      - overlap = intersection of edge_frames with center_frames
    """
    edge_path = os.path.join(EDGES_DIR, "ROTATION_GRAPH_EDGES.csv")
    if not os.path.exists(edge_path):
        return []

    with open(edge_path) as f:
        edge_rows = [r for r in csv.DictReader(f)
                     if r["sequence"] == seq_id and int(r["n_overlap"]) >= THRESHOLD]

    win_files = sorted(glob.glob(os.path.join(WINDOW_DIR, seq_id, "window_*.npz")))
    # load frame indices for each window
    window_frames = {}
    for k in range(len(win_files)):
        data = load_window_data(seq_id, k)
        if data is not None:
            window_frames[k] = set(data[1].tolist())

    rows = []
    for r in edge_rows:
        a, b = int(r["window_i"]), int(r["window_j"])
        if a not in window_frames or b not in window_frames:
            continue
        fa = sorted(window_frames[a])
        fb = sorted(window_frames[b])
        n_fa, n_fb = len(fa), len(fb)
        edge_shared = set(fa) & set(fb)
        # center frames: positions n//2-1, n//2 in each window
        center_a = set(fa[n_fa // 2 - 1 : n_fa // 2 + 1]) if n_fa >= 2 else set()
        center_b = set(fb[n_fb // 2 - 1 : n_fb // 2 + 1]) if n_fb >= 2 else set()
        overlap_a = edge_shared & center_a
        overlap_b = edge_shared & center_b
        rows.append({
            "window_i": a, "window_j": b,
            "hop": int(r["window_distance"]),
            "n_edge_frames": len(edge_shared),
            "n_center_frames_a": len(center_a),
            "n_center_frames_b": len(center_b),
            "n_edge_center_overlap_a": len(overlap_a),
            "n_edge_center_overlap_b": len(overlap_b),
        })

    csv_path = os.path.join(OUT_DIR, f"CENTRAL_OWNER_ANALYSIS_{seq_id}.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def cross_window_context_analysis(seq_id, R_ref_seq):
    """§17, §24-§25: same-image orientation dispersion across windows.

    For each image index appearing in multiple windows:
      - In each containing window, fit gauge G_k, then compute
        the gauge-aligned rotation of that image: (G_k @ R_vggt)^T @ R_ref
      - This is a gauge-invariant quantity (gauge cancels in the comparison)
      - Report the dispersion across windows: high dispersion → prediction
        depends on context window (non-invariant), low dispersion → context-invariant
    """
    win_files = sorted(glob.glob(os.path.join(WINDOW_DIR, seq_id, "window_*.npz")))
    n_w = len(win_files)

    # build per-image container: {frame_idx: [(window_k, R_vggt_frame, R_ref_frame)]}
    image_data = defaultdict(list)
    for k in range(n_w):
        data = load_window_data(seq_id, k)
        if data is None:
            continue
        R_v, fidx = data
        n_fr = len(fidx)
        R_ref = R_ref_seq[fidx]
        Gk = global_rotation_procrustes(R_v, R_ref)
        R_v_aligned = np.einsum("ab,sbc->sac", Gk, R_v)
        for p in range(n_fr):
            # gauge-invariant orientation residual
            res = R_v_aligned[p].T @ R_ref[p]
            image_data[int(fidx[p])].append({
                "window": k,
                "position": p,
                "residual": res,
                "residual_deg": rot_angle_deg(res),
            })

    # only images appearing in ≥2 windows are informative
    rows = []
    for img_idx in sorted(image_data.keys()):
        entries = image_data[img_idx]
        if len(entries) < 2:
            continue
        resids = [e["residual"] for e in entries]
        degs = [e["residual_deg"] for e in entries]
        # gauge-invariant dispersion: compare the residuals directly
        # since each is (G @ R)^T @ R_ref, and gauge is already removed,
        # dispersion = how different the residuals are across windows
        # pairwise dispersion: max angle between any pair of residuals
        max_disp = 0.0
        for a in range(len(resids)):
            for b in range(a + 1, len(resids)):
                d = rot_angle_deg(resids[a].T @ resids[b])
                if d > max_disp:
                    max_disp = d
        rows.append({
            "frame_index": img_idx,
            "n_windows": len(entries),
            "residual_median_deg": float(np.median(degs)),
            "residual_max_deg": float(np.max(degs)),
            "cross_window_dispersion_deg": max_disp,
        })

    csv_path = os.path.join(OUT_DIR, f"CROSS_WINDOW_CONTEXT_{seq_id}.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()
    sequences = args.seq if args.seq else list(SEQUENCES)

    summary = {}

    for seq_id in sequences:
        print(f"\n=== {seq_id} ===")

        meta = find_sequence_json(seq_id)
        ref_w2c = load_reference_poses(meta)
        if ref_w2c is None:
            print("  SKIP: no COLMAP reference")
            continue
        R_ref_seq = ref_w2c[:, :3, :3].transpose(0, 2, 1)

        # §15: local position
        pos_rows, pos_errors = local_position_analysis(seq_id, R_ref_seq)
        if pos_rows:
            # print boundary vs center comparison
            positions = [r["position"] for r in pos_rows]
            n = len(positions)
            boundary = [r for r in pos_rows if r["position"] <= 1 or r["position"] >= n - 2]
            center = [r for r in pos_rows if n // 2 - 2 <= r["position"] <= n // 2 + 1]
            b_med = np.median([r["median"] for r in boundary]) if boundary else -1
            c_med = np.median([r["median"] for r in center]) if center else -1
            print(f"  §15 position: boundary_med={b_med:.3f}° center_med={c_med:.3f}° "
                  f"(boundary/center ratio={b_med / c_med:.2f}x)" if c_med > 0 else "")

        # §16: central-owner overlap
        co_rows = central_owner_analysis(seq_id)
        if co_rows:
            # count how many edges have zero center overlap
            n_zero = sum(1 for r in co_rows
                         if r["n_edge_center_overlap_a"] == 0 and r["n_edge_center_overlap_b"] == 0)
            n_all = len(co_rows)
            print(f"  §16 central-owner: {n_all} edges, {n_zero}/{n_all} have ZERO center overlap "
                  f"({100*n_zero/max(n_all,1):.0f}%)")

        # §17, §24-§25: cross-window context
        xw_rows = cross_window_context_analysis(seq_id, R_ref_seq)
        if xw_rows:
            disps = [r["cross_window_dispersion_deg"] for r in xw_rows]
            print(f"  §17 cross-window: {len(xw_rows)} frames in ≥2 windows, "
                  f"dispersion median={np.median(disps):.3f}° max={np.max(disps):.3f}°")

        summary[seq_id] = {
            "local_position": {
                "boundary_median_deg": float(b_med) if boundary else None,
                "center_median_deg": float(c_med) if center else None,
            },
            "central_owner": {
                "n_edges": len(co_rows),
                "n_zero_center_overlap": n_zero if co_rows else 0,
            },
            "cross_window_context": {
                "n_multi_window_frames": len(xw_rows),
                "dispersion_median_deg": float(np.median(disps)) if xw_rows else None,
                "dispersion_max_deg": float(np.max(disps)) if xw_rows else None,
            },
        }

    with open(os.path.join(OUT_DIR, "LOCAL_POSITION_SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved LOCAL_POSITION_SUMMARY.json")


if __name__ == "__main__":
    main()