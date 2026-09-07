#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3.1 §5-§6: Corrected per-window reference gauge fit.

Repairs the per-window Procrustes bug from rotation_diagnostics.py:
  OLD (buggy):  H = einsum("sij,sik->jk", R_ref, R_local)  = Σ R_ref^T @ R_local
  NEW (correct): H = einsum("sij,skj->ik", R_local, R_ref)  = Σ R_local @ R_ref^T

Minimizes  Σ_f || G_k @ R_local,f - R_ref,f ||_F  →  R_ref,f ≈ G_k @ R_local,f.

For each window k in each sequence:
  1. Load VGGT c2w rotations from window npz (R_local)
  2. Load COLMAP reference c2w rotations for the same frames (R_ref)
  3. Fit corrected gauge G_k via global_rotation_procrustes
  4. Compute per-frame residual and window-level fit quality

Then for each pairwise edge (i,j) from the stride-4 graph:
  5. Compute Q_ref_ij = G_i^T @ G_j (corrected reference relative rotation)
  6. Compare to measured Q_ij from VGGT edges (angle error)

Outputs (in 02_window_gauge_fit/):
  CORRECTED_WINDOW_GAUGES_<seq>.npz     — G_k (N_w,3,3) + per-frame residuals
  CORRECTED_Q_VS_REF_<seq>.csv          — per-edge Q_ref vs Q_ij comparison
  WINDOW_GAUGE_FIT_SUMMARY.json         — per-sequence summary statistics

DIAGNOSTIC_ONLY: uses COLMAP reference as ground truth; G_k is not used in solver.

Usage:
    python 02_window_gauge_fit/corrected_window_gauge_fit.py [--seq ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
EDGES_DIR = os.path.join(SYNC_DIR, "02_rotation_edges")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4")
OUT_DIR = os.path.join(PHASE3C, "13_q_bias_audit_v331", "02_window_gauge_fit")
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

THRESHOLD = 8  # min_overlap_frames for edge inclusion


def load_window_c2w(seq_id, window_idx):
    """Load VGGT c2w rotation matrices from window npz.

    Returns:
        R_local_c2w: (16,3,3) — VGGT camera-to-world rotations
        frame_idx:   (16,)    — original frame indices in the sequence
    """
    path = os.path.join(WINDOW_DIR, seq_id, f"window_{window_idx:03d}.npz")
    if not os.path.exists(path):
        return None, None
    d = np.load(path)
    ext = d["ext_w2c_vggt"]  # (16,3,4) w2c
    R_c2w = ext[:, :3, :3].transpose(0, 2, 1)  # (16,3,3) c2w
    frame_idx = d["frame_idx"].astype(int)
    return R_c2w, frame_idx


def load_ref_c2w(seq_id):
    """Load COLMAP reference c2w rotations for entire sequence.

    Returns:
        R_ref_c2w: (N_total, 3, 3) — COLMAP camera-to-world rotations
    """
    meta = find_sequence_json(seq_id)
    ref_w2c = load_reference_poses(meta)  # (N, 3, 4) — w2c[:3,:4]
    if ref_w2c is None:
        return None
    return ref_w2c[:, :3, :3].transpose(0, 2, 1)  # (N, 3, 3) c2w


def load_edges(seq_id, threshold):
    """Load pairwise Q edges from the stride-4 graph CSV."""
    path = os.path.join(EDGES_DIR, "ROTATION_GRAPH_EDGES.csv")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = [r for r in csv.DictReader(f)
                if r["sequence"] == seq_id and int(r["n_overlap"]) >= threshold]
    if not rows:
        return None
    i_arr = np.array([int(r["window_i"]) for r in rows])
    j_arr = np.array([int(r["window_j"]) for r in rows])
    Q_arr = np.array([[[float(r[f"q{rr}{cc}"]) for cc in range(3)] for rr in range(3)]
                      for r in rows])
    hop_arr = np.array([int(r["window_distance"]) for r in rows])
    return i_arr, j_arr, Q_arr, hop_arr


def fit_all_windows(seq_id, R_ref_c2w):
    """Fit corrected gauge G_k for every window in a sequence.

    Returns:
        G:            (N_w, 3, 3) — corrected gauge per window
        residuals:    list of per-frame (16,) residual arrays
        residual_med: per-window median residual (N_w,)
        center_frame: per-window center frame index (for central-owner analysis)
    """
    win_dir = os.path.join(WINDOW_DIR, seq_id)
    npz_files = sorted(glob.glob(os.path.join(win_dir, "window_*.npz")))
    N_w = len(npz_files)
    G = np.zeros((N_w, 3, 3))
    residuals = []
    residual_med = np.zeros(N_w)
    center_frame = np.zeros(N_w, dtype=int)

    for k, wf in enumerate(npz_files):
        R_local, fidx = load_window_c2w(seq_id, k)
        if R_local is None:
            G[k] = np.eye(3)
            residuals.append(np.zeros(16))
            continue
        R_ref = R_ref_c2w[fidx]  # (16,3,3) — reference for these 16 frames
        Gk = global_rotation_procrustes(R_local, R_ref)
        G[k] = Gk
        # per-frame residual (handle variable window size; last window may have < 16 frames)
        aligned = np.einsum("ab,sbc->sac", Gk, R_local)
        n_fr = len(fidx)
        errs = np.array([rot_angle_deg(aligned[f].T @ R_ref[f]) for f in range(n_fr)])
        residuals.append(errs)
        residual_med[k] = float(np.median(errs))
        center_frame[k] = int(fidx[n_fr // 2])  # center of window

    return G, residuals, residual_med, center_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    ap.add_argument("--threshold", type=int, default=THRESHOLD)
    args = ap.parse_args()
    sequences = args.seq if args.seq else list(SEQUENCES)
    th = args.threshold

    summary = {}

    for seq_id in sequences:
        print(f"\n=== {seq_id} ===")

        # Load reference poses
        R_ref_c2w = load_ref_c2w(seq_id)
        if R_ref_c2w is None:
            print("  SKIP: no COLMAP reference")
            continue

        # Fit corrected gauges for all windows
        G, residuals, res_med, center_fr = fit_all_windows(seq_id, R_ref_c2w)
        N_w = len(G)
        print(f"  N_windows={N_w}, median_fit_err: "
              f"mean={np.mean(res_med):.4f}° med={np.median(res_med):.4f}° "
              f"max={np.max(res_med):.4f}°")

        # Save corrected gauges (DIAGNOSTIC_ONLY — not used in solver)
        npz_path = os.path.join(OUT_DIR, f"CORRECTED_WINDOW_GAUGES_{seq_id}.npz")
        np.savez(npz_path, G=G, residual_med=res_med, center_frame=center_fr)
        print(f"  Saved {os.path.basename(npz_path)}")

        # Load edges and compute Q-vs-ref comparison
        edge_data = load_edges(seq_id, th)
        if edge_data is None:
            print("  No edges found")
            continue
        i_arr, j_arr, Q_arr, hop_arr = edge_data

        rows = []
        for e in range(len(i_arr)):
            a, b = int(i_arr[e]), int(j_arr[e])
            if a >= N_w or b >= N_w:
                continue
            # Corrected Q_ref from fitted gauges
            Q_ref = G[a].T @ G[b]
            # Measured Q from VGGT edges
            Q_meas = Q_arr[e]
            # Angular error between Q_ref and Q_meas
            err = rot_angle_deg(Q_ref.T @ Q_meas)
            hop = int(hop_arr[e])
            rows.append({
                "window_i": a, "window_j": b,
                "hop": hop,
                "q_ref_vs_meas_deg": round(err, 4),
                "gauge_fit_resid_i_deg": round(float(res_med[a]), 4),
                "gauge_fit_resid_j_deg": round(float(res_med[b]), 4),
            })

        # Save Q-vs-ref comparison CSV
        csv_path = os.path.join(OUT_DIR, f"CORRECTED_Q_VS_REF_{seq_id}.csv")
        if rows:
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

        # Per-hop statistics
        hop_stats = {}
        for hop_k in sorted(set(r["hop"] for r in rows)):
            errs_h = [r["q_ref_vs_meas_deg"] for r in rows if r["hop"] == hop_k]
            hop_stats[hop_k] = {
                "n": len(errs_h),
                "median": round(float(np.median(errs_h)), 4),
                "p90": round(float(np.percentile(errs_h, 90)), 4),
                "mean": round(float(np.mean(errs_h)), 4),
            }
            print(f"  hop{hop_k}: n={len(errs_h)} "
                  f"Q_ref-vs-Q_meas med={np.median(errs_h):.3f}° "
                  f"p90={np.percentile(errs_h, 90):.3f}°")

        summary[seq_id] = {
            "n_windows": N_w,
            "gauge_fit_median_deg": round(float(np.median(res_med)), 4),
            "gauge_fit_p90_deg": round(float(np.percentile(res_med, 90)), 4),
            "gauge_fit_max_deg": round(float(np.max(res_med)), 4),
            "n_edges": len(rows),
            "hop_statistics": hop_stats,
            "q_ref_vs_meas_median_all": round(float(np.median(
                [r["q_ref_vs_meas_deg"] for r in rows])), 4),
            "q_ref_vs_meas_p90_all": round(float(np.percentile(
                [r["q_ref_vs_meas_deg"] for r in rows], 90)), 4),
        }

    # Save summary
    summary_path = os.path.join(OUT_DIR, "WINDOW_GAUGE_FIT_SUMMARY.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {summary_path}")


if __name__ == "__main__":
    main()
