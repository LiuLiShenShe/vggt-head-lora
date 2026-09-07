#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3.1 §11-§14: Gauge-invariant within-window relative rotation analysis.

Motivation: the per-window gauge fit (§5) assumes a single SO(3) gauge per window
is VALID. But that fit itself can mask gauge-independent error. Classic invariant:
    R_pred[f_a].T @ R_pred[f_b]   vs   R_ref[f_a].T @ R_ref[f_b]
is IDENTICAL under any global gauge G applied to the whole window, because
    (G R_a)^T (G R_b) = R_a^T G^T G R_b = R_a^T R_b.
So the pair (f_a, f_b) relative rotation needs NO gauge fit, NO Procrustes, and
any discrepancy is directly attributable to VGGT local orientation error —
NOT to gauge freedom.

For each window k and each frame pair (f_a, f_b):
    rel_err = angle( (R_v[f_a]^T R_v[f_b])^T (R_ref[f_a]^T R_ref[f_b]) )

Outputs (in 03_relative_rotation/):
  WINDOW_RELATIVE_ROT_ERRORS_<seq>.csv  — per-window, per-pair relative error
  RELATIVE_ERROR_BY_SEPARATION_<seq>.csv — bucketed by |f_a - f_b| (frame separation)
  RELATIVE_ROTATION_GLOBAL_SUMMARY.json  — per-sequence summary

Comparison to §5/§6: if relative-rotation error is LOW but per-window gauge
residual is HIGH, the violation is gauge-level (window gauge varies within a
window — the single-gauge assumption is invalid). If relative error is HIGH,
VGGT local rotations themselves disagree with the reference at the sub-window
level.

DIAGNOSTIC_ONLY: uses COLMAP reference for evaluation only; no gauge output here.

Usage:
    python 03_relative_rotation/relative_rotation_analysis.py [--seq ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np

from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4")
OUT_DIR = os.path.join(PHASE3C, "13_q_bias_audit_v331", "03_relative_rotation")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))
sys.path.insert(0, os.path.join(ROOT, "阶段3", "03_windowed_pose", "12_rotation_sync_v33", "06_evaluation"))
from evaluate_multoplant import rot_angle_deg
from evaluate_rotation_sync import find_sequence_json, load_reference_poses

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)


def load_window_c2w(seq_id, window_idx):
    path = os.path.join(WINDOW_DIR, seq_id, f"window_{window_idx:03d}.npz")
    if not os.path.exists(path):
        return None, None
    d = np.load(path)
    ext = d["ext_w2c_vggt"]
    R_c2w = ext[:, :3, :3].transpose(0, 2, 1)
    return R_c2w, d["frame_idx"].astype(int)


def analyze_window(R_v, R_ref, fidx):
    """Relative rotation error for all frame pairs within one window.

    Returns list of (f_a, f_b, separation, err_deg).
    """
    n = len(fidx)
    rows = []
    for a in range(n):
        for b in range(a + 1, n):
            rel_pred = R_v[a].T @ R_v[b]
            rel_ref = R_ref[a].T @ R_ref[b]
            err = rot_angle_deg(rel_pred.T @ rel_ref)
            sep = int(fidx[b]) - int(fidx[a])  # frame separation in original seq
            rows.append((int(fidx[a]), int(fidx[b]), sep, err))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    ap.add_argument("--max_sep", type=int, default=15,
                    help="max frame separation to report bucketed (default 15)")
    args = ap.parse_args()
    sequences = args.seq if args.seq else list(SEQUENCES)

    global_summary = {}

    for seq_id in sequences:
        print(f"\n=== {seq_id} ===")

        meta = find_sequence_json(seq_id)
        ref_w2c = load_reference_poses(meta)
        if ref_w2c is None:
            print("  SKIP: no COLMAP reference")
            continue
        R_ref_seq = ref_w2c[:, :3, :3].transpose(0, 2, 1)  # (N,3,3) c2w

        win_files = sorted(glob.glob(os.path.join(WINDOW_DIR, seq_id, "window_*.npz")))
        n_w = len(win_files)

        all_rows = []       # per-pair rows across all windows
        per_window_med = []  # median relative error per window

        for k in range(n_w):
            R_v, fidx = load_window_c2w(seq_id, k)
            if R_v is None or len(fidx) < 2:
                continue
            R_ref = R_ref_seq[fidx]
            rows = analyze_window(R_v, R_ref, fidx)
            per_window_med.append(np.median([r[3] for r in rows]))
            all_rows.extend(rows)

        all_rows = sorted(all_rows, key=lambda r: (r[2], r[0], r[1]))
        print(f"  windows={n_w}, pairs={len(all_rows)}, "
              f"median_rel_err={np.median([r[3] for r in all_rows]):.3f}°, "
              f"p90={np.percentile([r[3] for r in all_rows],90):.3f}°")

        # ---- Save full per-pair CSV ----
        pair_csv = os.path.join(OUT_DIR, f"WINDOW_RELATIVE_ROT_ERRORS_{seq_id}.csv")
        with open(pair_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_a", "frame_b", "frame_separation", "relative_err_deg"])
            w.writerows(all_rows)

        # ---- Bucket by frame separation (0..max_sep) ----
        sep_stats = {}
        sep_rows = []
        for sep in range(1, args.max_sep + 1):
            errs = np.array([r[3] for r in all_rows if r[2] == sep])
            if len(errs):
                sep_stats[sep] = {
                    "n": int(len(errs)),
                    "median": float(np.median(errs)),
                    "p90": float(np.percentile(errs, 90)),
                    "max": float(np.max(errs)),
                }
                sep_rows.append([sep, len(errs), np.median(errs),
                                 np.percentile(errs, 90), np.max(errs)])
        sep_csv = os.path.join(OUT_DIR, f"RELATIVE_ERROR_BY_SEPARATION_{seq_id}.csv")
        with open(sep_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame_separation", "n_pairs", "median_deg", "p90_deg", "max_deg"])
            w.writerows(sep_rows)

        # Print compact separation trend
        for sep in range(1, args.max_sep + 1):
            if sep in sep_stats:
                print(f"    sep={sep:2d}: n={sep_stats[sep]['n']:3d} "
                      f"med={sep_stats[sep]['median']:.3f}° "
                      f"p90={sep_stats[sep]['p90']:.3f}°")

        global_summary[seq_id] = {
            "n_windows": n_w,
            "n_frame_pairs": len(all_rows),
            "relative_err_median_deg": float(np.median([r[3] for r in all_rows])),
            "relative_err_p90_deg": float(np.percentile([r[3] for r in all_rows], 90)),
            "relative_err_max_deg": float(np.max([r[3] for r in all_rows])),
            "window_median_err_median_deg": float(np.median(per_window_med)),
            "window_median_err_max_deg": float(np.max(per_window_med)),
            "separation_stats": {str(k): v for k, v in sep_stats.items()},
        }

    with open(os.path.join(OUT_DIR, "RELATIVE_ROTATION_GLOBAL_SUMMARY.json"), "w") as f:
        json.dump(global_summary, f, indent=2)
    print(f"\nSaved RELATIVE_ROTATION_GLOBAL_SUMMARY.json")


if __name__ == "__main__":
    main()