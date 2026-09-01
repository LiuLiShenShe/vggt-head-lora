#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C P1: Window-to-Window Alignment via Sim(3).

Aligns consecutive windows using overlap-frame camera predictions.
No reference poses used — only VGGT predictions.

Two baselines:
  A. Camera-center Umeyama Sim(3)
  B. Camera-center + orientation constrained Sim(3)

Usage:
    python 04_window_alignment/run_alignment.py [--seq SEQUENCE_ID ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
OUT_DIR = os.path.join(PHASE3C, "04_window_alignment")

sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))
from evaluate_multoplant import horn_sim3, rot_angle_deg


def w2c_centers(ext_w2c_3x4):
    R = ext_w2c_3x4[:, :3, :3]
    t = ext_w2c_3x4[:, :3, 3]
    return np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)


def align_center_umeyama(src_centers, dst_centers):
    """Umeyama Sim(3) alignment from camera centers."""
    mu_s, mu_d = src_centers.mean(0), dst_centers.mean(0)
    sc, dc = src_centers - mu_s, dst_centers - mu_d
    cov = dc.T @ sc / len(src_centers)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / max((sc ** 2).sum() / len(src_centers), 1e-10)
    t = mu_d - s * R @ mu_s
    return s, R, t


def align_center_orientation(src_centers, dst_centers, src_Rs, dst_Rs):
    """Sim(3) from camera centers + orientation jointly."""
    # Stage 1: rotation alignment from orientations
    # Stack all rotation vectors (flattened rotation columns)
    H = np.einsum("sij,skj->ik", src_Rs, dst_Rs)
    U, _, Vt = np.linalg.svd(H)
    R_align = Vt.T @ U.T
    if np.linalg.det(R_align) < 0:
        Vt[-1, :] *= -1
        R_align = Vt.T @ U.T

    # Stage 2: apply rotation, then scale+translation from centers
    src_rotated = (R_align @ src_centers.T).T
    return R_align, *align_center_umeyama(src_rotated, dst_centers)[1:]


def check_degeneracy(centers, min_spread=1e-6):
    """Check if overlap camera centers are degenerate (nearly co-point)."""
    spread = np.linalg.norm(centers - centers.mean(0), axis=1).mean()
    centroid = centers.mean(0)
    # Condition number of relative vectors
    if len(centers) < 3:
        return True, spread
    vecs = centers - centroid
    cov = vecs.T @ vecs / len(vecs)
    eigvals = np.linalg.eigvalsh(cov)
    cond = eigvals[-1] / max(eigvals[0], 1e-12)
    return cond > 1e6 or spread < min_spread, spread


def load_window_windows(seq_dir):
    """Load all window extrinsics from npz files."""
    windows = []
    for f in sorted(glob.glob(os.path.join(seq_dir, "window_*.npz"))):
        data = np.load(f)
        windows.append({
            "path": f,
            "ext_w2c": data["ext_w2c_vggt"],       # (n, 3, 4)
            "frame_idx": data["frame_idx"],           # (n,)
        })
    return windows


def find_overlap_frames(frames_a, frames_b):
    """Find common frame indices between two windows."""
    set_a = set(frames_a.tolist()) if isinstance(frames_a, np.ndarray) else set(frames_a)
    set_b = set(frames_b.tolist()) if isinstance(frames_b, np.ndarray) else set(frames_b)
    return sorted(set_a & set_b)


def get_overlap_poses(ext_w2c, frame_idx, overlap_frames):
    """Extract camera poses for overlap frames."""
    centers = w2c_centers(ext_w2c)
    Rs = ext_w2c[:, :3, :3]  # w2c rotation
    frame_to_idx = {int(f): i for i, f in enumerate(frame_idx)}

    overlap_centers = []
    overlap_Rs = []
    for f in overlap_frames:
        if f in frame_to_idx:
            idx = frame_to_idx[f]
            overlap_centers.append(centers[idx])
            overlap_Rs.append(Rs[idx])

    return np.array(overlap_centers), np.array(overlap_Rs)


def run_alignment(seq_id, windows, method="center_umeyama"):
    """Run pairwise window alignment. Returns list of alignment results."""
    results = []
    manifest = {"sequence_id": seq_id, "method": method, "alignments": []}

    for i in range(len(windows) - 1):
        w_a = windows[i]
        w_b = windows[i + 1]

        overlap_frames = find_overlap_frames(w_a["frame_idx"], w_b["frame_idx"])
        n_overlap = len(overlap_frames)

        if n_overlap < 2:
            print(f"  W{i}->W{i+1}: FAIL (only {n_overlap} overlap frames)")
            results.append({
                "window_a": i, "window_b": i + 1,
                "n_overlap": n_overlap,
                "status": "FAIL_INSUFFICIENT_OVERLAP",
            })
            continue

        # Get overlap poses from each window
        centers_a, Rs_a = get_overlap_poses(w_a["ext_w2c"], w_a["frame_idx"], overlap_frames)
        centers_b, Rs_b = get_overlap_poses(w_b["ext_w2c"], w_b["frame_idx"], overlap_frames)

        # Check degeneracy
        degenerate, spread = check_degeneracy(centers_a)
        if degenerate:
            print(f"  W{i}->W{i+1}: FAIL (degenerate centers, spread={spread:.6f})")
            results.append({
                "window_a": i, "window_b": i + 1,
                "n_overlap": n_overlap,
                "status": "FAIL_DEGENERATE",
                "center_spread": spread,
            })
            continue

        # Compute alignment BEFORE
        rmse_before = float(np.sqrt(np.mean(np.linalg.norm(centers_a - centers_b, axis=1)**2)))
        rot_errs_before = [rot_angle_deg(Ra.T @ Rb) for Ra, Rb in zip(Rs_a, Rs_b)]
        rot_err_before = float(np.mean(rot_errs_before)) if rot_errs_before else -1

        # Compute alignment
        if method == "center_umeyama":
            s, R, t = align_center_umeyama(centers_a, centers_b)
        else:
            R_align, s, R, t = align_center_orientation(centers_a, centers_b, Rs_a, Rs_b)

        # Apply alignment to window A's overlap centers
        centers_a_aligned = s * (R @ centers_a.T).T + t

        # Compute alignment AFTER
        rmse_after = float(np.sqrt(np.mean(np.linalg.norm(centers_a_aligned - centers_b, axis=1)**2)))

        # Alignment parameters
        scale = float(s)
        rotation_deg = float(rot_angle_deg(R))
        translation_norm = float(np.linalg.norm(t))

        entry = {
            "window_a": i, "window_b": i + 1,
            "n_overlap": n_overlap,
            "overlap_frames": overlap_frames,
            "status": "OK",
            "alignment_scale": scale,
            "alignment_rotation_deg": rotation_deg,
            "alignment_translation_norm": translation_norm,
            "overlap_center_RMSE_before": rmse_before,
            "overlap_center_RMSE_after": rmse_after,
            "overlap_rotation_error_before": rot_err_before,
            "center_spread": spread,
            "s": s, "R": R.tolist(), "t": t.tolist(),
        }
        results.append(entry)
        manifest["alignments"].append(entry)

        print(f"  W{i:2d}->W{i+1:2d}: n_overlap={n_overlap:2d} "
              f"RMSE {rmse_before:.6f}->{rmse_after:.6f} "
              f"scale={scale:.4f} rot={rotation_deg:.2f}° "
              f"t_norm={translation_norm:.4f}")

    return results, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*", help="Specific sequences")
    ap.add_argument("--method", choices=["center_umeyama", "center_orientation"],
                    default="center_umeyama")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Find all sequences with window outputs
    if args.seq:
        seq_dirs = [os.path.join(WINDOW_DIR, s) for s in args.seq]
    else:
        seq_dirs = sorted([os.path.join(WINDOW_DIR, d)
                          for d in os.listdir(WINDOW_DIR)
                          if os.path.isdir(os.path.join(WINDOW_DIR, d))])

    all_alignments = []
    for seq_dir in seq_dirs:
        if not os.path.isdir(seq_dir):
            continue
        seq_id = os.path.basename(seq_dir)
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")
        print(f"{'='*60}")

        windows = load_window_windows(seq_dir)
        if len(windows) < 2:
            print(f"  SKIP: only {len(windows)} window(s)")
            continue

        results, manifest = run_alignment(seq_id, windows, method=args.method)

        # Save manifest
        mf_path = os.path.join(OUT_DIR, f"{seq_id}_ALIGNMENT_MANIFEST.json")
        with open(mf_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)

        all_alignments.extend([{**r, "sequence_id": seq_id} for r in results])

    # Save CSV
    if all_alignments:
        csv_path = os.path.join(OUT_DIR, "ALIGNMENT_RESULTS.csv")
        fields = ["sequence_id", "window_a", "window_b", "n_overlap", "status",
                  "alignment_scale", "alignment_rotation_deg", "alignment_translation_norm",
                  "overlap_center_RMSE_before", "overlap_center_RMSE_after",
                  "overlap_rotation_error_before", "center_spread"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_alignments)
        print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
