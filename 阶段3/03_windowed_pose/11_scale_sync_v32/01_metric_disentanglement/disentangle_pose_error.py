#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.2 Step 0: Disentangle orientation error from center/trajectory error.

Answers Q1: Is the 44-53° "rotation error" actually orientation error,
or is it trajectory shape distortion misinterpreted by the evaluation protocol?

Three evaluation modes:
A. ORIENTATION_ONLY — rotations only, no centers
B. CENTER_ONLY — centers only, no rotations
C. FULL — combined (existing protocol)

Usage:
    python 01_metric_disentanglement/disentangle_pose_error.py [--seq SEQ ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
GAUGE_DIR = os.path.join(PHASE3C, "05_global_stitching_v31")
OUT_DIR = os.path.join(PHASE3C, "11_scale_sync_v32", "01_metric_disentanglement")

sys.path.insert(0, os.path.join(PHASE3B, "03_pose_evaluation"))
from evaluate_multoplant import (
    global_rotation_procrustes, rot_angle_deg, w2c_centers, horn_sim3
)

os.makedirs(OUT_DIR, exist_ok=True)


def find_sequence_json(seq_id):
    for subdir in ["plant_view", "wheat3dgs", "mustc"]:
        for jp in glob.glob(os.path.join(SEQ_BASE, subdir, "*.json")):
            with open(jp) as f:
                meta = json.load(f)
            if meta["sequence_id"] == seq_id:
                return meta
    raise FileNotFoundError(f"Sequence JSON not found for {seq_id}")


def load_reference_poses(seq):
    ext_path = seq.get("extrinsics_path")
    if not ext_path or not os.path.exists(ext_path):
        return None
    with open(ext_path) as f:
        ext_data = json.load(f)
    ref_exts = ext_data.get("extrinsics", [])
    return np.array([np.array(e["w2c"])[:3, :4] for e in ref_exts])


def evaluate_orientation_only(ref_w2c, vggt_ext):
    """Evaluate using ONLY camera rotations (R_c2w). No centers involved."""
    n = min(len(ref_w2c), len(vggt_ext))
    R_ref = ref_w2c[:n, :3, :3].transpose(0, 2, 1)  # (n, 3, 3) c2w
    R_vggt = vggt_ext[:n, :3, :3].transpose(0, 2, 1)  # (n, 3, 3) c2w

    Rg = global_rotation_procrustes(R_vggt, R_ref)

    rot_errors = []
    for i in range(n):
        R_aligned = Rg @ R_vggt[i]
        rot_errors.append(rot_angle_deg(R_aligned.T @ R_ref[i]))
    rot_errors = np.array(rot_errors)

    return {
        "n_frames": n,
        "rot_median": float(np.median(rot_errors)),
        "rot_p90": float(np.percentile(rot_errors, 90)),
        "rot_mean": float(np.mean(rot_errors)),
        "rot_max": float(np.max(rot_errors)),
    }


def evaluate_center_only(ref_w2c, vggt_ext):
    """Evaluate using ONLY camera centers. No rotations involved."""
    n = min(len(ref_w2c), len(vggt_ext))
    centers_ref = w2c_centers(ref_w2c[:n])
    centers_vggt = w2c_centers(vggt_ext[:n])

    # Sim(3) alignment (includes scale, rotation, translation — but only on centers)
    s, R_sim3, t_sim3 = horn_sim3(centers_vggt, centers_ref)
    centers_aligned = s * (R_sim3 @ centers_vggt.T).T + t_sim3

    center_diff = centers_aligned - centers_ref
    ref_spread = np.linalg.norm(centers_ref - centers_ref.mean(0), axis=1).mean()
    cen_norm = np.linalg.norm(center_diff, axis=1) / max(ref_spread, 1e-10)

    # Trajectory cosine
    tc = 0.0
    if n > 1:
        dv = np.diff(centers_aligned, axis=0)
        dr = np.diff(centers_ref, axis=0)
        cosines = []
        for i in range(len(dv)):
            nv, nr = np.linalg.norm(dv[i]), np.linalg.norm(dr[i])
            if nv > 1e-10 and nr > 1e-10:
                cosines.append(np.dot(dv[i], dr[i]) / (nv * nr))
        tc = float(np.mean(cosines)) if cosines else 0.0

    # Scale error
    scale_error = abs(s - 1.0)

    return {
        "n_frames": n,
        "center_median_norm": float(np.median(cen_norm)),
        "center_p90_norm": float(np.percentile(cen_norm, 90)),
        "center_mean_norm": float(np.mean(cen_norm)),
        "trajectory_cosine": tc,
        "scale_estimated": float(s),
        "scale_error": float(scale_error),
    }


def evaluate_full(ref_w2c, vggt_ext):
    """Full pose evaluation (existing protocol — rotations + centers combined)."""
    n = min(len(ref_w2c), len(vggt_ext))
    ref_sub = ref_w2c[:n]
    vggt_sub = vggt_ext[:n]

    R_ref = ref_sub[:, :3, :3].transpose(0, 2, 1)
    R_vggt = vggt_sub[:, :3, :3].transpose(0, 2, 1)

    Rg = global_rotation_procrustes(R_vggt, R_ref)

    rot_errors = []
    for i in range(n):
        R_aligned = Rg @ R_vggt[i]
        rot_errors.append(rot_angle_deg(R_aligned.T @ R_ref[i]))
    rot_errors = np.array(rot_errors)

    centers_ref = w2c_centers(ref_sub)
    centers_vggt = w2c_centers(vggt_sub)
    s, R_sim3, t_sim3 = horn_sim3(centers_vggt, centers_ref)
    centers_aligned = s * (R_sim3 @ centers_vggt.T).T + t_sim3
    center_diff = centers_aligned - centers_ref
    ref_spread = np.linalg.norm(centers_ref - centers_ref.mean(0), axis=1).mean()
    cen_norm = np.linalg.norm(center_diff, axis=1) / max(ref_spread, 1e-10)

    tc = 0.0
    if n > 1:
        dv = np.diff(centers_aligned, axis=0)
        dr = np.diff(centers_ref, axis=0)
        cosines = []
        for i in range(len(dv)):
            nv, nr = np.linalg.norm(dv[i]), np.linalg.norm(dr[i])
            if nv > 1e-10 and nr > 1e-10:
                cosines.append(np.dot(dv[i], dr[i]) / (nv * nr))
        tc = float(np.mean(cosines)) if cosines else 0.0

    rot_median = float(np.median(rot_errors))
    rot_p90 = float(np.percentile(rot_errors, 90))
    gate = rot_median <= 10.0 and rot_p90 <= 20.0

    return {
        "n_frames": n,
        "rot_median": rot_median,
        "rot_p90": rot_p90,
        "rot_mean": float(np.mean(rot_errors)),
        "center_median_norm": float(np.median(cen_norm)),
        "center_p90_norm": float(np.percentile(cen_norm, 90)),
        "trajectory_cosine": tc,
        "pose_gate": "PASS" if gate else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()

    gauge_files = sorted(glob.glob(os.path.join(GAUGE_DIR, "*_GAUGE_GLOBAL_CAMERAS.npz")))
    if args.seq:
        gauge_files = [g for g in gauge_files if any(seq in g for seq in args.seq)]

    all_rows = []

    for gauge_path in gauge_files:
        seq_id = os.path.basename(gauge_path).replace("_GAUGE_GLOBAL_CAMERAS.npz", "")
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")

        try:
            seq = find_sequence_json(seq_id)
        except FileNotFoundError:
            print(f"  SKIP: no sequence JSON")
            continue
        ref_w2c = load_reference_poses(seq)
        if ref_w2c is None:
            print(f"  SKIP: no reference poses")
            continue

        data = np.load(gauge_path)
        gauge_ext = data["global_extrinsic"]
        orig_idx = data["original_frame_index"]
        ref_sub = ref_w2c[orig_idx]

        # Three evaluation modes
        orient = evaluate_orientation_only(ref_sub, gauge_ext)
        center = evaluate_center_only(ref_sub, gauge_ext)
        full = evaluate_full(ref_sub, gauge_ext)

        # Key diagnostic
        orient_gate = orient["rot_median"] <= 10.0 and orient["rot_p90"] <= 20.0
        center_gate = center["trajectory_cosine"] > 0.5
        diagnose = "ORIENTATION_OK_BUT_CENTER_DRIFT" if orient_gate and not center_gate \
            else "BOTH_FAIL" if not orient_gate and not center_gate \
            else "CENTER_OK_BUT_ORIENT_FAIL" if not orient_gate and center_gate \
            else "BOTH_OK"

        print(f"  Orientation-only: rot_med={orient['rot_median']:.2f}° "
              f"rot_p90={orient['rot_p90']:.2f}° gate={'PASS' if orient_gate else 'FAIL'}")
        print(f"  Center-only:      cen_med={center['center_median_norm']:.4f} "
              f"tc={center['trajectory_cosine']:.4f} scale={center['scale_estimated']:.4f}")
        print(f"  Full (combined):  rot_med={full['rot_median']:.2f}° "
              f"gate={full['pose_gate']}")
        print(f"  DIAGNOSIS: {diagnose}")

        row = {
            "sequence_id": seq_id,
            "n_frames": orient["n_frames"],
            # Orientation-only
            "orient_rot_median": orient["rot_median"],
            "orient_rot_p90": orient["rot_p90"],
            "orient_rot_mean": orient["rot_mean"],
            "orient_rot_max": orient["rot_max"],
            "orient_gate": "PASS" if orient_gate else "FAIL",
            # Center-only
            "center_median_norm": center["center_median_norm"],
            "center_p90_norm": center["center_p90_norm"],
            "center_trajectory_cosine": center["trajectory_cosine"],
            "center_scale_estimated": center["scale_estimated"],
            "center_scale_error": center["scale_error"],
            # Full
            "full_rot_median": full["rot_median"],
            "full_rot_p90": full["rot_p90"],
            "full_center_median_norm": full["center_median_norm"],
            "full_trajectory_cosine": full["trajectory_cosine"],
            "full_pose_gate": full["pose_gate"],
            # Diagnosis
            "diagnosis": diagnose,
        }
        all_rows.append(row)

    # Save CSV
    if all_rows:
        csv_path = os.path.join(OUT_DIR, "POSE_ERROR_DISENTANGLEMENT.csv")
        fields = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nSaved: {csv_path} ({len(all_rows)} rows)")

    # Summary table
    print(f"\n{'='*90}")
    print("DISENTANGLEMENT SUMMARY")
    print(f"{'='*90}")
    print(f"{'Sequence':<40s} {'Orient':>8s} {'Center':>8s} {'Full':>8s} {'Diagnosis'}")
    print("-" * 90)
    for r in all_rows:
        short = r["sequence_id"].replace("plantview__langdon_4__", "lang4__")
        print(f"  {short:<38s} {r['orient_rot_median']:5.1f}°{'+'if r['orient_gate']=='PASS' else '-':>2s} "
              f"  tc={r['center_trajectory_cosine']:+.3f} "
              f"  {r['full_rot_median']:5.1f}°{'+'if r['full_pose_gate']=='PASS' else '-':>2s} "
              f"  {r['diagnosis']}")


if __name__ == "__main__":
    main()
