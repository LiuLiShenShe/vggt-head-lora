#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.1: Evaluate gauge-aware stitched results against COLMAP GT."""
import argparse, csv, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
GAUGE_DIR = os.path.join(PHASE3C, "05_global_stitching_v31")
CENTER_DIR = os.path.join(PHASE3C, "05_global_stitching")
OUT_DIR = os.path.join(PHASE3C, "06_pose_evaluation_v31")

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


def evaluate_full(ref_w2c, vggt_ext):
    """Evaluate full trajectory pose against reference."""
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

    # Neighbor RPE
    nrr, ntr = [], []
    for i in range(n - 1):
        R_rel_ref = R_ref[i].T @ R_ref[i + 1]
        R_rel_vggt = (Rg @ R_vggt[i]).T @ (Rg @ R_vggt[i + 1])
        nrr.append(rot_angle_deg(R_rel_ref.T @ R_rel_vggt))
        t_rel_ref = R_ref[i].T @ (centers_ref[i + 1] - centers_ref[i])
        t_rel_vggt = (Rg @ R_vggt[i]).T @ (centers_aligned[i + 1] - centers_aligned[i])
        if np.linalg.norm(t_rel_ref) > 1e-10:
            ntr.append(np.linalg.norm(t_rel_vggt - t_rel_ref) / np.linalg.norm(t_rel_ref))

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
        "neighbor_rpe_rot": float(np.mean(nrr)) if nrr else -1,
        "neighbor_rpe_trans": float(np.mean(ntr)) if ntr else -1,
        "pose_gate": "PASS" if gate else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()

    # Find all gauge-aware results
    gauge_files = sorted(glob.glob(os.path.join(GAUGE_DIR, "*_GAUGE_GLOBAL_CAMERAS.npz")))
    if args.seq:
        gauge_files = [g for g in gauge_files if any(seq in g for seq in args.seq)]

    all_results = []

    for gauge_path in gauge_files:
        seq_id = os.path.basename(gauge_path).replace("_GAUGE_GLOBAL_CAMERAS.npz", "")
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")
        print(f"{'='*60}")

        # Load reference
        try:
            seq = find_sequence_json(seq_id)
        except FileNotFoundError:
            print(f"  SKIP: no sequence JSON")
            continue
        ref_w2c = load_reference_poses(seq)
        if ref_w2c is None:
            print(f"  SKIP: no reference poses")
            continue

        # --- Method B: Gauge-aware ---
        data = np.load(gauge_path)
        gauge_ext = data["global_extrinsic"]
        orig_idx = data["original_frame_index"]
        ref_sub = ref_w2c[orig_idx]

        result_b = evaluate_full(ref_sub, gauge_ext)
        result_b["method"] = "gauge_aware"
        result_b["sequence_id"] = seq_id
        all_results.append(result_b)
        print(f"  Gauge-aware: rot_med={result_b['rot_median']:.2f}° gate={result_b['pose_gate']} "
              f"center={result_b['center_median_norm']:.4f} tc={result_b['trajectory_cosine']:.4f}")

        # --- Method A: Center-only (from Phase 3C) ---
        center_path = os.path.join(CENTER_DIR, f"{seq_id}_WINDOWED_GLOBAL_CAMERAS.npz")
        if os.path.exists(center_path):
            cdata = np.load(center_path)
            center_ext = cdata["global_extrinsic"]
            c_orig_idx = cdata["original_frame_index"]
            c_ref_sub = ref_w2c[c_orig_idx]
            result_a = evaluate_full(c_ref_sub, center_ext)
            result_a["method"] = "center_only"
            result_a["sequence_id"] = seq_id
            all_results.append(result_a)
            print(f"  Center-only: rot_med={result_a['rot_median']:.2f}° gate={result_a['pose_gate']} "
                  f"center={result_a['center_median_norm']:.4f}")

        # --- Method C: Consecutive local (best window) ---
        window_dir = os.path.join(PHASE3C, "03_window_inference", "window_outputs", seq_id)
        if os.path.isdir(window_dir):
            w0_path = os.path.join(window_dir, "window_000.npz")
            if os.path.exists(w0_path):
                w0 = np.load(w0_path)
                w0_ext = w0["ext_w2c_vggt"]
                w0_idx = w0["frame_idx"]
                ref_w0 = ref_w2c[w0_idx]
                result_c = evaluate_full(ref_w0, w0_ext)
                result_c["method"] = "consecutive_local"
                result_c["sequence_id"] = seq_id
                all_results.append(result_c)
                print(f"  Local W0:    rot_med={result_c['rot_median']:.2f}° gate={result_c['pose_gate']}")

        # --- Method D: Uniform sparse (from Phase 3B.1) ---
        uniform_csv = os.path.join(PHASE3B, "03_pose_evaluation", "TRUE_VIEWCOUNT_RESULTS.csv")
        if os.path.exists(uniform_csv):
            with open(uniform_csv) as f:
                for r in csv.DictReader(f):
                    if r["sequence_id"] == seq_id and r["view_count"] == "16":
                        all_results.append({
                            "method": "uniform_sparse_16v",
                            "sequence_id": seq_id,
                            "n_frames": 16,
                            "rot_median": float(r["rot_median"]),
                            "rot_p90": float(r["rot_p90"]),
                            "center_median_norm": float(r["center_median_norm"]),
                            "trajectory_cosine": float(r.get("trajectory_cosine", 0)),
                            "pose_gate": r["pose_gate"],
                        })
                        print(f"  Uniform 16v: rot_med={r['rot_median']}° gate={r['pose_gate']}")
                        break

    # Save CSV
    if all_results:
        csv_path = os.path.join(OUT_DIR, "GAUGE_AWARE_GLOBAL_RESULTS.csv")
        fields = ["sequence_id", "method", "n_frames",
                  "rot_median", "rot_p90", "rot_mean",
                  "center_median_norm", "center_p90_norm",
                  "trajectory_cosine", "neighbor_rpe_rot", "neighbor_rpe_trans",
                  "pose_gate"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_results)
        print(f"\nSaved: {csv_path} ({len(all_results)} rows)")

    # Summary table
    print(f"\n{'='*80}")
    print("METHOD COMPARISON: CENTER-ONLY vs GAUGE-AWARE")
    print(f"{'='*80}")
    seq_ids = sorted(set(r["sequence_id"] for r in all_results))
    methods = ["uniform_sparse_16v", "consecutive_local", "center_only", "gauge_aware"]
    print(f"{'Sequence':<38s} {'Uniform':>10s} {'Local W0':>10s} {'Center':>10s} {'Gauge':>10s}")
    print("-" * 82)
    for sid in seq_ids:
        vals = []
        for m in methods:
            row = next((r for r in all_results
                       if r["sequence_id"] == sid and r["method"] == m), None)
            if row:
                gate = "+" if row["pose_gate"] == "PASS" else "-"
                vals.append(f"{row['rot_median']:5.1f}°{gate}")
            else:
                vals.append("—")
        short = sid.replace("plantview__langdon_4__", "lang4__").replace("wheat3dgs__plot_", "w").replace("mustc__plot198__230613__ugv__", "")
        print(f"  {short:<36s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s} {vals[3]:>10s}")


if __name__ == "__main__":
    main()
