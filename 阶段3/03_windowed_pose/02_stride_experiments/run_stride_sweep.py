#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C P0: Stride Sensitivity Experiment.

For each FAIL sequence, tests strides [1,2,4,6,8,10,12,16,20] with 5 start offsets.
Each combination runs independent VGGT forward on N=16 views.

Usage:
    conda activate vggt_lora
    python 02_stride_experiments/run_stride_sweep.py [--force]
"""
import argparse, csv, glob, json, os, sys, time
import numpy as np
import torch

ROOT = "/fj/VGGT+head+lora实验"
VGGT_ROOT = os.path.join(ROOT, "vggt")
sys.path.insert(0, VGGT_ROOT)
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
OUT_DIR = os.path.join(PHASE3C, "02_stride_experiments", "outputs")

# Reuse evaluation from Phase 3B
sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))
from evaluate_multoplant import (
    global_rotation_procrustes, rot_angle_deg, w2c_centers, horn_sim3
)

# Sequences
FAIL_DATES = ["12-03-24", "15-04-24", "19-03-24"]
PASS_DATES = ["05-03-24"]
WHEAT3DGS = ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
MUSTC = ["mustc__plot198__230613__ugv__pos00"]

ALL_SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in FAIL_DATES + PASS_DATES]
    + WHEAT3DGS + MUSTC
)

STRIDES = [1, 2, 4, 6, 8, 10, 12, 16, 20]
N_VIEWS = 16
START_FRAC = [0.0, 0.2, 0.4, 0.6, 0.8]


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


def evaluate_pose(ref_w2c, vggt_ext):
    n = len(vggt_ext)
    if n < 3:
        return None
    R_ref_c2w = ref_w2c[:, :3, :3].transpose(0, 2, 1)
    R_vggt_c2w = vggt_ext[:, :3, :3].transpose(0, 2, 1)
    Rg = global_rotation_procrustes(R_vggt_c2w, R_ref_c2w)
    rot_errors = []
    for i in range(n):
        R_aligned = Rg @ R_vggt_c2w[i]
        rot_errors.append(rot_angle_deg(R_aligned.T @ R_ref_c2w[i]))
    rot_errors = np.array(rot_errors)

    centers_ref = w2c_centers(ref_w2c)
    centers_vggt = w2c_centers(vggt_ext)
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
        R_rel_ref = R_ref_c2w[i].T @ R_ref_c2w[i + 1]
        R_rel_vggt = (Rg @ R_vggt_c2w[i]).T @ (Rg @ R_vggt_c2w[i + 1])
        nrr.append(rot_angle_deg(R_rel_ref.T @ R_rel_vggt))
        t_rel_ref = R_ref_c2w[i].T @ (centers_ref[i + 1] - centers_ref[i])
        t_rel_vggt = (Rg @ R_vggt_c2w[i]).T @ (centers_aligned[i + 1] - centers_aligned[i])
        if np.linalg.norm(t_rel_ref) > 1e-10:
            ntr.append(np.linalg.norm(t_rel_vggt - t_rel_ref) / np.linalg.norm(t_rel_ref))

    rot_median = float(np.median(rot_errors))
    rot_p90 = float(np.percentile(rot_errors, 90))
    gate = rot_median <= 10.0 and rot_p90 <= 20.0

    return {
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


def compute_view_geometry(frame_indices, ref_w2c):
    """Compute angular gap and coverage metrics from frame indices."""
    n = len(frame_indices)
    if n < 2 or ref_w2c is None:
        return {}

    # Camera centers and orientations from reference
    R = ref_w2c[:, :3, :3]
    t = ref_w2c[:, :3, 3]
    centers = np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)
    orientations = R.transpose(0, 2, 1)  # c2w rotation

    # Adjacent angular gaps
    gaps = []
    for i in range(n - 1):
        v1 = centers[i + 1] - centers[i]
        v2 = centers[min(i + 2, n - 1)] - centers[i + 1] if i + 2 < n else v1
        # Use view direction from center to centroid
        centroid = centers.mean(axis=0)
        d1 = centers[i] - centroid
        d2 = centers[i + 1] - centroid
        cos_a = np.clip(np.dot(d1 / max(np.linalg.norm(d1), 1e-10),
                                d2 / max(np.linalg.norm(d2), 1e-10)), -1, 1)
        gaps.append(float(np.degrees(np.arccos(cos_a))))

    # Total angular coverage
    centroid = centers.mean(axis=0)
    vecs = centers - centroid
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_n = vecs / np.maximum(norms, 1e-10)
    dots = vecs_n @ vecs_n.T
    coverage = float(np.degrees(np.arccos(np.clip(np.min(dots), -1, 1))))

    # Trajectory arc length
    diffs = np.diff(centers, axis=0)
    arc = float(np.sum(np.linalg.norm(diffs, axis=1)))

    return {
        "median_gap_deg": float(np.median(gaps)) if gaps else -1,
        "mean_gap_deg": float(np.mean(gaps)) if gaps else -1,
        "p90_gap_deg": float(np.percentile(gaps, 90)) if gaps else -1,
        "total_coverage_deg": coverage,
        "trajectory_arc_length": arc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--seq", nargs="*", help="Run specific sequences only")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16

    print("Loading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    print("Model loaded.")

    sequences = args.seq if args.seq else ALL_SEQUENCES
    all_rows = []

    for seq_id in sequences:
        print(f"\n{'='*60}")
        print(f"Sequence: {seq_id}")
        print(f"{'='*60}")

        try:
            seq = find_sequence_json(seq_id)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        rgb_paths = seq["rgb_paths"]
        S = len(rgb_paths)
        ref_w2c = load_reference_poses(seq)
        has_ref = ref_w2c is not None and len(ref_w2c) >= 3

        for stride in STRIDES:
            # Compute valid start offsets for this stride
            max_start = S - stride * (N_VIEWS - 1) - 1
            if max_start < 0:
                print(f"  stride={stride}: SKIP (S={S} too short for {N_VIEWS} views)")
                continue

            starts = [int(f * max_start) for f in START_FRAC]
            # Deduplicate and ensure valid
            starts = sorted(set(s for s in starts if 0 <= s <= max_start))

            for start in starts:
                out_path = os.path.join(OUT_DIR,
                    f"{seq_id}_stride{stride:02d}_start{start:04d}.npz")

                # Compute frame indices
                idx = np.array([start + stride * i for i in range(N_VIEWS)])
                assert idx[-1] < S, f"idx out of range: {idx[-1]} >= {S}"

                if os.path.exists(out_path) and not args.force:
                    data = np.load(out_path)
                    vggt_ext = data["ext_w2c_vggt"]
                    frame_idx = data["frame_idx"]
                else:
                    rgb = [rgb_paths[i] for i in idx]
                    images = load_and_preprocess_images(rgb, mode="crop").to(device)
                    H, W = images.shape[-2:]
                    torch.manual_seed(42)
                    t0 = time.time()
                    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                        tokens, ps_idx = model.aggregator(images.unsqueeze(0))
                        pose_enc = model.camera_head(tokens)[-1]
                        ext_w2c, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))
                    dt = time.time() - t0
                    vggt_ext = ext_w2c.squeeze(0).float().cpu().numpy()
                    frame_idx = idx
                    np.savez_compressed(out_path,
                        ext_w2c_vggt=vggt_ext, intr_vggt=intr.squeeze(0).float().cpu().numpy(),
                        frame_idx=frame_idx, stride=stride, start=start)

                # Evaluate
                row = {
                    "sequence_id": seq_id,
                    "stride": stride,
                    "start": start,
                    "n_views": N_VIEWS,
                    "frame_indices": ",".join(str(int(i)) for i in frame_idx),
                }

                # View geometry
                geo = compute_view_geometry(frame_idx, ref_w2c)
                row.update(geo)

                # Pose evaluation
                if has_ref:
                    ref_sub = ref_w2c[frame_idx]
                    result = evaluate_pose(ref_sub, vggt_ext)
                    if result:
                        row.update(result)
                        print(f"  s={stride:2d} start={start:4d}: gap={geo.get('median_gap_deg',-1):5.1f}° "
                              f"cov={geo.get('total_coverage_deg',-1):5.1f}° "
                              f"rot={result['rot_median']:6.2f}° gate={result['pose_gate']}")
                    else:
                        row["pose_gate"] = "SKIP"
                else:
                    row["pose_gate"] = "NO_REF"
                    print(f"  s={stride:2d} start={start:4d}: gap={geo.get('median_gap_deg',-1):5.1f}° "
                          f"cov={geo.get('total_coverage_deg',-1):5.1f}° (no ref)")

                all_rows.append(row)

    # Save CSV
    if all_rows:
        csv_path = os.path.join(PHASE3C, "02_stride_experiments", "STRIDE_POSE_RESULTS.csv")
        fields = ["sequence_id", "stride", "start", "n_views",
                  "median_gap_deg", "mean_gap_deg", "p90_gap_deg",
                  "total_coverage_deg", "trajectory_arc_length",
                  "rot_median", "rot_p90", "rot_mean",
                  "center_median_norm", "center_p90_norm",
                  "trajectory_cosine", "neighbor_rpe_rot", "neighbor_rpe_trans",
                  "pose_gate", "frame_indices"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nSaved: {csv_path} ({len(all_rows)} rows)")

    # Summary: pass rate by stride
    print(f"\n{'='*60}")
    print("PASS RATE BY STRIDE (averaged over starts)")
    print(f"{'='*60}")
    for seq_id in sequences:
        seq_rows = [r for r in all_rows if r["sequence_id"] == seq_id and r.get("pose_gate") in ("PASS", "FAIL")]
        if not seq_rows:
            continue
        print(f"\n  {seq_id}:")
        for stride in STRIDES:
            s_rows = [r for r in seq_rows if r["stride"] == stride]
            if not s_rows:
                continue
            n_pass = sum(1 for r in s_rows if r["pose_gate"] == "PASS")
            gaps = [r["median_gap_deg"] for r in s_rows if r.get("median_gap_deg", -1) >= 0]
            gap_str = f"{np.mean(gaps):.1f}°" if gaps else "N/A"
            print(f"    stride={stride:2d}: {n_pass}/{len(s_rows)} PASS (gap≈{gap_str})")


if __name__ == "__main__":
    main()
