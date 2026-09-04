#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.3 Step 6: Orientation-only evaluation of the 4 global-rotation methods.

For each sequence × method, load the global camera rotations produced by
05_global_stitching and evaluate against COLMAP reference rotations ONLY
(single global rotation Procrustes; no centers; GUARANTEED no GT in solver).

Methods:
  A. STRIDE8_SEQUENTIAL_CHAIN  — stride-8 windows, sequential Q chain (baseline ~44°)
  B. STRIDE8_SO3_GRAPH         — stride-8 windows, SO(3) sync (negative control, cr=0)
  C. STRIDE4_SEQUENTIAL_CHAIN  — stride-4 windows, sequential Q chain
  D. STRIDE4_SO3_GRAPH         — stride-4 windows, SO(3) sync (headline, redundant cycles)

Gate: rot_median <= 10° AND rot_p90 <= 20° → PASS (orientation-only Pose Gate).

All reference rotations are used ONLY here for evaluation, never in the solver
(solver input = window VGGT npz + shared-frame indices + edge Q's only).

Outputs (in 06_evaluation/):
  ROTATION_SYNC_METHOD_COMPARISON.csv   — per sequence × method metrics
  ROTATION_SYNC_PER_FRAME_ERRORS.csv    — per-frame rot error (for visualization)

Usage:
    python 06_evaluation/evaluate_rotation_sync.py [--seq ...]
"""
import argparse, csv, glob, json, os, sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SEQ_BASE = os.path.join(ROOT, "阶段2", "01_sequences", "sequences")
SYNC_DIR = os.path.join(PHASE3C, "12_rotation_sync_v33")
GLOBAL_DIR = os.path.join(SYNC_DIR, "05_global_stitching")
OUT_DIR = os.path.join(SYNC_DIR, "06_evaluation")
GRAPH_SUMMARY_PATH = os.path.join(SYNC_DIR, "03_graph_analysis", "ROTATION_GRAPH_SUMMARY.json")
os.makedirs(OUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, "阶段3", "02_pose_robustness", "03_pose_evaluation"))
from evaluate_multoplant import global_rotation_procrustes, rot_angle_deg

SEQUENCES = (
    [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
    + ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]
    + ["mustc__plot198__230613__ugv__pos00"]
)

# method tag -> (graph label in summary json, note)
METHODS = {
    "A_STRIDE8_CHAIN":     {"label": "STRIDE8", "note": "baseline chain"},
    "B_STRIDE8_SO3GRAPH":  {"label": "STRIDE8", "note": "negative control cr=0"},
    "C_STRIDE4_CHAIN":     {"label": "STRIDE4", "note": "chain more windows"},
    "D_STRIDE4_SO3GRAPH":  {"label": "STRIDE4", "note": "headline redundant sync"},
}

POSE_GATE_MEDIAN_DEG = 10.0
POSE_GATE_P90_DEG = 20.0


def find_sequence_json(seq_id):
    for subdir in ["plant_view", "wheat3dgs", "mustc"]:
        for jp in glob.glob(os.path.join(SEQ_BASE, subdir, "*.json")):
            with open(jp) as f:
                meta = json.load(f)
            if meta.get("sequence_id") == seq_id:
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


def load_graph_summary():
    """Load graph summary json → per (seq_id, label) graph metrics."""
    if not os.path.exists(GRAPH_SUMMARY_PATH):
        return {}
    with open(GRAPH_SUMMARY_PATH) as f:
        entries = json.load(f)
    out = {}
    for e in entries:
        out[(e["sequence_id"], e["label"])] = e
    return out


def evaluate_orientation_only(R_pred_c2w, ref_w2c_sub):
    """Orientation-only metrics: single global rotation Procrustes, no centers."""
    n = min(len(R_pred_c2w), len(ref_w2c_sub))
    R_ref_c2w = ref_w2c_sub[:n, :3, :3].transpose(0, 2, 1)

    Rg = global_rotation_procrustes(R_pred_c2w[:n], R_ref_c2w)

    rot_errors = np.array([
        rot_angle_deg((Rg @ R_pred_c2w[i]).T @ R_ref_c2w[i]) for i in range(n)
    ])

    gate = float(np.median(rot_errors)) <= POSE_GATE_MEDIAN_DEG and \
        float(np.percentile(rot_errors, 90)) <= POSE_GATE_P90_DEG

    return {
        "n_frames": n,
        "rot_median": float(np.median(rot_errors)),
        "rot_p90": float(np.percentile(rot_errors, 90)),
        "rot_mean": float(np.mean(rot_errors)),
        "rot_max": float(np.max(rot_errors)),
        "rot_errors": rot_errors,
        "pose_gate": "PASS" if gate else "FAIL",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", nargs="*")
    args = ap.parse_args()
    sequences = args.seq if args.seq else list(SEQUENCES)

    graph_summary = load_graph_summary()
    all_rows = []
    all_frame_rows = []

    files = sorted(glob.glob(os.path.join(GLOBAL_DIR, "*_METHOD_*_GLOBAL_CAMERAS.npz")))
    for npz_path in files:
        fname = os.path.basename(npz_path)
        # Parse sequence id + method tag from filename
        if "_METHOD_" not in fname:
            continue
        prefix = fname
        seq_id = None
        method_tag = None
        for cand_seq in sequences:
            if prefix.startswith(cand_seq + "_METHOD_"):
                seq_id = cand_seq
                method_tag = prefix[len(cand_seq) + len("_METHOD_"):-len("_GLOBAL_CAMERAS.npz")]
                break
        if seq_id is None or method_tag not in METHODS:
            continue
        if args.seq and seq_id not in sequences:
            continue

        print(f"\n=== {fname} ===")

        # Load predicted global rotations
        d = np.load(npz_path)
        R_pred = d["R_c2w_global"]
        idx = d["original_frame_index"]

        # Load reference (from sequence JSON)
        try:
            seq = find_sequence_json(seq_id)
        except FileNotFoundError:
            print(f"  SKIP: no sequence JSON")
            continue
        ref_w2c = load_reference_poses(seq)
        if ref_w2c is None:
            print(f"  SKIP: no reference poses")
            continue

        idx_int = np.asarray(idx, dtype=int)
        ref_sub = ref_w2c[idx_int]

        result = evaluate_orientation_only(R_pred, ref_sub)

        info = METHODS[method_tag]
        ginfo = graph_summary.get((seq_id, info["label"]), {})
        n_edges = ginfo.get("n_edges")
        cycle_rank = ginfo.get("cycle_rank")

        row = {
            "sequence": seq_id,
            "method": method_tag,
            "method_label": {"A_STRIDE8_CHAIN": "A", "B_STRIDE8_SO3GRAPH": "B",
                             "C_STRIDE4_CHAIN": "C", "D_STRIDE4_SO3GRAPH": "D"}[method_tag],
            "note": info["note"],
            "n_windows": len(idx),
            "n_frames_evaluated": result["n_frames"],
            "n_edges": n_edges if n_edges is not None else "",
            "cycle_rank": cycle_rank if cycle_rank is not None else "",
            "rot_median": result["rot_median"],
            "rot_p90": result["rot_p90"],
            "rot_mean": result["rot_mean"],
            "rot_max": result["rot_max"],
            "pose_gate": result["pose_gate"],
        }
        all_rows.append(row)
        print(f"  n_frames={result['n_frames']} rot_med={result['rot_median']:.2f}° "
              f"p90={result['rot_p90']:.2f}° gate={result['pose_gate']} "
              f"(method {row['method_label']}, {info['note']})")

        for k, f_idx in enumerate(idx_int):
            all_frame_rows.append({
                "sequence": seq_id,
                "method": method_tag,
                "frame_idx": int(f_idx),
                "rot_error_deg": float(result["rot_errors"][k]),
            })

    if all_rows:
        csv_path = os.path.join(OUT_DIR, "ROTATION_SYNC_METHOD_COMPARISON.csv")
        fields = ["sequence", "method", "method_label", "note", "n_windows",
                  "n_frames_evaluated", "n_edges", "cycle_rank",
                  "rot_median", "rot_p90", "rot_mean", "rot_max", "pose_gate"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nSaved: {csv_path} ({len(all_rows)} rows)")

    if all_frame_rows:
        csv_path = os.path.join(OUT_DIR, "ROTATION_SYNC_PER_FRAME_ERRORS.csv")
        fields = ["sequence", "method", "frame_idx", "rot_error_deg"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_frame_rows)
        print(f"Saved: {csv_path} ({len(all_frame_rows)} rows)")

    # Compact summary table
    print(f"\n{'='*110}")
    print("ROTATION SYNC METHOD COMPARISON  (orientation-only, COLMAP reference)")
    print(f"{'='*110}")
    print(f"{'Sequence':<38s} {'A':>17s} {'B':>17s} {'C':>17s} {'D':>17s}")
    print("-" * 110)
    for seq_id in sequences:
        seq_rows = [r for r in all_rows if r["sequence"] == seq_id]
        cells = {r["method_label"]: r for r in seq_rows}
        line = f"  {seq_id:<36s}"
        for label in ["A", "B", "C", "D"]:
            r = cells.get(label)
            if r is None:
                line += f" {'—':^15s}"
            else:
                line += f" {r['rot_median']:5.1f}°{'+'if r['pose_gate']=='PASS' else '-':>2s}"
        print(line)


if __name__ == "__main__":
    main()