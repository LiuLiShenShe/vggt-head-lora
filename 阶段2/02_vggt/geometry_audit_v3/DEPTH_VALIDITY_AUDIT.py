"""DEPTH_VALIDITY_AUDIT — depth range distribution for all plant_view sequences.

Reads depth PNGs (uint16, scale 0.001 -> meters), computes valid-pixel percentiles
and bin proportions per frame, then aggregates per-sequence.

Output:
  - DEPTH_VALIDITY_AUDIT.json
  - DEPTH_FILE_MAPPING_AUDIT.csv
"""
from __future__ import annotations

import csv
import glob
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image

# ── configuration ────────────────────────────────────────────────────────────
SEQS = [
    "langdon_4__05-03-24", "langdon_4__12-03-24",
    "langdon_4__13-02-24", "langdon_4__20-02-24",
    "langdon_4__19-03-24",  # scanner GT seq
]

DEPTH_BASE = "/fj/VGGT+head+lora实验/阶段1-数据集/3D Plant View"
SEQ_JSON_BASE = "/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences/plant_view"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPTH_SCALE = 0.001  # from DEPTH_UNIT_AUDIT.json

# depth unit audit path (read to confirm scale, as depth_audit_v3.py does)
UNIT_AUDIT_PATH = os.path.join(OUT_DIR, "DEPTH_UNIT_AUDIT.json")

# percentiles to compute
PERCENTILES = [1, 5, 50, 95, 99]

# bin edges (meters) for valid pixel proportions
BIN_EDGES = [
    (0, 2, "0_2m"),
    (2, 3, "2_3m"),
    (3, 5, "3_5m"),
    (5, 10, "5_10m"),
    (10, 65, "10_65m"),
]


def load_depth_scale() -> float:
    """Load depth_scale_to_meter from DEPTH_UNIT_AUDIT.json, default 0.001."""
    if os.path.exists(UNIT_AUDIT_PATH):
        a = json.load(open(UNIT_AUDIT_PATH))
        if a.get("status") == "VERIFIED":
            return float(a["depth_scale_to_meter"])
    return 0.001


def load_seq_json(seq_id: str) -> dict:
    """Load the source sequence JSON (v31 runner reads from 01_sequences)."""
    path = os.path.join(SEQ_JSON_BASE, f"{seq_id}.json")
    return json.load(open(path))


def compute_frame_stats(depth_m: np.ndarray) -> dict:
    """Compute percentiles and bin proportions for valid pixels in a depth map.

    Args:
        depth_m: 2-D float64 array in meters (already scaled from uint16).
    Returns:
        dict with percentile values (float) and bin proportions (float),
        or None if too few valid pixels.
    """
    # valid mask: raw was uint16; after *0.001, 0 and ~65.535 are invalid
    # Follow depth_audit_v3.py convention: >0 and <65.0 m
    valid = (depth_m > 0) & (depth_m < 65.0)
    n_valid = int(valid.sum())
    if n_valid < 10:
        return None

    vals = depth_m[valid]

    stats = {"n_valid_pixels": n_valid}
    pct_vals = np.percentile(vals, PERCENTILES)
    for p, v in zip(PERCENTILES, pct_vals):
        stats[f"P{p}"] = float(v)
    stats["max"] = float(np.max(vals))

    # bin proportions
    for lo, hi, name in BIN_EDGES:
        count = int(np.sum((vals >= lo) & (vals < hi)))
        stats[f"bin_{name}"] = count / n_valid

    # pct_far_plane_gt_10m: proportion of valid pixels > 10m
    stats["pct_far_plane_gt_10m"] = float(np.sum(vals > 10.0) / n_valid)

    return stats


def main():
    depth_scale = load_depth_scale()
    print(f"DEPTH_SCALE_TO_METER = {depth_scale}")

    results = {"depth_scale_to_meter": depth_scale, "sequences": {}}
    csv_rows = []

    for seq_id in SEQS:
        print(f"\n{'='*60}")
        print(f"Processing: {seq_id}")

        # load source seq JSON
        try:
            seq = load_seq_json(seq_id)
        except FileNotFoundError:
            print(f"  WARNING: seq JSON not found for {seq_id}")
            continue

        extra = seq.get("extra", {})
        depth_dir = extra.get("depth_dir")
        rgb_paths = seq.get("rgb_paths", [])
        n_rgb = len(rgb_paths)

        # check depth dir
        depth_dir_exists = os.path.isdir(depth_dir) if depth_dir else False
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png"))) if depth_dir_exists else []
        n_depth = len(depth_files)

        print(f"  depth_dir: {depth_dir}")
        print(f"  depth_dir_exists: {depth_dir_exists}")
        print(f"  n_rgb_in_json: {n_rgb}, n_depth_files: {n_depth}")

        # check mapping status
        if depth_dir_exists and n_rgb > 0:
            rgb_basenames = set(os.path.splitext(os.path.basename(p))[0] for p in rgb_paths)
            depth_basenames = set(os.path.splitext(os.path.basename(p))[0] for p in depth_files)
            common = rgb_basenames & depth_basenames
            mapping_status = "full_match" if len(common) == n_rgb else f"partial_match_{len(common)}/{n_rgb}"
        else:
            mapping_status = "no_depth_dir"

        # sample filenames
        sample_depths = [os.path.basename(f) for f in depth_files[:5]]
        csv_rows.append({
            "sequence_id": seq_id,
            "date": seq_id.split("__")[-1] if "__" in seq_id else "",
            "depth_dir_exists": depth_dir_exists,
            "n_depth_files": n_depth,
            "n_rgb_frames_in_json": n_rgb,
            "mapping_status": mapping_status,
            "sample_filenames": ";".join(sample_depths),
        })

        if not depth_dir_exists or n_depth == 0:
            print(f"  SKIPPING: no depth files")
            continue

        # compute per-frame stats
        frame_stats_list = []
        for i, rgb_path in enumerate(rgb_paths):
            basename = os.path.splitext(os.path.basename(rgb_path))[0]
            ref_path = os.path.join(depth_dir, basename + ".png")
            if not os.path.exists(ref_path):
                continue

            raw = np.asarray(Image.open(ref_path))
            depth_m = raw.astype(np.float64) * depth_scale
            fs = compute_frame_stats(depth_m)
            if fs is not None:
                frame_stats_list.append(fs)

        n_frames = len(rgb_paths)
        n_valid_frames = len(frame_stats_list)

        if n_valid_frames == 0:
            print(f"  WARNING: no valid frames after processing")
            continue

        # aggregate percentiles
        p_keys = [f"P{p}" for p in PERCENTILES] + ["max"]
        agg_pcts = {}
        for k in p_keys:
            vals = [fs[k] for fs in frame_stats_list]
            agg_pcts[k] = float(np.mean(vals))

        # aggregate bin proportions
        bin_keys = [name for _, _, name in BIN_EDGES]
        agg_bins = {}
        for name in bin_keys:
            vals = [fs[f"bin_{name}"] for fs in frame_stats_list]
            agg_bins[name] = float(np.mean(vals))

        # aggregate pct_far_plane_gt_10m
        far_vals = [fs["pct_far_plane_gt_10m"] for fs in frame_stats_list]
        agg_far = float(np.mean(far_vals))

        # aggregate n_valid_pixels
        nvp_vals = [fs["n_valid_pixels"] for fs in frame_stats_list]
        agg_nvp = {
            "mean": float(np.mean(nvp_vals)),
            "min": int(np.min(nvp_vals)),
            "max": int(np.max(nvp_vals)),
        }

        seq_summary = {
            "n_frames": n_frames,
            "n_valid_frames": n_valid_frames,
            "depth_range_m": agg_pcts,
            "bin_proportions_mean": agg_bins,
            "pct_far_plane_gt_10m": agg_far,
            "n_valid_pixels_per_frame": agg_nvp,
        }
        results["sequences"][seq_id] = seq_summary

        # console output
        print(f"  n_frames={n_frames}, n_valid_frames={n_valid_frames}")
        print(f"  depth_range_m: {json.dumps(agg_pcts, indent=4)}")
        print(f"  bin_proportions_mean: {json.dumps(agg_bins, indent=4)}")
        print(f"  pct_far_plane_gt_10m: {agg_far:.4f}")

    # ── write outputs ────────────────────────────────────────────────────────
    json_path = os.path.join(OUT_DIR, "DEPTH_VALIDITY_AUDIT.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"Wrote: {json_path}")

    csv_path = os.path.join(OUT_DIR, "DEPTH_FILE_MAPPING_AUDIT.csv")
    fieldnames = [
        "sequence_id", "date", "depth_dir_exists", "n_depth_files",
        "n_rgb_frames_in_json", "mapping_status", "sample_filenames",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
