#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3A shared configuration — paths, sequences, constants."""
import os, json

# ── Root paths ──────────────────────────────────────────────────────────
ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
VGGT_RERUN = os.path.join(ROOT, "阶段2", "02_vggt", "v2_clean_rerun", "plant_view_3d")
SEQ_DIR = os.path.join(ROOT, "阶段2", "01_sequences", "sequences", "plant_view")
GEOMETRY_AUDIT = os.path.join(ROOT, "阶段2", "02_vggt", "geometry_audit_v3")
DEPTH_UNIT_AUDIT = os.path.join(GEOMETRY_AUDIT, "DEPTH_UNIT_AUDIT.json")

# ── Depth constants ─────────────────────────────────────────────────────
with open(DEPTH_UNIT_AUDIT) as _f:
    _dau = json.load(_f)
DEPTH_SCALE_TO_METER = _dau["depth_scale_to_meter"]  # 0.001
assert DEPTH_SCALE_TO_METER == 0.001

# ── Sequences ───────────────────────────────────────────────────────────
SEQUENCES = [
    ("plantview__langdon_4__05-03-24", False),   # pose PASS
    ("plantview__langdon_4__12-03-24", True),     # pose FAIL
    ("plantview__langdon_4__13-02-24", False),    # pose PASS
    ("plantview__langdon_4__20-02-24", False),    # pose PASS
]

MODELS = ["vggt", "da3", "unidepth"]

# ── Conda env Python paths ──────────────────────────────────────────────
DA3_PYTHON = "/home/test/miniconda3/envs/da3/bin/python"
UNIDEPTH_PYTHON = "/home/test/miniconda3/envs/unidepth/bin/python"

# ── Model HF repo IDs ──────────────────────────────────────────────────
DA3_HF_REPO = "depth-anything/DA3METRIC-LARGE"
UNIDEPTH_HF_REPO = "lpiccinelli/unidepth-v2-vitl14"


def load_sequence_meta(seq_id):
    """Load sequence JSON and return dict with paths.

    seq_id is the VGGT-style ID (e.g. 'plantview__langdon_4__05-03-24').
    JSON files are named without the 'plantview__' prefix.
    """
    # Strip 'plantview__' prefix for JSON lookup
    json_name = seq_id
    if json_name.startswith("plantview__"):
        json_name = json_name[len("plantview__"):]
    json_path = os.path.join(SEQ_DIR, f"{json_name}.json")
    if not os.path.exists(json_path):
        # Fallback: try with full seq_id
        json_path = os.path.join(SEQ_DIR, f"{seq_id}.json")
    with open(json_path) as f:
        d = json.load(f)
    extra = d.get("extra", {})
    rgb_dir = os.path.dirname(d["rgb_paths"][0]) if d.get("rgb_paths") else None
    return {
        "seq_id": seq_id,
        "rgb_paths": d.get("rgb_paths", []),
        "rgb_dir": rgb_dir,
        "depth_dir": extra.get("depth_dir"),
        "mask_dir": extra.get("mask_dir"),
        "intrinsics_path": d.get("intrinsics_path"),
        "extrinsics_path": d.get("extrinsics_path"),
        "n_frames": len(d.get("rgb_paths", [])),
    }


def get_depth_path(depth_dir, rgb_path):
    """Derive reference depth PNG path from RGB path."""
    basename = os.path.splitext(os.path.basename(rgb_path))[0]  # e.g. "0000_eval"
    return os.path.join(depth_dir, basename + ".png")


def get_mask_path(mask_dir, rgb_path):
    """Derive plant mask PNG path from RGB path (double .png extension)."""
    basename = os.path.basename(rgb_path)  # e.g. "0000_eval.png"
    return os.path.join(mask_dir, basename + ".png")
