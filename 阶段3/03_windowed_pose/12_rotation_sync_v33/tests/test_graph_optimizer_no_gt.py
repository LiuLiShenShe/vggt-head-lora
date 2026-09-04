#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 8: The SO(3) graph optimizer does NOT use GT/COLMAP rotations as input.

The solver consumes, per edge, only:
  - window indices (i, j)          — from window frame overlaps
  - edge Q_ij (window-derived)     — R_c2w_i @ R_c2w_j^T from VGGT window outputs
  - edge weight w_ij               — n_overlap & dispersion, no GT
The COLMAP/reference extrinsics are never written into the edge CSV and never
read by the solver module.
"""
import csv
import json
import os
import sys
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC = os.path.join(PHASE3C, "12_rotation_sync_v33")

EDGES_CSV = os.path.join(SYNC, "02_rotation_edges", "ROTATION_GRAPH_EDGES.csv")
SOLVER_PY = os.path.join(SYNC, "04_so3_sync", "run_so3_sync.py")

LANGDON = [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]

# Reference data lives only in these files
REF_HINTS = ("extrinsics_path", "sequences/", "colmap", "_ref", "gt_", "reference")


def test_edge_csv_has_no_reference_columns():
    with open(EDGES_CSV) as f:
        rows = list(csv.DictReader(f))
    assert rows
    for r in rows:
        for k in r.keys():
            kk = k.lower()
            assert "ref" not in kk and "gt" not in kk and "colmap" not in kk and \
                "extrinsic" not in kk, f"edge CSV leaked reference data via column '{k}'"


def test_edge_csv_only_contains_vggt_window_derived_values():
    expected_cols = {
        "sequence", "window_i", "window_j", "window_distance", "n_overlap",
        "n_inliers", "Q_disp_med_deg", "Q_disp_p90_deg", "Q_disp_max_deg",
        "weight", "edge_status",
        "q00", "q01", "q02", "q10", "q11", "q12", "q20", "q21", "q22",
    }
    with open(EDGES_CSV) as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == expected_cols, set(rows[0].keys())


def test_solver_source_never_loads_reference():
    with open(SOLVER_PY) as f:
        src = f.read()
    for hint in ("extrinsics_path", "evaluate_multoplant", "reference", "COLMAP"):
        assert hint not in src, f"solver source references '{hint}' — must be GT-free"


def test_solver_npz_contains_no_reference_rotation():
    for seq in LANGDON:
        p = os.path.join(SYNC, "04_so3_sync", f"{seq}_SO3_SYNC_GAUGES_stride4_th8.npz")
        if not os.path.exists(p):
            continue
        with np.load(p, allow_pickle=True) as d:
            for k in d.files:
                kk = k.lower()
                assert "ref" not in kk and "gt" not in kk and "colmap" not in kk, \
                    f"solver npz leaked reference via key '{k}'"