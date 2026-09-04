#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 11: Controls (wheat 461/467, mustc pos00) must NOT regress — they pass the
orientation-only Pose Gate under ALL four methods."""
import csv
import os

ROOT = "/fj/VGGT+head+lora实验"
SYNC = os.path.join(ROOT, "阶段3", "03_windowed_pose", "12_rotation_sync_v33")
COMP = os.path.join(SYNC, "06_evaluation", "ROTATION_SYNC_METHOD_COMPARISON.csv")

CONTROLS = ["wheat3dgs__plot_461", "wheat3dgs__plot_467",
            "mustc__plot198__230613__ugv__pos00"]
ALL_METHODS = ["A", "B", "C", "D"]


def test_controls_pass_all_methods():
    with open(COMP) as f:
        rows = list(csv.DictReader(f))
    for seq in CONTROLS:
        for m in ALL_METHODS:
            r = next(x for x in rows if x["sequence"] == seq and x["method_label"] == m)
            assert r["pose_gate"] == "PASS", \
                f"control {seq} method {m} regressed → {r['pose_gate']}"


def test_controls_remain_accurate():
    """Control rot_median must stay small (≤ 6°) under method D."""
    with open(COMP) as f:
        rows = list(csv.DictReader(f))
    for seq in CONTROLS:
        r = next(x for x in rows if x["sequence"] == seq and x["method_label"] == "D")
        assert float(r["rot_median"]) <= 6.0, \
            f"control {seq} rot_median {r['rot_median']}° too high"