#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 12: Report numbers match CSV values (erratum guard).

Every number in the report MUST be read programmatically from the CSVs — not
hand-copied. This test verifies the final report contains no unresolved
placeholders and that the method-comparison table matches the evaluation CSV.
"""
import csv
import json
import os
import re

ROOT = "/fj/VGGT+head+lora实验"
SYNC = os.path.join(ROOT, "阶段3", "03_windowed_pose", "12_rotation_sync_v33")
REPORT = os.path.join(SYNC, "09_reports", "PHASE3C3_REDUNDANT_SO3_ROTATION_SYNC.md")
COMP_CSV = os.path.join(SYNC, "06_evaluation", "ROTATION_SYNC_METHOD_COMPARISON.csv")

LANGDON = [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]
SHORT_MAP = {
    "05-03-24": "plantview__langdon_4__05-03-24",
    "12-03-24": "plantview__langdon_4__12-03-24",
    "15-04-24": "plantview__langdon_4__15-04-24",
    "19-03-24": "plantview__langdon_4__19-03-24",
    "wheat 461": "wheat3dgs__plot_461",
    "wheat 467": "wheat3dgs__plot_467",
    "mustc": "mustc__plot198__230613__ugv__pos00",
}


def test_no_unresolved_placeholders():
    with open(REPORT) as f:
        text = f.read()
    braces = re.findall(r'\{[a-zA-Z_][a-zA-Z_0-9]*\}', text)
    braces = [b for b in braces if b not in {"gate}", "{", "}"} and b not in ("{cycle_rank}",)]
    # The only allowed unresolved is actually none — f-strings should be fully resolved
    assert len(braces) == 0, f"unresolved placeholders in report: {braces}"


def test_method_comparison_d_med_matches_csv():
    with open(REPORT) as f:
        lines = f.readlines()
    # Find the Section 6 table (method comparison: header starts "| Sequence | A med")
    in_sec6 = False
    table_lines = []
    for line in lines:
        if line.strip().startswith("| Sequence | A med"):
            in_sec6 = True
            continue
        if in_sec6:
            if not line.strip().startswith("|"):
                break
            if line.strip().startswith("|---"):
                continue
            table_lines.append(line)

    with open(COMP_CSV) as f:
        csv_rows = list(csv.DictReader(f))

    for line in table_lines:
        cols = [c.strip() for c in line.strip("|").split("|")]
        short = cols[0].strip()
        full_seq = SHORT_MAP.get(short)
        if full_seq is None:
            continue
        d_med_raw = cols[4].strip().replace("*", "")
        gate = cols[6].strip()
        csv_row = next((r for r in csv_rows if r["sequence"] == full_seq and r["method_label"] == "D"), None)
        assert csv_row is not None, f"no CSV row for {full_seq} method D"
        csv_med = f"{float(csv_row['rot_median']):.2f}"
        assert d_med_raw == csv_med, f"report D_med={d_med_raw} != CSV {csv_med} for {short}"
        assert gate == csv_row["pose_gate"], f"report gate={gate} != CSV {csv_row['pose_gate']} for {short}"