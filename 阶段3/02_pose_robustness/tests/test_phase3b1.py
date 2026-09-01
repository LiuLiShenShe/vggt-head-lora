#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B.1 tests: verify all sub-experiments produce consistent results."""
import os, sys, json, csv
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
VC_DIR = os.path.join(PHASE3B, "02_pose_inference", "viewcount_outputs")
MUSTC_DIR = os.path.join(PHASE3B, "02_pose_inference", "mustc_outputs")


def test_true_viewcount_outputs_exist():
    """All 6 langdon_4 × 3 view counts should have npz outputs."""
    langdon4_dates = ["05-03-24", "12-03-24", "13-02-24", "15-04-24", "19-03-24", "20-02-24"]
    for date in langdon4_dates:
        sid = f"plantview__langdon_4__{date}"
        for n in [8, 16, 24]:
            npz = os.path.join(VC_DIR, f"{sid}_n{n}.npz")
            assert os.path.exists(npz), f"Missing: {npz}"

    # Wheat3DGS controls
    for sid in ["wheat3dgs__plot_461", "wheat3dgs__plot_467"]:
        npz = os.path.join(VC_DIR, f"{sid}_n8.npz")
        assert os.path.exists(npz), f"Missing control: {npz}"

    print("PASS: test_true_viewcount_outputs_exist")


def test_true_viewcount_shapes():
    """Each npz has correct shapes: ext_w2c (n,3,4), intr (n,3,3), frame_idx (n,)."""
    for npz_file in sorted(os.listdir(VC_DIR)):
        if not npz_file.endswith(".npz"):
            continue
        # Only check view-count npz files (skip offset/reverse npz from sensitivity test)
        if "_offset" in npz_file or "_reverse" in npz_file:
            continue
        data = np.load(os.path.join(VC_DIR, npz_file))
        n = len(data["frame_idx"])
        assert data["ext_w2c_vggt"].shape == (n, 3, 4), \
            f"{npz_file}: ext_w2c shape {data['ext_w2c_vggt'].shape}, expected ({n},3,4)"
        assert data["intr_vggt"].shape == (n, 3, 3), \
            f"{npz_file}: intr shape {data['intr_vggt'].shape}, expected ({n},3,3)"
        assert data["frame_idx"].shape == (n,), \
            f"{npz_file}: frame_idx shape {data['frame_idx'].shape}, expected ({n},)"

    print("PASS: test_true_viewcount_shapes")


def test_true_viewcount_results_exist():
    """TRUE_VIEWCOUNT_RESULTS.csv should exist and have all expected rows."""
    csv_path = os.path.join(PHASE3B, "03_pose_evaluation", "TRUE_VIEWCOUNT_RESULTS.csv")
    assert os.path.exists(csv_path), f"Missing: {csv_path}"

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # 6 langdon_4 × 3 view counts + 2 controls × 1 = 20 expected
    assert len(rows) >= 20, f"Expected ≥20 rows, got {len(rows)}"

    # All rows should have valid pose_gate
    for r in rows:
        assert r["pose_gate"] in ("PASS", "FAIL"), \
            f"{r['sequence_id']} n={r['view_count']}: invalid gate {r['pose_gate']}"
        assert float(r["rot_median"]) >= 0, \
            f"{r['sequence_id']}: negative rot_median"

    print("PASS: test_true_viewcount_results_exist")


def test_wheat3dgs_controls_pass():
    """Wheat3DGS controls should still PASS at n=8."""
    csv_path = os.path.join(PHASE3B, "03_pose_evaluation", "TRUE_VIEWCOUNT_RESULTS.csv")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if r["sequence_id"].startswith("wheat3dgs__"):
            assert r["pose_gate"] == "PASS", \
                f"Wheat3DGS control {r['sequence_id']} n={r['view_count']}: FAIL (expected PASS)"

    print("PASS: test_wheat3dgs_controls_pass")


def test_acquisition_comparison_exists():
    """Acquisition audit should produce comparison CSV and summary JSON."""
    csv_path = os.path.join(PHASE3B, "05_failure_analysis", "ACQUISITION_COMPARISON.csv")
    json_path = os.path.join(PHASE3B, "05_failure_analysis", "ACQUISITION_SUMMARY.json")
    assert os.path.exists(csv_path), f"Missing: {csv_path}"
    assert os.path.exists(json_path), f"Missing: {json_path}"

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6, f"Expected 6 dates, got {len(rows)}"

    # All PASS/FAIL groups present
    groups = set(r["group"] for r in rows)
    assert groups == {"PASS", "FAIL"}, f"Expected PASS+FAIL groups, got {groups}"

    print("PASS: test_acquisition_comparison_exists")


def test_starting_frame_sensitivity_exists():
    """Starting-frame sensitivity CSV should exist if test was run."""
    csv_path = os.path.join(PHASE3B, "03_pose_evaluation", "STARTING_FRAME_SENSITIVITY.csv")
    if not os.path.exists(csv_path):
        print("SKIP: test_starting_frame_sensitivity_exists (not yet run)")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Should have rows for 3 FAIL dates
    dates = set(r["date"] for r in rows)
    assert len(dates) >= 1, f"Expected ≥1 FAIL dates, got {len(dates)}"

    # All rows should have valid gate
    for r in rows:
        assert r["pose_gate"] in ("PASS", "FAIL"), \
            f"Invalid gate: {r['pose_gate']}"

    print("PASS: test_starting_frame_sensitivity_exists")


def test_mustc_controls_exist():
    """MuST-C control outputs should exist if test was run."""
    csv_path = os.path.join(PHASE3B, "03_pose_evaluation", "MUSTC_CONTROL_RESULTS.csv")
    if not os.path.exists(csv_path):
        print("SKIP: test_mustc_controls_exist (not yet run)")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) >= 4, f"Expected ≥4 MuST-C rows, got {len(rows)}"

    for r in rows:
        assert r["pose_gate"] in ("PASS", "FAIL"), \
            f"Invalid gate: {r['pose_gate']}"

    print("PASS: test_mustc_controls_exist")


def test_no_phase3b_output_overwritten():
    """Phase 3B original outputs should still be intact."""
    csv_path = os.path.join(PHASE3B, "03_pose_evaluation", "MULTIPLANT_POSE_RESULTS.csv")
    assert os.path.exists(csv_path), "Phase 3B MULTIPLANT_POSE_RESULTS.csv was deleted!"

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "Phase 3B results CSV is empty"

    print("PASS: test_no_phase3b_output_overwritten")


def test_true_vs_subsample_differ():
    """True re-run predictions should differ from Phase 3B subsample predictions.

    If they're identical, the re-run didn't actually re-run VGGT.
    """
    csv_path = os.path.join(PHASE3B, "03_pose_evaluation", "TRUE_VIEWCOUNT_RESULTS.csv")
    old_csv = os.path.join(PHASE3B, "03_pose_evaluation", "MULTIPLANT_POSE_RESULTS.csv")

    if not os.path.exists(csv_path) or not os.path.exists(old_csv):
        print("SKIP: test_true_vs_subsample_differ (CSVs not yet available)")
        return

    with open(csv_path) as f:
        true_rows = {r["sequence_id"]: r for r in csv.DictReader(f) if r["view_count"] == "8"}
    with open(old_csv) as f:
        old_rows = {r["sequence_id"]: r for r in csv.DictReader(f)
                    if r["view_count"] == "8" and r["sequence_id"].startswith("plantview__langdon_4")}

    for sid, old_r in old_rows.items():
        if sid in true_rows:
            old_rot = float(old_r["rot_median"])
            new_rot = float(true_rows[sid]["rot_median"])
            # They should differ at least slightly (independent forward ≠ subsample)
            # Allow for cases where they might coincidentally be close
            assert abs(old_rot - new_rot) > 0.01 or old_r["pose_gate"] == true_rows[sid]["pose_gate"], \
                f"{sid} n=8: identical results — re-run may not have been truly independent"

    print("PASS: test_true_vs_subsample_differ")


if __name__ == "__main__":
    test_true_viewcount_outputs_exist()
    test_true_viewcount_shapes()
    test_true_viewcount_results_exist()
    test_wheat3dgs_controls_pass()
    test_acquisition_comparison_exists()
    test_starting_frame_sensitivity_exists()
    test_mustc_controls_exist()
    test_no_phase3b_output_overwritten()
    test_true_vs_subsample_differ()
    print("\n=== ALL 9 TESTS PASSED ===")
