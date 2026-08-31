#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B tests: view set nesting, plant overlap, pose-only evaluation."""
import os, sys, json, csv
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")


def test_view_sets_are_nested():
    """24-view set ⊃ 16-view set ⊃ 8-view set."""
    manifest_path = os.path.join(PHASE3B, "00_protocol", "VIEW_SAMPLING_MANIFEST.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    for seq_id, vm in manifest.items():
        s24 = set(vm.get("24", []))
        s16 = set(vm.get("16", []))
        s8 = set(vm.get("8", []))

        assert s8.issubset(s16), f"{seq_id}: 8v not subset of 16v"
        assert s16.issubset(s24), f"{seq_id}: 16v not subset of 24v"
        assert len(s8) == 8, f"{seq_id}: 8v has {len(s8)} entries"
        assert len(s16) == 16, f"{seq_id}: 16v has {len(s16)} entries"
        assert len(s24) <= 24, f"{seq_id}: 24v has {len(s24)} entries"

    print("PASS: test_view_sets_are_nested")


def test_no_plant_overlap():
    """Each plant_id appears in exactly one dataset."""
    inv_path = os.path.join(PHASE3B, "01_dataset_inventory", "MULTIPLANT_POSE_DATASET_INVENTORY.csv")
    with open(inv_path) as f:
        rows = list(csv.DictReader(f))

    plant_datasets = {}
    for r in rows:
        pid = r["plant_id"]
        ds = r["dataset"]
        if pid not in plant_datasets:
            plant_datasets[pid] = set()
        plant_datasets[pid].add(ds)

    for pid, datasets in plant_datasets.items():
        assert len(datasets) == 1, f"{pid} appears in multiple datasets: {datasets}"

    print("PASS: test_no_plant_overlap")


def test_pose_results_gate_consistent():
    """PASS/FAIL gate matches failure_type."""
    results_path = os.path.join(PHASE3B, "03_pose_evaluation", "MULTIPLANT_POSE_RESULTS.csv")
    with open(results_path) as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        gate = r["pose_gate"]
        ft = r["failure_type"]
        rot_med = float(r["rot_median"])
        rot_p90 = float(r["rot_p90"])

        if gate == "PASS":
            assert ft == "PASS", f"{r['sequence_id']} vc={r['view_count']}: PASS gate but failure_type={ft}"
            assert rot_med <= 10.0, f"{r['sequence_id']}: PASS but rot_median={rot_med:.2f} > 10"
            assert rot_p90 <= 20.0, f"{r['sequence_id']}: PASS but rot_p90={rot_p90:.2f} > 20"
        else:
            assert ft != "PASS", f"{r['sequence_id']} vc={r['view_count']}: FAIL gate but failure_type=PASS"

    print("PASS: test_pose_results_gate_consistent")


def test_no_view_rescue():
    """No sequence is RESCUED_BY_MORE_VIEWS (fail at 8 but pass at 24)."""
    results_path = os.path.join(PHASE3B, "03_pose_evaluation", "MULTIPLANT_POSE_RESULTS.csv")
    with open(results_path) as f:
        rows = list(csv.DictReader(f))

    view_counts = [8, 16, 24]
    for seq_id in set(r["sequence_id"] for r in rows):
        seq_rows = {int(r["view_count"]): r["pose_gate"] for r in rows
                    if r["sequence_id"] == seq_id and int(r["view_count"]) in view_counts}

        if seq_rows.get(8) == "FAIL" and seq_rows.get(24) == "PASS":
            assert False, f"{seq_id}: rescued by more views (8=FAIL, 24=PASS)"

    print("PASS: test_no_view_rescue")


def test_canopy_characterization_complete():
    """All inventory sequences with masks have canopy characterization.
    MuST-C excluded: no masks available."""
    inv_path = os.path.join(PHASE3B, "01_dataset_inventory", "MULTIPLANT_POSE_DATASET_INVENTORY.csv")
    canopy_path = os.path.join(PHASE3B, "04_scene_characterization", "CANOPY_CHARACTERIZATION.csv")

    with open(inv_path) as f:
        inv = {r["sequence_id"]: r for r in csv.DictReader(f)}
    with open(canopy_path) as f:
        canopy = {r["sequence_id"]: r for r in csv.DictReader(f)}

    for seq_id, r in inv.items():
        if r["plant_mask_available"] != "True":
            continue  # MuST-C: no masks, skip
        assert seq_id in canopy, f"{seq_id}: missing canopy characterization"
        cf = float(canopy[seq_id]["canopy_fraction_mean"])
        assert 0 <= cf <= 1, f"{seq_id}: canopy_fraction_mean={cf} out of [0,1]"

    print("PASS: test_canopy_characterization_complete")


def test_per_frame_errors_match_full():
    """Per-frame rotation errors exist for all PASS sequences at full view count."""
    results_path = os.path.join(PHASE3B, "03_pose_evaluation", "MULTIPLANT_POSE_RESULTS.csv")
    frame_path = os.path.join(PHASE3B, "03_pose_evaluation", "PER_FRAME_ROT_ERRORS.csv")

    with open(results_path) as f:
        full_rows = [r for r in csv.DictReader(f) if int(r["view_count"]) > 24 and r["pose_gate"] == "PASS"]
    with open(frame_path) as f:
        frame_rows = list(csv.DictReader(f))

    frame_seqs = set(r["sequence_id"] for r in frame_rows)
    for r in full_rows:
        assert r["sequence_id"] in frame_seqs, \
            f"{r['sequence_id']}: PASS at full but missing per-frame errors"

    print("PASS: test_per_frame_errors_match_full")


if __name__ == "__main__":
    test_view_sets_are_nested()
    test_no_plant_overlap()
    test_pose_results_gate_consistent()
    test_no_view_rescue()
    test_canopy_characterization_complete()
    test_per_frame_errors_match_full()
    print("\n=== ALL 6 TESTS PASSED ===")
