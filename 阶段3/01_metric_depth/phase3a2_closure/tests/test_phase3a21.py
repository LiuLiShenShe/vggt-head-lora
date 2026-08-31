#!/usr/bin/env python3
"""Tests for Phase 3A.2.1 — CalK compact indexing fix."""
import os, sys, json, csv
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
AUDIT_DIR = os.path.join(PHASE3_DIR, "phase3a2_closure")
sys.path.insert(0, PHASE3_DIR)
from configs import SEQUENCES, load_sequence_meta

SAMPLE_INTERVAL = 16
MAX_FRAMES = 20


def test_compact_index_mapping():
    """pilot_index i → original_index pilot_indices[i], NOT pred_stack[pilot_indices[i]]."""
    map_path = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_FRAME_MAP.json")
    assert os.path.exists(map_path), f"Missing: {map_path}"
    with open(map_path) as f:
        all_maps = json.load(f)

    for seq_id, _ in SEQUENCES:
        fm = all_maps[seq_id]
        pilot_indices = fm["pilot_indices"]
        mapping = fm["mapping"]
        assert len(mapping) == len(pilot_indices), f"{seq_id}: mapping len != pilot_indices len"
        for i, entry in enumerate(mapping):
            assert entry["pilot_index"] == i, f"pilot_index mismatch at {i}"
            assert entry["original_index"] == pilot_indices[i], \
                f"{seq_id}: mapping[{i}].original_index={entry['original_index']} != pilot_indices[{i}]={pilot_indices[i]}"
    print("  PASS: compact_index_mapping")


def test_all_20_frames_evaluated_per_sequence():
    """Each sequence must have exactly 20 evaluated frames."""
    csv_path = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_FRAME_METRICS_V321.csv")
    assert os.path.exists(csv_path), f"Missing: {csv_path}"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    per_seq = {}
    for row in rows:
        s = row["sequence"]
        per_seq[s] = per_seq.get(s, 0) + 1

    for seq_id, _ in SEQUENCES:
        count = per_seq.get(seq_id, 0)
        assert count == 20, f"{seq_id}: {count} frames evaluated (expected 20)"
    print("  PASS: all_20_frames_evaluated_per_sequence")


def test_total_80_frames():
    """Total evaluated frames must be 80."""
    comp_path = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_EVAL_COMPLETENESS.json")
    assert os.path.exists(comp_path), f"Missing: {comp_path}"
    with open(comp_path) as f:
        comp = json.load(f)
    assert comp["evaluated_total"] == 80, f"evaluated_total={comp['evaluated_total']} != 80"
    assert comp["status"] == "PASS", f"status={comp['status']} != PASS"
    print("  PASS: total_80_frames")


def test_frame_map_unique():
    """Both pilot_index and original_index must be unique per sequence."""
    map_path = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_FRAME_MAP.json")
    with open(map_path) as f:
        all_maps = json.load(f)

    for seq_id, _ in SEQUENCES:
        fm = all_maps[seq_id]
        pilot_idx = [e["pilot_index"] for e in fm["mapping"]]
        orig_idx = [e["original_index"] for e in fm["mapping"]]
        assert len(pilot_idx) == len(set(pilot_idx)), f"{seq_id}: duplicate pilot_index"
        assert len(orig_idx) == len(set(orig_idx)), f"{seq_id}: duplicate original_index"
    print("  PASS: frame_map_unique")


def test_matched_pilot_same_frames():
    """Autonomous and CalK must evaluate the exact same original frames."""
    csv_path = os.path.join(AUDIT_DIR, "UNIDEPTH_MATCHED_PILOT_COMPARISON.csv")
    assert os.path.exists(csv_path), f"Missing: {csv_path}"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 80, f"matched rows={len(rows)} != 80"

    per_seq = {}
    for row in rows:
        s = row["sequence"]
        per_seq[s] = per_seq.get(s, 0) + 1
    for seq_id, _ in SEQUENCES:
        short = seq_id.split("__")[-1]
        assert per_seq.get(short, per_seq.get(seq_id, 0)) == 20, \
            f"{seq_id}: matched rows={per_seq.get(seq_id, 0)} != 20"
    print("  PASS: matched_pilot_same_frames")


def test_report_frame_count():
    """Pose-PASS summary must say n_frames=60 for CalK."""
    pp_path = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_POSEPASS_SUMMARY_V321.json")
    assert os.path.exists(pp_path), f"Missing: {pp_path}"
    with open(pp_path) as f:
        pp = json.load(f)
    assert pp["n_frames"] == 60, f"pose_pass n_frames={pp['n_frames']} != 60"
    print("  PASS: report_frame_count (60 pose-PASS)")


def test_no_phase3a3_created():
    """No Phase 3A.3 directory should exist."""
    phase3a3 = os.path.join(PHASE3_DIR, "phase3a3_audit")
    assert not os.path.exists(phase3a3), f"Phase 3A.3 directory exists: {phase3a3}"
    print("  PASS: no_phase3a3_created")


def test_completeness_json_fields():
    """Completeness JSON must have required fields."""
    comp_path = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_EVAL_COMPLETENESS.json")
    with open(comp_path) as f:
        comp = json.load(f)
    required = ["expected_total", "prediction_total", "evaluated_total", "per_sequence", "status"]
    for field in required:
        assert field in comp, f"Missing field: {field}"
    assert comp["expected_total"] == 80
    assert comp["prediction_total"] == 80
    per_seq = comp["per_sequence"]
    assert len(per_seq) == 4, f"per_sequence has {len(per_seq)} entries (expected 4)"
    for short, count in per_seq.items():
        assert count == 20, f"{short}: {count} != 20"
    print("  PASS: completeness_json_fields")


def test_calK_not_6_frames():
    """CalK must NOT produce n_frames=6 (old bug)."""
    csv_path = os.path.join(AUDIT_DIR, "CORRECTED_COMPARISON_V321.csv")
    assert os.path.exists(csv_path), f"Missing: {csv_path}"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] == "unidepth_calK":
                n = int(row["n_frames"])
                assert n == 20, f"CalK {row['seq_id']}: n_frames={n} != 20 (old bug?)"
    print("  PASS: calK_not_6_frames")


if __name__ == "__main__":
    tests = [
        test_compact_index_mapping,
        test_all_20_frames_evaluated_per_sequence,
        test_total_80_frames,
        test_frame_map_unique,
        test_matched_pilot_same_frames,
        test_report_frame_count,
        test_no_phase3a3_created,
        test_completeness_json_fields,
        test_calK_not_6_frames,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        sys.exit(1)
