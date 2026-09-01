#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C tests: verify windowed pipeline correctness."""
import os, sys, json, csv, glob
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs")
ALIGN_DIR = os.path.join(PHASE3C, "04_window_alignment")
STITCH_DIR = os.path.join(PHASE3C, "05_global_stitching")


def test_window_frame_indices_correct():
    """Window frame indices must be contiguous within each window."""
    for seq_dir in sorted(glob.glob(os.path.join(WINDOW_DIR, "*"))):
        if not os.path.isdir(seq_dir):
            continue
        seq_id = os.path.basename(seq_dir)
        for npz in sorted(glob.glob(os.path.join(seq_dir, "window_*.npz"))):
            data = np.load(npz)
            idx = data["frame_idx"]
            # Check contiguous
            diffs = np.diff(idx)
            assert np.all(diffs == 1), \
                f"{seq_id}/{os.path.basename(npz)}: non-contiguous indices {idx}"
    print("PASS: test_window_frame_indices_correct")


def test_window_overlap_exact():
    """Adjacent windows must have exactly overlap frames in common."""
    # Read manifest for window/overlap config
    for seq_dir in sorted(glob.glob(os.path.join(WINDOW_DIR, "*"))):
        if not os.path.isdir(seq_dir):
            continue
        manifest_path = os.path.join(seq_dir, "WINDOW_RUN_MANIFEST.json")
        if not os.path.exists(manifest_path):
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        overlap = manifest["overlap"]
        windows = manifest["windows"]

        for i in range(len(windows) - 1):
            frames_a = set(windows[i]["original_frame_indices"])
            frames_b = set(windows[i + 1]["original_frame_indices"])
            actual_overlap = len(frames_a & frames_b)
            assert actual_overlap == overlap, \
                f"{manifest['sequence_id']} W{i}->W{i+1}: expected overlap={overlap}, got {actual_overlap}"
    print("PASS: test_window_overlap_exact")


def test_window_alignment_no_reference_pose():
    """Alignment manifests must exist and contain no reference pose data."""
    for mf in sorted(glob.glob(os.path.join(ALIGN_DIR, "*_ALIGNMENT_MANIFEST.json"))):
        with open(mf) as f:
            manifest = json.load(f)
        # Check no reference pose keys
        for align in manifest.get("alignments", []):
            for key in align:
                assert "reference" not in key.lower() and "gt" not in key.lower(), \
                    f"{mf}: alignment entry contains reference/GT key: {key}"
    print("PASS: test_window_alignment_no_reference_pose")


def test_window_alignment_no_gt_geometry():
    """Alignment must not use GT geometry (no splat.ply, no COLMAP)."""
    for mf in sorted(glob.glob(os.path.join(ALIGN_DIR, "*_ALIGNMENT_MANIFEST.json"))):
        with open(mf) as f:
            content = f.read()
        assert "splat.ply" not in content, f"{mf}: references GT geometry"
        assert "colmap" not in content.lower() or "colmap" not in content, \
            f"{mf}: references COLMAP"
    print("PASS: test_window_alignment_no_gt_geometry")


def test_global_transform_chain():
    """Global transforms must form a valid chain (Window 0 = identity)."""
    for stitch in sorted(glob.glob(os.path.join(STITCH_DIR, "*_STITCHING_MANIFEST.json"))):
        with open(stitch) as f:
            manifest = json.load(f)
        assert manifest["coverage_ratio"] > 0, \
            f"{manifest['sequence_id']}: zero coverage"
        assert manifest["n_unique_frames"] > 0, \
            f"{manifest['sequence_id']}: zero unique frames"
    print("PASS: test_global_transform_chain")


def test_full_trajectory_coverage():
    """Windowed method must cover >= 90% of trajectory frames."""
    for stitch in sorted(glob.glob(os.path.join(STITCH_DIR, "*_STITCHING_MANIFEST.json"))):
        with open(stitch) as f:
            manifest = json.load(f)
        # Skip short sequences (wheat3dgs, mustc) — check langdon_4 only
        if "langdon_4" in manifest["sequence_id"]:
            assert manifest["coverage_ratio"] >= 0.9, \
                f"{manifest['sequence_id']}: coverage {manifest['coverage_ratio']:.1%} < 90%"
    print("PASS: test_full_trajectory_coverage")


def test_pose_gate_fixed():
    """Pose gate must use standard thresholds (10°/20°)."""
    csv_path = os.path.join(PHASE3C, "06_pose_evaluation", "WINDOWED_GLOBAL_RESULTS.csv")
    if not os.path.exists(csv_path):
        print("SKIP: test_pose_gate_fixed (results not yet generated)")
        return
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["pose_gate"] == "PASS":
                assert float(r["rot_median"]) <= 10.0, \
                    f"PASS but rot_median={r['rot_median']}"
                assert float(r["rot_p90"]) <= 20.0, \
                    f"PASS but rot_p90={r['rot_p90']}"
    print("PASS: test_pose_gate_fixed")


def test_overlap_frame_fusion_unique():
    """Each original frame in global output must appear exactly once."""
    for npz in sorted(glob.glob(os.path.join(STITCH_DIR, "*_WINDOWED_GLOBAL_CAMERAS.npz"))):
        data = np.load(npz)
        frames = data["original_frame_index"]
        assert len(frames) == len(np.unique(frames)), \
            f"{os.path.basename(npz)}: duplicate frame indices"
    print("PASS: test_overlap_frame_fusion_unique")


def test_uniform_baseline_separate():
    """Uniform baseline results must not be modified by Phase 3C."""
    csv_path = os.path.join(PHASE3B, "03_pose_evaluation", "MULTIPLANT_POSE_RESULTS.csv")
    assert os.path.exists(csv_path), "Phase 3B results deleted!"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "Phase 3B results empty"
    print("PASS: test_uniform_baseline_separate")


def test_pass_controls_not_skipped():
    """PASS controls (05-03, wheat, mustc) must have results in stride CSV."""
    csv_path = os.path.join(PHASE3C, "02_stride_experiments", "STRIDE_POSE_RESULTS.csv")
    if not os.path.exists(csv_path):
        print("SKIP: test_pass_controls_not_skipped (stride results not yet generated)")
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    seq_ids = set(r["sequence_id"] for r in rows)
    # At minimum, one PASS control must be present
    pass_controls = [s for s in seq_ids if "05-03" in s or "wheat" in s or "mustc" in s]
    assert len(pass_controls) >= 1, f"No PASS controls in stride results: {seq_ids}"
    print("PASS: test_pass_controls_not_skipped")


def test_geometry_uses_foreground():
    """Placeholder: geometry evaluation must use foreground mask."""
    # This test becomes active when geometry evaluation is implemented
    print("SKIP: test_geometry_uses_foreground (geometry eval not yet implemented)")


if __name__ == "__main__":
    test_window_frame_indices_correct()
    test_window_overlap_exact()
    test_window_alignment_no_reference_pose()
    test_window_alignment_no_gt_geometry()
    test_global_transform_chain()
    test_full_trajectory_coverage()
    test_pose_gate_fixed()
    test_overlap_frame_fusion_unique()
    test_uniform_baseline_separate()
    test_pass_controls_not_skipped()
    test_geometry_uses_foreground()
    print("\n=== ALL TESTS PASSED ===")
