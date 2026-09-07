#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 13 (new): PHASE3C3_RUN_MANIFEST.json is complete per spec §51.

Verifies:
  - manifest exists
  - 7 sequences present (4 langdon + wheat461/467 + mustc)
  - window_size=16, stride=4, edge_threshold=8
  - n_inference_windows matches STRIDE4_WINDOW_MANIFEST.json
  - optimizer_settings has f_scale and anchor
  - git_commit is non-empty
"""
import json
import os

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC = os.path.join(PHASE3C, "12_rotation_sync_v33")

MANIFEST = os.path.join(SYNC, "PHASE3C3_RUN_MANIFEST.json")
WINDOW_MANIFEST = os.path.join(SYNC, "01_stride4_inference", "STRIDE4_WINDOW_MANIFEST.json")

EXPECTED_SEQS = [
    "plantview__langdon_4__05-03-24",
    "plantview__langdon_4__12-03-24",
    "plantview__langdon_4__15-04-24",
    "plantview__langdon_4__19-03-24",
    "wheat3dgs__plot_461",
    "wheat3dgs__plot_467",
    "mustc__plot198__230613__ugv__pos00",
]


def _load():
    with open(MANIFEST) as f:
        return json.load(f)


def _load_window_manifest():
    with open(WINDOW_MANIFEST) as f:
        return json.load(f)


def test_manifest_file_exists():
    assert os.path.exists(MANIFEST), f"PHASE3C3_RUN_MANIFEST.json missing: {MANIFEST}"


def test_all_7_sequences_present():
    m = _load()
    seqs = m.get("sequences", {})
    for s in EXPECTED_SEQS:
        assert s in seqs, f"sequence {s} missing from manifest"
    assert m.get("n_total_sequences") == 7


def test_window_size_stride_edge_threshold():
    m = _load()
    assert m["window_size"] == 16
    assert m["stride"] == 4
    assert m["edge_threshold_min_overlap_frames"] == 8


def test_n_inference_windows_matches_manifest():
    """manifest n_windows must equal the window inference manifest counts."""
    m = _load()
    wm = _load_window_manifest()
    for seq in EXPECTED_SEQS:
        expected = wm[seq]["n_windows"]
        actual = m["sequences"][seq]["n_windows"]
        assert actual == expected, (
            f"{seq}: manifest says {actual} windows but window manifest says {expected}"
        )


def test_optimizer_settings_complete():
    m = _load()
    opt = m.get("optimizer_settings", {})
    assert "huber_f_scale_rad" in opt, "missing huber_f_scale_rad"
    assert "anchor_window" in opt, "missing anchor_window"
    assert opt["huber_f_scale_rad"] == 0.05
    assert opt["anchor_window"] == 0
    assert opt.get("gt_used_in_solver") is False


def test_git_commit_nonempty():
    m = _load()
    gc = m.get("git_commit", "")
    assert gc and gc != "unknown", f"git_commit is empty or unknown: {gc!r}"


def test_artifact_hashes_present():
    """Each langdon sequence must have SHA256 hashes for its npz files."""
    m = _load()
    for seq in EXPECTED_SEQS[:4]:  # langdon only
        info = m["sequences"][seq]
        assert info["artifact_count"] >= 10, (
            f"{seq}: only {info['artifact_count']} hashes — expected ~77"
        )
        assert len(info["artifact_hashes"]) == info["artifact_count"]
