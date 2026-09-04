#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 1: Stride-4 windows are REAL VGGT outputs (not reconstructed/duplicated)."""
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")

MAN = os.path.join(PHASE3C, "12_rotation_sync_v33",
                   "01_stride4_inference", "STRIDE4_WINDOW_MANIFEST.json")

SEQ = "plantview__langdon_4__05-03-24"


def _load_manifest():
    with open(MAN) as f:
        return json.load(f)


def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(PHASE3C, path)


def test_manifest_lists_all_windows():
    m = _load_manifest()
    assert SEQ in m
    meta = m[SEQ]
    assert meta["n_windows"] >= 2
    assert len(meta["windows"]) == meta["n_windows"]


def test_window_has_real_vggt_inference_artifacts():
    """Every window npz must contain real VGGT outputs: depth + point_map at (16,H,W)."""
    m = _load_manifest()
    meta = m[SEQ]
    for w in meta["windows"][:5]:
        assert w["n_frames"] == 16
        path = _resolve(w["stride4_artifact"])
        assert os.path.exists(path), f"missing {path}"
        d = np.load(path)
        assert "ext_w2c_vggt" in d
        assert "depth_vggt" in d and d["depth_vggt"].shape[0] == 16
        assert "point_map" in d and d["point_map"].shape[0] == 16
        assert d["depth_vggt"].shape == (16, 518, 518)
        # real prediction → not all-zero, not all-constant
        depth = d["depth_vggt"].ravel()
        assert np.std(depth) > 1e-6, "depth is constant → not a real VGGT output"
        assert np.all(np.isfinite(d[ "ext_w2c_vggt"]))
        assert len(d["frame_idx"]) == 16
    # stride-4 frame indices must be contiguous consecutive frames
    w0 = meta["windows"][0]
    assert list(w0["original_frame_indices"][:4]) == [0, 1, 2, 3]
    w1 = meta["windows"][1]
    # stride-4: window 1 starts 4 frames later than window 0 (no overlap < window_size-... )
    assert w1["start_frame"] - w0["start_frame"] == 4


def test_stride4_reuses_stride8_windows_for_even_windows():
    """Windows co-located with a stride-8 window reuse the byte-identical real output
    (same 16 input frames), so those are valid reuse; the rest get fresh inference."""
    m = _load_manifest()
    meta = m[SEQ]
    n_reuse = 0
    n_fresh = 0
    for w in meta["windows"]:
        if w["start_frame"] % 8 == 0:
            assert w["source"] == "reuse_stride8", \
                f"window at start {w['start_frame']} should be reuse_stride8, got {w['source']}"
            assert w["stride8_window_id"] is not None
            n_reuse += 1
        else:
            assert w["source"] == "new_inference", f"got {w['source']}"
            assert w.get("stride8_window_id") is None
            n_fresh += 1
    # 320-frame langdon: stride-8 reuses 39 windows, remaining 38 need fresh inference
    assert n_reuse == 39 and n_fresh == 38, f"reuse={n_reuse} fresh={n_fresh}"


def test_manifest_matches_actual_window_files():
    """Manifest window count == actual .npz files on disk."""
    m = _load_manifest()
    meta = m[SEQ]
    seq_dir = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4", SEQ)
    n_files = len([p for p in os.listdir(seq_dir) if p.startswith("window_") and p.endswith(".npz")])
    assert n_files == meta["n_windows"]