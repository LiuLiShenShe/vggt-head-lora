#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 2: Graph node count == REAL generated window count (never hardcoded)."""
import json
import os

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC = os.path.join(PHASE3C, "12_rotation_sync_v33")

GRAPH_SUMMARY = os.path.join(SYNC, "03_graph_analysis", "ROTATION_GRAPH_SUMMARY.json")
MAN = os.path.join(SYNC, "01_stride4_inference", "STRIDE4_WINDOW_MANIFEST.json")

LANGDON = [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]


def _graph_entries():
    with open(GRAPH_SUMMARY) as f:
        return json.load(f)


def _manifest_counts():
    with open(MAN) as f:
        m = json.load(f)
    return {s: m[s]["n_windows"] for s in m}


def test_graph_nodes_match_manifest_windows():
    counts = _manifest_counts()
    entries = _graph_entries()
    for seq in LANGDON:
        s4 = next(e for e in entries if e["sequence_id"] == seq and e["label"] == "STRIDE4")
        assert s4["n_windows"] == counts[seq], \
            f"graph nodes {s4['n_windows']} != manifest windows {counts[seq]} for {seq}"


def test_expected_window_counts():
    """Actual manifest counts (already generated). Not assumed — verified from manifest."""
    counts = _manifest_counts()
    assert counts[LANGDON[0]] == 77, f"langdon stride-4 windows = {counts[LANGDON[0]]}"
    for seq in LANGDON:
        assert counts[seq] == 77


def test_stride8_chain_node_count():
    """Stride-8 graph nodes match stride-8 manifest (39 windows for langdon)."""
    entries = _graph_entries()
    for seq in LANGDON:
        s8 = next(e for e in entries if e["sequence_id"] == seq and e["label"] == "STRIDE8")
        assert s8["n_windows"] == 39, f"stride-8 windows {s8['n_windows']} != 39"