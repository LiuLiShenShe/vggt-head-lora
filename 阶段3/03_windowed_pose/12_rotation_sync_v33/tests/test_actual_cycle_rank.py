#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 3: stride-4 graph has real cycles (cycle_rank>0); stride-8 is a pure chain (rank=0)."""
import json
import os

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC = os.path.join(PHASE3C, "12_rotation_sync_v33")

GRAPH_SUMMARY = os.path.join(SYNC, "03_graph_analysis", "ROTATION_GRAPH_SUMMARY.json")

LANGDON = [f"plantview__langdon_4__{d}" for d in ["05-03-24", "12-03-24", "15-04-24", "19-03-24"]]


def _entries():
    with open(GRAPH_SUMMARY) as f:
        return json.load(f)


def test_stride4_graph_has_cycles():
    entries = _entries()
    for seq in LANGDON:
        s4 = next(e for e in entries if e["sequence_id"] == seq and e["label"] == "STRIDE4")
        assert s4["cycle_rank"] > 0, f"stride-4 {seq} cycle_rank={s4['cycle_rank']} (must be >0)"
        assert s4["is_connected"] is True
        assert s4["acceptance_gate"] is True
        assert not s4["is_tree"]


def test_stride8_graph_is_pure_chain():
    entries = _entries()
    for seq in LANGDON:
        s8 = next(e for e in entries if e["sequence_id"] == seq and e["label"] == "STRIDE8")
        assert s8["cycle_rank"] == 0, f"stride-8 {seq} cycle_rank={s8['cycle_rank']} (must be 0)"
        assert s8["is_tree"] is True


def test_cycle_rank_matches_edges_minus_nodes_plus_components():
    entries = _entries()
    s4 = next(e for e in entries if e["sequence_id"] == LANGDON[0] and e["label"] == "STRIDE4")
    expected = s4["n_edges"] - s4["n_windows"] + s4["n_components"]
    assert s4["cycle_rank"] == max(0, expected)


def test_langdon_edges_expected_magnitude():
    entries = _entries()
    for seq in LANGDON:
        s4 = next(e for e in entries if e["sequence_id"] == seq and e["label"] == "STRIDE4")
        # stride-4 adjacent (12-frame) + hop2 (8-frame) edges → ~2V
        assert 2 * s4["n_windows"] - 5 <= s4["n_edges"] <= 2 * s4["n_windows"] + 5, \
            f"edge count {s4['n_edges']} for V={s4['n_windows']} unexpected"