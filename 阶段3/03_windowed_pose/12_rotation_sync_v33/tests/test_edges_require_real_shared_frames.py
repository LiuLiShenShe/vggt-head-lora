#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test 4: every rotation edge requires REAL shared frames (>= threshold) — verified
by recomputing window overlaps from disk, not trusting the CSV."""
import csv
import glob
import os
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SYNC = os.path.join(PHASE3C, "12_rotation_sync_v33")

EDGES_CSV = os.path.join(SYNC, "02_rotation_edges", "ROTATION_GRAPH_EDGES.csv")
WINDOW_DIR = os.path.join(PHASE3C, "03_window_inference", "window_outputs_stride4")

SEQ = "plantview__langdon_4__05-03-24"


def _window_frame_sets():
    files = sorted(glob.glob(os.path.join(WINDOW_DIR, SEQ, "window_*.npz")))
    sets = []
    for f in files:
        d = np.load(f)
        sets.append(set(int(x) for x in d["frame_idx"]))
    return sets


def _load_edges():
    es = []
    with open(EDGES_CSV) as f:
        for r in csv.DictReader(f):
            if r["sequence"] == SEQ:
                es.append(r)
    return es


def test_all_headline_edges_have_real_large_shared_frames():
    """Headline threshold: edges with n_overlap>=8 (the solver's stride4 th8 set)."""
    frame_sets = _window_frame_sets()
    edges = [r for r in _load_edges() if int(r["n_overlap"]) >= 8]
    assert len(edges) == 151, f"expected 151 headline edges, got {len(edges)}"
    for e in edges:
        i, j = int(e["window_i"]), int(e["window_j"])
        real_overlap = len(frame_sets[i] & frame_sets[j])
        assert real_overlap >= 8, \
            f"edge ({i},{j}) claims n_overlap={e['n_overlap']} but real overlap={real_overlap}"
        assert real_overlap == int(e["n_overlap"]), \
            f"edge ({i},{j}) CSV n_overlap={e['n_overlap']} != real {real_overlap}"


def test_superset_edges_all_have_real_shared_frames_matching_csv():
    """The CSV is a min_overlap=4 superset (incl. hop-3); EVERY edge's claimed
    n_overlap must match the recomputed real overlap (a diagnostic-only set)."""
    frame_sets = _window_frame_sets()
    edges = _load_edges()
    assert len(edges) == 225, f"expected 225 superset edges, got {len(edges)}"
    for e in edges:
        i, j = int(e["window_i"]), int(e["window_j"])
        real_overlap = len(frame_sets[i] & frame_sets[j])
        assert int(e["n_overlap"]) == real_overlap, \
            f"edge ({i},{j}) CSV n_overlap={e['n_overlap']} != real {real_overlap}"
        assert real_overlap >= 4, "superset edges must have >= 4 real shared frames"


def test_no_edge_without_shared_frames():
    """No edge exists with zero shared frames — every edge is grounded in real overlap."""
    frame_sets = _window_frame_sets()
    for e in _load_edges():
        i, j = int(e["window_i"]), int(e["window_j"])
        assert len(frame_sets[i] & frame_sets[j]) >= 1