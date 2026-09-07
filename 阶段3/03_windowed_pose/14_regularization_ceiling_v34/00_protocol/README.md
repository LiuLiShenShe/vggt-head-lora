# 00_protocol — Phase 3C.4 Run Manifest

**Phase:** 3C.4 — Trajectory Regularization Ceiling Experiment
**Status:** IN PROGRESS
**Prerequisite:** Phase 3C.3.1 (drift analysis, coherence C=0.76, CASE C) + Phase 3C.3 SO(3) sync gauges
**Approved:** team-lead plan approval (plan_approval-1788775424160)

## Purpose

Measure the *regularization ceiling*: how much of the temporally-coherent drift
(161° over 76 hops, coherence C=0.76) can be suppressed by post-processing the
gauge trajectory — without GT entering the solver and without re-running VGGT.

Mathematical insight under test: naive smoothness cannot separate coherent drift
from true ~24°/step camera motion (both are smooth, near-constant-rate). The
experiment quantifies exactly how much each family of regularizers can (or cannot)
recover, on a synthetic testbed with known injected drift, then transfers the best
config to real data.

## Spec Compliance

| Stage | Requirement | Deliverable |
|---|---|---|
| Step 1 | Synthetic drift model matching real stats | `01_synthetic_drift_model/` |
| Step 2 | 4 regularization methods | `02_regularization_methods/` |
| Step 3 | Synthetic evaluation (ceiling table) | `03_synthetic_evaluation/SYNTHETIC_CEILING_RESULT.json` |
| Step 4 | Real-data application (7 sequences) | `04_real_data_application/REGULARIZATION_RESULT.json` |
| Step 5 | Decision framework (GREEN/YELLOW/RED) | `05_reports/PHASE3C4_REGULARIZATION_CEILING_REPORT.md` |
| Tests | 17 tests across 3 files | `tests/` |

## Constraints Observed

- ❌ No VGGT re-inference — reads existing `04_so3_sync/*_SO3_SYNC_GAUGES_stride4_th8.npz`
- ❌ No new long-range edges — hop-3 edges used only from existing `ROTATION_GRAPH_EDGES.csv` (th≥4 subset)
- ❌ No IMU / gravity / external anchor / LoRA / MSAM
- ✅ GT used only for synthetic test design + evaluation (never inside a method or the sync solve)
- ❌ Phase 3C.3 artifacts untouched (read-only imports from `12_rotation_sync_v33`)

## Data Provenance

- **Solver gauges (input):** `12_rotation_sync_v33/04_so3_sync/<seq>_SO3_SYNC_GAUGES_stride4_th8.npz` (G, anchor=0)
- **Edge topology + weights:** `12_rotation_sync_v33/02_rotation_edges/ROTATION_GRAPH_EDGES.csv`
- **GT gauges (evaluation only):** `13_q_bias_audit_v331/02_window_gauge_fit/CORRECTED_WINDOW_GAUGES_<seq>.npz`
- **Window outputs (global camera assembly):** `03_window_inference/window_outputs_stride4/<seq>/window_*.npz`
- **COLMAP reference (evaluation):** sequence JSONs via `evaluate_rotation_sync.load_reference_poses`

## Sequences

`plantview__langdon_4__{05-03-24,12-03-24,15-04-24,19-03-24}` (77 windows, drift 161-179°),
`wheat3dgs__plot_461/467` (6 windows, controls 3-5°), `mustc__plot198__230613__ugv__pos00` (2 windows, control ~0.6°).
