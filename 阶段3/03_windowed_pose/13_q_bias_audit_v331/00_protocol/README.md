# 00_protocol — Phase 3C.3.1 Run Manifest

**Phase:** 3C.3.1 — Q-Bias Diagnostic Repair + Window Gauge-Equivariance Audit
**Status:** COMPLETED (2026-09-07)
**Prerequisite:** Phase 3C.3 commit 7e2dd3a (PROCRUSTES_BUG_UNFIXED)

## Spec Compliance

| Spec section | Requirement | Deliverable |
|---|---|---|
| §1 | Fix Procrustes formula | `01_procrustes_repair/correct_gauge_fit.py` |
| §2 | Corrected gauge fit on real data | `02_window_gauge_fit/corrected_window_gauge_fit.py` |
| §3-§4 | Synthetic exact/noisy Procrustes tests | `tests/test_reference_gauge_procrustes_exact.py` (7 tests) |
| §5 | Per-window gauge fit + Q-vs-ref | `02_window_gauge_fit/WINDOW_GAUGE_FIT_SUMMARY.json` |
| §6 | Corrected Q-vs-reference edges | `02_window_gauge_fit/CORRECTED_Q_VS_REF_*.csv` |
| §11-§14 | Gauge-invariant relative rotation | `03_relative_rotation/RELATIVE_ROTATION_GLOBAL_SUMMARY.json` |
| §15-§17 | Local position + central-owner + cross-window | `04_edge_truth_diagnostic/LOCAL_POSITION_SUMMARY.json` |
| §18-§23 | Graph gauge vs GT-gauge diagnostic | `05_graph_gauge_diagnostic/GRAPH_GAUGE_DIAGNOSTIC_SUMMARY.json` |
| §20-§23 | Drift trajectory + temporal coherence | `05_graph_gauge_diagnostic/DRIFT_COHERENCE_SUMMARY.json` |
| §28-§31 | Report + Q1–Q15 | `07_reports/PHASE3C31_Q_BIAS_AUDIT_REPORT.md` |
| §32-§34 | Test suite (21 tests) + commit | `tests/test_phase3c31_audit_pipeline.py` + `test_reference_gauge_procrustes_exact.py` |

## Key Results

- **Procrustes bug confirmed:** old formula → 85.6° error; corrected → 0° (synthetic)
- **Corrected Q-vs-ref hop1: 1.4–2.7°** (was claimed 25° with buggy formula)
- **Q-bias claim RETRACTED** — artifact of Procrustes implementation bug
- **Root cause: temporally-coherent per-edge drift** — coherence C=0.76, 7.6–13× over random walk
- **Graph sync ≈ chain (CASE C):** chain-vs-graph < 0.6° for all sequences

## Data Provenance

- **VGGT window outputs:** `03_window_inference/window_outputs_stride4/<seq>/window_*.npz`
- **COLMAP reference:** loaded via `evaluate_rotation_sync.load_reference_poses()`
- **Graph edges:** `12_rotation_sync_v33/02_rotation_edges/ROTATION_GRAPH_EDGES.csv`
- **Solver gauges:** `12_rotation_sync_v33/04_so3_sync/<seq>_SO3_SYNC_GAUGES_stride4_th8.npz`

## Constraints Observed

- ❌ No VGGT re-inference
- ❌ No new edges added
- ❌ No GT used in solver (GT only in `02_window_gauge_fit` gauge fitting for evaluation)
- ❌ Phase 3C.3 artifacts untouched (verified via `git status`)
