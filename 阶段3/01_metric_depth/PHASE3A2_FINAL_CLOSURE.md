# Phase 3A.2.1 — Final Metric-Depth Closure (v3.2.1)

**Date**: 2026-08-31
**Status**: FROZEN — no further Phase 3A.x audits
**Version**: 3A.2.1 (corrects compact indexing bug from 3A.2)

---

## 1. Root Cause of Previous Error

The CalK pilot inference saved a **compact** prediction stack of shape `(20, 1080, 1080)` per sequence, where:
- `pred_stack[0]` = frame 0 (original index 0)
- `pred_stack[1]` = frame 16 (original index 16)
- `pred_stack[2]` = frame 32 (original index 32)
- ...
- `pred_stack[19]` = frame 304 (original index 304)

The old evaluator iterated `for idx in pilot_indices` (i.e. `[0, 16, 32, ...]`) and accessed `pred_stack[idx]`. This meant:
- `pred_stack[0]` → correct (original frame 0)
- `pred_stack[16]` → **wrong** — read pilot entry 16 (original frame 256), not original frame 16
- `pred_stack[32]` → **IndexError** (only 20 entries)

**Result**: Only 2 frames/sequence were actually evaluated. The old "AbsRel=0.640 / RMSE=0.637" was computed on corrupted data.

**Fix**: `pred_stack[local_idx]` paired with `ref_depths[original_idx]` and `fg_masks[original_idx]`, where `local_idx = pilot_indices.index(original_idx)`.

---

## 2. Corrected CalK Evaluation (80 frames)

### Per-Sequence (all 4, 20 frames each)

| Sequence | AbsRel | RMSE | Scale | δ1 | N |
|----------|--------|------|-------|-----|---|
| 05-03-24 | 0.5381 | 0.6260 | 0.7678 | 0.147 | 20 |
| 12-03-24 | 0.5299 | 0.7321 | 1.3915 | 0.163 | 20 |
| 13-02-24 | 0.3822 | 0.4374 | 1.1688 | 0.210 | 20 |
| 20-02-24 | 0.4764 | 0.5769 | 0.8715 | 0.161 | 20 |

### Pose-PASS Summary (60 frames, 3 sequences)

| Metric | Value |
|--------|-------|
| AbsRel mean | 0.4656 |
| RMSE mean | 0.5468 |
| Scale mean | 0.9360 |
| Scale CV | 0.4328 |
| N frames | **60** (not 6) |

---

## 3. Matched Autonomous vs CalK (80-frame paired comparison)

| Metric | Autonomous | CalK | Delta |
|--------|-----------|------|-------|
| AbsRel mean | 0.4418 | 0.4816 | +0.0398 |
| CalK wins AbsRel | — | 16/80 | — |
| CalK wins RMSE | — | 18/80 | — |

Per-sequence wins:
- 05-03-24: CalK wins 1/20
- 12-03-24: CalK wins 8/20
- 13-02-24: CalK wins 3/20
- 20-02-24: CalK wins 4/20

**CalK verdict: WORSE** — CalK is worse on 64/80 frames (AbsRel).

---

## 4. Final Depth Model Comparison

### Pose-PASS (full for VGGT/DA3/UniDepth; pilot for CalK)

| Model | AbsRel↓ | RMSE↓ | Scale | CV | Scope | N |
|-------|---------|-------|-------|-----|-------|---|
| VGGT | **0.1896** | 0.3813 | 1.1716 | 0.1816 | full | 998 |
| DA3Metric-official | 0.2117 | **0.3785** | **1.0994** | 0.1832 | full | 998 |
| UniDepth autonomous | 0.3855 | 0.4757 | 1.0177 | 0.4701 | full | 998 |
| UniDepth CalK-conditioned | 0.4656 | 0.5468 | 0.9360 | 0.4328 | pilot_60 | 60 |

**VGGT wins AbsRel on all 3 pose-PASS sequences. DA3Metric wins RMSE and scale proximity.**

---

## 5. DA3 Anchor Result (unchanged)

| Sequence | Anchor | Effect |
|----------|--------|--------|
| 05-03-24 | 1.0698 | improves |
| 13-02-24 | 1.0905 | roughly neutral |
| 20-02-24 | 1.0409 | worsens |
| 12-03-24 | 1.2170 | large improvement (but pose FAIL) |

Only 1/3 pose-PASS sequences improves with DA3 anchor. Anchor effect is marginal and inconsistent.

---

## 6. Final Route Decision

```
primary_depth = VGGT
da3metric_role = external metric-depth baseline + hard-case diagnostic
unidepth_role = external monocular baseline
unidepth_calk = WORSE (conditioning on calibrated K disrupts learned ray geometry)
scale_anchor = MARGINAL (1/3 pose-PASS improves, 1 worsens)
msam = NOT_PRIORITIZED
depth_branch = FROZEN
next_research_branch = POSE_ROBUSTNESS
loRA_pose_pilot = READY_FOR_POSE_PILOT
formal_lora = HOLD (until multi-plant pose validation)
```

---

## 7. Phase 3A Freeze State

```
phase3a_index_integrity = PASS
unidepth_calk_evaluation = VALID (80/80 frames, compact indexing correct)
phase3a_depth_benchmark = PASS
primary_depth = VGGT
da3metric_role = external baseline + diagnostic
unidepth_role = external baseline
unidepth_calk_role = REJECTED (WORSE than autonomous)
scale_anchor_result = MARGINAL
MSAM = NOT_PRIORITIZED
depth_branch = FROZEN
next_research_branch = POSE_ROBUSTNESS
loRA_pose_pilot = READY_FOR_POSE_PILOT
```

---

## 8. Q1–Q10 Answers

**Q1.** Old CalK evaluator used `pred_stack[original_idx]` on compact stack. Only `pred_stack[0]` and `pred_stack[1]` were valid (indices 0–1 in range 0–19). Frames at original indices 16, 32, … 304 were either wrong or IndexError.

**Q2.** YES — inference produced 20 frames/sequence, 80 total. All `depth_unidepth_calK_pilot.npy` files have shape `(20, 1080, 1080)`.

**Q3.** inference_rerun = **NO**. All 80 predictions already exist and are valid.

**Q4.** Corrected evaluation frame counts:
- 13-02-24: 20
- 20-02-24: 20
- 05-03-24: 20
- 12-03-24: 20
- total: **80**

**Q5.** Corrected CalK Pose-PASS (60 frames):
- AbsRel = 0.4656
- RMSE = 0.5468
- Scale = 0.9360
- CV = 0.4328

**Q6.** Matched 80-frame:
- Autonomous AbsRel = 0.4418
- CalK AbsRel = 0.4816

**Q7.** CalK better on **16/80** frames (AbsRel).

**Q8.** CalK = **WORSE** (+0.0398 AbsRel, only 20% win rate).

**Q9.** Final depth route: **VGGT** (primary). DA3Metric (baseline). UniDepth (baseline). VGGT+DA3 anchor (not justified). CalK rejected.

**Q10.** MSAM = **NOT_PRIORITIZED**. LoRA pose pilot = **READY_FOR_POSE_PILOT**.

---

## 9. Files Generated

| File | Description |
|------|-------------|
| `evaluate_corrected_v321.py` | Corrected evaluator with compact indexing |
| `UNIDEPTH_CALK_FRAME_MAP.json` | Compact→original frame mapping |
| `UNIDEPTH_CALK_EVAL_COMPLETENESS.json` | 80/80 completeness check |
| `UNIDEPTH_CALK_PROVENANCE.json` | K source and resolution details |
| `UNIDEPTH_CALK_FRAME_METRICS_V321.csv` | 80-frame per-frame metrics |
| `UNIDEPTH_CALK_SEQUENCE_SUMMARY_V321.csv` | Per-sequence summary |
| `UNIDEPTH_CALK_POSEPASS_SUMMARY_V321.json` | Pose-PASS summary (n=60) |
| `UNIDEPTH_MATCHED_PILOT_COMPARISON.csv` | Autonomous vs CalK paired comparison |
| `CORRECTED_COMPARISON_V321.csv` | 4-model comparison (corrected) |
| `PHASE3A21_MANIFEST.json` | All results manifest |
| `tests/test_phase3a21.py` | 9 tests, all passing |
