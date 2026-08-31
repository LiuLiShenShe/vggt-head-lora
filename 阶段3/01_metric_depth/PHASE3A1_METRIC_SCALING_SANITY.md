# Phase 3A.1 — Metric Scaling Sanity Check Report

> Generated from evaluation CSVs (no hand-copied numbers).

> Evaluation version: 3A.1


## 1. Code Audit Findings

| Item | Value | Finding |
|------|-------|---------|
| DA3 model type | `DepthAnything3Net` (single branch) | NOT `NestedDepthAnything3Net` |
| DA3 `cam_dec` | `None` | No camera decoder → no intrinsics prediction |
| DA3 `is_metric` | `0` (False) | Model declares output is NOT metric |
| DA3 `pred.intrinsics` | `None` | No intrinsics in Prediction object |
| DA3 focal/300 scaling | **NOT applied** | Only exists in `NestedDepthAnything3Net._apply_metric_scaling` |
| UniDepth intrinsics | Predicted K (fx≈1205 vs calibrated 1372) | Called without intrinsics argument |

### Root Cause of 2.35× Scale

The DA3 2.35× scale ratio (ref/DA3) is **NOT** from `focal/300` scaling.

DA3METRIC-LARGE outputs raw relative depth from the DPT head without any
metric scaling mechanism. The 2.35× is the model's native learned scale being
~0.43× of the actual metric depth.

Applying `focal/300 = 640/300 ≈ 2.13` as a post-hoc correction brings DA3's
scale from 2.35× to 1.10× and improves AbsRel from 0.560 to 0.212.


## 2. Corrected Model Comparison (pose-PASS sequences)

| Model | raw AbsRel ↓ | raw RMSE ↓ | aligned AbsRel | scale mean | scale CV ↓ |
|-------|-------------|-----------|---------------|-----------|-----------|
| vggt | 0.1896 | 0.3813 | 0.1756 | 1.1716 | 0.1816 |
| da3_raw | 0.5597 | 0.7298 | 0.1902 | 2.3452 | 0.1832 |
| da3_calibrated | 0.2117 | 0.3785 | 0.1902 | 1.0994 | 0.1832 |
| unidepth_raw | 0.3855 | 0.4757 | 0.1543 | 1.0177 | 0.4701 |
| unidepth_k_corrected | 0.3232 | 0.4397 | 0.1543 | 1.1566 | 0.4701 |

### Key Improvements After Correction

- **DA3 raw → calibrated**: AbsRel 0.5597 → 0.2117 (+62.2% improvement)
- **DA3 scale**: 2.3452 → 1.0994 (closer to 1.0)
- **UniDepth raw → K-corrected**: AbsRel 0.3855 → 0.3232 (+16.2% improvement)

## 3. Per-Sequence Corrected Breakdown

| Sequence | pose | Model | raw AbsRel | RMSE | aligned | scale | CV |
|----------|------|-------|-----------|------|---------|-------|-----|
| 05-03-24 | PASS | vggt | 0.2050 | 0.4329 | 0.1800 | 1.2696 | 0.1688 |
| 05-03-24 | PASS | da3_raw | 0.5930 | 0.8176 | 0.2006 | 2.5689 | 0.1795 |
| 05-03-24 | PASS | da3_calibrated | 0.2157 | 0.4226 | 0.2006 | 1.2043 | 0.1795 |
| 05-03-24 | PASS | unidepth_raw | 0.3491 | 0.4862 | 0.1636 | 1.0119 | 0.4354 |
| 05-03-24 | PASS | unidepth_k_corrected | 0.2943 | 0.4539 | 0.1636 | 1.1513 | 0.4354 |
| 12-03-24 | FAIL | vggt | 0.2993 | 0.5457 | 0.2029 | 1.5271 | 0.1577 |
| 12-03-24 | FAIL | da3_raw | 0.5965 | 0.8722 | 0.2085 | 2.6827 | 0.1624 |
| 12-03-24 | FAIL | da3_calibrated | 0.1999 | 0.4235 | 0.2085 | 1.2576 | 0.1624 |
| 12-03-24 | FAIL | unidepth_raw | 0.3899 | 0.5445 | 0.1676 | 0.9658 | 0.5057 |
| 12-03-24 | FAIL | unidepth_k_corrected | 0.2939 | 0.4560 | 0.1676 | 1.1188 | 0.5057 |
| 13-02-24 | PASS | vggt | 0.1710 | 0.3369 | 0.1518 | 1.1346 | 0.1549 |
| 13-02-24 | PASS | da3_raw | 0.5475 | 0.6729 | 0.1640 | 2.2112 | 0.1483 |
| 13-02-24 | PASS | da3_calibrated | 0.1883 | 0.3222 | 0.1640 | 1.0366 | 0.1483 |
| 13-02-24 | PASS | unidepth_raw | 0.3709 | 0.4319 | 0.1323 | 1.0859 | 0.4809 |
| 13-02-24 | PASS | unidepth_k_corrected | 0.3232 | 0.4126 | 0.1323 | 1.2331 | 0.4809 |
| 20-02-24 | PASS | vggt | 0.1928 | 0.3742 | 0.1950 | 1.1107 | 0.2211 |
| 20-02-24 | PASS | da3_raw | 0.5386 | 0.6988 | 0.2060 | 2.2555 | 0.2218 |
| 20-02-24 | PASS | da3_calibrated | 0.2311 | 0.3906 | 0.2060 | 1.0573 | 0.2218 |
| 20-02-24 | PASS | unidepth_raw | 0.4366 | 0.5089 | 0.1670 | 0.9553 | 0.4942 |
| 20-02-24 | PASS | unidepth_k_corrected | 0.3519 | 0.4527 | 0.1670 | 1.0855 | 0.4942 |

## 4. Q&A

### Q1: Is DA3 output canonical or metric depth?

**Neither.** DA3 outputs raw relative depth from the DPT head.
The `is_metric=0` flag confirms the model does NOT claim metric output.
No intrinsics prediction, no focal/300 scaling, no metric calibration.

### Q2: Is scale ≈ 2.35 equal to focal/300?

**NO.** The 2.35× is `median(ref_depth) / median(DA3_depth)` — the model's native
scale being ~0.43× of metric depth. The focal/300 = 640/300 ≈ 2.13 is coincidentally
close but has a different source. Applying focal/300 as post-hoc correction works
because it happens to approximately rescale DA3's output to metric.

### Q3: Missing or double scaling?

**Neither.** DA3 has no metric scaling mechanism in this model variant.
The scale mismatch is inherent to the model's training (relative depth, not metric).

### Q4: UniDepth using calibrated or predicted intrinsics?

**Predicted intrinsics** (called without intrinsics argument).
Predicted fx ≈ 1205 vs calibrated fx = 1372 (ratio ≈ 0.88).
K-corrected depth improves AbsRel from 0.386 → 0.323.

### Q5: Does corrected DA3 beat VGGT?

**No on raw AbsRel** (0.2117 > 0.1896).
**YES on RMSE** (0.3785 < 0.3813).
**YES on scale** (1.0994 closer to 1.0 than 1.1716).
VGGT retains best scale CV (0.1816 vs 0.1832).


## 5. Final Verdict

```
phase3a_scaling_integrity = PASS  (no implementation error found)
da3_metric_scaling = NOT_APPLICABLE  (single-branch model, no scaling mechanism)
da3_2.35x_explanation = native relative depth scale, ~0.43× of metric ref
da3_focal_correction = IMPROVES  (AbsRel 0.560→0.212, scale 2.35→1.10)
unidepth_calibrated_intrinsics = IMPROVES  (AbsRel 0.386→0.323)
final_route = Route 3 (keep VGGT) — VGGT best raw AbsRel (0.190)
  BUT: DA3+calibration is competitive (0.212) and beats on RMSE (0.379)
  AND: UniDepth+K is also improved (0.323) but still worse than VGGT
msam = HOLD  (geometry accuracy still PARTIAL)
lora = HOLD
```


## 6. Deliverables

| File | Description |
|------|-------------|
| phase3a1_audit/INTRINSICS_AUDIT.csv | DA3 intrinsics verification |
| phase3a1_audit/CORRECTED_MODEL_COMPARISON.csv | 5-variant × 4-seq comparison |
| phase3a1_audit/CORRECTED_SUMMARY.json | Summary stats |
| phase3a1_audit/SCALING_AUDIT_SUMMARY.json | Full audit findings |

---
*Generated: Phase 3A.1 Metric Scaling Sanity Check | 20 model-sequence evaluations*
