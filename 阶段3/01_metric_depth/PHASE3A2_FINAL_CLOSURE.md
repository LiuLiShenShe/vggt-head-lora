# Phase 3A.2 — Final Metric-Depth Closure

**Date**: 2026-08-31
**Status**: COMPLETE

---

## Corrected Model Comparison (Pose-PASS sequences)

| Model | AbsRel↓ | RMSE↓ | Scale | CV | δ1↑ | n_frames |
|-------|---------|-------|-------|-----|-----|----------|
| VGGT | 0.1896 | 0.3813 | 1.1716 | 0.1816 | 0.3945 | 998 |
| DA3Metric-official | 0.2117 | 0.3785 | 1.0994 | 0.1832 | 0.3124 | 998 |
| UniDepth autonomous | 0.3855 | 0.4757 | 1.0177 | 0.4701 | 0.1265 | 998 |
| UniDepth calK (pilot) | 0.6401 | 0.6365 | 0.7318 | 0.3207 | 0.1420 | 6 |

**Key**: DA3Metric-official uses `D_metric = D_net × focal_net / 300 = D_net × 2.1331` (README line 235).

---

## Anchor Analysis (DA3Metric-official → VGGT)

### Anchor Values per Sequence

| Sequence | Anchor | CV |
|----------|--------|-----|
| plantview__langdon_4__05-03-24 | 1.0698 | 0.0830 |
| plantview__langdon_4__12-03-24 | 1.2170 | 0.0522 |
| plantview__langdon_4__13-02-24 | 1.0905 | 0.0758 |
| plantview__langdon_4__20-02-24 | 1.0409 | 0.0674 |

### Anchor Comparison (Pose-PASS)

| Variant | AbsRel↓ | RMSE↓ | Scale | CV |
|---------|---------|-------|-------|-----|
| VGGT raw | 0.1896 | 0.3813 | 1.1716 | 0.1816 |
| VGGT+frame anchor | 0.1983 | 0.3638 | 1.0951 | 0.1857 |
| VGGT+seq anchor | 0.1847 | 0.3582 | 1.0981 | 0.1816 |
| DA3Metric direct | 0.2118 | 0.3732 | 1.0881 | 0.1819 |

**Sequence anchor improves VGGT by 2.6% AbsRel**.

---

## UniDepth Calibrated-K Pilot

80 frames (20 per sequence), using `model.infer(rgb, Pinhole(K=calibrated_K))`.

**Calibrated-K WORSE than autonomous**: AbsRel 0.6401 vs 0.3855 (+66.0%). Scale 0.7318 vs 1.0177.

The model's own predicted intrinsics produce better depth than calibrated camera parameters.
This suggests UniDepthV2's decoder is optimized for its predicted ray geometry,
and overriding with calibrated K disrupts this learned behavior.

---

## Phase 3A.1 Findings (Corrected Labels)

| Finding | Value |
|---------|-------|
| DA3 model type | DepthAnything3Net (single-branch, cam_dec=None) |
| DA3 official formula | `metric_depth = focal * net_output / 300` |
| DA3 conversion factor | 2.1331 (focal_net=639.94 at 504px) |
| DA3 official AbsRel | 0.2117 |
| DA3 official RMSE | 0.3785 |
| DA3 official scale | 1.0994 |
| VGGT AbsRel | 0.1896 |
| VGGT RMSE | 0.3813 |
| VGGT scale | 1.1716 |

---

## Final State

```
phase3a_scaling_integrity = PASS
da3_metric_conversion = APPLIED (official formula: focal*net/300)
da3_official_absrel = 0.212
da3_official_rmse = 0.379
da3_official_scale = 1.099
unidepth_autonomous_absrel = 0.386
unidepth_calK_pilot = WORSE (AbsRel=0.640, scale=0.732)
vggt_absrel = 0.190
vggt_rmse = 0.381
anchor_da3_official = seq_anchor_improves_by_2.6pct
msam = HOLD
lora = HOLD
final_route = Route 3 (keep VGGT as-is)
```

---

## Decision Log

1. **DA3 metric conversion APPLIED**: Official formula converts raw network output to metric depth. AbsRel improves from ~0.56 to 0.212.
2. **UniDepth calibrated-K REJECTED**: True calibrated-K inference (via Pinhole API) produces WORSE results than autonomous prediction.
3. **DA3 anchor with official metric**: Sequence anchor improves VGGT slightly (+2.6% AbsRel). Frame anchor hurts (-4.6%).
4. **MSAM = HOLD**: Anchor effect is marginal and inconsistent. Not sufficient to justify multi-source aggregation.
5. **Route**: Route 3 (keep VGGT as-is). VGGT AbsRel is better than DA3Metric; DA3Metric RMSE is better than VGGT.

---

## Files Generated

| File | Description |
|------|-------------|
| `apply_da3_metric_conversion.py` | Apply focal*net/300 to all DA3 depths |
| `run_unidepth_calibrated_k.py` | UniDepth calibrated-K pilot (80 frames) |
| `evaluate_corrected.py` | Unified eval with DA3-metric as 4th model |
| `recalc_anchor.py` | DA3-metric anchor recalculation |
| `CORRECTED_COMPARISON.csv` | 4-model comparison CSV |
| `DA3_METRIC_ANCHOR_VALUES.csv` | Per-sequence anchor values |
| `DA3_METRIC_ANCHOR_COMPARISON.csv` | Anchor comparison CSV |
| `UNIDEPTH_CALK_PILOT_MANIFEST.json` | CalK pilot results |