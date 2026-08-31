# Phase 3A — Metric Depth Benchmark Report

> Generated from evaluation CSVs (no hand-copied numbers).

> Evaluation version: 3A.0


## 1. Model Comparison (pose-PASS sequences)

| Model | raw AbsRel ↓ | raw RMSE ↓ | aligned AbsRel | scale mean | scale CV ↓ |
|-------|-------------|-----------|---------------|-----------|-----------|
| vggt | 0.1896 | 0.3813 | 0.1756 | 1.1716 | 0.1816 |
| da3 | 0.5597 | 0.7298 | 0.1902 | 2.3452 | 0.1832 |
| unidepth | 0.3855 | 0.4757 | 0.1543 | 1.0177 | 0.4701 |

## 2. Per-Sequence Breakdown

| Sequence | pose | Model | raw AbsRel | raw RMSE | scale CV |
|----------|------|-------|-----------|---------|---------|
| 05-03-24 | PASS | vggt | 0.2050 | 0.4329 | 0.1688 |
| 05-03-24 | PASS | da3 | 0.5930 | 0.8176 | 0.1795 |
| 05-03-24 | PASS | unidepth | 0.3491 | 0.4862 | 0.4354 |
| 12-03-24 | FAIL | vggt | 0.2993 | 0.5457 | 0.1577 |
| 12-03-24 | FAIL | da3 | 0.5965 | 0.8722 | 0.1624 |
| 12-03-24 | FAIL | unidepth | 0.3899 | 0.5445 | 0.5057 |
| 13-02-24 | PASS | vggt | 0.1710 | 0.3369 | 0.1549 |
| 13-02-24 | PASS | da3 | 0.5475 | 0.6729 | 0.1483 |
| 13-02-24 | PASS | unidepth | 0.3709 | 0.4319 | 0.4809 |
| 20-02-24 | PASS | vggt | 0.1928 | 0.3742 | 0.2211 |
| 20-02-24 | PASS | da3 | 0.5386 | 0.6988 | 0.2218 |
| 20-02-24 | PASS | unidepth | 0.4366 | 0.5089 | 0.4942 |

## 3. Scale Anchor Analysis

Anchors estimated using DA3/UniDepth as proxy metric (NOT GT).

| Sequence | Proxy | Anchor | AbsRel (raw) | AbsRel (anchored) | Change |
|----------|-------|--------|-------------|------------------|--------|
| 05-03-24 | da3 | 0.7679 | 0.2050 | 0.3779 | +84.4% |
| 05-03-24 | unidepth | 1.5814 | 0.2050 | 0.3734 | +82.2% |
| 12-03-24 | da3 | 0.8176 | 0.2993 | 0.4255 | +42.1% |
| 12-03-24 | unidepth | 2.0816 | 0.2993 | 0.5000 | +67.0% |
| 13-02-24 | da3 | 0.8081 | 0.1710 | 0.2901 | +69.6% |
| 13-02-24 | unidepth | 1.5607 | 0.1710 | 0.4427 | +158.9% |
| 20-02-24 | da3 | 0.7388 | 0.1928 | 0.3060 | +58.7% |
| 20-02-24 | unidepth | 1.6674 | 0.1928 | 0.6216 | +222.4% |

### Q-A: Which model has better depth quality?

- **Best raw AbsRel**: vggt (mean 0.1896 on pose-PASS)
- **Best aligned AbsRel (shape only)**: unidepth (mean 0.1543)
- **Most stable scale**: vggt (CV=0.1816)
- **Scale closest to 1.0**: unidepth (mean ~1.02) vs VGGT (~1.17) vs DA3 (~2.35)
- DA3 has severely wrong scale (2.35× overestimate) — raw AbsRel is dominated by scale error, not shape.
- VGGT has best raw quality due to moderate scale (1.17×) combined with good shape.
- UniDepth has best shape but very high scale variance (CV=0.47).


### Q-B: Does metric anchoring VGGT improve depth?

**NO — anchoring degrades ALL metrics.**
- DA3 anchors: [0.767867964402843, 0.8175867634705305, 0.8081273968572199, 0.7387561844046133] (mean=0.7831). VGGT's raw scale (1.17×) is already better than DA3 (0.77×).
- UniDepth anchors have very high variance (std ~0.45), producing unstable anchors.
- Conclusion: VGGT's built-in metric scale is already closer to reference than proxy models.


### Q-C: Route 1/2/3?

**Route 3: No benefit from scale anchoring.**
- Route 1 (direct replace with DA3/UniDepth): NOT recommended — VGGT has better raw quality.
- Route 2 (scale anchor VGGT): NOT recommended — makes metrics worse.
- Route 3 (keep VGGT as-is): RECOMMENDED — best raw AbsRel and scale stability.


## 4. Failure Case: 12-03-24 (pose-FAIL)

- **vggt**: AbsRel=0.2993, scale=1.5271, CV=0.1577
- **da3**: AbsRel=0.5965, scale=2.6827, CV=0.1624
- **unidepth**: AbsRel=0.3899, scale=0.9658, CV=0.5057
- Pose-FAIL does not significantly degrade depth quality (depth is frame-local).
- 12-03-24 has higher AbsRel than pose-PASS sequences for all models, suggesting the image content
  (possibly imaging angle/quality) affects depth more than pose estimation.


## 5. MSAM Readiness Gate

| Condition | Status |
|-----------|--------|
| Ref depth unit verified | ✅ (0.001, DEPTH_UNIT_AUDIT.json) |
| FG depth evaluator working | ✅ (unified_depth_evaluator.py, 1318 frames) |
| 3-tier scanner provenance correct | ✅ (v3.2.1, no identity leak) |
| Artifacts match report | ✅ (CSVs generate this report) |
| Scale semantics documented | ✅ (VGGT ≈ metric, not guaranteed metric) |
| Metric depth comparison done | ✅ (this report) |

**MSAM: HOLD** — geometry accuracy still PARTIAL (see v3.2.1 scanner GT, only 1 plant locally).
Metric depth comparison complete. VGGT is the best available depth source for this dataset.


## 6. Deliverables

| File | Description |
|------|-------------|
| da3/{seq}/depth_da3.npy | DA3Metric depth (504×504, float32, meters) |
| unidepth_v2/{seq}/depth_unidepth.npy | UniDepthV2 depth (1080×1080, float32, meters) |
| unidepth_v2/{seq}/intrinsics_unidepth.npy | UniDepthV2 predicted intrinsics (3×3) |
| evaluation/DEPTH_MODEL_COMPARISON_FRAME.csv | Per-frame metrics (3954 rows) |
| evaluation/DEPTH_MODEL_COMPARISON_SEQ.csv | Per-sequence summary (12 rows) |
| evaluation/DEPTH_MODEL_COMPARISON_SUMMARY.json | Cross-model comparison |
| anchor/SCALE_ANCHOR_VALUES.csv | Per-frame anchor values |
| anchor/ANCHORED_VGGT_METRICS.csv | Anchored vs raw comparison |
| figures/*.png | Visualizations |
| smoke_test_results.json | Model smoke test results |
| configs.py | Shared configuration |
| run_da3_inference.py | DA3 batch inference |
| run_unidepth_inference.py | UniDepth batch inference |
| unified_depth_evaluator.py | Core evaluation |
| scale_anchor_pilot.py | Scale anchor analysis |

---
*Generated: Phase 3A Metric Depth Benchmark | 12 model-sequence evaluations*
