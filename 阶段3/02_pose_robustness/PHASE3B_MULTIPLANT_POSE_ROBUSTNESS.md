# Phase 3B: Multi-Plant Pose Robustness Benchmark — Final Report

**Date**: 2026-08-31  
**Status**: COMPLETE (erratum issued 2026-09-01 — see below)  
**Branch**: POSE_ROBUSTNESS  
**Depth branch**: FROZEN (Phase 3A.2.1)

---

> **⚠️ ERRATUM (2026-09-01):** The 8/16/24 view-count experiment in this report was invalid.
> Phase 3B subsampled predicted cameras from full-view inference output rather than running
> independent VGGT forward passes. The conclusion "view-count rescue = ALWAYS_FAIL" is
> unsupported. See `PHASE3B1_ROOT_CAUSE_AUDIT.md` for corrected results with independent
> inference. Key correction: consecutive-frame 16-view windows rescue ALL failures (269/269 PASS).
> The failure was caused by uniform subsampling from a long trajectory, not by image content.

---

## Executive Summary

VGGT pose catastrophic failure **does NOT generalize** beyond langdon_4. Across 8 independent plants (7 Wheat3DGS plots + 1 langdon_4), only langdon_4 exhibits pose failure — and only on 3 of 6 dates. All 7 Wheat3DGS plots PASS at all view counts (8/16/24/full). Therefore, **LoRA Pose Rescue is NOT JUSTIFIED** at this time.

**Key answer to the scientific question**: Dense-canopy pose failure is a **langdon_4-specific longitudinal phenomenon**, not a general limitation of VGGT across plant diversity.

---

## Q1–Q15 Answers

### Q1: Does VGGT pose catastrophic failure generalize beyond langdon_4?
**Answer: NO.** 1/8 plant_ids (12.5%) exhibits failure. 7/8 plants PASS at all view counts.

### Q2: Is failure correlated with canopy density?
**Answer: Weakly, but confounded with plant identity.** Spearman(ρ=0.63, p=0.02) between canopy_fraction and pose failure. However, this correlation is entirely driven by langdon_4 being both denser AND the only failing plant. Within wheat3dgs (canopy 0.064–0.082), all pass. Within langdon_4 (canopy 0.086–0.112), 3/6 fail — showing intra-plant variation unrelated to density alone.

### Q3: Is failure correlated with growth stage / acquisition date?
**Answer: YES — strong longitudinal pattern.** langdon_4 failures cluster at dates 12-03, 15-04, 19-03 (all rot_med > 78°). PASS dates: 05-03, 13-02, 20-02 (all rot_med < 3.6°). The failing dates span mid-March to mid-April, suggesting a temporal/capture-condition factor rather than canopy density alone.

### Q4: What is the failure mode?
**Answer: ROTATION_AND_TRANSLATION in all 3 cases.** No pure rotation collapse, no center collapse, no translation-only. The VGGT pose is catastrophically wrong in both rotation and position for langdon_4 dates 12-03, 15-04, 19-03.

### Q5: Does increasing views (8→16→24) rescue failures?
**Answer: NO.** All 3 failing sequences are ALWAYS_FAIL across 8, 16, 24, and full views. No sequence shows RESCUED_BY_MORE_VIEWS.

### Q6: What is the overall pose gate pass rate?
**Answer: 10/13 = 76.9%** (full views). Per-plant: langdon_4 = 3/6 = 50%, wheat3dgs = 7/7 = 100%.

### Q7: What are the wheat3dgs rot_median values?
**Answer: 2.7–3.8° across all 7 plots, all view counts.** Excellent pose quality. Range: plot_465 at 24v (2.75°) to plot_461 at 16v (3.79°).

### Q8: What are the langdon_4 PASS rot_median values?
**Answer: 1.5–3.6°** — comparable to wheat3dgs when it works.

### Q9: What are the langdon_4 FAIL rot_median values?
**Answer: 67–90°** — catastrophic rotation error, roughly random orientation.

### Q10: Does failure type vary with view count?
**Answer: No.** All failures are ROTATION_AND_TRANSLATION at all view counts.

### Q11: Is there a density class effect on failure?
**Answer:表面上 yes, 实际 confounded.**
- LOW density: 0/5 fail (0%)
- MEDIUM density: 1/4 fail (25%)
- HIGH density: 2/4 fail (50%)

But all failures are langdon_4. The density effect is a plant-identity confound.

### Q12: Canopy fraction range?
**Answer:**
- Wheat3DGS: 0.064–0.082 (LOW to MEDIUM)
- langdon_4: 0.086–0.112 (MEDIUM to HIGH)

### Q13: Correlation statistics?
**Answer:**
- Spearman(rot_median, canopy_fraction): ρ=0.214, p=0.482 (NOT significant)
- Spearman(pose_fail, canopy_fraction): ρ=0.634, p=0.020 (significant, but confounded)

### Q14: View-count rescue evidence?
**Answer: ZERO.** 0/3 failing sequences rescued. 13/13 sequences show ALWAYS_PASS or ALWAYS_FAIL pattern.

### Q15: Should we start LoRA Pose Rescue?
**Answer: NOT JUSTIFIED.** Failure is specific to 3 dates of 1 plant (langdon_4). The 7 independent wheat3dgs plants all pass. Insufficient evidence of a general problem requiring architectural intervention.

---

## Final State Format

```
phase3b_status = COMPLETE
n_plants_evaluated = 8
n_sequences_evaluated = 13
n_view_counts = 4 (8, 16, 24, full)
pose_gate_pass_rate = 0.7692
n_plants_with_failure = 1 (langdon_4 only)
n_dates_with_failure = 3/6 (langdon_4)
failure_type = ROTATION_AND_TRANSLATION (all 3 cases)
view_rescue_effect = NONE (ALWAYS_FAIL for all failures)
canopy_fraction_range_wheat3dgs = [0.064, 0.082]
canopy_fraction_range_langdon4 = [0.086, 0.112]
density_class_distribution = LOW:5, MEDIUM:4, HIGH:4
spearman_rot_vs_canopy = 0.214 (p=0.482, NOT significant)
spearman_fail_vs_canopy = 0.634 (p=0.020, significant but confounded)
loRA_pose_rescue_readiness = NOT_JUSTIFIED
DENSE_CANOPY_POSE_FAILURE_GENERALIZES = NO
next_action = Investigate langdon_4 acquisition anomaly (camera metadata, lighting, motion, trajectory)
```

---

## Detailed Results Tables

### Per-Plant Summary (Full Views)

| Plant ID | Dataset | Dates | Pass | Fail | Fail Rate | Density Range |
|----------|---------|-------|------|------|-----------|---------------|
| wheat_461 | Wheat3DGS | 1 | 1 | 0 | 0% | LOW |
| wheat_462 | Wheat3DGS | 1 | 1 | 0 | 0% | LOW |
| wheat_463 | Wheat3DGS | 1 | 1 | 0 | 0% | MEDIUM |
| wheat_464 | Wheat3DGS | 1 | 1 | 0 | 0% | LOW |
| wheat_465 | Wheat3DGS | 1 | 1 | 0 | 0% | LOW |
| wheat_466 | Wheat3DGS | 1 | 1 | 0 | 0% | LOW |
| wheat_467 | Wheat3DGS | 1 | 1 | 0 | 0% | MEDIUM |
| langdon_4 | 3DPlantView | 6 | 3 | 3 | 50% | MEDIUM–HIGH |
| **Total** | | **13** | **10** | **3** | **23%** | |

### langdon_4 Longitudinal Results (Full Views)

| Date | rot_med (°) | rot_p90 (°) | center_norm | Gate | Failure Type | Canopy | Density |
|------|------------|------------|-------------|------|-------------|--------|---------|
| 05-03 | 1.71 | 4.10 | 0.031 | PASS | PASS | 0.106 | HIGH |
| 12-03 | 84.19 | 158.48 | 0.906 | FAIL | ROTATION_AND_TRANSLATION | 0.109 | HIGH |
| 13-02 | 2.22 | 4.69 | 0.039 | PASS | PASS | 0.086 | MEDIUM |
| 15-04 | 89.82 | 163.31 | 0.954 | FAIL | ROTATION_AND_TRANSLATION | 0.101 | MEDIUM |
| 19-03 | 78.95 | 159.53 | 0.871 | FAIL | ROTATION_AND_TRANSLATION | 0.112 | HIGH |
| 20-02 | 3.59 | 7.51 | 0.059 | PASS | PASS | 0.103 | HIGH |

### Wheat3DGS Rot-Median by Plot and View Count

| Plot | 8 views | 16 views | 24 views | Full |
|------|---------|----------|----------|------|
| 461 | 3.59° | 3.79° | 3.50° | 3.60° |
| 462 | 3.18° | 3.70° | 3.30° | 3.66° |
| 463 | 3.21° | 3.67° | 3.08° | 3.45° |
| 464 | 3.15° | 3.59° | 2.92° | 3.21° |
| 465 | 3.08° | 3.12° | 2.75° | 2.95° |
| 466 | 3.20° | 3.36° | 3.22° | 3.23° |
| 467 | 3.34° | 3.33° | 3.31° | 3.33° |

All values ≤ 3.8° — well within PASS gate (≤10°).

### View-Count Effect Classification

| Sequence | 8v | 16v | 24v | Full | Classification |
|----------|-----|------|------|------|----------------|
| langdon_4 05-03 | PASS | PASS | PASS | PASS | ALWAYS_PASS |
| langdon_4 12-03 | FAIL | FAIL | FAIL | FAIL | ALWAYS_FAIL |
| langdon_4 13-02 | PASS | PASS | PASS | PASS | ALWAYS_PASS |
| langdon_4 15-04 | FAIL | FAIL | FAIL | FAIL | ALWAYS_FAIL |
| langdon_4 19-03 | FAIL | FAIL | FAIL | FAIL | ALWAYS_FAIL |
| langdon_4 20-02 | PASS | PASS | PASS | PASS | ALWAYS_PASS |
| wheat3dgs 461–467 | PASS | PASS | PASS | PASS | ALWAYS_PASS (×7) |

**0/13 sequences show RESCUED_BY_MORE_VIEWS.**

---

## Files Generated

| File | Description |
|------|-------------|
| `01_dataset_inventory/MULTIPLANT_POSE_DATASET_INVENTORY.csv` | 13 sequences, 8 plants |
| `01_dataset_inventory/DATASET_SUMMARY.json` | Dataset inventory summary |
| `00_protocol/VIEW_SAMPLING_MANIFEST.json` | Nested view sets (24/16/8) |
| `04_scene_characterization/CANOPY_CHARACTERIZATION.csv` | Canopy fractions + density classes |
| `03_pose_evaluation/MULTIANT_POSE_RESULTS.csv` | 52 rows (13 seq × 4 view counts) |
| `03_pose_evaluation/PER_FRAME_ROT_ERRORS.csv` | 2208 per-frame rotation errors |
| `07_statistics/POSE_FAILURE_SUMMARY.json` | Statistical analysis results |
| `07_statistics/POSE_PLANT_LEVEL_SUMMARY.csv` | Per-plant pass/fail counts |
| `07_statistics/POSE_DENSITY_CORRELATION.csv` | Spearman correlations |
| `PHASE3B_MULTIPLANT_POSE_ROBUSTNESS.md` | This report |

---

## Scripts

| Script | Purpose |
|--------|---------|
| `01_dataset_inventory/build_inventory.py` | Scan all datasets |
| `00_protocol/generate_view_sampling.py` | Generate nested view sets |
| `04_scene_characterization/characterize_canopy.py` | Canopy density from masks |
| `03_pose_evaluation/evaluate_multoplant.py` | Multi-plant pose evaluation |
| `07_statistics/analyze_failure.py` | Statistical analysis |

---

## Recommended Next Steps

1. **Investigate langdon_4 acquisition anomaly**: Check camera metadata, lighting conditions, capture trajectory, motion blur for dates 12-03, 15-04, 19-03 vs PASS dates (05-03, 13-02, 20-02)
2. **Do NOT start LoRA Pose Rescue**: Insufficient evidence of a general problem
3. **Depth branch remains FROZEN**: Phase 3A.2.1 closure is final
4. **Optional**: Add more plant species/datasets to strengthen the generalization argument
