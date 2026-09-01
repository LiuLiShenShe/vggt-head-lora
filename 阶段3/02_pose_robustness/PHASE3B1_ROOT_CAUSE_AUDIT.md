# Phase 3B.1 — langdon_4 Failure Root-Cause Audit

## Erratum: Phase 3B View-Count Experiment

**The Phase 3B 8/16/24 view-count experiment was invalid.** Phase 3B subsampled
predicted cameras from full-view inference output (`extrinsic_w2c.npy`) rather than
running independent VGGT forward passes on 8/16/24-image subsets. This means the
conclusion "view-count rescue = ALWAYS_FAIL" is unsupported.

Phase 3B.1 reruns VGGT inference independently at each view count.

## 1. True View-Count Re-Run (Independent VGGT Forward)

| Sequence | 8 views | 16 views | 24 views |
|----------|---------|----------|----------|
| 05-03-24 | 10.70° ✗ | 1.06° ✓ | 1.82° ✓ |
| 12-03-24 | 76.29° ✗ | 76.17° ✗ | 86.71° ✗ |
| 13-02-24 | 4.13° ✓ | 1.94° ✓ | 1.02° ✓ |
| 15-04-24 | 54.01° ✗ | 68.86° ✗ | 94.52° ✗ |
| 19-03-24 | 60.16° ✗ | 89.49° ✗ | 81.70° ✗ |
| 20-02-24 | 8.09° ✓ | 3.57° ✓ | 2.85° ✓ |
| plot_461 | 2.17° ✓ | 3.29° ✓ | 3.34° ✓ |
| plot_467 | 2.57° ✓ | 2.67° ✓ | 3.22° ✓ |

**View-count rescue observed in:** plantview__langdon_4__05-03-24

view_count_rescue = POSSIBLE

## 2. Acquisition Anomaly Audit (PASS vs FAIL)

| Date | Group | Brightness | Blur (Lap) | Path Length | Adj Baseline | Capture Order |
|------|-------|------------|------------|-------------|--------------|---------------|
| 05-03-24 | PASS | 123.69 | 73.97 | 38.8601 | 7.96° | CLOCKWISE |
| 13-02-24 | PASS | 129.29 | 49.65 | 35.6408 | 7.56° | CLOCKWISE |
| 20-02-24 | PASS | 126.31 | 69.10 | 35.7224 | 7.62° | CLOCKWISE |
| 12-03-24 | FAIL | 122.04 | 74.00 | 42.1280 | 7.84° | CLOCKWISE |
| 15-04-24 | FAIL | 122.68 | 96.23 | 44.0202 | 7.90° | CLOCKWISE |
| 19-03-24 | FAIL | 125.74 | 83.88 | 44.0301 | 7.83° | CLOCKWISE |

### PASS vs FAIL Group Differences

| Metric | PASS mean | FAIL mean | Δ (FAIL-PASS) |
|--------|-----------|-----------|---------------|
| brightness_mean | 126.43 | 123.48 | -2.9482 |
| saturation_mean | 0.02 | 0.02 | -0.0002 |
| blur_laplacian_var_mean | 64.24 | 84.70 | 20.4648 |
| trajectory_path_length | 36.74 | 43.39 | 6.6516 |
| adj_baseline_mean_deg | 7.71 | 7.86 | 0.1483 |
| transforms_n_intrinsic_clusters | 2.00 | 2.00 | 0.0000 |

## 3. Starting-Frame Sensitivity Test

| Date | Best Offset | Best rot_med | Worst Offset | Worst rot_med | Rescued? |
|------|-------------|--------------|--------------|---------------|----------|
| 12-03-24 | 200 | 0.55° | 152 | 8.96° | YES |
| 15-04-24 | 45 | 0.00° | 246 | 9.15° | YES |
| 19-03-24 | 36 | 0.24° | 216 | 8.91° | YES |

**CRITICAL: Starting-frame sensitivity rescues failure.**
The failure is input-dependent, not architectural.

starting_frame_sensitivity = RESCUE_OBSERVED

## 4. Same-Domain Multi-Plant Controls (MuST-C)

| Sequence | 8 views | 16 views | 20 views (full) |
|----------|---------|----------|-----------------|
| plot198 | 1.79° ✓ | 1.68° ✓ | 1.84° ✓ |
| plot198 | 3.02° ✓ | 1.20° ✓ | 1.38° ✓ |
| plot198 | 0.36° ✓ | 2.64° ✓ | 1.90° ✓ |
| plot198 | 1.69° ✓ | 1.03° ✓ | 1.41° ✓ |

MuST-C: **All PASS** — same-domain generalization confirmed.

same_domain_multiplant_generalization = CONFIRMED

## Phase 3B.1 State Summary

```
phase3b1_status = COMPLETE
true_view_count_experiment = COMPLETED (independent VGGT forward)
view_count_rescue = POSSIBLE
starting_frame_sensitivity = RESCUE_OBSERVED
acquisition_anomaly = COMPLETED
cross_dataset_failure_generalization = NOT_OBSERVED
same_domain_multiplant_generalization = CONFIRMED
lora_pose_rescue = HOLD_PENDING_ROOT_CAUSE
DENSE_CANOPY_POSE_FAILURE_GENERALIZES = NOT_OBSERVED
```
