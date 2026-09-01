# Phase 3C: Windowed Pose - Final Report

## Summary

Windowed pose estimation (stride=8, window=16, overlap=8) with Sim(3) center-based stitching achieves 100% frame coverage for all 7 test sequences.

**Key Finding**: The windowed approach works for sequences where VGGT rotations are globally consistent (wheat3dgs, mustc) but fails for langdon_4 where VGGT rotations are inconsistent across windows.

## Results

| Sequence | Method | Frames | rot_median | rot_p90 | Center Norm | Gate |
|---|---|---|---|---|---|---|
| mustc__pos00 | Windowed Global | 20 | 3.2 deg | 5.8 deg | 0.101 | PASS |
| mustc__pos00 | Local W0 | 16 | 1.8 deg | 4.1 deg | 0.069 | PASS |
| langdon_4__05-03 | Windowed Global | 320 | 48.8 deg | 69.2 deg | 0.514 | FAIL |
| langdon_4__05-03 | Local W0 | 16 | 2.7 deg | 4.1 deg | 0.053 | PASS |
| langdon_4__05-03 | Uniform 16v | 16 | 1.1 deg | 3.9 deg | 0.055 | PASS |
| langdon_4__12-03 | Windowed Global | 320 | 28.9 deg | 55.3 deg | 0.501 | FAIL |
| langdon_4__12-03 | Local W0 | 16 | 2.9 deg | 4.9 deg | 0.048 | PASS |
| langdon_4__12-03 | Uniform 16v | 16 | 76.2 deg | 148.7 deg | 0.957 | FAIL |
| langdon_4__15-04 | Windowed Global | 318 | 59.9 deg | 119.8 deg | 0.632 | FAIL |
| langdon_4__15-04 | Local W0 | 16 | 2.4 deg | 4.8 deg | 0.058 | PASS |
| langdon_4__15-04 | Uniform 16v | 16 | 68.9 deg | 162.3 deg | 0.818 | FAIL |
| langdon_4__19-03 | Windowed Global | 320 | 27.9 deg | 49.0 deg | 0.457 | FAIL |
| langdon_4__19-03 | Local W0 | 16 | 2.5 deg | 3.7 deg | 0.041 | PASS |
| langdon_4__19-03 | Uniform 16v | 16 | 89.5 deg | 161.1 deg | 1.178 | FAIL |
| wheat3dgs__461 | Windowed Global | 36 | 6.5 deg | 10.4 deg | 0.178 | PASS |
| wheat3dgs__461 | Local W0 | 16 | 2.5 deg | 5.7 deg | 0.120 | PASS |
| wheat3dgs__461 | Uniform 16v | 16 | 3.3 deg | 5.0 deg | 0.162 | PASS |
| wheat3dgs__467 | Windowed Global | 36 | 3.9 deg | 6.0 deg | 0.190 | PASS |
| wheat3dgs__467 | Local W0 | 16 | 2.4 deg | 5.5 deg | 0.108 | PASS |
| wheat3dgs__467 | Uniform 16v | 16 | 2.7 deg | 4.2 deg | 0.131 | PASS |

## Root Cause Analysis: Why Windowed Stitching Fails for langdon_4

### Per-Window Rotation Consistency Diagnostic

Each window predicts poses in its own coordinate frame. To check consistency, we compute the Procrustes rotation needed to align each window to the COLMAP reference. If VGGT rotations are globally consistent, all windows need the SAME Procrustes rotation.

| Sequence | Procrustes Angle Mean | Std | Spread (max-min) | Stitched Gate |
|---|---|---|---|---|
| wheat3dgs__461 | 176.2 deg | 2.1 deg | 5.7 deg | PASS |
| wheat3dgs__467 | 173.9 deg | 2.5 deg | 6.3 deg | PASS |
| mustc__pos00 | 153.8 deg | 7.9 deg | 15.9 deg | PASS |
| langdon_4__05-03 | 146.3 deg | 28.1 deg | 99.1 deg | FAIL |
| langdon_4__12-03 | 145.5 deg | 27.8 deg | 100.5 deg | FAIL |
| langdon_4__15-04 | 145.9 deg | 24.9 deg | 85.8 deg | FAIL |
| langdon_4__19-03 | 146.7 deg | 27.7 deg | 97.5 deg | FAIL |

**Interpretation**: For wheat3dgs/mustc, all windows need nearly the same rotation correction (spread < 16 deg), so a single global alignment works. For langdon_4, each window needs a DIFFERENT rotation correction (spread 86-100 deg), so no single global alignment can work.

### Individual Window Quality

Critically, each individual window has EXCELLENT rotation accuracy when evaluated alone:

| Window | rot_median (per-window eval) | rot_median (global eval) |
|---|---|---|
| langdon_4 W0 | 2.7 deg | 49.5 deg |
| langdon_4 W1 | 0.7 deg | 58.3 deg |
| langdon_4 W2 | 1.4 deg | 64.1 deg |
| langdon_4 W3 | 8.5 deg | 56.8 deg |
| ... | all < 9 deg | all 33-84 deg |

The per-window Procrustes alignment compensates for the different rotation offsets. But when we apply the SAME global alignment to all windows, the compensation is wrong for most windows.

### Scale Drift

Scale drift is moderate and NOT the primary failure cause:

| Sequence | Mean Scale | CV | Max Jump |
|---|---|---|---|
| langdon_4__05-03 | 0.9994 | 8.5% | 0.429 |
| langdon_4__12-03 | 1.0092 | 6.0% | 0.238 |
| langdon_4__15-04 | 1.0160 | 5.2% | 0.159 |
| langdon_4__19-03 | 1.0036 | 9.2% | 0.349 |

### Overlap Alignment Quality

Sim(3) alignment RMSE on camera centers is excellent (all < 0.035m):

| Sequence | Mean Overlap RMSE | Max Overlap RMSE |
|---|---|---|
| langdon_4__05-03 | 0.010m | 0.031m |

The POSITION alignment is correct. The problem is purely ROTATIONAL.

## Conclusions

1. **VGGT rotations are NOT globally consistent across windows for langdon_4**. Each window has its own arbitrary rotation offset (80-179 deg from reference), varying by up to 100 deg between windows.

2. **VGGT rotations ARE globally consistent for wheat3dgs and mustc** (spread < 16 deg). Windowed stitching works well for these.

3. **The position alignment (Sim(3) on camera centers) is correct** (RMSE < 0.035m for all sequences).

4. **The failure is NOT caused by**: scale drift (CV < 10%), insufficient overlap (8 frames), or bad position alignment.

5. **The failure IS caused by**: VGGT predicting different rotation offsets for the same scene in different local windows, which makes it impossible to find a single global rotation that corrects all windows simultaneously.

6. **Individual 16-frame windows always work** (rot_median < 9 deg for all windows, most < 3 deg). The model is accurate on local sequences.

## Implications

- **For PASS sequences (wheat3dgs, mustc)**: Windowed stitching provides full trajectory coverage with acceptable accuracy. Use the stitched result.

- **For FAIL sequences (langdon_4)**: Windowed stitching cannot recover global rotation consistency. The per-window predictions are excellent but incommensurable. This is a fundamental VGGT limitation, not a stitching algorithm limitation.

- **No known remedy without external supervision**: Since the rotation offset varies per window, correcting it requires either (a) ground truth poses, (b) cross-window feature matching with PnP, or (c) a VGGT architectural change that enforces global rotation consistency.

## Files Generated

-  - Sim(3) center-based stitching
-  - Stitched camera poses
-  - Stitching metadata
-  - Evaluation script
-  - Full results CSV
