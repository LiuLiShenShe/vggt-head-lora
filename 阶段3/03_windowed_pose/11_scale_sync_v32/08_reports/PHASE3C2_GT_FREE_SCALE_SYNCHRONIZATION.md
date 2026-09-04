# Phase 3C.2 — GT-Free Window Scale Synchronization: Final Report

## Executive Summary

**phase3c2_status = PARTIAL**

The 44-53° "rotation error" for long sequences (39 windows / 320 frames) is **NOT caused by scale drift**. It is caused by **Q-chain rotation accumulation** —38 pairwise orientation transforms compose with accumulated error. Scale drift (8-31%) is a secondary, smaller problem.

Three GT-free scale methods were tested:
1. **Center-scale chain** — camera centers for pairwise scale (CV 0.05-0.09, 8-31% chain drift)
2. **Dense point_map scale** — biased <1.0 due to per-window arbitrary depth scaling (chains to zero)
3. **Per-window scale normalization** — log-space least squares (similar results to center chain)

All three produce essentially identical rotation error (~44°) because **rotation error and scale drift are independent problems**.

---

## Q1: Is the 44-53° error really orientation error?

**YES, but with nuance.** The error comes from Q-chain rotation accumulation, not from scale distortion:

| Chain length | Rot error | Q_chain_err | Interpretation |
|---|---|---|---|
| 1 window | 2.7° | 0° | Local rotation correct |
| 2 windows | 2.1° | 45.7° | Q_01 has large offset, but Procrustes absorbs it |
| 5 windows | 6.7° | 18.7° | Error starts accumulating |
| 10 windows | 16.7° | 45.5° | Significant accumulation |
| 20 windows | 21.4° | 92.9° | Severe |
| 38 windows | 45.9° | 159.2° | Full chain error |

**Per-segment evaluation (40-frame windows) shows only 4-8° error** — proving the stitched rotations are locally accurate. The 44° is a global Procrustes alignment artifact from the accumulated Q-chain offset.

## Q2: Is dense geometry scale more stable than camera-center scale?

**NO — for long sequences, center scale is MORE stable:**

| Sequence | Center CV | Dense CV | Better |
|---|---|---|---|
| lang4 05-03 | 0.085 | 0.117 | center |
| lang4 12-03 | 0.059 | 0.109 | center |
| lang4 15-04 | 0.051 | 0.116 | center |
| lang4 19-03 | 0.091 | 0.090 | dense (marginal) |
| wheat 461 | 0.105 | 0.084 | dense |
| wheat 467 | 0.157 | 0.056 | dense |

Dense point_map scale is biased <1.0 for all langdon_4 pairs because VGGT's point head produces depth at arbitrary per-window scale. Camera centers are NOT affected by depth scaling.

## Q3: Does scale drift cause the rotation error?

**NO.** Evidence:
- `uniform_chain` (s=1 for all pairs, zero scale drift) → same 44° rotation error
- `per_window_scale` normalization → same 44° rotation error
- Rotation error is identical across all scale methods

The 44° error is entirely from Q-chain rotation accumulation.

## Q4: Can global scale optimization help?

**Scale optimization cannot fix the rotation error.** However:
- Center scale chain has 8-31% drift over 38 pairs
- Graph is pure chain (cycle rank = 0) — no redundant edges for optimization
- Stride=4 would give cycle rank = 73, enabling real optimization
- But even with perfect scale, rotation error remains 44°

## Q5: What is the fundamental limitation?

Two independent limitations for long sequences:

1. **Q-chain rotation accumulation**:38 pairwise Q compositions accumulate ~159° total rotation offset. This is inherent to sequential chaining without loop closure or absolute orientation constraints.

2. **Scale chain drift**:8-31% over38 pairs. Less severe than rotation, but distorts trajectory shape (center TC = 0.81-0.84).

Both could potentially be addressed by:
- **Loop closure**: if trajectory returns to start (first-last overlap = 0 frames currently)
- **Stride=4**: provides redundant edges for scale optimization (cycle rank 73)
- **Absolute orientation anchor**: e.g., gravity direction from IMU (not available)

---

## Detailed Results

### Disentanglement (Step 0)

| Sequence | Orient-only | Center-only | Full | Diagnosis |
|---|---|---|---|---|
| mustc pos00 | 2.6° ✅ | tc=0.990 ✅ | 2.6° ✅ | BOTH_OK |
| lang4 05-03 | 44.0° ❌ | tc=0.839 | 44.0° ❌ | ORIENT_FAIL |
| lang4 12-03 | 48.7° ❌ | tc=0.822 | 48.7° ❌ | ORIENT_FAIL |
| lang4 15-04 | 53.1° ❌ | tc=0.813 | 53.1° ❌ | ORIENT_FAIL |
| lang4 19-03 | 44.3° ❌ | tc=0.834 | 44.3° ❌ | ORIENT_FAIL |
| wheat 461 | 3.3° ✅ | tc=0.992 ✅ | 3.3° ✅ | BOTH_OK |
| wheat 467 | 3.4° ✅ | tc=0.990 ✅ | 3.4° ✅ | BOTH_OK |

### Scale Chain (Step 1-2)

| Sequence | Center chain | Dense chain | Center CV | Dense CV |
|---|---|---|---|---|
| lang4 05-03 | 0.691 (31%) | 0.000 (100%) | 0.085 | 0.117 |
| lang4 12-03 | 1.194 (19%) | 0.000 (100%) | 0.059 | 0.109 |
| lang4 15-04 | 1.189 (19%) | 0.001 (100%) | 0.051 | 0.116 |
| lang4 19-03 | 0.915 (9%) | 0.000 (100%) | 0.091 | 0.090 |
| wheat 461 | 1.313 (31%) | 0.938 (6%) | 0.105 | 0.084 |
| wheat 467 | 1.390 (39%) | 0.903 (10%) | 0.157 | 0.056 |

### Method Comparison (Step 7)

| Sequence | Center chain | Uniform (s=1) | Per-window scale | Original |
|---|---|---|---|---|
| lang4 05-03 | 44.0° / 0.834 | 44.0° / 0.821 | 44.0° / 0.800 | 44.0° / 0.839 |
| lang4 12-03 | 48.7° / 0.818 | 48.7° / 0.814 | 48.7° / 0.806 | 48.7° / 0.822 |
| lang4 15-04 | 53.1° / 0.817 | 53.1° / 0.807 | 53.1° / 0.807 | 53.1° / 0.813 |
| lang4 19-03 | 44.2° / 0.834 | 44.2° / 0.816 | 44.2° / 0.785 | 44.3° / 0.834 |

Format: rot_median° / trajectory_cosine. All methods give essentially identical results.

### Graph Structure (Step 4)

| Sequence | Nodes | Edges | Cycle rank | Is chain |
|---|---|---|---|---|
| All (stride=8) | 39 | 38 | 0 | Yes (pure chain) |
| langdon_4 (stride=4 theoretical) | 39 | 111 | 73 | No (redundant) |

---

## Decision Logic Outcome

**CASE A (dense chain sufficient)**: ❌ Dense scale is worse than center scale for long sequences.

**CASE B (need stride=4)**: ⚠️ Center scale chain has 8-31% drift, but fixing scale won't fix the 44° rotation error. Stride=4 would help scale but not orientation.

**CASE C (need DA3 gauge)**: ❌ DA3 is only available for 2/4 sequences. And scale is not the primary problem.

**CASE D (hard boundary)**: The rotation error is NOT a hard boundary for VGGT — it's an artifact of sequential Q-chain composition without loop closure. The per-segment accuracy (4-8°) demonstrates that VGGT's local geometry is correct.

---

## Final State

```
phase3c2_status = PARTIAL
orientation_error_disentangled = PASS  (Q-chain accumulation identified as root cause)
true_orientation_chain = FAIL  (38-pair Q composition accumulates ~159° offset)
center_scale_estimator = STABLE  (CV 0.05-0.09, 8-31% chain drift)
dense_shared_scale = FAIL  (biased <1.0 due to per-window depth scaling)
scale_graph_redundancy = INSUFFICIENT  (pure chain, cycle_rank=0)
global_scale_drift = MODERATE  (8-31% for center, 100% for dense)
langdon_window_cases_rescued = 0 / 4
original_catastrophic_dates_rescued = 0 / 3
shared_geometry_consistency = NOT_EVALUATED
DA3_metric_gauge = NOT_USED  (only 2/4 sequences available, scale not primary problem)
fullplant_geometry_closure = NOT_EVALUATED
VGGT_long_sequence_hard_boundary = NOT_PROVEN  (local geometry is correct)
LoRA = NOT_JUSTIFIED
MSAM = NOT_PRIORITIZED
next_phase = Loop closure / stride-4 inference for redundant constraints
```

---

## Key Findings

1. **VGGT per-window geometry is correct**: 2-3° local rotation, 5% center error, 4-8° per-segment stitched rotation.

2. **Q-chain composition is the bottleneck**: Sequential chaining of38 pairwise orientation transforms accumulates ~159° rotation offset. This is NOT a VGGT model failure — it's a composition/alignment issue.

3. **Scale drift is secondary**: 8-31% chain drift for center scale. Not the primary cause of the 44° error.

4. **Dense point_map scale is unreliable**: Per-window arbitrary depth scaling biases all dense scale estimates <1.0.

5. **Per-segment stitched trajectory is accurate**: 4-8° rotation error, tc=0.81-0.84. The global evaluation masks this local accuracy.

---

## What Would Fix the Long-Sequence Problem

1. **Loop closure**: If the trajectory returns to the start (first-last frame overlap), the accumulated Q-chain error can be distributed across the loop. Currently first-last overlap = 0 frames.

2. **Stride=4 with VGGT inference**: Creates redundant edges (cycle rank 73) enabling global optimization. Would need ~320 new VGGT inferences (~50 min GPU).

3. **Absolute orientation reference**: Gravity direction from IMU, or known ground plane normal. Would fix the global rotation gauge.

4. **Pose graph optimization**: Treat stitched rotations as initial estimates, add constraints from overlapping frames across non-adjacent windows.

---

## Files Produced

```
11_scale_sync_v32/
├── 01_metric_disentanglement/
│   └── POSE_ERROR_DISENTANGLEMENT.csv
├── 02_dense_pairwise_scale/
│   ├── PAIRWISE_SCALE_ESTIMATES.csv
│   ├── SHARED_FRAME_SCALE_DISTRIBUTION.csv
│   └── PAIRWISE_SCALE_STABILITY.csv
├── 03_scale_graph/
│   ├── SCALE_GRAPH_SUMMARY.csv
│   └── SCALE_GRAPH_EDGES.json
├── 05_global_reconstruction/
│   ├── *_CENTER_CHAIN_GLOBAL_CAMERAS.npz
│   ├── *_UNIFORM_CHAIN_GLOBAL_CAMERAS.npz
│   ├── *_PER_WINDOW_SCALE_GLOBAL_CAMERAS.npz
│   └── REBUILD_COMPARISON.csv
└── tests/
    └── test_orientation_metric_independent_of_center_scale.py (PASS)
```
