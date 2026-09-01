# Phase 3C Protocol — Windowed Global Pose Reconstruction

**Date**: 2026-09-01
**Branch**: POSE_ROBUSTNESS
**Status**: ACTIVE

## Core Research Question

How to maintain local VGGT pose stability via overlapping windows
and stitch them into a globally consistent full-plant trajectory,
without reference poses or LoRA?

## Sequences

### P0 FAIL targets
- langdon_4 12-03-24 (320 frames)
- langdon_4 19-03-24 (320 frames)
- langdon_4 15-04-24 (318 frames)

### Controls (must not degrade)
- langdon_4 05-03-24 (320 frames, PASS)
- wheat3dgs plot_461 (36 frames, PASS)
- wheat3dgs plot_467 (36 frames, PASS)
- mustc plot198 pos00 (20 frames, PASS)

## Three comparison modes

| Mode | Description | Coverage |
|------|-------------|----------|
| A. Uniform sparse | np.linspace every Nth frame | Full trajectory, non-local |
| B. Consecutive local | Single 16-frame window | Partial, local only |
| C. Windowed full | Overlapping windows stitched | Full trajectory, local overlap |

The primary comparison is **A vs C**.
B is shown only to prove local recoverability.

## Phase 3C Phases

### P0: Stride Sensitivity (Sections 4-11)
- Strides: [1, 2, 4, 6, 8, 10, 12, 16, 20]
- 5 start offsets per stride (0%, 20%, 40%, 60%, 80%)
- Independent VGGT forward per (stride, start) combination
- Output: STRIDE_POSE_RESULTS.csv

### P1: Windowed Full-Trajectory (Sections 12-24)
- Config A: window=16, overlap=8 (first priority)
- Config B: window=16, overlap=12 (if A fails)
- Config C: window=24, overlap=12 (if B fails)
- Window alignment: Sim(3) from camera centers
- Global stitching: sequential propagation

### P2: Geometry Evaluation (Sections 31-38)
- VGGT depth unprojection with stitched cameras
- Chamfer, F-score at multiple thresholds
- Window seam diagnostics

### P3: Evaluation & Reporting (Sections 28-65)
- Anti-regression on PASS controls
- Full Q1-Q15 answering
- Final architecture decision

## Constraints
- Frozen VGGT (no LoRA, no weight modification)
- No reference poses during stitching (only for final evaluation)
- No scanner/GT geometry for alignment
- All reports from CSVs
