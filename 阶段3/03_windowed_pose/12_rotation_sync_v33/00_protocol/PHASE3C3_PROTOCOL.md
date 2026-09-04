# Phase 3C.3 Protocol — Redundant SO(3) Rotation Synchronization

## Hypothesis (Q1)

The 44-53° long-sequence rotation error comes from reliable local Q measurements
being chained (38 sequential compositions → ~159° drift) **without redundancy**.
Adding real overlap cycles (stride=4, cycle_rank=75) + robust SO(3) graph
synchronization should eliminate accumulated rotation drift **without GT anchor**.

## Frozen Formulas

### Pairwise Q
For shared frame f:
```
Q_ij,f = R_c2w_i,f @ R_c2w_j,f^T
```
Both are VGGT-extracted c2w rotations, **frozen** since Phase 3C.1.

### Edge Q (robust mean)
```
Q_ij = so3_robust_mean([Q_ij,f])   # quaternion avg + MAD rejection
```
Reuses `so3_robust_mean()` from `run_gauge_stitching.py`.

### Edge confidence
```
w_overlap = min(n_overlap / 12, 1)
w_disp    = 1 / (1 + median_disp_deg)
w_ij      = w_overlap * w_disp
```

### SO(3) sync residual
For edge (i,j) with synced gauge rotations G_i, G_j:
```
E_ij = Q_ij^T @ G_i^T @ G_j
r_ij = Log_SO3(E_ij) = Rotation.from_matrix(E_ij).as_rotvec()  ∈ R^3
```
**Manifold residual** (NOT 9 matrix elements).

## Optimization Model

```
min_G  Σ_(i,j)  w_ij * ρ_huber( ||r_ij||_2 / f_scale )
subject to  G_0 = I   (anchor)
```

| Parameter | Value | Source |
|---|---|---|
| Huber f_scale | 2-5° (0.035-0.087 rad) | Protocol, not tuned |
| Anchor | G_0 = I | Spec §17 |
| Init | Maximum-spanning-tree (by edge weight) from anchor | Spec §17 |
| Solver | scipy.optimize.least_squares, loss='huber', method='trf' | Available |

## Prohibitions

| # | Prohibition | Justification |
|---|---|---|
| 1 | LoRA / VGGT fine-tuning | Spec §0 |
| 2 | GT pose in graph optimization | GT-free requirement |
| 3 | COLMAP rotation in sync | Only for final evaluation |
| 4 | Scanner GT, IMU, gravity, known orientation | Not available |
| 5 | DA3, MSAM | Not in pipeline |
| 6 | Geometry reconstruction | Orientation-only headline |
| 7 | Depth scale re-study | Completed in Phase 3C.2 |
| 8 | Pure chain disguised as drift correction | Redundant cycles required |
| 9 | Interpolated/fabricated stride-4 windows | Real VGGT forward only |
| 10 | PASS from loss decrease alone | Independent reference eval required |

## Success Criteria

- ≥3/4 langdon dates pass Pose Gate (rot_median ≤ 10°, P90 ≤ 20°)
- Controls (wheat/mustc) no regression
- Original catastrophic dates rescued: X/3 (12-03, 15-04, 19-03)

## Negative Control

Stride-8 SO(3) graph (cycle_rank=0) must show **no improvement** over chain,
validating redundancy as the mechanism.

## Evidence Integrity

All report numbers must be **read from CSV programmatically** (never hand-copied).
Phase 3C.1 PASS claims invalidated — see `10_reports_v31/PHASE3C1_EVIDENCE_ERRATUM.md`.
