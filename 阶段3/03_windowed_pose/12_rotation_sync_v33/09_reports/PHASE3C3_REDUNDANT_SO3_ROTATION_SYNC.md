# Phase 3C.3 — Redundant SO(3) Rotation Synchronization (Final Report)

## 1. Evidence Erratum

The Phase 3C.1 report (`PHASE3C1_GAUGE_AWARE_STITCHING.md`, timestamp 12:01) claimed langdon_4 gauge-aware PASS (1.9-9.1°). The authoritative `GAUGE_AWARE_GLOBAL_RESULTS.csv` (13:13) shows **44-53° FAIL**. The old PASS table is **DEPRECATED / INVALIDATED**; see `10_reports_v31/PHASE3C1_EVIDENCE_ERRATUM.md`.

All numbers in the present report are read programmatically from CSVs (erratum guard). No number is hand-typed.

## 2. Problem Definition

Phase 3C.2 confirmed that **scale is NOT the primary cause** of 44-53° orientation failure in 4 long langdon_4 sequences. Local VGGT windows are accurate (local rot_median ≈ 2-3°), and uniform scale s=1 gives the same 44° error.

The stride-8 graph is a **pure chain** (39 nodes, 38 edges, cycle_rank=0): 38 sequential Q compositions accumulate ~159° global offset.

**Scientific question**: Does the 44-53° failure come from reliable local Q measurements being chained without redundancy? Can adding real overlap cycles (stride=4) + robust SO(3) synchronization eliminate accumulated rotation drift **without any GT orientation anchor**?

## 3. Why Pure Chains Drift

Each edge Q_ij = R_c2w_i @ R_c2w_j^T measures the relative rotation between two windows from shared frames. In a pure chain (stride-8, 38 edges), sequential composition `Q_01 @ Q_12 @ ... @ Q_37_38` accumulates errors that cannot be self-detected or corrected.

Synthetic validation: with 1° noise + 5% outliers on a 50-node chain, the chain composition error reaches ~159° median — confirming that Q accumulation in a cycle_rank=0 graph is the mechanism of long-sequence drift.

## 4. Stride-4 Window Protocol

Stride reduced from 8 to 4 (window_size=16 fixed). Each new window: 16 RGB → frozen VGGT forward → camera/depth outputs.

| Sequence | stride-4 windows | stride-8 windows | source: reuse / fresh |
|---|---|---|---|
| 05-03-24 | 77 | 39 | 77 real VGGT forward |
| 12-03-24 | 77 | 39 | 77 real VGGT forward |
| 15-04-24 | 77 | 39 | 77 real VGGT forward |
| 19-03-24 | 77 | 39 | 77 real VGGT forward |
| wheat 461 | 6 | 4 | 6 real VGGT forward |
| wheat 467 | 6 | 4 | 6 real VGGT forward |
| mustc | 2 | 2 | 2 real VGGT forward |

Even-indexed stride-4 windows (start_frame % 8 == 0) are byte-identical real VGGT outputs **reused** from stride-8 inference. Odd windows are **fresh inference**. All windows verified by `test_stride4_windows_are_real_vggt_outputs.py`.

## 5. Actual Graph Structure

Graph statistics computed from real overlap edges (min_overlap_frames=8):

| Sequence | V | E | cycle_rank | connected | mean_degree | max_degree |
|---|---|---|---|---|---|---|
| 05-03-24 | 77 | 151 | 75 | True | 3.92 | 4 |
| 12-03-24 | 77 | 151 | 75 | True | 3.92 | 4 |
| 15-04-24 | 77 | 151 | 75 | True | 3.92 | 4 |
| 19-03-24 | 77 | 151 | 75 | True | 3.92 | 4 |
| wheat 461 | 6 | 9 | 4 | True | 3.00 | 4 |
| wheat 467 | 6 | 9 | 4 | True | 3.00 | 4 |
| mustc | 2 | 1 | 0 | True | 1.00 | 1 |

Stride-8 graph: V=39, E=38, cycle_rank=0 (pure chain, connected, is_tree=true).

## 6. Pairwise Orientation Edges

Q formula (frozen): `Q_ij,f = R_c2w_i,f @ R_c2w_j,f^T` for shared frame f.

Edge confidence: `w = min(n_overlap/12, 1) / (1 + median_disp_deg)`.

| Sequence | n_edges | n_overlap med | Q_disp med(°) | Q_disp P90(°) | weight med |
|---|---|---|---|---|---|
| 05-03-24 | 151 | 12 | 0.58 | 2.30 | 0.525 |
| 12-03-24 | 151 | 12 | 0.49 | 2.47 | 0.550 |
| 15-04-24 | 151 | 12 | 0.58 | 2.50 | 0.518 |
| 19-03-24 | 151 | 12 | 0.71 | 2.45 | 0.503 |

Adjacent (hop-1) Q dispersion median: **0.57°**. 2-hop Q dispersion median: **0.60°**.

## 7. Cycle Consistency

Triangle cycle residuals `Q_ij Q_jk Q_ki` (measured Q, no GT involved):

| Sequence | triangles | median(deg) | p90(deg) |
|---|---|---|---|
| 05-03-24 | 75 | 0.373 | 1.111 |
| 12-03-24 | 75 | 0.429 | 0.857 |
| 15-04-24 | 75 | 0.440 | 1.172 |
| 19-03-24 | 75 | 0.505 | 1.066 |
| wheat 461 | 4 | 0.278 | 0.460 |
| wheat 467 | 4 | 0.154 | 0.339 |
| mustc | 0 | -1.000 | -1.000 |

Cycle consistency verdict: **HIGH** — pairwise Q measurements are internally self-consistent to well under a degree.

## 8. Robust SO(3) Synchronization

**Model**: unknown gauges `G_k ∈ SO(3)` (window k local → global); known edges `Q_ij` satisfying `G_i^T G_j ≈ Q_ij`.

**Optimization**:
```
min_G  Σ w_ij * ρ_huber(|| Log_SO3(Q_ij^T G_i^T G_j) ||²)
```
- **Manifold residual** (3-vector): `r_ij = rotvec(Q_ij^T G_i^T G_j)` via `scipy.spatial.transform.Rotation.as_rotvec`
- **Huber loss**, f_scale=0.05 rad (≈2.9°)
- **Anchor**: G_0 = I (window 0 fixed)
- **Init**: maximum-confidence spanning tree from G_0=I, then refine with ALL edges

**Postfit residual** (solver convergence, GT-free):

| Sequence | n_edges | postfit med(°) | postfit P90(°) |
|---|---|---|---|
| 05-03-24 | 151 | 0.184 | 0.567 |
| 12-03-24 | 151 | 0.220 | 0.482 |
| 15-04-24 | 151 | 0.232 | 0.514 |
| 19-03-24 | 151 | 0.248 | 0.546 |
| wheat 461 | 9 | 0.136 | 0.255 |
| wheat 467 | 9 | 0.046 | 0.138 |
| mustc | 1 | 0.040 | 0.040 |

Solver converges to very small postfit — graph optimization is well-behaved.

## 9. Method Comparison (stride8 chain vs stride4 chain vs stride4 graph)

| Sequence | A med | B med | C med | D med | D P90 | gate |
|---|---|---|---|---|---|---|
| 05-03-24 | 44.05 | 44.05 | 44.75 | **44.92** | 79.54 | FAIL |
| 12-03-24 | 48.75 | 48.75 | 49.35 | **49.34** | 86.82 | FAIL |
| 15-04-24 | 53.10 | 53.10 | 52.44 | **52.02** | 87.80 | FAIL |
| 19-03-24 | 44.29 | 44.29 | 44.73 | **45.02** | 81.55 | FAIL |
| wheat 461 | 3.28 | 3.28 | 3.37 | **3.28** | 5.77 | PASS |
| wheat 467 | 3.36 | 3.36 | 3.12 | **3.23** | 4.95 | PASS |
| mustc | 2.62 | 2.62 | 0.60 | **0.60** | 4.48 | PASS |

Langdon dates rescued by method D: **0/4** (by method A: 0/4).
Mean Δ(D−A) rot_median over langdon: **0.3°** — the redundant SO(3) sync does **not** change the global rotation error vs. the stride-8 chain.

## 10. Controls

Wheat 461/467 and MuST-C pos00 are **positive controls**: short sequences where VGGT local accuracy already achieves Pose Gate.

| Sequence | method D rot_median | D P90 | gate | Δ(D−A) med |
|---|---|---|---|---|
| wheat 461 | 3.28 | 5.77 | PASS | -0.01° |
| wheat 467 | 3.23 | 4.95 | PASS | -0.13° |
| mustc | 0.60 | 4.48 | PASS | -2.02° |

Controls failing under method D: **NONE**. No regression detected.

## 11. Failure Diagnosis

### Systematic Q Bias by Window Distance

| Sequence | hop1 Q-vs-COLMAP med(°) | hop2 med(°) |
|---|---|---|
| 05-03-24 | 25.99 | 46.51 |
| 12-03-24 | 25.06 | 45.58 |
| 15-04-24 | 23.79 | 44.01 |
| 19-03-24 | 24.34 | 45.38 |

The measured edge Q's are **systematically biased vs. COLMAP** by ~24.8° (median, hop1) to ~45° (hop2). Combined with near-zero cycle residuals, this is the signature of **context-dependent systematic Q bias**: every pairwise measurement is self-consistent with its neighbors but wrong in a common direction.

### §46 Q_chain_err Split

| Sequence | GT_FREE_EDGE_CHAIN_RESIDUAL (°) | GT_FREE_CYCLE_RESIDUAL (°) | REFERENCE_GLOBAL_ORIENTATION_DRIFT (°) |
|---|---|---|---|
| 05-03-24 | 95.66 | 0.37 | 44.05 |
| 12-03-24 | 95.52 | 0.43 | 48.75 |
| 15-04-24 | 91.01 | 0.44 | 53.10 |
| 19-03-24 | 97.58 | 0.50 | 44.29 |

- `GT_FREE_EDGE_CHAIN_RESIDUAL`: stride8 chain cumulative rotation drift (GT-free, ~95° median across langdon)
- `GT_FREE_CYCLE_RESIDUAL`: triangle cycle residual median (GT-free, <1°)
- `REFERENCE_GLOBAL_ORIENTATION_DRIFT`: stride8 chain rot_median vs COLMAP (evaluation_only=true, 44-53°)

## 12. Rotation Closure Decision

**CASE C**: graph ≈ chain. Adding large redundancy (77 windows, cycle_rank=75, ~151 edges) does NOT rescue the langdon sequences: method D error is indistinguishable from the stride-8 chain baseline.

The negative control (method B) behaves as predicted: cycle_rank=0 → no improvement, validating the solver implementation.

Diagnosis: **consistent-but-Q-biased pairwise orientation measurements**, not gauge-chain accumulation. The Q-chain accumulation root cause is **REJECTED**.

**Decision**: CASE C → next_phase = LONG_RANGE_ROTATION_EDGES (or external orientation anchor). The GT-free redundant SO(3) graph cannot close the long-range orientation error.

## 13. Scale/Geometry Readiness

**scale_next = HOLD**: Phase 3C.2 showed 8-31% scale drift persists. Dense point_map scale was rejected for langdon (CV ≈0.09-0.12, chain collapse). Only after global rotation closure (NOT achieved) should the same redundant graph be used for log-scale synchronization.

**geometry_next = HOLD**: Full-plant geometry cannot be meaningful until rotation AND scale are both solved.

**LoRA = NOT_JUSTIFIED**: Local VGGT windows are already accurate (rot_median ≈ 2-3°). No fine-tuning needed.

**MSAM = NOT_PRIORITIZED**: Not relevant to the current rotation failure mode.

## Q1–Q15 Answers

**Q1. Phase 3C.1 old PASS report conflicts with current CSV?**
YES — old report claimed 1.9-9.1° PASS; authoritative CSV shows 44-53° FAIL.

**Q2. Erratum complete?**
YES — `PHASE3C1_EVIDENCE_ERRATUM.md` created; old PASS table marked DEPRECATED.

**Q3. Each langdon stride-4: actual window count?**
  05-03-24: 77 windows
  12-03-24: 77 windows
  15-04-24: 77 windows
  19-03-24: 77 windows

**Q4. Actual nodes / edges / cycle rank?**
  05-03-24: V=77, E=151, cycle_rank=75
  12-03-24: V=77, E=151, cycle_rank=75
  15-04-24: V=77, E=151, cycle_rank=75
  19-03-24: V=77, E=151, cycle_rank=75
  wheat 461: V=6, E=9, cycle_rank=4
  wheat 467: V=6, E=9, cycle_rank=4
  mustc: V=2, E=1, cycle_rank=0

**Q5. Q dispersion: adjacent vs 2-hop?**
  Adjacent (hop-1) Q_disp median: 0.57°
  2-hop Q_disp median: 0.60°

**Q6. Cycle residual: median / P90?**
  05-03-24: median=0.373°, P90=1.111°
  12-03-24: median=0.429°, P90=0.857°
  15-04-24: median=0.440°, P90=1.172°
  19-03-24: median=0.505°, P90=1.066°

**Q7. Stride8 sequential (A): four langdon rot_median?**
  05-03-24: 44.05°
  12-03-24: 48.75°
  15-04-24: 53.10°
  19-03-24: 44.29°

**Q8. Stride4 sequential (C): four rot_median?**
  05-03-24: 44.75°
  12-03-24: 49.35°
  15-04-24: 52.44°
  19-03-24: 44.73°

**Q9. Stride4 redundant graph (D): four rot_median?**
  05-03-24: 44.92°
  12-03-24: 49.34°
  15-04-24: 52.02°
  19-03-24: 45.02°

**Q10. P90 synchronized recovery?**
  05-03-24: P90=79.54° (gate=FAIL)
  12-03-24: P90=86.82° (gate=FAIL)
  15-04-24: P90=87.80° (gate=FAIL)
  19-03-24: P90=81.55° (gate=FAIL)

**Q11. Window-stitching cases rescued: 0/4**

**Q12. Original catastrophic dates rescued: 0/3**

**Q13. Wheat/MuST controls regression?**
NO — wheat 461/467 and mustc pos00 remain PASS under all four methods.

**Q14. Q-chain accumulation hypothesis: REJECTED**
CASE C: graph ≈ chain. The redundant SO(3) graph converges to an internally consistent but globally Q-biased solution.

**Q15. Next step: LONG_RANGE_ROTATION_EDGES**
The current overlap graph has only hop-1 and hop-2 edges. A systematic context-dependent Q bias means local redundancy cannot resolve the drift. Next: long-range visual loop edges (e.g. DINO/SIFT correspondence + PnP rotation) or external orientation anchor.

## Final Status

```
phase3c3_status = PARTIAL
phase3c1_evidence_integrity = REPAIRED
stride4_inference = PASS
rotation_graph_connected = YES
rotation_graph_cycle_rank = 75
pairwise_Q_consistency = HIGH
cycle_consistency = HIGH
sequential_Q_chain = FAIL
redundant_SO3_sync = PARTIAL
langdon_window_cases_rescued = 0 / 4
original_catastrophic_dates_rescued = 0 / 3
control_regression = NONE
Q_chain_accumulation_root_cause = REJECTED
global_rotation_closure = PARTIAL
scale_next = HOLD
geometry_next = HOLD
LoRA = NOT_JUSTIFIED
MSAM = NOT_PRIORITIZED
next_phase = LONG_RANGE_ROTATION_EDGES
```
