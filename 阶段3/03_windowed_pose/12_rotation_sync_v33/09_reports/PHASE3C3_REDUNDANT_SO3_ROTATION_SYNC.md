# Phase 3C.3 — Redundant SO(3) Rotation Synchronization (Final Report)

## 1. Evidence Integrity

The Phase 3C.1 report (`PHASE3C1_GAUGE_AWARE_STITCHING.md`, timestamp 12:01) claimed langdon_4 gauge-aware PASS (1.9-9.1°). The authoritative `GAUGE_AWARE_GLOBAL_RESULTS.csv` (13:13) shows **44-53° FAIL**. The old PASS is **DEPRECATED / INVALIDATED**; see `10_reports_v31/PHASE3C1_EVIDENCE_ERRATUM.md`. All numbers in the present report are read programmatically from CSVs (erratum guard).

## 2. Methods

| Method | Windows | Graph | Description |
|---|---|---|---|
| **A** | 39 | chain (cycle_rank=0) | 39-window pure chain (cycle_rank=0), 38 sequential Q compositions |
| **B** | 39 | chain (cycle_rank=0) | stride-8 SO(3) sync — negative control, cycle_rank=0 |
| **C** | 77 | chain (cycle_rank=0) | 77-window chain — tests 'more windows alone' |
| **D** | 77 | redundant (cycle_rank>0) | 77-window redundant SO(3) sync — headline |

All methods use the same frozen Q formula `Q_ij,f = R_c2w_i,f @ R_c2w_j,f^T`; COLMAP rotations are used ONLY for final evaluation, never in the solver.

## 3. Rotation Graph Structure (real)

Headline stride-4 graph: V=77, E=151, cycle_rank=75, connected=True. The stride-8 graph has cycle_rank=0 (pure chain).

## 4. Cycle Consistency (GT-free)

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

Cycle consistency verdict: **HIGH** — the pairwise Q measurements are internally self-consistent to well under a degree.

## 5. Postfit Residual (solver convergence, GT-free)

| Sequence | n_edges | postfit med(deg) | postfit p90(deg) |
|---|---|---|---|
| 05-03-24 | 151 | 0.184 | 0.567 |
| 12-03-24 | 151 | 0.220 | 0.482 |
| 15-04-24 | 151 | 0.232 | 0.514 |
| 19-03-24 | 151 | 0.248 | 0.546 |
| wheat 461 | 9 | 0.136 | 0.255 |
| wheat 467 | 9 | 0.046 | 0.138 |
| mustc | 1 | 0.040 | 0.040 |

The SO(3) sync solver converges to a **very small** postfit residual — the graph optimization itself is well-behaved.

## 6. Method Comparison (orientation-only, COLMAP reference)

| Sequence | A med | B med | C med | D med | D p90 | gate |
|---|---|---|---|---|---|---|
| 05-03-24 | 44.05 | 44.05 | 44.75 | **44.92** | 79.54 | FAIL |
| 12-03-24 | 48.75 | 48.75 | 49.35 | **49.34** | 86.82 | FAIL |
| 15-04-24 | 53.10 | 53.10 | 52.44 | **52.02** | 87.80 | FAIL |
| 19-03-24 | 44.29 | 44.29 | 44.73 | **45.02** | 81.55 | FAIL |
| wheat 461 | 3.28 | 3.28 | 3.37 | **3.28** | 5.77 | PASS |
| wheat 467 | 3.36 | 3.36 | 3.12 | **3.23** | 4.95 | PASS |
| mustc | 2.62 | 2.62 | 0.60 | **0.60** | 4.48 | PASS |

Langdon dates rescued by method D: **0/4** (by method A: 0/4). Controls failing under D: NONE.

Mean Δ(D−A) rot_median over langdon: **0.3°** — the redundant SO(3) sync does **not** change the global rotation error vs. the stride-8 chain.

## 7. Systematic Q Bias by Window Distance (hop)

| Sequence | hop1 Q-vs-COLMAP med(deg) | hop2 med(deg) |
|---|---|---|
| 05-03-24 | 25.99 | 46.51 |
| 12-03-24 | 25.06 | 45.58 |
| 15-04-24 | 23.79 | 44.01 |
| 19-03-24 | 24.34 | 45.38 |

The measured edge Q's are **systematically biased vs. COLMAP** by ~24.8° (median, hop1) to ~45° (hop2). Combined with near-zero cycle residuals, this is the signature of **context-dependent systematic Q bias**: every pairwise measurement is self-consistent with its neighbors but wrong in a common direction. A redundant graph can remove noise/outliers, but **cannot remove a bias shared by all measurements** — it merely reproduces the same consistent-yet-biased gauge, which is why D ≈ C ≈ A.

## 8. Conclusion

**Q_chain_accumulation root cause: REJECTED.** Adding large redundancy (77 windows, cycle_rank=75, ~151 edges) does NOT rescue the langdon sequences: method D error is indistinguishable from the stride-8 chain baseline. The negative control (B) behaves exactly as predicted (cycle_rank=0 → no change), which validates the solver. The diagnosis is therefore **consistent-but-Q-biased pairwise orientation measurements**, not gauge-chain accumulation.

Global rotation closure: **PARTIAL** — the GT-free redundant SO(3) graph cannot close the long-range orientation error.

## 9. Final Status

```
phase3c3_status = PARTIAL
phase3c1_evidence_integrity = REPAIRED
stride4_inference = PASS
rotation_graph_connected = YES
rotation_graph_cycle_rank = 75
pairwise_Q_consistency = HIGH (cycle residual ~0.50°)
cycle_consistency = HIGH
sequential_Q_chain = FAIL
redundant_SO3_sync = PARTIAL (converges cleanly, no rescue)
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

Decision case: **CASE C** — graph ≈ chain. The pairwise Q measurements are internally consistent (cycle residual < 1°) but systematically biased vs COLMAP. Next avenue is **long-range rotation edges** or an **external orientation anchor** (GT), both of which are outside the current GT-free graph-only scope.
