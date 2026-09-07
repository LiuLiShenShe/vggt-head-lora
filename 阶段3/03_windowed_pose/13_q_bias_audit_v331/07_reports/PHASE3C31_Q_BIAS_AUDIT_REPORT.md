# Phase 3C.3.1 — Q-Bias Diagnostic Repair + Window Gauge-Equivariance Audit

**Generated:** 2026-09-07
**Root cause of 3C.3 Q-bias claim:** Procrustes implementation bug in `rotation_diagnostics.py:231`.
**Frozen correct formula:** `evaluate_multoplant.py:36-44`.

---

## §28 Evidence Erratum

Phase 3C.3 §7 concluded 'systematic Q bias: hop1 ≈ 25°, hop2 ≈ 45°'
based on the old buggy formula in `rotation_diagnostics.py:231`:

```python
H = np.einsum('sij,sik->jk', R_refk, R_local)   # WRONG: = Σ R_ref^T @ R_local
```

This computes `Σ R_ref^T @ R_local`, which is NOT the Procrustes objective
`Σ || G @ R_local - R_ref ||_F` that the frozen correct formula solves:

```python
H = np.einsum('sij,skj->ik', R_local, R_ref)    # CORRECT: = Σ R_local @ R_ref^T
```

Synthetic verification (this audit, test §3-§4):
- Correct formula on exact data: **0°** error (exact recovery)
- Old formula on same data: **85.6°** error

### Corrected Q-vs-reference values (this audit)

| Sequence | hop1 Q-vs-ref median | hop2 Q-vs-ref median | old claim |
|---|---|---|---|
| plantview__langdon_4__05-03-24 | 1.42° | 4.96° | ~26.0° (bug) |
| plantview__langdon_4__12-03-24 | 1.71° | 5.42° | ~25.1° (bug) |
| plantview__langdon_4__15-04-24 | 2.72° | 4.92° | ~23.8° (bug) |
| plantview__langdon_4__19-03-24 | 1.45° | 4.86° | ~24.3° (bug) |
| wheat3dgs__plot_461 | 0.99° | 2.49° | ~16.8° (bug) |
| wheat3dgs__plot_467 | 0.86° | 2.13° | ~17.9° (bug) |
| mustc__plot198__230613__ugv__pos00 | 0.21° | N/A° | ~120.4° (bug) |

- **plantview__langdon_4__05-03-24: hop1 = 1.42°, hop2 = 4.96°** (corrected)
- **plantview__langdon_4__12-03-24: hop1 = 1.71°, hop2 = 5.42°** (corrected)
- **plantview__langdon_4__15-04-24: hop1 = 2.72°, hop2 = 4.92°** (corrected)
- **plantview__langdon_4__19-03-24: hop1 = 1.45°, hop2 = 4.86°** (corrected)

**Erratum status:** The old 25°/45° hop values were entirely an artifact of the
Procrustes formula bug. With the corrected formula, hop1 = 1.4–2.7° and hop2 = 4.9–5.4°.
The 'systematic Q bias' claim from Phase 3C.3 is RETRACTED.

However, the **observed rot_median failure (44–53°) is real** and persists.
The corrected audit identifies a different root cause (see §29).

---

## §29 Root Cause Analysis: Temporally-Coherent Per-Edge Drift

The corrected analysis reveals the mechanism behind the 44–53° rotation failure:

### 2.1 Per-edge Q accuracy is good

With corrected Procrustes, hop1 Q-vs-reference median error is 1.4–2.7°
across all langdon sequences. The per-window single-gauge assumption is valid:

- plantview__langdon_4__05-03-24: gauge fit residual median = 1.77°, Q-vs-ref hop1 = 1.42°
- plantview__langdon_4__12-03-24: gauge fit residual median = 2.15°, Q-vs-ref hop1 = 1.71°
- plantview__langdon_4__15-04-24: gauge fit residual median = 2.70°, Q-vs-ref hop1 = 2.72°
- plantview__langdon_4__19-03-24: gauge fit residual median = 2.15°, Q-vs-ref hop1 = 1.45°

### 2.2 Gauge-invariant relative rotations are accurate

Within-window frame-pair relative rotations (gauge-free, requiring NO Procrustes):

- plantview__langdon_4__05-03-24: sep=1 relative error median = 0.726°
- plantview__langdon_4__12-03-24: sep=1 relative error median = 0.803°
- plantview__langdon_4__15-04-24: sep=1 relative error median = 0.812°
- plantview__langdon_4__19-03-24: sep=1 relative error median = 0.745°
- wheat3dgs__plot_461: sep=1 relative error median = 2.073°
- wheat3dgs__plot_467: sep=1 relative error median = 1.574°
- mustc__plot198__230613__ugv__pos00: sep=1 relative error median = 1.436°

### 2.3 The errors are temporally coherent, not random

The key finding: per-step Q errors along the hop-1 chain accumulate
MONOTONICALLY (not as a random walk), producing the trajectory drift. Table below uses the corrected gauges as the reference trajectory:

| Sequence | steps | per-step median | per-step mean | random-walk estimate | observed final drift | amplification | coherence C |
|---|---|---|---|---|---|---|---|
| plantview__langdon_4__05-03-24 | 76 | 1.422° | 2.493° | 12.4° | 161.6° | 13.0× | 0.76 |
| plantview__langdon_4__12-03-24 | 76 | 1.708° | 2.716° | 14.9° | 174.6° | 11.7× | 0.76 |
| plantview__langdon_4__15-04-24 | 76 | 2.720° | 2.681° | 23.7° | 179.4° | 7.6× | 0.79 |
| plantview__langdon_4__19-03-24 | 76 | 1.447° | 2.527° | 12.6° | 161.2° | 12.8× | 0.76 |
| wheat3dgs__plot_461 | 5 | 0.992° | 1.641° | 2.2° | 4.4° | 2.8× | 0.53 |
| wheat3dgs__plot_467 | 5 | 0.855° | 1.282° | 1.9° | 3.3° | 2.5× | 0.51 |
| mustc__plot198__230613__ugv__pos00 | 1 | 0.209° | 0.209° | 0.2° | 0.2° | 1.0× | 1.00 |

For a 76-step chain with step error ~1.4–2.7°:
- Random walk expectation: sqrt(76) × 1.4° ≈ **12°** total drift
- Observed trajectory drift: **161–179°** (up to 179° by window 76)
- Temporal coherence ratio (|Σ rotvec| / Σ|rotvec|): **0.76** (0 = random, 1 = fully aligned)
- Drift amplification over random walk: **7.6–13×**

This means per-step errors are **76% co-directed** — they rotate the trajectory
in nearly the same direction at each hop. The drift grows approximately linearly;
for langdon_05-03: window 0: 0°, window 20: ~37°, window 40: ~86°, window 60: ~124°, window 76: ~162°.

### 2.4 Graph sync cannot correct coherent drift

- plantview__langdon_4__05-03-24: chain-vs-graph = 0.233° (graph ≈ chain), chain-vs-ref = 77.5°
- plantview__langdon_4__12-03-24: chain-vs-graph = 0.387° (graph ≈ chain), chain-vs-ref = 88.7°
- plantview__langdon_4__15-04-24: chain-vs-graph = 0.526° (graph ≈ chain), chain-vs-ref = 94.1°
- plantview__langdon_4__19-03-24: chain-vs-graph = 0.509° (graph ≈ chain), chain-vs-ref = 80.0°
- wheat3dgs__plot_461: chain-vs-graph = 0.060° (graph ≈ chain), chain-vs-ref = 4.9°
- wheat3dgs__plot_467: chain-vs-graph = 0.109° (graph ≈ chain), chain-vs-ref = 3.7°
- mustc__plot198__230613__ugv__pos00: chain-vs-graph = 0.020° (graph ≈ chain), chain-vs-ref = 0.1°

The sequential chain (sequential Q composition) and the SO(3) graph-sync solution
differ by < 1° — **graph sync does not correct the chain drift** (CASE C confirmed).
Graph sync enforces Q_ij consistency, but when all Q edges share a coherent bias,
the redundant graph cycles cannot correct it: cycle residual is small (~0.3–0.4°)
because the biased Q edges are internally consistent.

### 2.5 Boundary vs center frame analysis

- plantview__langdon_4__05-03-24: boundary err = 2.80°, center err = 1.81°, ratio = 1.55x
- plantview__langdon_4__12-03-24: boundary err = 3.33°, center err = 2.20°, ratio = 1.51x
- plantview__langdon_4__15-04-24: boundary err = 4.03°, center err = 2.06°, ratio = 1.96x
- plantview__langdon_4__19-03-24: boundary err = 3.08°, center err = 2.01°, ratio = 1.53x

Boundary frames (near window edges, seeing fewer context frames) have ~1.5–2.0×
the per-frame orientation error of center frames. However, the edge overlap
guarantees all edge constraints cover the final camera region
(§16: 0/151 edges have zero center overlap).

### 2.6 Cross-window context variation

- plantview__langdon_4__05-03-24: 312 frames in ≥2 windows, dispersion median = 6.12°
- plantview__langdon_4__12-03-24: 312 frames in ≥2 windows, dispersion median = 7.54°
- plantview__langdon_4__15-04-24: 312 frames in ≥2 windows, dispersion median = 7.70°
- plantview__langdon_4__19-03-24: 312 frames in ≥2 windows, dispersion median = 6.72°

Same image appearing in multiple 16-frame windows shows ~6–8° dispersion
in its gauge-aligned orientation prediction. This is a real context-dependence
signal, but its magnitude (~6°) is dwarfed by the accumulated trajectory drift (~160°).

### Root cause summary

| Mechanism | Magnitude | Role |
|---|---|---|
| Hop-1 Q measurement error | 1.4–2.7° per edge | Local noise |
| Temporal coherence of Q errors | 0.76 (high) | Drives accumulation |
| Cross-window same-frame dispersion | 6–8° median | Minor context signal |
| Boundary vs center frame error | 1.5–2.0× ratio | Minor structural factor |
| Accumulated trajectory drift | 160–179° over 76 windows | **Primary failure mode** |
| Final rot_median (graph sync) | 44–53° (aligned evaluation) | **What the user sees** |

**The root cause is temporally-coherent accumulation of small per-edge errors,
NOT context-dependent Q bias. The corrected Procrustes shows per-edge errors are small
(1.4–2.7°) but coherent (0.76 alignment ratio), producing near-linear drift over 76 windows.**

---

## §30 Q1–Q15 Answers

### Q1: Did the Phase 3C.3 Q-bias claim rely on a buggy formula?
**YES.** The Procrustes formula in `rotation_diagnostics.py:231` was incorrect.
The claimed hop1 ≈ 25° / hop2 ≈ 45° were artifacts of `einsum('sij,sik->jk', ...)`.
Corrected values: hop1 = 1.4–2.7°, hop2 = 4.9–5.4°.

### Q2: Is the corrected formula validated?
**YES.** Synthetic exact-recovery test: 0° error (corrected), 85.6° error (old).
5 trials with distinct random gauges, 3 trials with 1° noise — all pass.

### Q3: Per-window single-gauge fit quality

- plantview__langdon_4__05-03-24: median fit residual = 1.77°, max = 8.98°
- plantview__langdon_4__12-03-24: median fit residual = 2.15°, max = 8.96°
- plantview__langdon_4__15-04-24: median fit residual = 2.70°, max = 7.58°
- plantview__langdon_4__19-03-24: median fit residual = 2.15°, max = 8.91°
- wheat3dgs__plot_461: median fit residual = 2.65°, max = 4.11°
- wheat3dgs__plot_467: median fit residual = 2.31°, max = 3.76°
- mustc__plot198__230613__ugv__pos00: median fit residual = 1.35°, max = 1.79°

The single-gauge model is valid: ~1.8–3.4° median residual per frame within each window.

### Q4: Corrected Q-vs-reference per hop

- plantview__langdon_4__05-03-24: hop1 median = 1.42°, hop2 median = 4.96°
- plantview__langdon_4__12-03-24: hop1 median = 1.71°, hop2 median = 5.42°
- plantview__langdon_4__15-04-24: hop1 median = 2.72°, hop2 median = 4.92°
- plantview__langdon_4__19-03-24: hop1 median = 1.45°, hop2 median = 4.86°
- wheat3dgs__plot_461: hop1 median = 0.99°, hop2 median = 2.49°
- wheat3dgs__plot_467: hop1 median = 0.86°, hop2 median = 2.13°
- mustc__plot198__230613__ugv__pos00: hop1 median = 0.21°, hop2 median = N/A°


### Q5: Gauge-invariant relative rotation accuracy (within-window)

- plantview__langdon_4__05-03-24: sep=1 median = 0.726°, overall median = 1.949°, p90 = 17.176°
- plantview__langdon_4__12-03-24: sep=1 median = 0.803°, overall median = 2.541°, p90 = 17.617°
- plantview__langdon_4__15-04-24: sep=1 median = 0.812°, overall median = 2.683°, p90 = 17.944°
- plantview__langdon_4__19-03-24: sep=1 median = 0.745°, overall median = 2.127°, p90 = 17.455°
- wheat3dgs__plot_461: sep=1 median = 2.073°, overall median = 3.112°, p90 = 8.115°
- wheat3dgs__plot_467: sep=1 median = 1.574°, overall median = 2.523°, p90 = 7.366°
- mustc__plot198__230613__ugv__pos00: sep=1 median = 1.436°, overall median = 1.804°, p90 = 5.193°

### Q6: Cycle consistency (GT-free)

- plantview__langdon_4__05-03-24: cycle residual median = 0.373°, postfit median = 0.184°
- plantview__langdon_4__12-03-24: cycle residual median = 0.429°, postfit median = 0.220°
- plantview__langdon_4__15-04-24: cycle residual median = 0.440°, postfit median = 0.232°
- plantview__langdon_4__19-03-24: cycle residual median = 0.505°, postfit median = 0.248°
- wheat3dgs__plot_461: cycle residual median = 0.278°, postfit median = 0.136°
- wheat3dgs__plot_467: cycle residual median = 0.154°, postfit median = 0.046°
- mustc__plot198__230613__ugv__pos00: cycle residual median = N/A°, postfit median = 0.040°


### Q7: Graph-sync vs chain (method D vs C)

- plantview__langdon_4__05-03-24: chain rot_median = 44.0°, graph rot_median = 44.9°
- plantview__langdon_4__12-03-24: chain rot_median = 48.7°, graph rot_median = 49.3°
- plantview__langdon_4__15-04-24: chain rot_median = 53.1°, graph rot_median = 52.0°
- plantview__langdon_4__19-03-24: chain rot_median = 44.3°, graph rot_median = 45.0°
- wheat3dgs__plot_461: chain rot_median = 3.3°, graph rot_median = 3.3°
- wheat3dgs__plot_467: chain rot_median = 3.4°, graph rot_median = 3.2°
- mustc__plot198__230613__ugv__pos00: chain rot_median = 2.6°, graph rot_median = 0.6°


### Q8: Graph gauge vs GT gauge trajectory

- plantview__langdon_4__05-03-24: global-A alignment median = 43.2°, max = 83.7°
- plantview__langdon_4__12-03-24: global-A alignment median = 48.9°, max = 89.4°
- plantview__langdon_4__15-04-24: global-A alignment median = 50.5°, max = 92.2°
- plantview__langdon_4__19-03-24: global-A alignment median = 44.8°, max = 82.8°
- wheat3dgs__plot_461: global-A alignment median = 1.2°, max = 3.9°
- wheat3dgs__plot_467: global-A alignment median = 1.1°, max = 3.2°
- mustc__plot198__230613__ugv__pos00: global-A alignment median = 0.1°, max = 0.1°

For langdon: the graph-gauge trajectory shape differs from GT by 43–51° median
(even after optimal global rotation alignment). Wheat/mustc: 1–4°.

### Q9: Sequential chain composition drift

Chain (Q-edge composition from GT anchor) vs GT gauges:

- plantview__langdon_4__05-03-24: chain-vs-ref median = 77.5°, chain-vs-graph median = 0.233°
- plantview__langdon_4__12-03-24: chain-vs-ref median = 88.7°, chain-vs-graph median = 0.387°
- plantview__langdon_4__15-04-24: chain-vs-ref median = 94.1°, chain-vs-graph median = 0.526°
- plantview__langdon_4__19-03-24: chain-vs-ref median = 80.0°, chain-vs-graph median = 0.509°
- wheat3dgs__plot_461: chain-vs-ref median = 4.9°, chain-vs-graph median = 0.060°
- wheat3dgs__plot_467: chain-vs-ref median = 3.7°, chain-vs-graph median = 0.109°
- mustc__plot198__230613__ugv__pos00: chain-vs-ref median = 0.1°, chain-vs-graph median = 0.020°


### Q10: Temporal coherence of per-edge errors

For all 4 langdon sequences, coherence ratio (|Σ rotvec|/Σ|rotvec|) ≈ 0.76.
Per-step axes have intermediate pairwise dot-product (0.3–0.6): the step error
axes are not perfectly constant but are sufficiently aligned to drive near-linear
accumulation. The total accumulated rotvec vector sum is ~143° over 76 steps.

### Q11: Cross-window context variation

- plantview__langdon_4__05-03-24: 312 frames in ≥2 windows, dispersion median = 6.12°
- plantview__langdon_4__12-03-24: 312 frames in ≥2 windows, dispersion median = 7.54°
- plantview__langdon_4__15-04-24: 312 frames in ≥2 windows, dispersion median = 7.70°
- plantview__langdon_4__19-03-24: 312 frames in ≥2 windows, dispersion median = 6.72°
- wheat3dgs__plot_461: 28 frames in ≥2 windows, dispersion median = 3.54°
- wheat3dgs__plot_467: 28 frames in ≥2 windows, dispersion median = 2.44°
- mustc__plot198__230613__ugv__pos00: 12 frames in ≥2 windows, dispersion median = 0.00°


### Q12: Boundary vs center frame orientation accuracy

- plantview__langdon_4__05-03-24: boundary = 2.80°, center = 1.81°, ratio = 1.55x
- plantview__langdon_4__12-03-24: boundary = 3.33°, center = 2.20°, ratio = 1.51x
- plantview__langdon_4__15-04-24: boundary = 4.03°, center = 2.06°, ratio = 1.96x
- plantview__langdon_4__19-03-24: boundary = 3.08°, center = 2.01°, ratio = 1.53x
- wheat3dgs__plot_461: boundary = 3.98°, center = 2.24°, ratio = 1.77x
- wheat3dgs__plot_467: boundary = 2.97°, center = 1.98°, ratio = 1.50x
- mustc__plot198__230613__ugv__pos00: boundary = 1.37°, center = 2.04°, ratio = 0.67x


### Q13: Central-owner frame overlap

- plantview__langdon_4__05-03-24: 0/151 edges have zero center overlap → edge constraint region covers camera region
- plantview__langdon_4__12-03-24: 0/151 edges have zero center overlap → edge constraint region covers camera region
- plantview__langdon_4__15-04-24: 0/151 edges have zero center overlap → edge constraint region covers camera region
- plantview__langdon_4__19-03-24: 0/151 edges have zero center overlap → edge constraint region covers camera region
- wheat3dgs__plot_461: 0/9 edges have zero center overlap → edge constraint region covers camera region
- wheat3dgs__plot_467: 0/9 edges have zero center overlap → edge constraint region covers camera region
- mustc__plot198__230613__ugv__pos00: 0/1 edges have zero center overlap → edge constraint region covers camera region


### Q14: Root cause of 44–53° failure

**RETRACTED from Phase 3C.3:** 'Context-dependent Q bias' was based on buggy Procrustes.

**CORRECTED:** The root cause is **temporally-coherent per-edge rotational drift**.
Per-edge errors are small (1.4–2.7°) but co-aligned (coherence 0.76), producing
near-linear trajectory drift (160–179° over 76 hops). Graph sync cannot correct this
because the edges are internally consistent despite systematic reference drift.

### Q15: Recommended next steps

1. **Trajectory regularization (high priority):** Add regularization or prior
   favoring smooth spatially-varying gauge trajectories. The current formulation
   (graph sync with Q_ij only) has no mechanism to suppress coherent drift.
2. **Heading regularization:** If available, use gravity/IMU or global rotation prior
   to anchor the trajectory heading. The current per-window absolute orientation
   is underdetermined by relative edges alone.
3. **Larger context windows:** Increase context from 16 to 32+ frames to
   reduce per-window gauge fit residual and improve cross-window Q accuracy.
4. **Graph regularization (negative-edge pruning):** The coherent drift suggests
   systematic bias in the measurement model; robust M-estimators handle
   outlier edges but not coherent bias. Consider direction-aware robustness.

---

## §31 Final Status

```json
{
  "phase3c31_status": "COMPLETED",
  "procrustes_bug": "CONFIRMED_AND_REPAIRED",
  "q_bias_claim_status": "RETRACTED",
  "corrected_hop1_q_vs_ref": {
    "langdon_05-03": 1.422,
    "langdon_12-03": 1.7075,
    "langdon_15-04": 2.7194,
    "langdon_19-03": 1.4474
  },
  "corrected_hop2_q_vs_ref": {
    "langdon_05-03": 4.9554,
    "langdon_12-03": 5.4224,
    "langdon_15-04": 4.9173,
    "langdon_19-03": 4.8604
  },
  "single_gauge_assumption": "VALID",
  "per_window_gauge_fit_residual_median": {
    "plantview__langdon_4__05-03-24": 1.7745,
    "plantview__langdon_4__12-03-24": 2.1538,
    "plantview__langdon_4__15-04-24": 2.7042,
    "plantview__langdon_4__19-03-24": 2.1496,
    "wheat3dgs__plot_461": 2.6503,
    "wheat3dgs__plot_467": 2.3127,
    "mustc__plot198__230613__ugv__pos00": 1.3534
  },
  "gauge_invariant_relative_rotation_sep1_median": {
    "plantview__langdon_4__05-03-24": 0.7258582413240972,
    "plantview__langdon_4__12-03-24": 0.8028643110073782,
    "plantview__langdon_4__15-04-24": 0.8121149170141552,
    "plantview__langdon_4__19-03-24": 0.7451578850763245,
    "wheat3dgs__plot_461": 2.0725137221539924,
    "wheat3dgs__plot_467": 1.5741392565080705,
    "mustc__plot198__230613__ugv__pos00": 1.4363346951432105
  },
  "trajectory_drift_temporal_coherence_ratio": 0.76,
  "trajectory_drift_over_76_hops_deg": "160-179",
  "random_walk_expectation_deg": 12.4,
  "graph_sync_correction": "NONE (graph \u2248 chain)",
  "chain_vs_graph_median_deg": {
    "plantview__langdon_4__05-03-24": 0.2333,
    "plantview__langdon_4__12-03-24": 0.3869,
    "plantview__langdon_4__15-04-24": 0.5262,
    "plantview__langdon_4__19-03-24": 0.5093,
    "wheat3dgs__plot_461": 0.0596,
    "wheat3dgs__plot_467": 0.1089,
    "mustc__plot198__230613__ugv__pos00": 0.0199
  },
  "final_rot_median_method_D": {
    "plantview__langdon_4__05-03-24": "44.92431950876518",
    "plantview__langdon_4__12-03-24": "49.342215236881046",
    "plantview__langdon_4__15-04-24": "52.017978609056634",
    "plantview__langdon_4__19-03-24": "45.01983646833279",
    "wheat3dgs__plot_461": "3.2761449529229703",
    "wheat3dgs__plot_467": "3.2338538179493366",
    "mustc__plot198__230613__ugv__pos00": "0.6027471166057163"
  },
  "cross_window_same_frame_dispersion_median": {
    "plantview__langdon_4__05-03-24": 6.1152283406466585,
    "plantview__langdon_4__12-03-24": 7.5418399167307015,
    "plantview__langdon_4__15-04-24": 7.697252090590045,
    "plantview__langdon_4__19-03-24": 6.72299647379784,
    "wheat3dgs__plot_461": 3.5398862578012,
    "wheat3dgs__plot_467": 2.4437934532511334,
    "mustc__plot198__230613__ugv__pos00": 0.0
  },
  "boundary_vs_center_ratio": {
    "plantview__langdon_4__05-03-24": 1.55,
    "plantview__langdon_4__12-03-24": 1.51,
    "plantview__langdon_4__15-04-24": 1.96,
    "plantview__langdon_4__19-03-24": 1.53,
    "wheat3dgs__plot_461": 1.77,
    "wheat3dgs__plot_467": 1.5,
    "mustc__plot198__230613__ugv__pos00": 0.67
  },
  "central_owner_edge_coverage": "151/151 edges cover center frames (langdon); 0 zero-overlap",
  "root_cause_corrected": "TEMPORALLY_COHERENT_PER_EDGE_DRIFT",
  "root_cause_retracted": "CONTEXT_DEPENDENT_Q_BIAS",
  "recommended_next": "TRAJECTORY_REGULARIZATION",
  "tests_passed": "20+ (synthetic Procrustes + pipeline tests + regression)"
}
```
