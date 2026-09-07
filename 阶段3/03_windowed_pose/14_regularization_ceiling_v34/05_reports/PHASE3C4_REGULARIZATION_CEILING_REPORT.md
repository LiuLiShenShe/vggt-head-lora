# Phase 3C.4 — GT-Free Regularization Ceiling Experiment (Report)

_Generated 2026-09-07 · `14_regularization_ceiling_v34` · branch decision: **RED**_

## 1. Objective

Phase 3C.3.1 (`6f69c5b`) established the root cause of langdon 44–53° failure:
**temporally-coherent per-edge rotational drift** — temporal coherence
C = 0.756, a constant-rate bias accumulating ~2°/step along a stable axis,
7.6–13× amplification over a random walk. This phase answers the plan's Q15
follow-up question empirically:

> **Can GT-free trajectory regularization suppress the coherent drift, and if
> so, how much?**

Answer: **no — the ceiling is ~35° synthetic / ~45° real, and no GT-free
regularizer moves it by more than ~0.5°.** The branch is closed (RED).

Decision bands (per approved protocol): GREEN rot_median < 20° ·
YELLOW 20–35° · RED > 35°.

## 2. Error structure (restated from Phase 3C.3.1)

| Quantity | Value | Source |
|---|---|---|
| Temporal coherence C | 0.756–0.794 | langdon 05-03/12-03/15-04/19-03 |
| Per-step error median | 1.42–1.53° | pairwise VGGT relative rotations |
| Bias axis vs GT rotation axis | 26° apart | rotvec correlation |
| Over-rotation fraction | 63–96 % | drift sign consistency |
| Final drift (no regularization) | 161–168° | window-0-anchored gauges |
| Amplification vs random walk | 7.6–13× | ‖Σ rv‖ / (√N · rmse) |

Two structural facts determine the experiment's outcome:

1. **The drift is constant-rate** (low-frequency, near-zero 2nd derivative). Low-pass
   smoothing preserves DC (kernel unit mass) and 2nd-order acceleration penalties
   have nothing to penalize (zero 2nd difference). Both families are provably
   orthogonal to the bias — the measured survival ratio of the 2nd-order penalty
   is 0.997–1.013 across λ.
2. **The edge graph is CASE-C consistent**: the redundant hop-2/hop-3 edges agree
   with the drifted chain to ≈0° (scenario A) — the graph contains **no
   corrective information** against the bias direction, so topology/weight
   modifications move the solution by < 0.2°.

## 3. Synthetic drift model

`01_synthetic_drift_model/synthetic_drift_generator.py` builds a testbed with
**known ground truth**, fitted on `plantview__langdon_4__05-03-24` per-step
error statistics: median magnitude 1.53°, coherence target 0.756, 96 %
over-rotation along a stable bias axis. Three scenarios:

| Scenario | hop-3 edge construction | Models |
|---|---|---|
| A | chain product of drifted steps | real CASE-C consistency (residual ≈ 0.0°) |
| B | chain product + Exp(η), η~N(0, (0.37°)²) | real transitivity noise (~0.52° triangle residual) |
| C | bias axis rotates 2°/window | alternative drift regime — NOT observed on real data |

Validation (`tests/test_synthetic_recovery.py`, 7/7 pass): coherence |Δ|<0.02,
magnitudes within ±0.15°, over-rotation ≥ 0.93, final drift > 100°, SO(3)
membership, deterministic seeds, monotonicity ≥ 0.92, triangle residuals ≈0 (A)
/ 0.2–0.9° (B).

## 4. Methods tested

All methods are GT-free (GT appears only in synthetic scoring). Parameters were
swept on synthetic, then transferred **1 config per method** to real data (no
per-sequence re-sweep).

| ID | Method | Parameter sweep | File |
|---|---|---|---|
| R | Reference — raw chain (no regularization) | — | baseline |
| A | Gaussian increment smoothing (increment-domain, re-integrated) | σ ∈ {0.5, 1, 2, 3, 5, 8, 12, 16} | `smoothing_gaussian.py` |
| B | 2nd-order (acceleration) penalty (closed-form tridiagonal ridge) | λ ∈ {0.1, 1, 10, 100} | `smoothing_gaussian.py` |
| C | Lower-threshold re-solve, th≥4 (full 225-edge set incl. 74 hop-3) | th=4 | `lower_threshold_solver.py` |
| D | Multi-anchor diagnostic (pin mid/end gauges to identity) | {0}, {0,mid}, {0,end}, {0,mid,end} | `multi_anchor_solver.py` (NOT DEPLOYABLE) |

All operations act on the **increment domain** (per-step rotvecs
δ_k = Log(G_kᵀ G_{k+1})) to avoid the π-aliasing of the 161° accumulated drift.
Warm start via MST; vectorized residual evaluation.

Supplementary methods from the wider sweep (per-window convention, kept as a
consistency check): SO(3) GP smoothing (increments / local-trend) and edge
reweighting (temporal attenuation / bias alignment) — all reached the same RED
verdict (`REGULARIZATION_RESULT.json`).

## 5. Synthetic ceiling (final, frame-level convention)

Protocol: 30 scenarios/seed × scenario, single global rotation Procrustes vs
G_true, median across scenarios of per-window rot_median. Threshold 4 (full
225-edge set).

### Scenario A (CASE-C consistent) and B (with transitivity noise) — realistic regimes

Baseline raw-chain rot_median: **35.67°** (both scenarios).

| Method | Best param | A median | B median | Decision |
|---|---:|---:|---:|:--:|
| R — raw chain | 0.0 | 35.67° | 35.67° | **RED** |
| A — Gaussian smoothing | σ=2.0 | 35.40° | 35.40° | **RED** |
| B — 2nd-order penalty | λ=0.1 | 35.20° | 35.20° | **RED** |
| C — lower threshold (225 edges) | th=4 | 35.67° | 35.77° | **RED** |
| D — multi-anchor [0,end] | [0,76] | 17.06° | 17.24° | GREEN ⚠ diagnostic |

The best deployable suppression is **35.20°** (B@0.1) — a gain of 0.47° over the
35.67° baseline. C is *identical* to R (35.67°): the 74 hop-3 edges are
chain-consistent with the drift and add zero corrective power. No deployable
method approaches the 20° GREEN band.

**D is diagnostic-only.** Its GREEN median (17°) is a metric artifact of the
chain-consistent testbed: pinning window-76's gauge to identity forces an open
pan trajectory (~1800° accumulated rotation) to "close"; global Procrustes then
spreads the endpoint misfit uniformly across windows, lowering the median
without suppressing drift. `multi_anchor_solver.py` is marked NOT DEPLOYABLE.
**Real data (§6) confirms this decisively**: D[0,76] makes langdon *worse*
(45→74–78°), because real hop-3 edges carry independent measurement noise and
real trajectories genuinely accumulate rotation — the closure assumption is
false.

### Scenario C (rotating bias axis) — NOT the observed regime

Baseline **16.52°** (already GREEN). A@σ=16 → 12.99°, B@λ=100 → 14.29°,
C=16.52°, D[0]=15.36°. Smoothing *does* help here — but scenario C does not
reproduce real data, whose bias axis is stable (C=0.756 across all 4 langdon
sequences). It is reported for completeness and as the boundary of the finding.

## 6. Real-data application (7 sequences, frame-level)

Evaluation follows the Phase 3C.3 convention exactly: per-frame global cameras
R_c2w_global[f] = G_reg[owner] @ R_c2w_local[f] (central-window ownership),
single global rotation Procrustes vs COLMAP reference, per-frame rot_median.
**All 7 baselines reproduce Phase 3C.3's D-stride4 values to < 1e-6°** —
validating the harness.

### 6.1 Headline — no deployable method suppresses langdon drift

| Sequence | Baseline D | Best deployable | rot_median | Δ vs baseline | gate |
|---|---|---:|---|---:|---:|---:|
| langdon 05-03-24 | 44.92° | B 2nd-order @0.1 | 44.74° | −0.18° | FAIL |
| langdon 12-03-24 | 49.34° | B 2nd-order @0.1 | 49.25° | −0.09° | FAIL |
| langdon 15-04-24 | 52.02° | B 2nd-order @0.1 | 51.56° | −0.46° | FAIL |
| langdon 19-03-24 | 45.02° | C lower-threshold @4 | 44.86° | −0.16° | FAIL |
| wheat plot_461 | 3.28° | A Gaussian @0.5 | 2.75° | −0.53° | PASS |
| wheat plot_467 | 3.23° | A Gaussian @0.5 | 2.54° | −0.69° | PASS |
| mustc pos00 | 0.60° | (no improvement) | 0.60° | 0.00° | PASS |

The maximum real-data gain is **0.69°** (wheat 467, already accurate). langdon
stays 44.7–52.0° — 24–32° above the GREEN threshold.

### 6.2 Representative full sweep — langdon 05-03-24 (baseline 44.92°)

| Param | A Gaussian | B 2nd-order | C / D |
|---|---:|---:|---:|
| mild (0.5 / 0.1 / th4) | 45.40° | **44.74°** | C: 45.18° |
| σ=1 / λ=1 | 49.33° | 49.04° | D[0]: 45.18° |
| σ=2 / λ=10 | 58.24° | 58.97° | D[0,38]: 56.86° |
| σ=3 / λ=100 | 61.47° | 62.41° | **D[0,76]: 74.32°** |
| σ=5 | 62.16° | — | D[0,38,76]: 74.71° |
| σ=8/12/16 | 62.87/63.22/63.20° | — | — |

Pattern (identical for all 4 langdon sequences): mild smoothing does nothing,
σ≥1 / λ≥1 **degrades** the result (+4…+19°); C is baseline-neutral (≤0.16°);
D end-pinning is catastrophic (74–78°). The drift survives every deployable
operation; over-regularization only adds damage.

### 6.3 Controls gate (wheat461 / wheat467 / mustc)

Tolerance Δrot_median ≤ +1.0° vs baseline AND pose gate PASS.

| Control | Baseline | A@0.5 | B@0.1 | C@4 | σ≥1 family |
|---|---:|---:|---:|---:|---|
| wheat 461 | 3.28° | 2.75° PASS | 2.79° PASS | 3.29° PASS | **VIOLATES** (σ≥1: 4.2–6.1°) |
| wheat 467 | 3.23° | 2.54° PASS | 3.03° PASS | 3.30° PASS | **VIOLATES** (σ≥1: 4.4–6.2°) |
| mustc | 0.60° | 0.60° PASS | 0.60° PASS | 0.60° PASS | PASS |

Deployable configs (A@0.5, B@0.1, C@4) pass all controls; the over-smoothing
family (σ≥1) violates the wheat tolerance and is filtered (counts RED). mustc
is untouched by every deployable method (already 0.60°).

## 7. Decision

**RED — branch closed.**

The two structural facts from §2 are empirically confirmed end-to-end:

1. **Constant-rate drift survives trajectory regularization.** On realistic
   synthetic scenarios (A/B), every deployable method lands in 35.2–35.8° vs a
   35.67° baseline — ceiling ≈ 0.5° of achievable suppression. On real data the
   best gain is 0.46° (langdon), and over-smoothing actively worsens the result.
2. **The graph has nothing corrective to exploit.** CASE-C consistency (hop-3
   edges agree with the drifted chain) makes C == R to 0.00–0.10°. Redundancy
   without independent information cannot anchor the trajectory.

The only GREEN outcome (D[0,76] = 17°) is a diagnostic artifact of pinning an
open trajectory to a closed gauge — confirmed non-deployable by real data,
where the same operation destroys the solution (74–78°).

## 8. Implications for the Phase 3C roadmap

- **Regularization branch (this phase): CLOSED.** GT-free post-hoc trajectory
  smoothing, penalty, edge-set expansion, and anchor redistribution cannot lower
  the 45–52° langdon error.
- **The ceiling is information-theoretic, not algorithmic**: under a constant-rate
  coherent bias with CASE-C consistent edges, no function of the *same* edge set
  can recover the drift direction. Corrective information must enter from
  outside the current edge set.
- **Remaining levers (both out of scope by constraint):**
  - *Larger temporal support / re-inference* (longer windows, more frames per
    window, or shared-frame alignment over a longer baseline) — attacks the bias
    at its source (VGGT per-window relative rotation error).
  - *An external absolute-orientation anchor* (IMU/gravity, ground-control
    orientation, or a loop-closure to a known pose) — provides exactly the
    corrective direction the graph lacks. Explicitly excluded here.
- **Report fidelity note**: the stride-4 windowing creates a 4-step ramp/plateau
  drift structure in window-increment space (each new window shares 12/16 frames
  with its predecessor); this was modeled implicitly (per-step i.i.d. errors)
  and does not change the verdict, but should be revisited if windowing changes
  in a future phase.

## 9. Q1–Q10

- **Q1 — What is the drift's structure?** Temporally coherent (C=0.756),
  constant-rate (~2°/step), stable bias axis 26° from the GT rotation axis,
  96 % over-rotation, 161–168° total. It is a low-frequency signal.
- **Q2 — Does Gaussian smoothing suppress it?** No. Increment-domain Gaussian
  low-pass preserves DC by construction; σ=0.5 is neutral (±0.5°), σ≥1 degrades
  real data (+4…+19°) by flattening the true trajectory faster than it removes
  bias.
- **Q3 — Does the 2nd-order acceleration penalty suppress it?** No. A
  constant-rate bias has zero 2nd difference; measured survival ratio
  0.997–1.013. Best λ=0.1 gives 35.20° synthetic / 44.74° real — within noise
  of baseline.
- **Q4 — Do the redundant hop-3 edges (th≥4, 225 edges) carry corrective
  information?** No. C == R to 0.00° (synthetic A) and ≤0.16° (real). The edges
  are CASE-C consistent with the drifted chain — redundancy without independence.
- **Q5 — Can anchor redistribution (Method D) suppress drift?** As a diagnostic,
  it reveals the drift direction and magnitude; as a deployable fix it cannot —
  the synthetic GREEN (17°) is an artifact of forcing an open pan trajectory to
  close, and real data degrades to 74–78°. NOT DEPLOYABLE.
- **Q6 — Does the ceiling hold under realistic edge noise (scenario B)?** Yes —
  B is identical to A within 0.10° (best 35.20° vs 35.77° C). The verdict is
  robust to transitivity noise.
- **Q7 — Is the ceiling regime-dependent?** Partially. Under scenario C (bias
  axis rotating 2°/window — NOT observed on real data) smoothing reaches 12.99°
  GREEN. The RED verdict applies to the *observed* stable-axis regime only.
- **Q8 — What is the maximum achievable GT-free suppression?** 0.47° synthetic
  (35.67→35.20°) and 0.46° real langdon (52.02→51.56°). Negligible.
- **Q9 — Do the controls survive?** Yes for deployable configs (A@0.5, B@0.1,
  C@4: all PASS, within +1.0°); the over-smoothing family (σ≥1) violates the
  wheat tolerance and is filtered.
- **Q10 — Verdict and roadmap?** RED: close the GT-free regularization branch.
  Next: attack the bias at its source (larger temporal support / re-inference)
  or provide an external absolute-orientation anchor — both outside this
  experiment's constraints.

## 10. Next steps

1. **Register the branch decision** (RED) in the Phase 3C status ledger; the
   regularization branch is closed and not a candidate for the 3C.5 baseline.
2. **Re-inference study (out of scope here, highest expected value)**: quantify
   the per-window error statistics as a function of window length / frame
   support. The coherent bias per step (1.4–1.5°) is the quantity to shrink;
   stride-4 ramp/plateau structure should be re-examined under longer windows.
3. **External absolute-orientation anchor** (IMU/gravity or ground-control
   orientation): the only in-graph-free mechanism that introduces the missing
   corrective direction.
4. **Keep the D diagnostic** as a cheap drift-direction probe in future windowed
   pipelines (it isolates the accumulated bias direction at ~1/100th the cost of
   a re-solve), but never as a solver constraint.
5. **Archive** the wider method sweep (SO(3) GP, edge reweighting — per-window
   convention, same RED verdict) in `REGULARIZATION_RESULT.json` as a
   reproducibility record; the frame-level pipeline
   (`REGULARIZATION_REAL_RESULT.json`, `SYNTHETIC_CEILING_RESULT_scen{A,B,C}.json`)
   is the canonical result set for this phase.

## Artifacts

| Artifact | Path |
|---|---|
| Protocol | `00_protocol/README.md` |
| Synthetic model | `01_synthetic_drift_model/synthetic_drift_generator.py` |
| Methods (A/B/C/D) | `02_regularization_methods/{smoothing_gaussian,lower_threshold_solver,multi_anchor_solver}.py` |
| Synthetic results | `03_synthetic_evaluation/SYNTHETIC_CEILING_RESULT_scen{A,B,C}.json` |
| Real-data results | `04_real_data_application/REGULARIZATION_REAL_RESULT.json` |
| Supplementary sweep | `04_real_data_application/REGULARIZATION_RESULT.json` |
| Tests | `tests/test_{synthetic_recovery,methods_reduce_drift,controls_no_regression}.py` |
