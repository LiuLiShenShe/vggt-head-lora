# PHASE3C1 Evidence Erratum — INVALIDATED PASS Claims

> **Status: ACTIVE CORRECTION.** This erratum supersedes conflicting numbers in
> `PHASE3C1_GAUGE_AWARE_STITCHING.md`. The old report's PASS claims for langdon_4
> are **DEPRECATED / INVALIDATED** and must not be quoted as evidence.
>
> **Authoritative source = `06_pose_evaluation_v31/GAUGE_AWARE_GLOBAL_RESULTS.csv`**
> (written 2026-09-01 13:13:37, from the current stitching code `run_gauge_stitching.py`
> written 13:12:47).

---

## 1. The Conflict

| Sequence | Reported in old report (12:01) | Authoritative CSV (13:13) |
|---|---|---|
| langdon_4 05-03 | **2.0° PASS** (claimed "rescued") | **44.05° FAIL** (gauge_aware) |
| langdon_4 12-03 | **2.6° PASS** (claimed "rescued") | **48.75° FAIL** (gauge_aware) |
| langdon_4 15-04 | **9.1° PASS** (claimed "rescued") | **53.10° FAIL** (gauge_aware) |
| langdon_4 19-03 | **1.9° PASS** (claimed "rescued") | **44.29° FAIL** (gauge_aware) |
| mustc pos00 | 2.6° PASS | 2.65° PASS ✅ (consistent) |
| wheat3dgs 461 | 3.3° PASS | 3.28° PASS ✅ (consistent) |
| wheat3dgs 467 | 3.4° PASS | 3.37° PASS ✅ (consistent) |

The 4 langdon_4 long-sequence claims are **contradicted**. The 3 short-sequence
controls are consistent with the CSV.

## 2. Root Cause — Timeline

```
2026-09-01 12:01:59  old report PHASE3C1_GAUGE_AWARE_STITCHING.md written (claims PASS 1.9-9.1°)
2026-09-01 13:12:47  run_gauge_stitching.py REWRITTEN (current code, 15744-byte current logic)
2026-09-01 13:13:26  *_GAUGE_GLOBAL_CAMERAS.npz regenerated
2026-09-01 13:13:37  GAUGE_AWARE_GLOBAL_RESULTS.csv regenerated (authoritative: 44-53° FAIL)
```

The old report **predates** the current stitching implementation by ~71 minutes.
Its langdon_4 PASS numbers came from either (a) an earlier overwritten version of
the stitching algorithm, or (b) a local/consecutive-window evaluation instead of the
full 39-window global chain.

**Smoking-gun cross-check:** the old report's values (2.0 / 2.6 / 9.1 / 1.9) closely
match the CSV's `consecutive_local` rows (2.69 / 2.89 / 2.39 / 2.46) — a 16-frame
**local** evaluation that does NOT test global chain stitching. The old report
reported local accuracy as if it were global chain accuracy.

## 3. What Is Still TRUE

- **Rotation convention is verified** (12/12 synthetic tests): `Q = R_c2w_A @ R_c2w_B^T`.
- **Overlap Q dispersion is HIGH consistency** (< 2° median, max ~8.16°) — VGGT local
  rotations are NOT inconsistent. Gauge freedom confirmed.
- **Short sequences (≤3 windows) genuinely PASS** gauge-aware stitching: mustc 2.65°,
  wheat 3.28°/3.37° — reproduced in the authoritative CSV.
- **COLMAP/GT never entered the stitching algorithm** (leakage test passes).

## 4. What Is FALSE / Must NOT Be Quoted

- ~~"Gauge-Aware 05-03: 2.0° PASS"~~
- ~~"Catastrophic Dates Rescued: 4/4"~~
- ~~"Gauge-aware P90 < 15 deg for all dates"~~
- ~~"Center median norm < 0.21 for all dates"~~
- ~~"global_pose_gate = PASS"~~
- ~~"pose_graph_needed = NO"~~ → actually NEEDED (see Phase 3C.2/3C.3)

## 5. Where The Truth Stands (from CSV + Phase 3C.2)

- 4 langdon_4 dates: **44-53° global rotation error = FAIL** (gauge_aware chain).
- Phase 3C.2 root cause: **Q-chain rotation accumulation** (~159° over 38 pairs),
  NOT scale drift. Scale is secondary (8-31% center drift).
- This is the motivation for **Phase 3C.3** (redundant SO(3) rotation sync, stride-4).

## 6. Authoritative Numbers (from CSV, rot_median)

| Method | 05-03 | 12-03 | 15-04 | 19-03 | mustc | wheat461 | wheat467 |
|---|---|---|---|---|---|---|---|
| gauge_aware | 44.05 FAIL | 48.75 FAIL | 53.10 FAIL | 44.29 FAIL | 2.65 PASS | 3.28 PASS | 3.37 PASS |
| center_only | 48.76 FAIL | 28.93 FAIL | 59.89 FAIL | 27.88 FAIL | 3.25 PASS | 6.46 PASS | 3.85 PASS |
| consecutive_local | 2.69 PASS | 2.89 PASS | 2.39 PASS | 2.46 PASS | 1.79 PASS | 2.51 PASS | 2.39 PASS |

---

**Bottom line:** langdon_4 was NOT rescued by Phase 3C.1. The 44-53° long-sequence
rotation error is a real, unresolved problem caused by Q-chain accumulation, and is
the target of Phase 3C.3. Short-sequence gauge-aware stitching remains validated.
