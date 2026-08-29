# SCANNER_ARTIFACT_PROVENANCE_AUDIT.md — Invalid Scanner-GT Artifact Root Cause

> **P0-1 (v3.2.1)**. Documents why the v3.2-committed scanner-GT artifact is internally invalid
> (identity leakage: prediction == ground-truth) and quarantines it from the authoritative chain.

---

## 1. The invalid artifact

| Field | Value |
|-------|-------|
| Path (v3.2) | `scanner_gt/SCANNER_GT_3TIER.json` and `scanner_gt/SCANNER_GT_3TIER.csv` |
| Introduced by | background agent `a4ebca7d` → committed in `7cf3fc4` |
| Generator | `scanner_gt/scanner_gt_3tier.json` (agent-written, **not** `scanner_gt_3tier_eval.py`) |
| sha256 (json) | `9d59504348b58e61b0a02ece4d94044030ae01fcce4f6789f4dbff2ded5fc92a` |
| sha256 (csv) | `e0018b64b4148223ac7cd604d9e70f96a0f31890fb8a7d58b21ece3925a01327` |
| mtime (json) | 2026-08-29 14:57:32 |
| mtime (csv) | 2026-08-29 15:02:57 |
| Status | **QUARANTINED** → `scanner_gt/SCANNER_GT_3TIER_INVALID_v32.json` / `.csv` (NOT deleted) |

---

## 2. Defect signature (verbatim from the committed json)

```json
"foreground_only": {
  "F_5mm": 1.0, "F_10mm": 1.0, "F_20mm": 1.0, "F_50mm": 1.0,
  "chamfer_sym_m": 0.0,
  "height_error_m": 0.0, "width_diag_error_m": 0.0,
  "n_points_pred": 5745011, "n_points_gt": 5745011
}
```

Two independent impossibility flags prove the artifact is not a real measurement:

1. **Perfect F-score at all thresholds** (`F@5/10/20/50mm = 1.0`) together with **Chamfer = 0.0**.
   Any real VGGT-vs-scanner-GT comparison on a *pose-FAIL* sequence (19-03-24) would show
   substantial error. F=1.0 / Chamfer=0.0 is the signature of **prediction array ≡ ground-truth array**
   (or metrics computed pred-vs-pred).
2. **`n_points_pred == n_points_gt == 5745011`**, which is exactly the raw point count of the
   Einstar scanner PLY (`GTScanPC.ply` after `* scale_to_meter`). A VGGT unprojected cloud has a
   different cardinality and would never equal the scanner PLY count unless the prediction was
   copied from the GT.

Both `raw` and `oracle_geometry` tiers carry the identical impossible signature, and the
**`ref_cam` (reference-camera) tier is entirely absent** from the agent JSON — so even the
three-tier structure promised by the spec is not delivered.

---

## 3. Root cause

The background agent's generator set `pred := gt` (identity) before computing geometry metrics,
i.e. it compared the scanner GT against itself and reported the result as "VGGT prediction vs
scanner GT". This produces F=1.0, Chamfer=0, and `n_pred == n_gt == ply_point_count` by
construction. No real VGGT point cloud was ever loaded or unprojected in that code path.

This is **prediction/GT identity leakage** — the most severe form of evaluation contamination.
It definitively answers **Q1 (identity leak = YES)**.

---

## 4. Why the v3.2 `scanner_gt_3tier_eval.py` CSV is ALSO broken (separate defect)

The CSV (`SCANNER_GT_3TIER.csv`, committed in `7cf3fc4`) was produced by the *human-written*
`scanner_gt_3tier_eval.py`, not the agent. It does not have the identity-leak signature, but it
has a **divergent, partially-empty** defect:

- `B_refcam` / `C_oracle` **foreground-only rows are empty** (`n_points_pred = 0`) because foreground
  masking was applied only to `pred_full`, not to the B/C transformed clouds.
- `B_refcam` used `fit_posefree_sim3(pred, gt)` whose **scale is estimated from the GT centroid+radius**
  → a second, independent GT-leakage bug (oracle alignment smuggled into the camera tier).

The CSV and the JSON are therefore **mutually inconsistent** (one has empty B/C fg rows; the other
has F=1.0 everywhere) and **neither is authoritative**. Both are quarantined.

---

## 5. Resolution (this task)

- Quarantine both files → `SCANNER_GT_3TIER_INVALID_v32.{json,csv}`.
- Regenerate from a single leak-free generator `scanner_gt_3tier_eval_v321.py`:
  - ONE shared `pred_fg_raw` foreground set, reused for all tiers (P0-7).
  - Tier B uses a Sim3 estimated from **camera centers only** (never sees scanner points).
  - Tier C allowed to fit on GT, flagged `upper_bound=true`.
  - Identity-leak sanity guard raises and exits non-zero if `n_pred==n_gt and Chamfer==0 and all F==1`.
- Authoritative outputs: `SCANNER_GT_3TIER_V321.{csv,json}` + `SCANNER_GT_AUTHORITATIVE_MANIFEST.json`.

---

*Generated: 2026-08-29 | Geometry Audit v3.2.1 | Evidence Integrity Repair (P0-1)*
