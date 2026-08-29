# DEPRECATED — figures_v31/*_precision_recall_explanation.png

These three figures (05/13/20-03-24) were moved here because they used **full-scene**
`per_seq/{sid}_pred_aligned.npy` points that were **mislabeled as "foreground"** in the
v3.2 `precision_recall_visual_v32.py` pipeline.

The mislabeled pipeline reported `precision@10mm ≈ 0.06–0.08` (because full-scene VGGT points
spread far beyond the plant). The corrected v3.2.1 pipeline (`precision_recall_visual_v321.py`)
loads only the **true** plant-foreground arrays (`*_pred_foreground_aligned.npy` /
`*_reference_foreground.npy`) and reports the real `precision@10mm ≈ 0.37–0.44` with recall ≈ 0.96–1.00.

**Do NOT cite these figures.** Use `figures_v321/*_precision_recall_explanation.png`.

Root cause: `run_geometry_audit_v31.py` persisted only `*_pred_aligned.npy` (full scene); the
real foreground arrays `P_fore`/`Q_fore` were in-memory only. v3.2.1 now persists them and the
PR figures consume them exclusively.

*Generated: 2026-08-29 | Geometry Audit v3.2.1 | Evidence Integrity Repair (P1-4)*
