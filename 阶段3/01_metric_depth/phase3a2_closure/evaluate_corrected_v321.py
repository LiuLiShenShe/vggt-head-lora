#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3A.2.1 — Corrected UniDepth CalK pilot evaluator.

Bug: original evaluator used pred_stack[original_idx] on compact pilot stack.
Fix: pred_stack[local_idx] maps to original_frame[pilot_indices[local_idx]].

Reads existing CalK predictions (NO re-inference).
Saves frame mapping, completeness check, per-frame metrics, and all comparison CSVs.

Usage:
  /home/test/miniconda3/envs/unidepth/bin/python evaluate_corrected_v321.py
"""
import os, sys, csv, json
import numpy as np
from PIL import Image

ROOT = "/fj/VGGT+head+lora实验"
PHASE3_DIR = os.path.join(ROOT, "阶段3", "01_metric_depth")
AUDIT_DIR = os.path.join(PHASE3_DIR, "phase3a2_closure")
sys.path.insert(0, PHASE3_DIR)
from configs import SEQUENCES, load_sequence_meta, get_depth_path, get_mask_path, VGGT_RERUN
from unified_depth_evaluator import evaluate_one_frame

DEPTH_SCALE_TO_METER = 0.001

# Must match inference script
SAMPLE_INTERVAL = 16
MAX_FRAMES = 20


def compute_frame_map(seq_id):
    """Compute frame mapping: compact pilot index → original frame index."""
    meta = load_sequence_meta(seq_id)
    n_frames = len(meta["rgb_paths"])
    pilot_indices = list(range(0, min(n_frames, MAX_FRAMES * SAMPLE_INTERVAL), SAMPLE_INTERVAL))[:MAX_FRAMES]
    mapping = []
    for pilot_idx, orig_idx in enumerate(pilot_indices):
        mapping.append({
            "pilot_index": pilot_idx,
            "original_index": orig_idx,
            "rgb_filename": os.path.basename(meta["rgb_paths"][orig_idx]),
        })
    return {
        "sequence": seq_id,
        "n_original_frames": n_frames,
        "n_pilot_frames": len(pilot_indices),
        "pilot_indices": pilot_indices,
        "mapping": mapping,
    }


def main():
    os.makedirs(AUDIT_DIR, exist_ok=True)

    # ── Step 1: Generate frame mappings ────────────────────────────────────
    all_mappings = {}
    for seq_id, _ in SEQUENCES:
        fm = compute_frame_map(seq_id)
        all_mappings[seq_id] = fm
        print(f"  {seq_id}: {fm['n_pilot_frames']} pilot frames, indices={fm['pilot_indices'][:5]}...")

    map_path = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_FRAME_MAP.json")
    with open(map_path, "w") as f:
        json.dump(all_mappings, f, indent=2, ensure_ascii=False)
    print(f"Saved frame map: {map_path}")

    # ── Step 2: Verify existing predictions ─────────────────────────────────
    prediction_total = 0
    for seq_id, _ in SEQUENCES:
        calK_path = os.path.join(PHASE3_DIR, "unidepth_v2", seq_id, "depth_unidepth_calK_pilot.npy")
        if not os.path.exists(calK_path):
            print(f"  FATAL: Missing prediction: {calK_path}")
            sys.exit(1)
        d = np.load(calK_path)
        expected = all_mappings[seq_id]["n_pilot_frames"]
        assert d.shape[0] == expected, f"{seq_id}: shape[0]={d.shape[0]} != {expected}"
        prediction_total += d.shape[0]
    assert prediction_total == 80, f"Total predictions {prediction_total} != 80"
    print(f"\nPrediction integrity: {prediction_total}/80 frames OK")

    # ── Step 3: Evaluate with CORRECT compact indexing ──────────────────────
    frame_metrics_all = []  # per-frame rows
    seq_summary = []        # per-sequence summary

    for seq_id, pose_fail in SEQUENCES:
        meta = load_sequence_meta(seq_id)
        rgb_paths = meta["rgb_paths"]
        depth_dir = meta["depth_dir"]
        mask_dir = meta["mask_dir"]
        n_frames = len(rgb_paths)
        fm = all_mappings[seq_id]
        pilot_indices = fm["pilot_indices"]

        # Pre-load ALL reference depths and masks (indexed by original frame)
        ref_depths = []
        fg_masks = []
        for rp in rgb_paths:
            dp = get_depth_path(depth_dir, rp)
            ref_depths.append(np.asarray(Image.open(dp)) if os.path.exists(dp) else None)
            mp = get_mask_path(mask_dir, rp)
            fg_masks.append(
                np.asarray(Image.open(mp).convert("L")) > 0 if os.path.exists(mp) else None
            )

        # Load ALL model predictions
        models_data = {}
        model_paths = {
            "vggt": os.path.join(VGGT_RERUN, seq_id, "depth_vggt.npy"),
            "da3_metric": os.path.join(PHASE3_DIR, "da3", seq_id, "depth_da3_metric.npy"),
            "unidepth_auto": os.path.join(PHASE3_DIR, "unidepth_v2", seq_id, "depth_unidepth.npy"),
            "unidepth_calK": os.path.join(PHASE3_DIR, "unidepth_v2", seq_id, "depth_unidepth_calK_pilot.npy"),
        }
        for mname, mpath in model_paths.items():
            if os.path.exists(mpath):
                models_data[mname] = np.load(mpath)

        print(f"\n{seq_id}: {n_frames} frames, {len(pilot_indices)} pilot frames")

        # Evaluate full-frame models (VGGT, DA3Metric, UniDepth autonomous)
        for model in ["vggt", "da3_metric", "unidepth_auto"]:
            if model not in models_data:
                print(f"  SKIP {model}: not loaded")
                continue
            pred_stack = models_data[model]
            out_h, out_w = pred_stack.shape[1], pred_stack.shape[2]
            n_eval = min(n_frames, pred_stack.shape[0])

            absrels, rmses, delta1s, scales = [], [], [], []
            for i in range(n_eval):
                if ref_depths[i] is None or fg_masks[i] is None:
                    continue
                result = evaluate_one_frame(pred_stack[i], ref_depths[i], fg_masks[i], out_h, out_w)
                if result is None:
                    continue
                absrels.append(result["raw"]["absrel"])
                rmses.append(result["raw"]["rmse"])
                delta1s.append(result["raw"]["delta1"])
                scales.append(result["scale_ratio"])

            if absrels:
                row = {
                    "seq_id": seq_id, "pose_fail": pose_fail, "model": model,
                    "evaluation_scope": "full",
                    "n_frames": len(absrels),
                    "abs_rel_mean": float(np.mean(absrels)),
                    "abs_rel_median": float(np.median(absrels)),
                    "rmse_mean": float(np.mean(rmses)),
                    "rmse_median": float(np.median(rmses)),
                    "delta1_mean": float(np.mean(delta1s)),
                    "scale_mean": float(np.mean(scales)),
                    "scale_median": float(np.median(scales)),
                    "scale_cv": float(np.std(scales) / np.mean(scales)),
                }
                seq_summary.append(row)
                print(f"  {model:20s}: AbsRel={row['abs_rel_mean']:.4f} RMSE={row['rmse_mean']:.4f} "
                      f"scale={row['scale_mean']:.4f} n={row['n_frames']}")

        # Evaluate CalK with CORRECT compact indexing
        if "unidepth_calK" not in models_data:
            print(f"  SKIP unidepth_calK: not loaded")
            continue
        calK_stack = models_data["unidepth_calK"]
        out_h, out_w = calK_stack.shape[1], calK_stack.shape[2]

        calK_absrels, calK_rmses, calK_delta1s, calK_scales = [], [], [], []
        auto_absrels_matched, auto_rmses_matched = [], []  # matched autonomous

        for local_idx, orig_idx in enumerate(pilot_indices):
            if local_idx >= calK_stack.shape[0]:
                print(f"  FATAL: local_idx={local_idx} >= calK shape[0]={calK_stack.shape[0]}")
                sys.exit(1)
            if ref_depths[orig_idx] is None or fg_masks[orig_idx] is None:
                print(f"  SKIP pilot[{local_idx}]=orig[{orig_idx}]: missing ref/mask")
                continue

            # CalK prediction (compact indexed)
            calK_pred = calK_stack[local_idx]
            result_calK = evaluate_one_frame(calK_pred, ref_depths[orig_idx], fg_masks[orig_idx], out_h, out_w)
            if result_calK is None:
                print(f"  SKIP pilot[{local_idx}]=orig[{orig_idx}]: not enough valid pixels (CalK)")
                continue

            # Matched autonomous prediction (full indexed by orig_idx)
            auto_pred = models_data["unidepth_auto"][orig_idx]
            out_h_auto, out_w_auto = auto_pred.shape[0], auto_pred.shape[1]
            result_auto = evaluate_one_frame(auto_pred, ref_depths[orig_idx], fg_masks[orig_idx], out_h_auto, out_w_auto)

            rgb_filename = os.path.basename(rgb_paths[orig_idx])

            # Frame-level metrics
            frame_row = {
                "sequence": seq_id,
                "pilot_index": local_idx,
                "original_index": orig_idx,
                "filename": rgb_filename,
                "raw_absrel": result_calK["raw"]["absrel"],
                "raw_mae_m": result_calK["raw"]["mae"],
                "raw_rmse_m": result_calK["raw"]["rmse"],
                "raw_delta1": result_calK["raw"]["delta1"],
                "scale_ratio": result_calK["scale_ratio"],
                "aligned_absrel": result_calK["aligned"]["absrel"],
                "aligned_rmse_m": result_calK["aligned"]["rmse"],
            }
            if result_auto:
                frame_row["auto_raw_absrel"] = result_auto["raw"]["absrel"]
                frame_row["auto_raw_rmse_m"] = result_auto["raw"]["rmse"]
                frame_row["auto_scale_ratio"] = result_auto["scale_ratio"]
            frame_metrics_all.append(frame_row)

            calK_absrels.append(result_calK["raw"]["absrel"])
            calK_rmses.append(result_calK["raw"]["rmse"])
            calK_delta1s.append(result_calK["raw"]["delta1"])
            calK_scales.append(result_calK["scale_ratio"])

            if result_auto:
                auto_absrels_matched.append(result_auto["raw"]["absrel"])
                auto_rmses_matched.append(result_auto["raw"]["rmse"])

        if calK_absrels:
            row = {
                "seq_id": seq_id, "pose_fail": pose_fail, "model": "unidepth_calK",
                "evaluation_scope": "pilot_20",
                "n_frames": len(calK_absrels),
                "abs_rel_mean": float(np.mean(calK_absrels)),
                "abs_rel_median": float(np.median(calK_absrels)),
                "rmse_mean": float(np.mean(calK_rmses)),
                "rmse_median": float(np.median(calK_rmses)),
                "delta1_mean": float(np.mean(calK_delta1s)),
                "scale_mean": float(np.mean(calK_scales)),
                "scale_median": float(np.median(calK_scales)),
                "scale_cv": float(np.std(calK_scales) / np.mean(calK_scales)),
            }
            seq_summary.append(row)
            print(f"  unidepth_calK       : AbsRel={row['abs_rel_mean']:.4f} RMSE={row['rmse_mean']:.4f} "
                  f"scale={row['scale_mean']:.4f} n={row['n_frames']} [CORRECTED]")

    # ── Step 4: Completeness check ─────────────────────────────────────────
    per_seq_counts = {}
    for row in frame_metrics_all:
        s = row["sequence"]
        per_seq_counts[s] = per_seq_counts.get(s, 0) + 1
    total_evaluated = sum(per_seq_counts.values())

    short_names = {s: s.split("__")[-1] for s in per_seq_counts}
    completeness = {
        "expected_total": 80,
        "prediction_total": prediction_total,
        "evaluated_total": total_evaluated,
        "per_sequence": {short_names[s]: c for s, c in per_seq_counts.items()},
        "status": "PASS" if total_evaluated == 80 else "FAIL",
    }
    comp_path = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_EVAL_COMPLETENESS.json")
    with open(comp_path, "w") as f:
        json.dump(completeness, f, indent=2, ensure_ascii=False)
    print(f"\nCompleteness: {completeness['status']} ({total_evaluated}/80)")

    if total_evaluated != 80:
        print("FATAL: completeness check failed")
        sys.exit(1)

    # ── Step 5: Save K provenance ──────────────────────────────────────────
    provenance = {
        "calibrated_K_source": "Plant View calibration (fx=1371.82, fy=1370.79, cx=cy=540.0)",
        "fx": 1371.82, "fy": 1370.79, "cx": 540.0, "cy": 540.0,
        "original_resolution": [1080, 1080],
        "model_input_resolution": "handled internally by UniDepthV2 infer()",
        "pinhole_api": "from unidepth.utils.camera import Pinhole; Pinhole(K=K_tensor)",
        "note": "CameraHead still predicts intrinsics independently; calibrated K only conditions rays via decoder.py:400",
    }
    prov_path = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_PROVENANCE.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)

    # ── Step 6: Save per-frame metrics CSV ─────────────────────────────────
    frame_csv = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_FRAME_METRICS_V321.csv")
    if frame_metrics_all:
        fields = list(frame_metrics_all[0].keys())
        with open(frame_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(frame_metrics_all)
        print(f"Frame metrics CSV: {frame_csv} ({len(frame_metrics_all)} rows)")

    # ── Step 7: Sequence summary CSV ───────────────────────────────────────
    seq_csv = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_SEQUENCE_SUMMARY_V321.csv")
    if seq_summary:
        fields = list(seq_summary[0].keys())
        with open(seq_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(seq_summary)
        print(f"Sequence summary CSV: {seq_csv} ({len(seq_summary)} rows)")

    # ── Step 8: Pose-PASS summary ─────────────────────────────────────────
    pass_rows = [r for r in seq_summary if not r["pose_fail"]]
    calK_pass = [r for r in pass_rows if r["model"] == "unidepth_calK"]
    total_calK_pass = sum(r["n_frames"] for r in calK_pass)

    posepass_summary = {}
    if calK_pass:
        posepass_summary = {
            "model": "unidepth_calK",
            "evaluation_scope": "pilot_60_posepass",
            "n_sequences": len(calK_pass),
            "n_frames": total_calK_pass,
            "abs_rel_mean": float(np.mean([r["abs_rel_mean"] for r in calK_pass])),
            "abs_rel_median": float(np.mean([r["abs_rel_median"] for r in calK_pass])),
            "rmse_mean": float(np.mean([r["rmse_mean"] for r in calK_pass])),
            "rmse_median": float(np.mean([r["rmse_median"] for r in calK_pass])),
            "delta1_mean": float(np.mean([r["delta1_mean"] for r in calK_pass])),
            "scale_mean": float(np.mean([r["scale_mean"] for r in calK_pass])),
            "scale_median": float(np.mean([r["scale_median"] for r in calK_pass])),
            "scale_cv": float(np.mean([r["scale_cv"] for r in calK_pass])),
        }
    pp_path = os.path.join(AUDIT_DIR, "UNIDEPTH_CALK_POSEPASS_SUMMARY_V321.json")
    with open(pp_path, "w") as f:
        json.dump(posepass_summary, f, indent=2, ensure_ascii=False)
    print(f"Pose-PASS summary: n_frames={total_calK_pass} (expected=60)")

    # ── Step 9: Matched pilot comparison (autonomous vs CalK) ─────────────
    matched_rows = []
    for fm in frame_metrics_all:
        if "auto_raw_absrel" in fm:
            matched_rows.append({
                "sequence": fm["sequence"],
                "pilot_index": fm["pilot_index"],
                "original_index": fm["original_index"],
                "filename": fm["filename"],
                "auto_absrel": fm["auto_raw_absrel"],
                "auto_rmse": fm["auto_raw_rmse_m"],
                "auto_scale": fm["auto_scale_ratio"],
                "calK_absrel": fm["raw_absrel"],
                "calK_rmse": fm["raw_rmse_m"],
                "calK_scale": fm["scale_ratio"],
                "delta_absrel": fm["raw_absrel"] - fm["auto_raw_absrel"],
                "delta_rmse": fm["raw_rmse_m"] - fm["auto_raw_rmse_m"],
                "calK_wins_absrel": 1 if fm["raw_absrel"] < fm["auto_raw_absrel"] else 0,
                "calK_wins_rmse": 1 if fm["raw_rmse_m"] < fm["auto_raw_rmse_m"] else 0,
            })

    matched_csv = os.path.join(AUDIT_DIR, "UNIDEPTH_MATCHED_PILOT_COMPARISON.csv")
    if matched_rows:
        fields = list(matched_rows[0].keys())
        with open(matched_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(matched_rows)
        print(f"Matched pilot CSV: {matched_csv} ({len(matched_rows)} rows)")

    # Paired delta summary
    n_total = len(matched_rows)
    n_wins_absrel = sum(r["calK_wins_absrel"] for r in matched_rows)
    n_wins_rmse = sum(r["calK_wins_rmse"] for r in matched_rows)
    mean_delta_absrel = float(np.mean([r["delta_absrel"] for r in matched_rows]))
    mean_delta_rmse = float(np.mean([r["delta_rmse"] for r in matched_rows]))
    auto_mean_absrel = float(np.mean([r["auto_absrel"] for r in matched_rows]))
    calK_mean_absrel = float(np.mean([r["calK_absrel"] for r in matched_rows]))

    # Per-sequence matched summary
    per_seq_matched = {}
    for r in matched_rows:
        s = r["sequence"]
        if s not in per_seq_matched:
            per_seq_matched[s] = {"deltas_absrel": [], "deltas_rmse": [], "wins_absrel": 0, "total": 0}
        per_seq_matched[s]["deltas_absrel"].append(r["delta_absrel"])
        per_seq_matched[s]["deltas_rmse"].append(r["delta_rmse"])
        per_seq_matched[s]["wins_absrel"] += r["calK_wins_absrel"]
        per_seq_matched[s]["total"] += 1

    print(f"\n=== Matched 80-frame Paired Comparison ===")
    print(f"  Autonomous AbsRel: {auto_mean_absrel:.4f}")
    print(f"  CalK AbsRel:       {calK_mean_absrel:.4f}")
    print(f"  Delta AbsRel:      {mean_delta_absrel:+.4f} ({'CalK worse' if mean_delta_absrel > 0 else 'CalK better'})")
    print(f"  CalK wins AbsRel:  {n_wins_absrel}/{n_total}")
    print(f"  CalK wins RMSE:    {n_wins_rmse}/{n_total}")
    for s, ps in per_seq_matched.items():
        short = s.split("__")[-1]
        d = np.mean(ps["deltas_absrel"])
        print(f"    {short}: delta={d:+.4f} wins={ps['wins_absrel']}/{ps['total']}")

    # ── Step 10: CalK verdict ──────────────────────────────────────────────
    if mean_delta_absrel < -0.01 and n_wins_absrel > n_total * 0.6:
        calK_verdict = "IMPROVES"
    elif mean_delta_absrel > 0.01 and n_wins_absrel < n_total * 0.4:
        calK_verdict = "WORSE"
    else:
        calK_verdict = "NEUTRAL"

    print(f"\n  CalK verdict: {calK_verdict}")

    # ── Step 11: Updated CORRECTED_COMPARISON_V321.csv ────────────────────
    final_rows = []
    for row in seq_summary:
        final_rows.append(row)

    comp_v321 = os.path.join(AUDIT_DIR, "CORRECTED_COMPARISON_V321.csv")
    if final_rows:
        fields = list(final_rows[0].keys())
        with open(comp_v321, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(final_rows)
        print(f"\nCorrected comparison CSV: {comp_v321} ({len(final_rows)} rows)")

    # Pose-PASS summary print
    print("\n=== Pose-PASS Summary (Corrected) ===")
    for model in ["vggt", "da3_metric", "unidepth_auto", "unidepth_calK"]:
        m = [r for r in pass_rows if r["model"] == model]
        if m:
            mean_abs = np.mean([r["abs_rel_mean"] for r in m])
            mean_rmse = np.mean([r["rmse_mean"] for r in m])
            mean_scale = np.mean([r["scale_mean"] for r in m])
            mean_cv = np.mean([r["scale_cv"] for r in m])
            n = sum(r["n_frames"] for r in m)
            print(f"  {model:20s}: AbsRel={mean_abs:.4f} RMSE={mean_rmse:.4f} "
                  f"scale={mean_scale:.4f} CV={mean_cv:.4f} (n={n})")

    # ── Save all results manifest ──────────────────────────────────────────
    manifest = {
        "inference_rerun": False,
        "prediction_total": prediction_total,
        "evaluated_total": total_evaluated,
        "calK_verdict": calK_verdict,
        "matched_comparison": {
            "n_frames": n_total,
            "auto_mean_absrel": auto_mean_absrel,
            "calK_mean_absrel": calK_mean_absrel,
            "mean_delta_absrel": mean_delta_absrel,
            "mean_delta_rmse": mean_delta_rmse,
            "calK_wins_absrel": n_wins_absrel,
            "calK_wins_rmse": n_wins_rmse,
        },
        "completeness": completeness,
    }
    manifest_path = os.path.join(AUDIT_DIR, "PHASE3A21_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
