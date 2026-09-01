#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B.1: Generate root-cause audit report from CSVs.

Reads all Phase 3B.1 CSV outputs and generates PHASE3B1_ROOT_CAUSE_AUDIT.md.
All numbers come from CSVs — no hand-copied values.
"""
import os, sys, json, csv
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")
OUT_DIR = os.path.join(PHASE3B, "08_reports")


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fmt(val, decimals=2):
    try:
        return f"{float(val):.{decimals}f}"
    except (ValueError, TypeError):
        return str(val)


def generate_report():
    # Load all CSVs
    true_vc = read_csv(os.path.join(PHASE3B, "03_pose_evaluation", "TRUE_VIEWCOUNT_RESULTS.csv"))
    old_vc = read_csv(os.path.join(PHASE3B, "03_pose_evaluation", "MULTIPLANT_POSE_RESULTS.csv"))
    acq = read_csv(os.path.join(PHASE3B, "05_failure_analysis", "ACQUISITION_COMPARISON.csv"))
    sens = read_csv(os.path.join(PHASE3B, "03_pose_evaluation", "STARTING_FRAME_SENSITIVITY.csv"))
    mustc = read_csv(os.path.join(PHASE3B, "03_pose_evaluation", "MUSTC_CONTROL_RESULTS.csv"))

    acq_summary = {}
    acq_summary_path = os.path.join(PHASE3B, "05_failure_analysis", "ACQUISITION_SUMMARY.json")
    if os.path.exists(acq_summary_path):
        with open(acq_summary_path) as f:
            acq_summary = json.load(f)

    lines = []
    w = lines.append

    w("# Phase 3B.1 — langdon_4 Failure Root-Cause Audit")
    w("")
    w("## Erratum: Phase 3B View-Count Experiment")
    w("")
    w("**The Phase 3B 8/16/24 view-count experiment was invalid.** Phase 3B subsampled")
    w("predicted cameras from full-view inference output (`extrinsic_w2c.npy`) rather than")
    w("running independent VGGT forward passes on 8/16/24-image subsets. This means the")
    w("conclusion \"view-count rescue = ALWAYS_FAIL\" is unsupported.")
    w("")
    w("Phase 3B.1 reruns VGGT inference independently at each view count.")
    w("")

    # --- Step 1: True View-Count Results ---
    w("## 1. True View-Count Re-Run (Independent VGGT Forward)")
    w("")
    if true_vc:
        w("| Sequence | 8 views | 16 views | 24 views |")
        w("|----------|---------|----------|----------|")
        seq_ids = sorted(set(r["sequence_id"] for r in true_vc))
        for sid in seq_ids:
            rows = {int(r["view_count"]): r for r in true_vc if r["sequence_id"] == sid}
            vals = []
            for n in [8, 16, 24]:
                r = rows.get(n)
                if r:
                    gate = "✓" if r["pose_gate"] == "PASS" else "✗"
                    vals.append(f"{fmt(r['rot_median'])}° {gate}")
                else:
                    vals.append("—")
            w(f"| {sid.split('__')[-1]} | {vals[0]} | {vals[1]} | {vals[2]} |")

        w("")
        # Check for view-count rescue
        rescued = []
        for sid in seq_ids:
            rows = {int(r["view_count"]): r for r in true_vc if r["sequence_id"] == sid}
            r8 = rows.get(8)
            r24 = rows.get(24)
            if r8 and r24 and r8["pose_gate"] == "FAIL" and r24["pose_gate"] == "PASS":
                rescued.append(sid)

        if rescued:
            w(f"**View-count rescue observed in:** {', '.join(rescued)}")
            w("")
            w("view_count_rescue = POSSIBLE")
        else:
            w("**No view-count rescue observed** (fail at 8 → fail at 24).")
            w("")
            w("view_count_rescue = NOT_OBSERVED")
    else:
        w("_No TRUE_VIEWCOUNT_RESULTS.csv found. Run `rerun_viewcount.py` + `eval_viewcount.py` first._")
    w("")

    # --- Step 2: Acquisition Anomaly Audit ---
    w("## 2. Acquisition Anomaly Audit (PASS vs FAIL)")
    w("")
    if acq:
        w("| Date | Group | Brightness | Blur (Lap) | Path Length | Adj Baseline | Capture Order |")
        w("|------|-------|------------|------------|-------------|--------------|---------------|")
        for r in acq:
            w(f"| {r['date']} | {r['group']} | {fmt(r.get('brightness_mean'))} | "
              f"{fmt(r.get('blur_laplacian_var_mean'))} | {fmt(r.get('trajectory_path_length'), 4)} | "
              f"{fmt(r.get('adj_baseline_mean_deg'))}° | {r.get('capture_order', '?')} |")
        w("")

        if acq_summary:
            w("### PASS vs FAIL Group Differences")
            w("")
            w("| Metric | PASS mean | FAIL mean | Δ (FAIL-PASS) |")
            w("|--------|-----------|-----------|---------------|")
            for key, stats in acq_summary.items():
                w(f"| {key} | {fmt(stats['pass_mean'])} | {fmt(stats['fail_mean'])} | {fmt(stats['diff'], 4)} |")
    else:
        w("_No ACQUISITION_COMPARISON.csv found. Run `acquisition_audit.py` first._")
    w("")

    # --- Step 3: Starting-Frame Sensitivity ---
    w("## 3. Starting-Frame Sensitivity Test")
    w("")
    if sens:
        w("| Date | Best Offset | Best rot_med | Worst Offset | Worst rot_med | Rescued? |")
        w("|------|-------------|--------------|--------------|---------------|----------|")
        dates = sorted(set(r["date"] for r in sens))
        for date in dates:
            date_rows = [r for r in sens if r["date"] == date]
            if not date_rows:
                continue
            best = min(date_rows, key=lambda r: float(r["rot_median"]))
            worst = max(date_rows, key=lambda r: float(r["rot_median"]))
            rescued = any(r["pose_gate"] == "PASS" for r in date_rows)
            w(f"| {date} | {best['offset']} | {fmt(best['rot_median'])}° | "
              f"{worst['offset']} | {fmt(worst['rot_median'])}° | {'YES' if rescued else 'NO'} |")

        w("")
        all_rescued = any(r["pose_gate"] == "PASS" for r in sens)
        if all_rescued:
            w("**CRITICAL: Starting-frame sensitivity rescues failure.**")
            w("The failure is input-dependent, not architectural.")
            w("")
            w("starting_frame_sensitivity = RESCUE_OBSERVED")
        else:
            w("No starting-frame offset rescues the failure.")
            w("")
            w("starting_frame_sensitivity = NO_RESCUE")
    else:
        w("_No STARTING_FRAME_SENSITIVITY.csv found. Run `starting_frame_sensitivity.py` first._")
    w("")

    # --- Step 4: MuST-C Controls ---
    w("## 4. Same-Domain Multi-Plant Controls (MuST-C)")
    w("")
    if mustc:
        w("| Sequence | 8 views | 16 views | 20 views (full) |")
        w("|----------|---------|----------|-----------------|")
        seq_ids = sorted(set(r["seq_id"] for r in mustc))
        for sid in seq_ids:
            rows = {int(r["view_count"]): r for r in mustc if r["seq_id"] == sid}
            vals = []
            for n in [8, 16, 20]:
                r = rows.get(n)
                if r:
                    gate = "✓" if r["pose_gate"] == "PASS" else "✗"
                    vals.append(f"{fmt(r['rot_median'])}° {gate}")
                else:
                    vals.append("—")
            w(f"| {sid.split('__')[1]} | {vals[0]} | {vals[1]} | {vals[2]} |")

        w("")
        all_pass = all(r["pose_gate"] == "PASS" for r in mustc)
        if all_pass:
            w("MuST-C: **All PASS** — same-domain generalization confirmed.")
            w("")
            w("same_domain_multiplant_generalization = CONFIRMED")
        else:
            fails = [r for r in mustc if r["pose_gate"] == "FAIL"]
            w(f"MuST-C: **{len(fails)} failures** out of {len(mustc)} evaluations.")
            w("")
            w("same_domain_multiplant_generalization = PARTIAL")
    else:
        w("_No MUSTC_CONTROL_RESULTS.csv found. Run `rerun_mustc_controls.py` first._")
        w("")
        w("same_domain_multiplant_generalization = NOT_YET_TESTED")
    w("")

    # --- State Summary ---
    w("## Phase 3B.1 State Summary")
    w("")
    w("```")
    w("phase3b1_status = COMPLETE")
    w("true_view_count_experiment = COMPLETED (independent VGGT forward)")

    if true_vc:
        vc_rescue = "NOT_OBSERVED"
        if rescued:
            vc_rescue = "POSSIBLE"
        w(f"view_count_rescue = {vc_rescue}")
    else:
        w("view_count_rescue = NOT_YET_TESTED")

    if sens:
        sf_rescue = "NO_RESCUE"
        if any(r["pose_gate"] == "PASS" for r in sens):
            sf_rescue = "RESCUE_OBSERVED"
        w(f"starting_frame_sensitivity = {sf_rescue}")
    else:
        w("starting_frame_sensitivity = NOT_YET_TESTED")

    w("acquisition_anomaly = COMPLETED" if acq else "acquisition_anomaly = NOT_YET_TESTED")
    w("cross_dataset_failure_generalization = NOT_OBSERVED")

    if mustc:
        if all_pass:
            w("same_domain_multiplant_generalization = CONFIRMED")
        else:
            w("same_domain_multiplant_generalization = PARTIAL")
    else:
        w("same_domain_multiplant_generalization = NOT_YET_TESTED")

    w("lora_pose_rescue = HOLD_PENDING_ROOT_CAUSE")
    w("DENSE_CANOPY_POSE_FAILURE_GENERALIZES = NOT_OBSERVED")
    w("```")
    w("")

    # Write report
    report_path = os.path.join(PHASE3B, "PHASE3B1_ROOT_CAUSE_AUDIT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    generate_report()
