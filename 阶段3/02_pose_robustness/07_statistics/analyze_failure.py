#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3B Step 5: Statistical analysis — failure rates, correlations, view-count effect."""
import os, sys, json, csv
import numpy as np
from scipy import stats

ROOT = "/fj/VGGT+head+lora实验"
PHASE3B = os.path.join(ROOT, "阶段3", "02_pose_robustness")


def main():
    # Load results
    results_path = os.path.join(PHASE3B, "03_pose_evaluation", "MULTIPLANT_POSE_RESULTS.csv")
    with open(results_path) as f:
        all_rows = list(csv.DictReader(f))

    for r in all_rows:
        r["rot_median"] = float(r["rot_median"])
        r["rot_p90"] = float(r["rot_p90"])
        r["center_median_norm"] = float(r["center_median_norm"])
        r["canopy_fraction"] = float(r["canopy_fraction"])
        r["view_count"] = int(r["view_count"])

    # ── 1. Overall pass rate ──────────────────────────────────────────────
    full_rows = [r for r in all_rows if r["view_count"] > 24]
    n_total = len(full_rows)
    n_pass = sum(1 for r in full_rows if r["pose_gate"] == "PASS")
    print(f"Overall pass rate (full views): {n_pass}/{n_total} = {n_pass/n_total:.1%}")

    # ── 2. Failure rate by plant ──────────────────────────────────────────
    plant_results = {}
    for r in full_rows:
        pid = r["plant_id"]
        if pid not in plant_results:
            plant_results[pid] = {"pass": 0, "fail": 0, "n_dates": 0}
        plant_results[pid]["n_dates"] += 1
        if r["pose_gate"] == "PASS":
            plant_results[pid]["pass"] += 1
        else:
            plant_results[pid]["fail"] += 1

    print(f"\n=== Failure Rate by Plant (full views) ===")
    for pid, pr in sorted(plant_results.items()):
        rate = pr["fail"] / pr["n_dates"] if pr["n_dates"] > 0 else 0
        print(f"  {pid:25s}: {pr['fail']}/{pr['n_dates']} fail ({rate:.0%})")

    n_plants_with_fail = sum(1 for pr in plant_results.values() if pr["fail"] > 0)
    print(f"\nPlants with ≥1 failure: {n_plants_with_fail}/{len(plant_results)}")

    # ── 3. Failure rate by density class ──────────────────────────────────
    density_results = {}
    for r in full_rows:
        dc = r["density_class"]
        if dc not in density_results:
            density_results[dc] = {"pass": 0, "fail": 0}
        if r["pose_gate"] == "PASS":
            density_results[dc]["pass"] += 1
        else:
            density_results[dc]["fail"] += 1

    print(f"\n=== Failure Rate by Density Class ===")
    for dc in ["LOW", "MEDIUM", "HIGH"]:
        if dc in density_results:
            dr = density_results[dc]
            total = dr["pass"] + dr["fail"]
            print(f"  {dc:8s}: {dr['fail']}/{total} fail ({dr['fail']/total:.0%})")

    # ── 4. Canopy fraction correlation ────────────────────────────────────
    canopy_vals = np.array([r["canopy_fraction"] for r in full_rows])
    rot_vals = np.array([r["rot_median"] for r in full_rows])
    fail_vals = np.array([1 if r["pose_gate"] == "FAIL" else 0 for r in full_rows])

    rho_rot, p_rot = stats.spearmanr(canopy_vals, rot_vals)
    rho_fail, p_fail = stats.spearmanr(canopy_vals, fail_vals)

    print(f"\n=== Correlations ===")
    print(f"  Spearman(rot_median, canopy_fraction): rho={rho_rot:.3f}, p={p_rot:.4f}, N={len(canopy_vals)}")
    print(f"  Spearman(pose_fail, canopy_fraction):  rho={rho_fail:.3f}, p={p_fail:.4f}, N={len(fail_vals)}")

    # ── 5. View-count effect ──────────────────────────────────────────────
    view_counts = [8, 16, 24]
    print(f"\n=== View-Count Effect ===")
    for seq_id in sorted(set(r["sequence_id"] for r in all_rows)):
        seq_rows = [r for r in all_rows if r["sequence_id"] == seq_id]
        full = [r for r in seq_rows if r["view_count"] > 24]
        full_gate = full[0]["pose_gate"] if full else "N/A"

        vc_gates = {}
        for r in seq_rows:
            if r["view_count"] in view_counts:
                vc_gates[r["view_count"]] = r["pose_gate"]

        # Classify
        all_fail = all(g == "FAIL" for g in vc_gates.values())
        all_pass = all(g == "PASS" for g in vc_gates.values())

        if all_pass:
            effect = "ALWAYS_PASS"
        elif all_fail:
            effect = "ALWAYS_FAIL"
        elif vc_gates.get(8) == "FAIL" and vc_gates.get(24) == "PASS":
            effect = "RESCUED_BY_MORE_VIEWS"
        elif vc_gates.get(8) == "PASS" and vc_gates.get(24) == "FAIL":
            effect = "WORSENED"
        else:
            effect = "NO_EFFECT"

        print(f"  {seq_id}: full={full_gate} 8v={vc_gates.get(8,'?')} 16v={vc_gates.get(16,'?')} 24v={vc_gates.get(24,'?')} → {effect}")

    # ── 6. Failure type distribution ──────────────────────────────────────
    type_counts = {}
    for r in full_rows:
        ft = r["failure_type"]
        type_counts[ft] = type_counts.get(ft, 0) + 1

    print(f"\n=== Failure Types (full views) ===")
    for ft, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ft:30s}: {count}")

    # ── 7. Save statistics ────────────────────────────────────────────────
    stats_data = {
        "n_sequences": n_total,
        "n_unique_plants": len(plant_results),
        "pose_gate_pass_rate": round(n_pass / n_total, 4) if n_total else 0,
        "n_plants_with_failure": n_plants_with_fail,
        "failure_by_plant": {pid: pr for pid, pr in plant_results.items()},
        "failure_by_density": density_results,
        "correlations": {
            "rot_vs_canopy_spearman_rho": round(float(rho_rot), 4),
            "rot_vs_canopy_p_value": round(float(p_rot), 6),
            "fail_vs_canopy_spearman_rho": round(float(rho_fail), 4),
            "fail_vs_canopy_p_value": round(float(p_fail), 6),
        },
        "failure_types": type_counts,
    }

    stats_path = os.path.join(PHASE3B, "07_statistics", "POSE_FAILURE_SUMMARY.csv")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)

    # Save per-plant summary
    plant_csv = os.path.join(PHASE3B, "07_statistics", "POSE_PLANT_LEVEL_SUMMARY.csv")
    with open(plant_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["plant_id", "n_dates", "n_pass", "n_fail", "fail_rate"])
        w.writeheader()
        for pid, pr in sorted(plant_results.items()):
            w.writerow({
                "plant_id": pid,
                "n_dates": pr["n_dates"],
                "n_pass": pr["pass"],
                "n_fail": pr["fail"],
                "fail_rate": round(pr["fail"] / pr["n_dates"], 4) if pr["n_dates"] else 0,
            })
    print(f"\nSaved: {plant_csv}")

    # Save correlation results
    corr_csv = os.path.join(PHASE3B, "07_statistics", "POSE_DENSITY_CORRELATION.csv")
    with open(corr_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "rho", "p_value", "n"])
        w.writeheader()
        w.writerow({"metric": "rot_median_vs_canopy", "rho": round(float(rho_rot), 4),
                     "p_value": round(float(p_rot), 6), "n": len(canopy_vals)})
        w.writerow({"metric": "pose_fail_vs_canopy", "rho": round(float(rho_fail), 4),
                     "p_value": round(float(p_fail), 6), "n": len(fail_vals)})
    print(f"Saved: {corr_csv}")

    # Save stats JSON
    json_path = os.path.join(PHASE3B, "07_statistics", "POSE_FAILURE_SUMMARY.json")
    with open(json_path, "w") as f:
        json.dump(stats_data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
