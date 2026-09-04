#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3C.2 Step 3: Scale stability analysis.

Compare center scale vs dense scale stability metrics.
"""
import csv, os
import numpy as np

ROOT = "/fj/VGGT+head+lora实验"
PHASE3C = os.path.join(ROOT, "阶段3", "03_windowed_pose")
SCALE_DIR = os.path.join(PHASE3C, "11_scale_sync_v32", "02_dense_pairwise_scale")
OUT_DIR = os.path.join(PHASE3C, "11_scale_sync_v32", "02_dense_pairwise_scale")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    csv_path = os.path.join(SCALE_DIR, "PAIRWISE_SCALE_ESTIMATES.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run pairwise_scale_estimates.py first.")
        return

    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append(r)

    # Group by sequence
    seq_ids = sorted(set(r["sequence_id"] for r in rows))

    stability_rows = []
    for sid in seq_ids:
        seq_rows = [r for r in rows if r["sequence_id"] == sid]
        s_center = np.array([float(r["s_center"]) for r in seq_rows])
        s_dense_raw = [r["s_dense"] for r in seq_rows if r["s_dense"] != ""]
        s_dense = np.array([float(x) for x in s_dense_raw]) if s_dense_raw else np.array([])

        def metrics(s, name):
            if len(s) < 2:
                return {}
            log_s = np.log(s)
            return {
                f"{name}_mean": float(np.mean(s)),
                f"{name}_std": float(np.std(s)),
                f"{name}_cv": float(np.std(s) / np.mean(s)),
                f"{name}_median": float(np.median(s)),
                f"{name}_mad_log": float(np.median(np.abs(log_s - np.median(log_s)))),
                f"{name}_range_min": float(np.min(s)),
                f"{name}_range_max": float(np.max(s)),
                f"{name}_range_ratio": float(np.max(s) / np.min(s)),
                f"{name}_max_adjacent_jump": float(np.max(np.abs(np.diff(s)))),
                f"{name}_chain_composed": float(np.prod(s)),
                f"{name}_chain_drift_pct": float(abs(np.prod(s) - 1) * 100),
            }

        m_c = metrics(s_center, "center")
        m_d = metrics(s_dense, "dense")

        # Decision
        if m_c and m_d:
            cv_c = m_c["center_cv"]
            cv_d = m_d["dense_cv"]
            better = "dense" if cv_d < cv_c else "center"
        else:
            better = "center_only"

        row = {"sequence_id": sid, "n_pairs": len(seq_rows), "better_method": better}
        row.update(m_c)
        row.update(m_d)
        stability_rows.append(row)

        print(f"\n{sid}:")
        if m_c:
            print(f"  Center: cv={m_c['center_cv']:.4f} "
                  f"range=[{m_c['center_range_min']:.4f}, {m_c['center_range_max']:.4f}] "
                  f"chain_drift={m_c['center_chain_drift_pct']:.1f}%")
        if m_d:
            print(f"  Dense:  cv={m_d['dense_cv']:.4f} "
                  f"range=[{m_d['dense_range_min']:.4f}, {m_d['dense_range_max']:.4f}] "
                  f"chain_drift={m_d['dense_chain_drift_pct']:.1f}%")
        print(f"  → {better} is MORE STABLE")

    # Save
    if stability_rows:
        out_path = os.path.join(OUT_DIR, "PAIRWISE_SCALE_STABILITY.csv")
        fields = list(stability_rows[0].keys())
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(stability_rows)
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
