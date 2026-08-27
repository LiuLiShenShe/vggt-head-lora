"""Clean rerun Gate 汇总(da3):只经 eval_version.get_active_pose_eval 读取 v2 评价。

门槛(与 PHASE22_REPORT v2 一致):rot median<=10 且 P90<=20;
坍缩判定:sanity.camera_shape_scale_normalized.camera_collapse_scale_relative。

用法: python gate_stats_clean_rerun.py <seq_dir> [...] --out <gate_stats_clean_rerun.json>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, "/fj/VGGT+head+lora实验/阶段2/00_environment")
from eval_version import get_active_pose_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, n_pass = [], 0
    for sd in sorted(args.seq_dirs):
        meta = json.load(open(os.path.join(sd, "prediction_meta.json")))
        pe = get_active_pose_eval(meta)          # 非 v2 一律报错
        rot_med = pe["rotation_error_deg"]["median"]
        rot_p90 = pe["rotation_error_deg"]["p90"]
        collapse = (meta.get("sanity", {})
                      .get("camera_shape_scale_normalized", {})
                      .get("camera_collapse_scale_relative", False))
        nan_inf = sum(meta.get("sanity", {}).get("nan_inf_counts", {}).values())
        ok = (rot_med <= 10 and rot_p90 <= 20)
        n_pass += ok
        rows.append({"sequence_id": meta["sequence_id"], "seq_dir": sd,
                     "rot_median_deg": rot_med, "rot_p90_deg": rot_p90,
                     "center_rel": pe["center_error_aligned"]["relative_to_spread"],
                     "camera_collapse_scale_relative": collapse,
                     "nan_inf_total": nan_inf,
                     "gate_pass": ok})
    summary = {"n_sequences": len(rows), "n_gate_pass": n_pass,
               "gate_rate": round(n_pass / len(rows), 4),
               "criteria": "rot median<=10deg AND P90<=20deg; 坍缩/NaN 单列",
               "sequences": rows}
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for r in rows:
        print(f"{'PASS' if r['gate_pass'] else 'FAIL'}  {r['sequence_id']}: "
              f"med {r['rot_median_deg']} p90 {r['rot_p90_deg']} "
              f"collapse={r['camera_collapse_scale_relative']}")
    print(f"-> {args.out}: {n_pass}/{len(rows)}")


if __name__ == "__main__":
    main()
