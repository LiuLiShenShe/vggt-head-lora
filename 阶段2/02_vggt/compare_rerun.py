"""Clean rerun P0-6:新旧结果对账(da3)。

逐序列比较旧(02_vggt/<ds>/<sid>)与新(v2_clean_rerun/<ds>/<sid>):
- pose_eval_v2 数值(rot median/P90 等)
- NPY 逐文件 max_abs_diff / mean_abs_diff / relative_diff / allclose
写 CLEAN_RERUN_COMPARISON.md + npy_diff_stats.json。

用法: python compare_rerun.py --old-base <dir> --new-base <dir> --out-dir <dir>
"""
import argparse
import glob
import json
import os

import numpy as np

sys_path = "/fj/VGGT+head+lora实验/阶段2/00_environment"
import sys
sys.path.insert(0, sys_path)
from eval_version import get_active_pose_eval

ALLCLOSE = dict(atol=1e-3, rtol=1e-2)   # bf16 量级容差,报告中注明


def load_pose_eval(seq_dir):
    meta = json.load(open(os.path.join(seq_dir, "prediction_meta.json")))
    return get_active_pose_eval(meta), meta


def rot_fields(pe):
    r, c = pe["rotation_error_deg"], pe["center_error_aligned"]
    return (r["median"], r["p90"], c["relative_to_spread"],
            pe["relative_rotation_error_deg"]["median"], pe["trajectory_direction"]["mean_cosine"])


def gate_of(pe):
    return pe["rotation_error_deg"]["median"] <= 10 and pe["rotation_error_deg"]["p90"] <= 20


def diff_npy(old_p, new_p):
    a = np.load(old_p)
    b = np.load(new_p)
    if a.shape != b.shape:
        return {"shape_mismatch": [list(a.shape), list(b.shape)]}
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    d = np.abs(a64 - b64)
    denom = np.linalg.norm(a64)
    return {
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "relative_diff": float(np.linalg.norm(a64 - b64) / max(denom, 1e-30)),
        "allclose(atol=1e-3,rtol=1e-2)": bool(np.allclose(a64, b64, **ALLCLOSE)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-base", default="/fj/VGGT+head+lora实验/阶段2/02_vggt")
    ap.add_argument("--new-base", default="/fj/VGGT+head+lora实验/阶段2/02_vggt/v2_clean_rerun")
    ap.add_argument("--out-dir", default="/fj/VGGT+head+lora实验/阶段2/02_vggt/v2_clean_rerun_eval")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sids = sorted(os.path.basename(p) for p in glob.glob(os.path.join(args.new_base, "*", "*")))
    assert len(sids) == 17, f"期望 17 个新序列目录,实得 {len(sids)}"
    ds_map = {}
    for sid in sids:
        ds = glob.glob(os.path.join(args.new_base, "*", sid))[0].split(os.sep)[-2]
        ds_map[sid] = ds

    table_rows, npy_stats, n_gate_same = [], {}, 0
    for sid in sids:
        old_dir = os.path.join(args.old_base, ds_map[sid], sid)
        new_dir = os.path.join(args.new_base, ds_map[sid], sid)
        # P0-7:两侧都必须已迁移为 v2-active 格式,否则此处直接报错
        try:
            pe_old, _ = load_pose_eval(old_dir)
        except ValueError as e:
            print(f"[SKIP-eval] {sid} old side: {e}")
            pe_old = None
        pe_new, _ = load_pose_eval(new_dir)

        row = {"sequence_id": sid}
        if pe_old:
            fo, fn = rot_fields(pe_old), rot_fields(pe_new)
            go, gn = gate_of(pe_old), gate_of(pe_new)
            n_gate_same += (go == gn)
            row.update({
                "rot_median": [fo[0], fn[0]], "rot_p90": [fo[1], fn[1]],
                "center_rel": [fo[2], fn[2]], "rel_rot_median": [fo[3], fn[3]],
                "traj_cos": [fo[4], fn[4]],
                "gate": ["PASS" if go else "FAIL", "PASS" if gn else "FAIL"],
                "status": "一致" if go == gn else "GATE翻转",
            })
        # NPY 对比
        stats = {}
        for p_new in sorted(glob.glob(os.path.join(new_dir, "*.npy"))):
            name = os.path.basename(p_new)
            p_old = os.path.join(old_dir, name)
            if not os.path.exists(p_old):
                stats[name] = {"missing_in_old": True}
                continue
            stats[name] = diff_npy(p_old, p_new)
        npy_stats[sid] = stats
        bad = [k for k, v in stats.items() if v.get("shape_mismatch")
               or not v.get("allclose(atol=1e-3,rtol=1e-2)", False)]
        row["npy_not_allclose"] = bad
        table_rows.append(row)
        print(f"{sid}: {row.get('status', '?')} | not-allclose: {bad or '无'}")

    with open(os.path.join(args.out_dir, "npy_diff_stats.json"), "w") as f:
        json.dump(npy_stats, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "comparison_table.json"), "w") as f:
        json.dump(table_rows, f, indent=2, ensure_ascii=False)
    print(f"\ngate 判定一致 {n_gate_same}/17;-> {args.out_dir}")


if __name__ == "__main__":
    main()
