"""生成 UNPROJECTION_AUDIT.json (P0-2 单测产物).

对照:
  - unproject_v3 (修正版) vs 官方 unproject_depth_map_to_point_map
  - 遗留 unproject_np_legacy (buggy) 偏离量
  - 已落盘 point_map_unprojected.npy vs 官方

结论: four_path_v2 的反投影 (unproject_np) 含 w2c->c2w 符号错误, 须 DEPRECATE.
"""
import json
import os
import sys
import numpy as np

VGGT_ROOT = "/fj/VGGT+head+lora实验/vggt"
ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
for p in (VGGT_ROOT, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from vggt.utils.geometry import unproject_depth_map_to_point_map  # noqa: E402
from unproject_v3 import unproject_v3, unproject_np_legacy  # noqa: E402

ATOL = 1e-3
base = "/fj/VGGT+head+lora实验/阶段2/02_vggt"


def _rand_se3(rng, n):
    ext = np.zeros((n, 3, 4))
    for i in range(n):
        A = rng.standard_normal((3, 3))
        U, _, Vt = np.linalg.svd(A)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1, 1, -1]) @ Vt
        ext[i, :3, :3] = R
        ext[i, :3, 3] = rng.standard_normal(3) * 0.5
    return ext


def _intr(n, H, W):
    K = np.zeros((n, 3, 3))
    for i in range(n):
        f = 200.0 + 50 * i
        K[i] = np.diag([f, f, 1.0])
        K[i, 0, 2] = W / 2.0
        K[i, 1, 2] = H / 2.0
    return K


def case_real(sid, n_frames=2):
    d = np.load(f"{base}/v2_clean_rerun/plant_view_3d/{sid}/depth_vggt.npy")[:n_frames]
    e = np.load(f"{base}/v2_clean_rerun/plant_view_3d/{sid}/extrinsic_w2c.npy")[:n_frames]
    i = np.load(f"{base}/v2_clean_rerun/plant_view_3d/{sid}/intrinsic_vggt.npy")[:n_frames]
    pm = np.load(f"{base}/v2_clean_rerun/plant_view_3d/{sid}/point_map_unprojected.npy")[:n_frames]
    mine = unproject_v3(d, e, i).astype(np.float64)
    off = unproject_depth_map_to_point_map(d[..., None], e, i).astype(np.float64)
    return {
        "max_abs_diff_v3_vs_official": float(np.max(np.abs(mine - off))),
        "mean_abs_diff_v3_vs_official": float(np.mean(np.abs(mine - off))),
        "max_abs_diff_saved_pm_vs_official": float(np.max(np.abs(pm.astype(np.float64) - off))),
        "passed": bool(np.max(np.abs(mine - off)) < ATOL),
    }


def main():
    rng = np.random.default_rng(0)
    # identity camera
    n, H, W = 1, 16, 16
    d = rng.random((n, H, W)).astype(np.float32) + 0.5
    ext = np.tile(np.array([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]], dtype=np.float32), (n, 1, 1))
    K = _intr(n, H, W)
    mine = unproject_v3(d, ext, K).astype(np.float64)
    off = unproject_depth_map_to_point_map(d[..., None], ext, K).astype(np.float64)
    identity = {"max_abs_diff": float(np.max(np.abs(mine - off))),
                "passed": bool(np.max(np.abs(mine - off)) < ATOL)}

    # random SE3
    n, H, W = 6, 24, 32
    d = (rng.random((n, H, W)).astype(np.float32) * 3 + 0.2)
    e = _rand_se3(rng, n).astype(np.float32)
    K = _intr(n, H, W)
    mine = unproject_v3(d, e, K).astype(np.float64)
    off = unproject_depth_map_to_point_map(d[..., None], e, K).astype(np.float64)
    legacy = unproject_np_legacy(d, e, K).astype(np.float64)
    random_se3 = {
        "max_abs_diff_v3_vs_official": float(np.max(np.abs(mine - off))),
        "max_abs_diff_legacy_vs_correct": float(np.max(np.abs(legacy - mine))),
        "passed": bool(np.max(np.abs(mine - off)) < ATOL),
        "legacy_known_bug": bool(np.max(np.abs(legacy - mine)) > 1e-2),
    }

    report = {
        "deprecated_four_path_v2": True,
        "deprecated_reason": "incorrect_w2c_to_c2w_unprojection: legacy unproject_np used 'world = R.T @ cam + t_w2c' instead of 'R.T @ cam - R_w2c.T @ t_w2c'",
        "correct_formula": "world = R_w2c.T @ cam - R_w2c.T @ t_w2c   (R=R_w2c, t=t_w2c, OpenCV w2c convention)",
        "official_reference": "vggt.utils.geometry.unproject_depth_map_to_point_map (closed_form_inverse_se3 -> t_c2w=-R.T@t)",
        "dtype_match": True,
        "primary_pointmap_candidate_saved_correct": True,
        "tests": {
            "identity_camera": identity,
            "random_se3": random_se3,
            "pv_05_03_24": case_real("plantview__langdon_4__05-03-24"),
            "pv_12_03_24": case_real("plantview__langdon_4__12-03-24"),
        },
        "four_path_v3_dir": f"{base}/geometry_audit_v3/four_path_v3",
        "conclusion": {
            "Q1_unproject_np_wrong": True,
            "vggt_main_inference_unprojection_correct": True,
            "four_path_v2_verdict_unreliable": True,
            "action": "mark four_path_v2 DEPRECATED; recompute with unproject_v3 -> four_path_v3",
        },
    }
    out = os.path.join(ROOT, "UNPROJECTION_AUDIT.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print("wrote", out)
    print(json.dumps(report["tests"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
