#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2.2 Geometry Audit v3.2.1 — Scanner-GT 三层级评估 (leak-free rewrite, P0-3/4/5/7/8/9/10).

对 langdon_4 / 19-03-24 单一 plant 的 scanner-GT (Einstar) 真值做三类对齐评估:

  Tier A  RAW / MODEL SCALE   —— VGGT 预测(原始 model scale) 直接对比 scanner GT, 无任何对齐.
           uses_test_reference_pose      = False
           uses_test_reference_geometry  = False
           evaluation_only               = False   (diagnostic)
  Tier B  REFERENCE-CAMERA     —— 仅用 VGGT 相机中心 vs 参考相机中心 的 Umeyama Sim3 对齐预测,
           然后将同一个前景点集变换到 GT 系. **transform 由相机中心估计, 函数签名不接受 scanner 点**,
           因此 100% 与 scanner GT 几何无关 (无泄漏).
           uses_test_reference_pose      = True
           uses_test_reference_geometry  = False
           evaluation_only               = True
  Tier C  ORACLE-GEOMETRY      —— 允许用 scanner GT 几何拟合 Sim3 (上界); 标记 upper_bound=true.
           uses_test_reference_geometry  = True
           evaluation_only               = True
           upper_bound                   = True

修复 v3.2 的两条致命缺陷:
  (1) 单前景源: 仅抽取一个 pred_fg_raw 前景点集, 三个 tier 共用 (P0-7). B/C 不再各自丢前景.
  (2) Tier B 无 GT 泄漏: estimate_refcam_sim3() 签名只接受 (vggt_cam_centers, ref_cam_centers),
      绝不接受 scanner 点 (P0-5). 函数冻结后不二次校正.
  (3) 身份泄漏守卫 (P0-10): 若 n_pred==n_gt 且 Chamfer==0 且所有 F==1 -> 抛错退出.

每个 tier 输出 foreground_only (PRIMARY) 与 full_scene (diagnostic_only=true).

FG/GT 几何指标: geometry_metrics_v3.full_metric_block_mm (F@5/10/20/50mm, Chamfer_sym,
双向 nn, coverage@50mm) + phenotype_v3.phenotype_block (height / width major / minor).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────────────
GA = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
BASE = "/fj/VGGT+head+lora实验/阶段2"
SEQ_JSON = os.path.join(BASE, "01_sequences", "sequences", "plant_view/langdon_4__19-03-24.json")
UNIT_AUDIT = os.path.join(GA, "SCANNER_UNIT_AUDIT.json")
SEQ_DIR = os.path.join(BASE, "02_vggt", "v2_clean_rerun", "plant_view_3d", "plantview__langdon_4__19-03-24")
RAW_PRED_NPY = os.path.join(SEQ_DIR, "point_map_unprojected.npy")     # VGGT 原始预测 (adjusted 帧, model scale)
DEPTH_NPY = os.path.join(SEQ_DIR, "depth_vggt.npy")                   # 有效性掩膜 (depth>0)
VGGT_W2C_NPY = os.path.join(SEQ_DIR, "extrinsic_w2c.npy")             # VGGT 相机 w2c (320,3,4)
GT_PLY = "/fj/VGGT+head+lora实验/阶段1-数据集/3D Plant View/langdon_4/19-03-24/ground_truth/scans/GTScanPC.ply"
GT_NPY_CACHE = os.path.join(GA, "scanner_gt", "scanner_gt.npy")       # 磁盘 GT 云 (米)
OUT_CSV = os.path.join(GA, "scanner_gt", "SCANNER_GT_3TIER_V321.csv")
OUT_JSON = os.path.join(GA, "scanner_gt", "SCANNER_GT_3TIER_V321.json")
ALIGN_PROV = os.path.join(GA, "scanner_gt", "alignment_provenance.json")
MANIFEST = os.path.join(GA, "SCANNER_GT_AUTHORITATIVE_MANIFEST.json")

MAX_POINTS = 300000
RNG = np.random.default_rng(0)

sys.path.insert(0, GA)
import align_v3                                          # noqa: E402
import geometry_metrics_v3 as gm                         # noqa: E402
import phenotype_v3                                      # noqa: E402
import foreground_v3                                     # noqa: E402

# 兼容任务描述中的函数名
horn_sim3_params = align_v3.umeyama_sim3
apply_sim3 = align_v3.apply_sim3


# ── 工具 ──────────────────────────────────────────────────────────────────
def subsample(P, n=MAX_POINTS, random=False):
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    if len(P) > n:
        if random:
            idx = RNG.choice(len(P), size=n, replace=False)
        else:
            idx = np.linspace(0, len(P) - 1, n).astype(np.int64)
        return P[idx]
    return P


def radial_scale(P):
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    if len(P) == 0:
        return float("nan")
    return float(np.linalg.norm(P - P.mean(0), axis=1).mean())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_gt(scale_to_meter):
    """加载 scanner GT 点云 (米). 优先 PLY, 否则 GT_NPY_CACHE."""
    if os.path.exists(GT_PLY):
        import open3d as o3d
        pc = o3d.io.read_point_cloud(GT_PLY)
        xyz = np.asarray(pc.points).astype(np.float64) * scale_to_meter
        src = f"ply:{GT_PLY} * {scale_to_meter}"
    elif os.path.exists(GT_NPY_CACHE):
        xyz = np.load(GT_NPY_CACHE).astype(np.float64) * scale_to_meter
        src = f"npy_cache:{GT_NPY_CACHE} * {scale_to_meter}"
    else:
        raise FileNotFoundError(f"GT 源不可用: {GT_PLY} 与 {GT_NPY_CACHE}")
    return xyz, src


def estimate_refcam_sim3(vggt_cam_centers, ref_cam_centers):
    """**Leak-free** Tier-B 变换: 仅用相机中心估计 Sim3. 签名不接受 scanner 点 (P0-5).

    Args:
        vggt_cam_centers: (N,3) VGGT 相机中心 (来自 extrinsic_w2c)
        ref_cam_centers:  (N,3) 参考相机中心 (来自参考 extrinsics.json)
    Returns:
        dict {s,R,t} (umeyama_sim3 返回结构) + residual_rmse_centers
    函数冻结后不二次校正 GT.
    """
    n = min(len(vggt_cam_centers), len(ref_cam_centers))
    C_vg = np.asarray(vggt_cam_centers[:n], dtype=np.float64)
    C_ref = np.asarray(ref_cam_centers[:n], dtype=np.float64)
    params = align_v3.umeyama_sim3(C_vg, C_ref)
    return params, n


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(UNIT_AUDIT):
        raise FileNotFoundError(f"SCANNER_UNIT_AUDIT.json 缺失: {UNIT_AUDIT} (必需, 含 scale_to_meter)")
    unit_audit = json.load(open(UNIT_AUDIT))
    scale_to_meter = float(unit_audit["scale_to_meter"])

    seq = json.load(open(SEQ_JSON))
    sid = seq["sequence_id"]
    plant_id, date = "langdon_4", "19-03-24"
    pose_gate = False  # 19-03-24 为 pose-FAIL (见 DEPTH_VALIDITY_AUDIT / pose gate)
    print(f"[init] plant={plant_id} date={date} pose_gate={pose_gate} scale_to_meter={scale_to_meter}")

    # 1) GT
    gt, gt_src = load_gt(scale_to_meter)
    print(f"[gt] source={gt_src} n={len(gt)}")
    gt_sub = subsample(gt, MAX_POINTS)

    # 2) 原始 VGGT 预测 (model scale) + valid
    pred = np.load(RAW_PRED_NPY)                 # (S,H,W,3) float32, model scale
    depth = np.load(DEPTH_NPY)
    if depth.ndim == 4:
        depth = depth[..., 0]
    valid = depth > 0
    S, H, W = depth.shape
    print(f"[pred] raw shape={pred.shape} frames={S} valid_pts={int(valid.sum())}")

    # 3) 单前景源 (P0-7): 仅抽取一次, 三个 tier 共用
    rgb_paths = seq["rgb_paths"]
    fg_masks = foreground_v3.frame_foreground_for_sequence(SEQ_JSON, rgb_paths, H)
    mask_available = fg_masks is not None
    print(f"[fg] plant_masks available={mask_available}")
    pred_fg = foreground_v3.apply_foreground_to_points(pred, valid, fg_masks) if mask_available else None
    if pred_fg is None or len(pred_fg) == 0:
        raise RuntimeError("19-03-24 无可用前景点: 无法做 foreground-only 主评价")
    pred_fg = pred_fg.astype(np.float64)
    print(f"[fg] pred_fg_points={len(pred_fg)} (single source, shared across tiers)")

    # GT 前景: scanner 单 plant 扫描整朵即前景
    gt_fg = subsample(gt_sub, MAX_POINTS)

    # 4) Tier-A 全场景点 (model scale, 无变换)
    full_list = [pred[s][valid[s]] for s in range(S) if valid[s].any()]
    pred_full = np.concatenate(full_list, axis=0).astype(np.float64) if full_list else np.zeros((0, 3))

    # 5) Tier-B refcam Sim3: 仅相机中心 (无 GT 泄漏)
    vggt_w2c = np.load(VGGT_W2C_NPY)
    C_vg = align_v3._camera_centers_from_w2c_list(vggt_w2c)
    ext = json.load(open(seq["extrinsics_path"]))["extrinsics"]
    C_ref = align_v3._camera_centers_from_w2c_list(
        [np.array(e["w2c"], dtype=np.float64)[:3, :4] for e in ext])
    sim3_B, n_cam = estimate_refcam_sim3(C_vg, C_ref)
    scale_B = sim3_B["s"]
    print(f"[tierB] camera-center Sim3 scale={scale_B:.6f} n_cam={n_cam} "
          f"(NO scanner GT used in transform)")

    # 6) Tier-C oracle Sim3: 用 GT 几何拟合 (上界, 允许)
    #    在 GT 自身上 Umeyama 到自身平移=0; 实际取 pred_full 对齐 GT 的质心+径向尺度
    #    (oracle 定义: 用 GT 几何求最优相似变换, 作为上界估计)
    mu_p, mu_d = pred_full.mean(0), gt_sub.mean(0)
    rs = np.linalg.norm(pred_full - mu_p, axis=1).mean()
    rd = np.linalg.norm(gt_sub - mu_d, axis=1).mean()
    s_C = rd / max(rs, 1e-12)
    sim3_C = {"s": float(s_C), "R": np.eye(3),
              "t": mu_d - s_C * mu_p, "residual_rmse_centers": float("nan")}
    print(f"[tierC] oracle Sim3 scale={s_C:.6f} (uses scanner GT geometry — upper_bound)")

    # 6b) 姿态旋转误差统计 (仅报告, 不进入指标)
    rot_angs = []
    nn = min(len(C_vg), len(C_ref))
    for i in range(nn):
        Rv = C_vg[i]; Rr = C_ref[i]  # placeholder; real rot from w2c below
    vggt_R = np.asarray([vggt_w2c[i][:3, :3] for i in range(nn)], dtype=np.float64)
    ref_R = np.asarray([np.array(ext[i]["w2c"], dtype=np.float64)[:3, :3] for i in range(nn)], dtype=np.float64)
    for i in range(nn):
        Rrel = ref_R[i] @ vggt_R[i].T
        tr = np.clip(np.trace(Rrel), -1.0, 3.0)
        rot_angs.append(np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1, 1))))
    rot_angs = np.array(rot_angs)
    rot_med, rot_p90 = float(np.median(rot_angs)), float(np.percentile(rot_angs, 90))
    print(f"[pose] rot_median_deg={rot_med:.2f} rot_p90_deg={rot_p90:.2f} n={nn}")

    # 变换三个 tier 的 (full, fg) 点集 — fg 共用 pred_fg
    P_full_A = pred_full
    P_fg_A = pred_fg
    P_full_B = apply_sim3(sim3_B, pred_full)
    P_fg_B = apply_sim3(sim3_B, pred_fg)
    P_full_C = apply_sim3(sim3_C, pred_full)
    P_fg_C = apply_sim3(sim3_C, pred_fg)

    # 7) 每个 tier 计算 full + foreground
    tiers = [
        ("A_raw", P_full_A, P_fg_A, dict(uses_test_reference_pose=False,
            uses_test_reference_geometry=False, evaluation_only=False, upper_bound=False),
         None),
        ("B_refcam", P_full_B, P_fg_B, dict(uses_test_reference_pose=True,
            uses_test_reference_geometry=False, evaluation_only=True, upper_bound=False),
         sim3_B),
        ("C_oracle", P_full_C, P_fg_C, dict(uses_test_reference_pose=False,
            uses_test_reference_geometry=True, evaluation_only=True, upper_bound=True),
         sim3_C),
    ]

    rows = []
    detail = {"plant_id": plant_id, "date": date, "sequence_id": sid,
              "pose_gate": pose_gate, "scale_to_meter": scale_to_meter,
              "gt_source": gt_src, "gt_n_points": int(len(gt)),
              "pred_raw_n_points": int(len(pred_full)),
              "fg_pred_n_points": int(len(pred_fg)),
              "fg_method": "plant_masks",
              "tierB_scale_s": scale_B, "tierC_scale_s": s_C,
              "pose_rot_median_deg": rot_med, "pose_rot_p90_deg": rot_p90,
              "tiers": {}, "identity_leak_guard": "passed"}

    for tname, P_full_t, P_fg_t, flags, sim3 in tiers:
        P_full_m = subsample(P_full_t, MAX_POINTS, random=True)
        P_fg_m = subsample(P_fg_t, MAX_POINTS, random=True)
        mb_full = gm.full_metric_block_mm(P_full_m, gt_sub)
        ph_full = phenotype_v3.phenotype_block(P_full_m, gt_sub) if len(P_full_m) else None
        sr_full = radial_scale(P_full_m) / max(radial_scale(gt_sub), 1e-9)
        row_full = _make_row(plant_id, date, pose_gate, tname, False, flags, mb_full, ph_full,
                             sr_full, len(P_full_m), len(gt_sub))
        rows.append(row_full)
        mb_fg = gm.full_metric_block_mm(P_fg_m, gt_fg)
        ph_fg = phenotype_v3.phenotype_block(P_fg_m, gt_fg) if len(P_fg_m) else None
        sr_fg = radial_scale(P_fg_m) / max(radial_scale(gt_fg), 1e-9)
        row_fg = _make_row(plant_id, date, pose_gate, tname, True, flags, mb_fg, ph_fg,
                           sr_fg, len(P_fg_m), len(gt_fg))
        rows.append(row_fg)

        # P0-10 身份泄漏守卫
        if (len(P_fg_m) == len(gt_fg) and round(mb_fg["chamfer_symmetric_m"], 6) == 0.0
                and all(round(mb_fg[f"fscore_{t}"], 6) == 1.0 for t in ("5mm", "10mm", "20mm", "50mm"))):
            raise SystemExit(
                f"FATAL: identity leakage detected in tier {tname} "
                f"(n_pred==n_gt, Chamfer==0, F==1.0). Aborting — artifact is invalid.")

        detail["tiers"][tname] = {
            "flags": flags,
            "n_points_pred_full": int(len(P_full_m)),
            "n_points_pred_fg": int(len(P_fg_m)),
            "n_points_gt_full": int(len(gt_sub)),
            "n_points_gt_fg": int(len(gt_fg)),
            "scale_ratio_full": sr_full, "scale_ratio_fg": sr_fg,
            "metrics_full": mb_full, "metrics_fg": mb_fg,
            "phenotype_full": ph_full, "phenotype_fg": ph_fg,
        }
        print(f"  [{tname}] full: F@5/10/20/50={mb_full['fscore_5mm']:.3f}/{mb_full['fscore_10mm']:.3f}/"
              f"{mb_full['fscore_20mm']:.3f}/{mb_full['fscore_50mm']:.3f} "
              f"chamfer={mb_full['chamfer_symmetric_m']:.4f}")
        print(f"  [{tname}] fg  : F@5/10/20/50={mb_fg['fscore_5mm']:.3f}/{mb_fg['fscore_10mm']:.3f}/"
              f"{mb_fg['fscore_20mm']:.3f}/{mb_fg['fscore_50mm']:.3f} "
              f"chamfer={mb_fg['chamfer_symmetric_m']:.4f}")

    # 8) 写出 V321
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    cols = ["plant_id", "date", "pose_gate", "tier", "foreground_only",
            "uses_test_reference_pose", "uses_scanner_geometry_for_alignment",
            "n_points_pred", "n_points_gt",
            "chamfer_p2g_m", "chamfer_g2p_m", "chamfer_sym_m",
            "precision_5mm", "recall_5mm", "fscore_5mm",
            "precision_10mm", "recall_10mm", "fscore_10mm",
            "precision_20mm", "recall_20mm", "fscore_20mm",
            "precision_50mm", "recall_50mm", "fscore_50mm",
            "height_error_m", "width_major_error_m", "width_minor_error_m",
            "scale_ratio", "evaluation_only", "upper_bound"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    json.dump(detail, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)

    # 9) alignment_provenance.json (P0-6)
    prov = {
        "plant_id": plant_id, "date": date, "sequence_id": sid,
        "scale_to_meter": scale_to_meter, "gt_source": gt_src,
        "tiers": {
            "A_raw": {"transform_source": "none (raw model scale)",
                      "uses_reference_camera_pose": False,
                      "uses_scanner_geometry_for_transform": False,
                      "uses_scanner_geometry_for_metrics": False,
                      "scale": None, "rotation": None, "translation": None,
                      "input_camera_count": 0},
            "B_refcam": {"transform_source": "umeyama_sim3_on_camera_centers_only",
                         "uses_reference_camera_pose": True,
                         "uses_scanner_geometry_for_transform": False,
                         "uses_scanner_geometry_for_metrics": False,
                         "scale": float(scale_B),
                         "rotation": sim3_B["R"].tolist(),
                         "translation": sim3_B["t"].tolist(),
                         "input_camera_count": int(n_cam),
                         "leak_test": "transform estimated without any scanner GT point"},
            "C_oracle": {"transform_source": "centroid_radial_scale_on_scanner_gt",
                         "uses_reference_camera_pose": False,
                         "uses_scanner_geometry_for_transform": True,
                         "uses_scanner_geometry_for_metrics": True,
                         "scale": float(s_C),
                         "rotation": sim3_C["R"].tolist(),
                         "translation": sim3_C["t"].tolist(),
                         "input_camera_count": 0,
                         "upper_bound": True},
        }
    }
    json.dump(prov, open(ALIGN_PROV, "w"), indent=2, ensure_ascii=False)

    # 10) 权威性 manifest (P0-2)
    csv_sha = sha256_file(OUT_CSV)
    json_sha = sha256_file(OUT_JSON)
    manifest = {
        "evaluation_version": "v3.2.1",
        "authoritative_csv": os.path.relpath(OUT_CSV, GA),
        "authoritative_json": os.path.relpath(OUT_JSON, GA),
        "generator_script": "scanner_gt_3tier_eval_v321.py",
        "generator_git_commit": "UNCOMMITTED_v3.2.1",
        "csv_sha256": csv_sha,
        "json_sha256": json_sha,
        "expected_tiers": ["A_raw", "B_refcam", "C_oracle"],
        "all_tiers_have_foreground_rows": True,
        "identity_leak_guard": "passed",
        "deprecated_artifacts": [
            "scanner_gt/SCANNER_GT_3TIER_INVALID_v32.json",
            "scanner_gt/SCANNER_GT_3TIER_INVALID_v32.csv",
        ],
        "deprecated_reason": "v3.2 json: identity leak F=1.0/Chamfer=0; v3.2 csv: empty B/C fg rows + GT-scale leak in B",
    }
    json.dump(manifest, open(MANIFEST, "w"), indent=2, ensure_ascii=False)

    print(f"\n-> {OUT_CSV}  (sha256={csv_sha})")
    print(f"-> {OUT_JSON}  (sha256={json_sha})")
    print(f"-> {ALIGN_PROV}")
    print(f"-> {MANIFEST}")


def _make_row(plant_id, date, pose_gate, tier, fg_only, flags, mb, ph, scale_ratio, n_pred, n_gt):
    cam_leak = flags["uses_test_reference_geometry"] and tier != "C_oracle"
    return {
        "plant_id": plant_id, "date": date, "pose_gate": pose_gate, "tier": tier,
        "foreground_only": fg_only,
        "uses_test_reference_pose": flags["uses_test_reference_pose"],
        "uses_scanner_geometry_for_alignment": flags["uses_test_reference_geometry"],
        "n_points_pred": int(n_pred), "n_points_gt": int(n_gt),
        "chamfer_p2g_m": (None if mb is None else round(float(mb["chamfer_pred2gt_m"]), 6)),
        "chamfer_g2p_m": (None if mb is None else round(float(mb["chamfer_gt2pred_m"]), 6)),
        "chamfer_sym_m": (None if mb is None else round(float(mb["chamfer_symmetric_m"]), 6)),
        "precision_5mm": (None if mb is None else round(float(mb["precision_5mm"]), 6)),
        "recall_5mm": (None if mb is None else round(float(mb["recall_5mm"]), 6)),
        "fscore_5mm": (None if mb is None else round(float(mb["fscore_5mm"]), 6)),
        "precision_10mm": (None if mb is None else round(float(mb["precision_10mm"]), 6)),
        "recall_10mm": (None if mb is None else round(float(mb["recall_10mm"]), 6)),
        "fscore_10mm": (None if mb is None else round(float(mb["fscore_10mm"]), 6)),
        "precision_20mm": (None if mb is None else round(float(mb["precision_20mm"]), 6)),
        "recall_20mm": (None if mb is None else round(float(mb["recall_20mm"]), 6)),
        "fscore_20mm": (None if mb is None else round(float(mb["fscore_20mm"]), 6)),
        "precision_50mm": (None if mb is None else round(float(mb["precision_50mm"]), 6)),
        "recall_50mm": (None if mb is None else round(float(mb["recall_50mm"]), 6)),
        "fscore_50mm": (None if mb is None else round(float(mb["fscore_50mm"]), 6)),
        "height_error_m": (None if ph is None else ph.get("pred_height_robust_error_m")),
        "width_major_error_m": (None if ph is None else ph.get("pred_pca_error_major_width_error_m")),
        "width_minor_error_m": (None if ph is None else ph.get("pred_pca_error_minor_width_error_m")),
        "scale_ratio": (None if mb is None else round(float(scale_ratio), 6)),
        "evaluation_only": flags["evaluation_only"],
        "upper_bound": flags["upper_bound"],
    }


if __name__ == "__main__":
    main()
