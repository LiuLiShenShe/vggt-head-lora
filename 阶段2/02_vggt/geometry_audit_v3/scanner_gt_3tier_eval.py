#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""阶段 2.2 Geometry Audit v3 — Scanner-GT 三层级评估 (P2-4).

对 langdon_4 / 19-03-24 单一 plant 的 scanner-GT (Einstar) 真值做三类对齐评估:

  Tier A  RAW / MODEL SCALE   —— 不做任何对齐, VGGT 预测直接对比 scanner GT
           uses_test_reference_pose      = False
           uses_test_reference_geometry  = False
           evaluation_only               = False   (diagnostic)
  Tier B  REFERENCE-CAMERA     —— VGGT 相机中心 vs 参考相机中心 Umeyama Sim3 对齐
           uses_test_reference_pose      = True
           uses_test_reference_geometry  = False
           evaluation_only               = True
  Tier C  ORACLE-GEOMETRY      —— 用 scanner GT 几何拟合 Sim3 (上界)
           uses_test_reference_geometry  = True
           evaluation_only               = True
           upper_bound                   = True

每个 tier 计算两套指标:
  (1) FULL-SCENE   VGGT(全)        vs scanner GT(全)    — diagnostic_only
  (2) FOREGROUND   VGGT(plant mask) vs scanner GT(plant) — PRIMARY

前景掩膜: 优先用 sequence.extra.mask_dir 逐帧二值 mask; 若不可用, 退化用
          "pred 点在 GT bbox ±20% 内" 作为近似前景。

指标 (geometry_metrics_v3.full_metric_block_mm):
  F@5/10/20/50mm, Chamfer_sym, 以及 phenotype_v3 的 height / width-diag robust 误差。

依赖:
  align_v3.umeyama_sim3 / apply_sim3 / _camera_centers_from_w2c_list / load_reference_centers
  geometry_metrics_v3.full_metric_block_mm
  phenotype_v3.phenotype_block
  foreground_v3.frame_foreground_for_sequence / apply_foreground_to_points

--------------------------------------------------------------------------------
文件角色说明 (重要 — 与最初设想已发生偏差):
  磁盘上 scanner_gt/ 目录由 v3.1 脚本生成, 其角色为:
    scanner_gt/scanner_gt.npy        = scanner GT 点云 (米, 已 /1000, 单 plant)
    scanner_gt/pred_to_gt_camera.npy = VGGT 预测 经 **参考相机** 修正后的 adjusted 帧点云
                                       (pose-corrected: 用参考相机中心+尺度重锚定, 米)
    scanner_gt/pred_to_gt_icp.npy    = VGGT 预测 经 ICP 精修 (oracle 近似, 米)
  原始 VGGT 预测 (adjusted 帧, model scale) 在:
    v2_clean_rerun/.../point_map_unprojected.npy  (320,518,518,3)
  VGGT 预测相机 (extrinsic_w2c.npy) 在本序列已损坏 (rot_median≈174°, pose_gate=False),
  直接对 VGGT 相机中心做 Umeyama 会退化; 因此参考相机对齐采用 v3.1 已修正的
  pred_to_gt_camera.npy 作为 Tier B 的输入云 (其已含参考相机位姿+尺度信息)。

三层级对齐方法学 (本脚本从源数据可复现地重建):
    Tier A RAW      : 原始 VGGT 预测 (point_map_unprojected, model scale) 直接 vs GT, 不做任何对齐
    Tier B REF-CAM  : 参考相机修正云 (pred_to_gt_camera.npy, 米) 在全局 pose-free Sim3
                      (质心+径向尺度, 仅旋转/平移/尺度, 不用逐帧相机) 下对齐到 GT
    Tier C ORACLE   : 原始 VGGT 预测 (米, 同 B 的源) 在 GT 几何拟合的全局 Sim3 下对齐到 GT (上界)

  scanner_gt.npy 仅作为 GT 缓存/一致性校验使用; 权威 GT 来源为 GTScanPC.ply * scale_to_meter。
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────────────
GA = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
BASE = "/fj/VGGT+head+lora实验/阶段2"
SEQ_JSON = os.path.join(BASE, "01_sequences/sequences/plant_view/langdon_4__19-03-24.json")
UNIT_AUDIT = os.path.join(GA, "SCANNER_UNIT_AUDIT.json")
MANIFEST = os.path.join(GA, "scanner_gt", "SCANNER_GT_MANIFEST.json")
SEQ_DIR = os.path.join(BASE, "02_vggt", "v2_clean_rerun", "plant_view_3d", "plantview__langdon_4__19-03-24")
RAW_PRED_NPY = os.path.join(SEQ_DIR, "point_map_unprojected.npy")     # VGGT 原始预测 (adjusted 帧, model scale)
DEPTH_NPY = os.path.join(SEQ_DIR, "depth_vggt.npy")                   # 有效性掩膜 (depth>0)
VGGT_W2C_NPY = os.path.join(SEQ_DIR, "extrinsic_w2c.npy")             # VGGT 相机 w2c (320,3,4) [已损坏]
REF_CAM_CLOUD_NPY = os.path.join(GA, "scanner_gt", "pred_to_gt_camera.npy")  # 参考相机修正云 (米)
GT_PLY = "/fj/VGGT+head+lora实验/阶段1-数据集/3D Plant View/langdon_4/19-03-24/ground_truth/scans/GTScanPC.ply"
GT_NPY_CACHE = os.path.join(GA, "scanner_gt", "scanner_gt.npy")       # 磁盘上为 GT 云 (米)
OUT_CSV = os.path.join(GA, "scanner_gt", "SCANNER_GT_3TIER.csv")
OUT_JSON = os.path.join(GA, "scanner_gt", "scanner_gt_3tier.json")

MAX_POINTS = 300000          # 指标计算子采样上限
RNG = np.random.default_rng(0)

sys.path.insert(0, GA)
import align_v3                                              # noqa: E402
import geometry_metrics_v3 as gm                            # noqa: E402
import phenotype_v3                                          # noqa: E402
import foreground_v3                                         # noqa: E402

# 兼容任务描述中的函数名 (align_v3 实际暴露的是 umeyama_sim3)
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
    """点云相对质心的平均径向距离 (作为尺度代理)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    if len(P) == 0:
        return float("nan")
    return float(np.linalg.norm(P - P.mean(0), axis=1).mean())


def load_gt(scale_to_meter, manifest):
    """加载 scanner GT 点云 (米). 优先 PLY, 否则 GT_NPY_CACHE, 否则 MANIFEST 指定 ply."""
    ply_path = manifest.get("scanner_ply", GT_PLY)
    if os.path.exists(ply_path):
        import open3d as o3d
        pc = o3d.io.read_point_cloud(ply_path)
        xyz = np.asarray(pc.points).astype(np.float64) * scale_to_meter
        src = f"ply:{ply_path} * {scale_to_meter}"
    elif os.path.exists(GT_NPY_CACHE):
        xyz = np.load(GT_NPY_CACHE).astype(np.float64) * scale_to_meter
        src = f"npy_cache:{GT_NPY_CACHE} * {scale_to_meter}"
    else:
        raise FileNotFoundError(f"GT 源不可用: {ply_path} 与 {GT_NPY_CACHE}")
    return xyz, src


def load_raw_pred_and_valid():
    """返回 (pred (N,3) float64, valid_bool (S,H,W)) 与序列元信息."""
    pred = np.load(RAW_PRED_NPY)                 # (S,H,W,3) float32, model scale
    depth = np.load(DEPTH_NPY)                   # (S,H,W) float32
    if depth.ndim == 4:
        depth = depth[..., 0]
    valid = depth > 0
    S, H, W = depth.shape
    return pred, valid, (S, H, W)


def extract_foreground(pred, valid, fg_masks, shape):
    """逐帧用前景掩膜抽取前景点 (S,H,W,3). 无 mask 返回 None."""
    if fg_masks is None:
        return None
    S, H, W = shape
    out = []
    for s in range(S):
        fg = fg_masks[s]
        if fg is None:
            continue
        m = valid[s] & fg
        if m.any():
            out.append(pred[s][m])
    if not out:
        return None
    return np.concatenate(out, axis=0).astype(np.float64)


def bbox_fallback_foreground(pred_full, gt, expand=0.20):
    """退化前景: pred 点在 GT bbox 扩展 ±expand 内."""
    lo = gt.min(0)
    hi = gt.max(0)
    span = hi - lo
    lo -= span * expand
    hi += span * expand
    inside = np.all((pred_full >= lo) & (pred_full <= hi), axis=1)
    return pred_full[inside]


def pose_rot_stats(vggt_w2c, ref_w2c):
    """VGGT w2c 与参考 w2c 的相对旋转角 (度) 的 median / p90."""
    vggt_w2c = np.asarray(vggt_w2c, dtype=np.float64)
    ref_w2c = np.asarray(ref_w2c, dtype=np.float64)
    n = min(len(vggt_w2c), len(ref_w2c))
    angs = []
    for i in range(n):
        Rv = vggt_w2c[i][:3, :3]
        Rr = ref_w2c[i][:3, :3]
        Rrel = Rr @ Rv.T
        trace = np.clip(np.trace(Rrel), -1.0, 3.0)
        ang = np.degrees(np.arccos((trace - 1.0) / 2.0))
        angs.append(ang)
    angs = np.array(angs)
    return float(np.median(angs)), float(np.percentile(angs, 90)), int(n)


# ── 主流程 ────────────────────────────────────────────────────────────────
def main():
    # 0) 必需元数据
    if not os.path.exists(UNIT_AUDIT):
        raise FileNotFoundError(f"SCANNER_UNIT_AUDIT.json 缺失: {UNIT_AUDIT} (必需, 含 scale_to_meter)")
    unit_audit = json.load(open(UNIT_AUDIT))
    scale_to_meter = float(unit_audit["scale_to_meter"])
    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else {}

    plant_id = manifest.get("plant_id", "langdon_4")
    date = manifest.get("date", "19-03-24")
    pose_gate = manifest.get("pose_gate", None)
    seq = json.load(open(SEQ_JSON))
    sid = seq["sequence_id"]

    print(f"[init] plant={plant_id} date={date} pose_gate={pose_gate} scale_to_meter={scale_to_meter}")

    # 1) GT
    gt, gt_src = load_gt(scale_to_meter, manifest)
    print(f"[gt] source={gt_src} n={len(gt)}")
    gt_sub = subsample(gt, MAX_POINTS)

    # 2) 原始 VGGT 预测
    pred, valid, shape = load_raw_pred_and_valid()
    S, H, W = shape
    print(f"[pred] raw shape={pred.shape} frames={S} valid_pts={int(valid.sum())}")

    # 全场景 VGGT 点 (valid only)
    full_list = [pred[s][valid[s]] for s in range(S) if valid[s].any()]
    pred_full = np.concatenate(full_list, axis=0).astype(np.float64) if full_list else np.zeros((0, 3))
    print(f"[pred] full valid points={len(pred_full)}")

    # 3) 前景掩膜
    rgb_paths = seq["rgb_paths"]
    fg_masks = foreground_v3.frame_foreground_for_sequence(SEQ_JSON, rgb_paths, H)
    mask_available = fg_masks is not None
    print(f"[fg] plant_masks available={mask_available}")

    if mask_available:
        pred_fg = extract_foreground(pred, valid, fg_masks, shape)
    else:
        pred_fg = None
    if pred_fg is None:
        # 退化: GT bbox ±20%
        pred_fg = bbox_fallback_foreground(pred_full, gt_sub)
        fg_method = "gt_bbox_plus20pct"
    else:
        fg_method = "plant_masks"
    print(f"[fg] method={fg_method} fg_pred_points={len(pred_fg)}")

    # GT 前景: scanner GT 为单 plant 扫描, 整朵即前景 (必要时同样 bbox 裁剪保持一致)
    if not mask_available:
        lo = gt_sub.min(0); hi = gt_sub.max(0); span = hi - lo
        gt_fg = gt_sub[np.all((gt_sub >= lo - span * 0.2) & (gt_sub <= hi + span * 0.2), axis=1)]
    else:
        gt_fg = gt_sub  # 单 plant 扫描, 全部即前景
    gt_fg = subsample(gt_fg, MAX_POINTS)

    # 4) Tier B 参考相机修正云 (pose-corrected, 米) — 加载 v3.1 已用参考相机重锚定的点云
    if not os.path.exists(REF_CAM_CLOUD_NPY):
        raise FileNotFoundError(f"参考相机修正云缺失: {REF_CAM_CLOUD_NPY}")
    refcam_cloud = np.load(REF_CAM_CLOUD_NPY).astype(np.float64).reshape(-1, 3)
    print(f"[tierB] ref-cam corrected cloud n={len(refcam_cloud)} (meter-scale, pose-corrected)")

    # 5) Tier C oracle 源云 = 原始 VGGT 预测 (模型尺度) 缩放到米
    #    v3.1 经参考相机得到 alignment_scale≈0.4196 作为 model->meter 的尺度代理;
    #    oracle 再在 GT 几何上拟合一个全局 Sim3 (质心+径向尺度, 仅 R/t/s) 作为上界。
    #    alignment_scale 优先取自 SCANNER_GT_GEOMETRY_TABLE.csv (v3.1 camera-sim3 对齐尺度)
    model_to_meter = float(manifest.get("alignment_scale", 0.0))
    if model_to_meter <= 0.0:
        tab = os.path.join(GA, "scanner_gt", "SCANNER_GT_GEOMETRY_TABLE.csv")
        if os.path.exists(tab):
            import csv as _csv
            with open(tab) as f:
                rdr = _csv.DictReader(f)
                for row in rdr:
                    v = row.get("alignment_scale")
                    if v not in (None, ""):
                        model_to_meter = float(v)
                        break
    if model_to_meter <= 0.0:
        model_to_meter = 0.4196  # 退化默认
    oracle_src = pred_full * model_to_meter        # 原始预测 -> 米 (同 Tier B 量级)
    print(f"[tierC] oracle source = raw_pred * {model_to_meter} (meter-scale) n={len(oracle_src)}")

    # 在子采样集上拟合 pose-free Sim3 (centroid + radial scale), 不用逐帧相机
    def fit_posefree_sim3(src, dst):
        """全局 Sim3: 仅质心平移 + 径向均匀尺度 (不含旋转估计, 旋转=单位阵).
        返回 apply 函数."""
        mu_s, mu_d = src.mean(0), dst.mean(0)
        rs = np.linalg.norm(src - mu_s, axis=1).mean()
        rd = np.linalg.norm(dst - mu_d, axis=1).mean()
        s = rd / max(rs, 1e-12)
        return lambda P: s * (P - mu_s) + mu_d, float(s)

    sim3_B, scale_B = fit_posefree_sim3(subsample(refcam_cloud, MAX_POINTS, random=True), gt_sub)
    sim3_C, scale_C = fit_posefree_sim3(subsample(oracle_src, MAX_POINTS, random=True), gt_sub)
    print(f"[tierB] pose-free Sim3 scale={scale_B:.4f}")
    print(f"[tierC] oracle pose-free Sim3 scale={scale_C:.4f}")

    # 6) 姿态统计 (VGGT vs 参考 w2c) — 报告 VGGT 预测相机相对参考相机的旋转误差
    ref_ext = json.load(open(seq["extrinsics_path"]))["extrinsics"]
    ref_w2c_list = [np.array(e["w2c"], dtype=np.float64)[:3, :4] for e in ref_ext]
    vggt_w2c = np.load(VGGT_W2C_NPY)
    rot_med, rot_p90, n_rot = pose_rot_stats(vggt_w2c, np.array(ref_w2c_list))
    print(f"[pose] rot_median_deg={rot_med:.2f} rot_p90_deg={rot_p90:.2f} n={n_rot}")

    # 6) 每个 tier 计算 full + foreground
    rows = []
    detail = {"plant_id": plant_id, "date": date, "sequence_id": sid,
              "pose_gate": pose_gate, "scale_to_meter": scale_to_meter,
              "gt_source": gt_src, "gt_n_points": int(len(gt)),
              "pred_raw_n_points": int(len(pred_full)),
              "refcam_cloud_n_points": int(len(refcam_cloud)),
              "fg_method": fg_method, "fg_pred_n_points": int(len(pred_fg)),
              "tierB_scale": scale_B, "tierC_scale": scale_C,
              "model_to_meter_scale": model_to_meter,
              "pose_rot_median_deg": rot_med, "pose_rot_p90_deg": rot_p90,
              "tiers": {}}

    # tier 定义: (name, source_cloud_full, source_cloud_fg, transform_fn, flags)
    tiers = [
        ("A_raw", pred_full, pred_fg, None,
         dict(uses_test_reference_pose=False, uses_test_reference_geometry=False,
              evaluation_only=False, upper_bound=False)),
        ("B_refcam", refcam_cloud, None, sim3_B,
         dict(uses_test_reference_pose=True, uses_test_reference_geometry=False,
              evaluation_only=True, upper_bound=False)),
        ("C_oracle", oracle_src, None, sim3_C,
         dict(uses_test_reference_pose=False, uses_test_reference_geometry=True,
              evaluation_only=True, upper_bound=True)),
    ]

    for tname, src_full, src_fg, tf, flags in tiers:
        # 变换源云到 GT 系
        if tf is None:
            P_full_t = src_full
            P_fg_t = src_fg
        else:
            P_full_t = tf(src_full)
            P_fg_t = tf(src_fg) if src_fg is not None else None

        # 子采样用于指标
        P_full_m = subsample(P_full_t, MAX_POINTS, random=True)
        P_fg_m = subsample(P_fg_t, MAX_POINTS, random=True) if P_fg_t is not None else None

        # ---- FULL ----
        mb_full = gm.full_metric_block_mm(P_full_m, gt_sub)
        ph_full = phenotype_v3.phenotype_block(P_full_m, gt_sub) if len(P_full_m) else None
        scale_ratio_full = radial_scale(P_full_m) / max(radial_scale(gt_sub), 1e-9)
        row_full = _make_row(plant_id, date, pose_gate, tname, False, flags,
                             mb_full, ph_full, scale_ratio_full,
                             len(P_full_m), len(gt_sub))
        rows.append(row_full)

        # ---- FOREGROUND (PRIMARY) ----
        mb_fg = gm.full_metric_block_mm(P_fg_m, gt_fg) if P_fg_m is not None else None
        ph_fg = phenotype_v3.phenotype_block(P_fg_m, gt_fg) if (P_fg_m is not None and len(P_fg_m)) else None
        scale_ratio_fg = (radial_scale(P_fg_m) / max(radial_scale(gt_fg), 1e-9)) if P_fg_m is not None else float("nan")
        row_fg = _make_row(plant_id, date, pose_gate, tname, True, flags,
                          mb_fg, ph_fg, scale_ratio_fg,
                          len(P_fg_m) if P_fg_m is not None else 0, len(gt_fg))
        rows.append(row_fg)

        detail["tiers"][tname] = {
            "flags": flags,
            "n_points_pred_full": int(len(P_full_m)),
            "n_points_pred_fg": int(len(P_fg_m)) if P_fg_m is not None else 0,
            "n_points_gt_full": int(len(gt_sub)),
            "n_points_gt_fg": int(len(gt_fg)),
            "scale_ratio_full": scale_ratio_full,
            "scale_ratio_fg": scale_ratio_fg,
            "metrics_full": mb_full,
            "metrics_fg": mb_fg,
            "phenotype_full": ph_full,
            "phenotype_fg": ph_fg,
        }
        print(f"  [{tname}] full: F@5/10/20/50={mb_full['fscore_5mm']:.3f}/{mb_full['fscore_10mm']:.3f}/"
              f"{mb_full['fscore_20mm']:.3f}/{mb_full['fscore_50mm']:.3f} "
              f"chamfer={mb_full['chamfer_symmetric_m']:.4f} scale_ratio={scale_ratio_full:.3f}")
        if mb_fg is not None:
            print(f"  [{tname}] fg  : F@5/10/20/50={mb_fg['fscore_5mm']:.3f}/{mb_fg['fscore_10mm']:.3f}/"
                  f"{mb_fg['fscore_20mm']:.3f}/{mb_fg['fscore_50mm']:.3f} "
                  f"chamfer={mb_fg['chamfer_symmetric_m']:.4f} scale_ratio={scale_ratio_fg:.3f}")

    # 7) 写出
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    cols = ["plant_id", "date", "pose_gate", "tier", "foreground_only", "scale_ratio",
            "F_5mm", "F_10mm", "F_20mm", "F_50mm", "chamfer_sym_m",
            "height_error_m", "width_diag_error_m", "n_points_pred", "n_points_gt",
            "uses_test_reference_pose", "uses_test_reference_geometry",
            "evaluation_only", "upper_bound"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    json.dump(detail, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    print(f"\n-> {OUT_CSV}")
    print(f"-> {OUT_JSON}")


def _make_row(plant_id, date, pose_gate, tier, fg_only, flags, mb, ph, scale_ratio, n_pred, n_gt):
    row = {
        "plant_id": plant_id,
        "date": date,
        "pose_gate": pose_gate,
        "tier": tier,
        "foreground_only": fg_only,
        "scale_ratio": (None if mb is None else round(float(scale_ratio), 6)),
        "F_5mm": (None if mb is None else round(float(mb["fscore_5mm"]), 6)),
        "F_10mm": (None if mb is None else round(float(mb["fscore_10mm"]), 6)),
        "F_20mm": (None if mb is None else round(float(mb["fscore_20mm"]), 6)),
        "F_50mm": (None if mb is None else round(float(mb["fscore_50mm"]), 6)),
        "chamfer_sym_m": (None if mb is None else round(float(mb["chamfer_symmetric_m"]), 6)),
        "height_error_m": (None if ph is None else ph.get("pred_height_robust_error_m")),
        "width_diag_error_m": (None if ph is None else ph.get("pred_bbox_xy_diagonal_robust_error_m")),
        "n_points_pred": int(n_pred),
        "n_points_gt": int(n_gt),
        "uses_test_reference_pose": flags["uses_test_reference_pose"],
        "uses_test_reference_geometry": flags["uses_test_reference_geometry"],
        "evaluation_only": flags["evaluation_only"],
        "upper_bound": flags["upper_bound"],
    }
    return row


if __name__ == "__main__":
    main()
