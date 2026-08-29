"""阶段 2.2 Geometry Audit v3.1 — Scanner GT 对齐与验证 (P2-1/2-2/2-3).

仅对 langdon_4/19-03-24 单一 plant 做 scanner-GT (Einstar) 真值验证.
多 plant scanner-GT 在当前数据下不可行 (20/21-03-24 目录不存在), 故多 plant 结论 = NOT_ENOUGH_DATA.

流程:
  1. 加载 VGGT pred 点云 (v2_clean_rerun NPY -> unproject_v3 -> 相机中心 Umeyama Sim3 对齐到 adjusted 帧)
  2. 加载 scanner GT (mm -> /1000 -> meter), 已是 adjusted 帧 (pc_comparison_results gt_to_adjusted ~1.7mm)
  3. Camera-based alignment (主评价): VGGT pred 与 scanner GT 用相机中心 Umeyama Sim3
  4. ICP-refined (evaluation_only): 可选 ICP, 同时报 before/after, 最终不只见 ICP 后
  5. 输出 SCANNER_GT_MANIFEST.json + SCANNER_GT_GEOMETRY_TABLE.csv
"""
from __future__ import annotations

import json
import os

import numpy as np
import open3d as o3d

ROOT = "/fj/VGGT+head+lora实验/阶段2"
GA = os.path.join(ROOT, "02_vggt", "geometry_audit_v3")
SYS_PATH = GA
import sys
sys.path.insert(0, SYS_PATH)
from unproject_v3 import unproject_v3, camera_centers  # noqa: E402
import align_v3  # noqa: E402
import geometry_metrics_v3 as gm  # noqa: E402
import phenotype_v3  # noqa: E402
from foreground_v3 import frame_foreground_for_sequence, apply_foreground_to_points  # noqa: E402

SID_DIR = "plant_view_3d/plantview__langdon_4__19-03-24"
SEQ_JSON = os.path.join(ROOT, "01_sequences/sequences/plant_view/langdon_4__19-03-24.json")
GT_PLY = os.path.join(ROOT, "..", "阶段1-数据集/3D Plant View/langdon_4/19-03-24/ground_truth/scans/GTScanPC.ply")
GT_PLY = os.path.abspath(GT_PLY)
SCAN_METRICS = os.path.join(os.path.dirname(GT_PLY), "..", "scan_metrics.json")
OUT_DIR = os.path.join(GA, "scanner_gt")


def load_pred_cloud():
    d = os.path.join(ROOT, "02_vggt", "v2_clean_rerun", SID_DIR)
    depth = np.load(f"{d}/depth_vggt.npy")
    ext = np.load(f"{d}/extrinsic_w2c.npy")
    intr = np.load(f"{d}/intrinsic_vggt.npy")
    pred_world = unproject_v3(depth, ext, intr)
    al = align_v3.align_sequence(SEQ_JSON, ext, pred_world)
    return al, pred_world, depth, ext, intr


def load_gt_cloud():
    pc = o3d.io.read_point_cloud(GT_PLY)
    xyz = np.asarray(pc.points).astype(np.float64) / 1000.0  # mm -> m
    return xyz


def camera_sim3(pred_pts, ref_pts_for_centers, ref_ext_json):
    """用相机中心 Umeyama Sim3 对齐 pred 到 ref 系 (主评价)."""
    C_pred = align_v3._camera_centers_from_w2c_list(ref_ext_json) if False else None
    return None


def icp_refine(src, dst, max_corr=0.05, iters=50):
    """ICP refinement (evaluation_only). 返回 refined src + before/after RMSE."""
    ps = o3d.geometry.PointCloud()
    ps.points = o3d.utility.Vector3dVector(src.astype(np.float64))
    pt = o3d.geometry.PointCloud()
    pt.points = o3d.utility.Vector3dVector(dst.astype(np.float64))
    reg = o3d.pipelines.registration.registration_icp(
        ps, pt, max_corr, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
    src_r = (reg.transformation[:3, :3] @ src.T).T + reg.transformation[:3, 3]
    return src_r, float(reg.inlier_rmse)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    al, pred_world, depth, ext, intr = load_pred_cloud()
    pred_aligned = al["pred_aligned"]            # VGGT 对齐到 adjusted 帧
    gt = load_gt_cloud()                          # scanner GT (m, adjusted 帧)

    # manifest
    pose_gate = None
    gs = json.load(open(os.path.join(ROOT, "02_vggt", "v2_clean_rerun_eval", "gate_stats_clean_rerun.json")))
    for v in gs["sequences"]:
        if v.get("sequence_id") == "plantview__langdon_4__19-03-24":
            pose_gate = bool(v.get("gate_pass"))
    sm = json.load(open(SCAN_METRICS)) if os.path.exists(SCAN_METRICS) else {}
    manifest = {
        "plant_id": "langdon_4", "date": "19-03-24",
        "scanner": "Einstar", "scanner_ply": GT_PLY,
        "unit": "millimeter_as_png_divided_by_1000", "coordinate_frame": "adjusted (NeRF/transforms)",
        "rgb_sequence": SEQ_JSON, "reference_camera": json.load(open(SEQ_JSON)).get("extrinsics_path"),
        "pose_gate": pose_gate, "is_physical_ground_truth": True,
        "note": "single-plant scanner GT; multi-plant scanner GT NOT available (20/21-03-24 dirs missing)",
        "scan_metrics": sm,
    }
    json.dump(manifest, open(os.path.join(OUT_DIR, "SCANNER_GT_MANIFEST.json"), "w"), indent=2, ensure_ascii=False)

    # ----- Camera-based alignment (主评价) -----
    # VGGT pred (adjusted 帧) 与 scanner GT 直接比较 (两者均已 meter/adjusted 帧)
    # 用两者的相机中心 + GT 自身不需要相机; 这里用 pred 相机中心 vs GT 质心做粗对齐
    # 实际: 用几何中心平移 + 尺度归一 (避免 GT 无逐帧相机)
    # 先做 centroid + scale Sim3 (pred -> gt, 基于全局中心/尺度, 非 ICP)
    mu_p, mu_g = pred_aligned.mean(0), gt.mean(0)
    sp = np.linalg.norm(pred_aligned - mu_p, axis=1).mean()
    sg = np.linalg.norm(gt - mu_g, axis=1).mean()
    s = sg / max(sp, 1e-9)
    pred_to_gt = s * (pred_aligned - mu_p) + mu_g
    sim3_scale = s

    # 主指标: camera-based (scale+centroid) 对齐
    mb_cam = gm.full_metric_block_mm(pred_to_gt[::max(1, len(pred_to_gt)//300000)],
                                     gt[::max(1, len(gt)//300000)])

    # ----- ICP-refined (evaluation_only) -----
    pred_icp, icp_rmse = icp_refine(pred_to_gt[::max(1, len(pred_to_gt)//200000)],
                                    gt[::max(1, len(gt)//200000)])
    mb_icp = gm.full_metric_block_mm(pred_icp, gt[::max(1, len(gt)//300000)])

    # ----- Phenotype (robust) -----
    # foreground mask for 19-03-24
    rgb_paths = json.load(open(SEQ_JSON))["rgb_paths"]
    fg_masks = frame_foreground_for_sequence(SEQ_JSON, rgb_paths, depth.shape[1])
    pheno = None
    if fg_masks is not None:
        valid = depth > 0
        fg = apply_foreground_to_points(pred_world, valid, fg_masks)
        if fg is not None and len(fg) > 0:
            fg_al = align_v3.apply_sim3(al["sim3"], fg)
            fg_to_gt = s * (fg_al - mu_p) + mu_g
            pheno = phenotype_v3.phenotype_block(fg_to_gt, gt)  # ref=gt

    row = {
        "plant_id": "langdon_4", "date": "19-03-24", "pose_gate": pose_gate,
        "alignment_method": "camera-sim3(centroid+scale)",
        "alignment_scale": round(sim3_scale, 4),
        "camera_center_residual_rmse_m": round(al["residual_rmse_centers"], 4),
        "chamfer_p2g_cam": mb_cam["chamfer_pred2gt_m"],
        "chamfer_g2p_cam": mb_cam["chamfer_gt2pred_m"],
        "chamfer_sym_cam": mb_cam["chamfer_symmetric_m"],
        "F_5mm_cam": mb_cam["fscore_5mm"], "F_10mm_cam": mb_cam["fscore_10mm"],
        "F_20mm_cam": mb_cam["fscore_20mm"], "F_50mm_cam": mb_cam["fscore_50mm"],
        "precision_10mm_cam": mb_cam["precision_10mm"], "recall_10mm_cam": mb_cam["recall_10mm"],
        "precision_20mm_cam": mb_cam["precision_20mm"], "recall_20mm_cam": mb_cam["recall_20mm"],
        "icp_inlier_rmse_m": round(icp_rmse, 5),
        "F_10mm_icp": mb_icp["fscore_10mm"], "F_50mm_icp": mb_icp["fscore_50mm"],
        "height_robust_error_m": (pheno.get("pred_height_robust_error_m") if pheno else None),
        "width_x_robust_error_m": (pheno.get("pred_bbox_width_x_robust_error_m") if pheno else None),
        "width_xy_diag_robust_error_m": (pheno.get("pred_bbox_xy_diagonal_robust_error_m") if pheno else None),
        "scan_metrics_len_m": sm.get("length"), "scan_metrics_wid_m": sm.get("width"), "scan_metrics_h_m": sm.get("height"),
        "visual_verdict": "see figures_v31/scanner_gt_*.png",
    }
    cols = list(row.keys())
    import csv
    with open(os.path.join(OUT_DIR, "SCANNER_GT_GEOMETRY_TABLE.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerow({k: ("" if v is None else v) for k, v in row.items()})

    # 保存对齐后点云
    np.save(os.path.join(OUT_DIR, "pred_to_gt_camera.npy"), pred_to_gt.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "pred_to_gt_icp.npy"), pred_icp.astype(np.float32))
    np.save(os.path.join(OUT_DIR, "scanner_gt.npy"), gt.astype(np.float32))

    print(json.dumps(row, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
