"""阶段 2.2 Geometry Audit v3.1 — 主驱动 (P0-1/2/3/4/6/7/8/9 集成).

对 6 个代表序列:
  1. 加载 clean-rerun NPY -> unproject_v3 -> pred 世界点云
  2. align_v3 相机中心 Umeyama (保存 before/after)
  3. foreground_v3 逐帧前景掩膜 -> FULL 与 FOREGROUND 两套点集
  4. 双向指标: Chamfer / F@5-10-20-50mm / F@1-2-5%D
  5. depth_audit_v3 (修正单位) + 真实 depth montage
  6. phenotype_v3 RAW+ROBUST+PCA + 离群敏感审计
  7. 可视化: figures_v31 (接收 metric 用的 P_fore/Q_fore, 修复 v3 漏 Sim3 bug)
  8. 汇总: FOREGROUND_METRICS_V31.csv / PHENOTYPE_OUTLIER_SENSITIVITY.csv / per_seq JSON

原则: metric 输入点 === figure 输入点 (P_fore, Q_fore, P_full, Q_full 传给画图函数).
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

BASE = "/fj/VGGT+head+lora实验/阶段2"
ROOT = os.path.join(BASE, "02_vggt", "geometry_audit_v3")

REPRESENTATIVES = [
    ("plant_view_3d/plantview__langdon_4__05-03-24", "plant_view/langdon_4__05-03-24.json"),
    ("plant_view_3d/plantview__langdon_4__12-03-24", "plant_view/langdon_4__12-03-24.json"),
    ("plant_view_3d/plantview__langdon_4__13-02-24", "plant_view/langdon_4__13-02-24.json"),
    ("plant_view_3d/plantview__langdon_4__20-02-24", "plant_view/langdon_4__20-02-24.json"),
    ("wheat3dgs/wheat3dgs__plot_463", "wheat3dgs/plot_463.json"),
    ("mustc/mustc__plot198__230613__ugv__pos00", "mustc/plot198__230613__ugv__pos00.json"),
]

SUBSAMPLE = 300000


def load_seq(sid_dir):
    d = os.path.join(BASE, "02_vggt", "v2_clean_rerun", sid_dir)
    depth = np.load(f"{d}/depth_vggt.npy")
    ext = np.load(f"{d}/extrinsic_w2c.npy")
    intr = np.load(f"{d}/intrinsic_vggt.npy")
    return depth, ext, intr


def subsample(P, n=SUBSAMPLE):
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    if len(P) > n:
        idx = np.linspace(0, len(P) - 1, n).astype(int)
        return P[idx]
    return P


def robust_bbox_diag(Q, frac=0.95):
    Qc = np.asarray(Q, dtype=np.float64) - np.asarray(Q, dtype=np.float64).mean(0)
    lo = np.quantile(Qc, (1 - frac) / 2, axis=0)
    hi = np.quantile(Qc, 1 - (1 - frac) / 2, axis=0)
    return float(np.linalg.norm(hi - lo))


def load_pose_gate(sid):
    gs = json.load(open(os.path.join(BASE, "02_vggt", "v2_clean_rerun_eval",
                                     "gate_stats_clean_rerun.json")))
    for v in gs.get("sequences", []):
        if isinstance(v, dict) and v.get("sequence_id") == sid:
            return bool(v.get("gate_pass"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames-depth", type=int, default=30)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, ROOT)
    from unproject_v3 import unproject_v3
    import align_v3, foreground_v3, geometry_metrics_v3 as gm, depth_audit_v3, phenotype_v3, figures_v31

    figdir = os.path.join(ROOT, "figures_v31")
    if args.figures:
        os.makedirs(figdir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "per_seq"), exist_ok=True)

    fg_rows = []          # FOREGROUND_METRICS_V31.csv
    pheno_sens_rows = []  # PHENOTYPE_OUTLIER_SENSITIVITY.csv
    table_rows = []

    for sid_dir, seqjson_rel in REPRESENTATIVES:
        seqjson = os.path.join(BASE, "01_sequences", "sequences", seqjson_rel)
        sid = os.path.basename(sid_dir)
        ds = json.load(open(seqjson))["dataset_id"]
        print(f"\n=== {sid} ({ds}) ===")

        depth, ext, intr = load_seq(sid_dir)
        S, Hout, W = depth.shape
        valid = depth > 0
        pred_world = unproject_v3(depth, ext, intr)
        pred_points = pred_world[valid]

        al = align_v3.align_sequence(seqjson, ext, pred_world)
        ref = al["ref_cloud"]
        pred_aligned = al["pred_aligned"]
        comparable = (ds == "plant_view_3d")

        # 中心化非可比数据集
        if not comparable:
            pred_aligned = pred_aligned - pred_aligned.mean(0)
            ref = ref - ref.mean(0)
        D = robust_bbox_diag(ref, 0.95) if comparable else None

        rgb_paths = json.load(open(seqjson))["rgb_paths"]
        fg_masks = foreground_v3.frame_foreground_for_sequence(seqjson, rgb_paths, Hout)

        # ---- FULL-SCENE ----
        P_full = subsample(pred_aligned)
        Q_full = subsample(ref)
        mb_full = gm.full_metric_block(P_full, Q_full, D)
        if not comparable:
            for k in list(mb_full.keys()):
                if k != "truncated_inlier_diagnostic_only":
                    mb_full[k] = None

        # ---- PLANT-FOREGROUND-ONLY ----
        P_fore = Q_fore = block_fore = pheno = mm_fore = None
        if fg_masks is not None:
            fg_pts_world = foreground_v3.apply_foreground_to_points(pred_world, valid, fg_masks)
            if fg_pts_world is not None and len(fg_pts_world) > 0:
                fg_pred_aligned = align_v3.apply_sim3(al["sim3"], fg_pts_world)
                if not comparable:
                    fg_pred_aligned = fg_pred_aligned - fg_pred_aligned.mean(0)
                P_fore = subsample(fg_pred_aligned, SUBSAMPLE)
                Q_fore = subsample(ref, SUBSAMPLE)
                # P1-1 (v3.2.1): 持久化真实前景点集 (raw/aligned/reference) 供 PR 图与测试使用
                np.save(os.path.join(ROOT, "per_seq", f"{sid}_pred_foreground_raw.npy"),
                        subsample(fg_pts_world, SUBSAMPLE).astype(np.float32))
                np.save(os.path.join(ROOT, "per_seq", f"{sid}_pred_foreground_aligned.npy"),
                        P_fore.astype(np.float32))
                np.save(os.path.join(ROOT, "per_seq", f"{sid}_reference_foreground.npy"),
                        Q_fore.astype(np.float32))
                block_fore = gm.full_metric_block(P_fore, Q_fore, D)
                if not comparable:
                    for k in list(block_fore.keys()):
                        if k != "truncated_inlier_diagnostic_only":
                            block_fore[k] = None
                # 严格 mm 级指标 (P0-7)
                mm_fore = gm.full_metric_block_mm(P_fore, Q_fore) if comparable else None
                if mm_fore is not None:
                    row = {"sequence_id": sid, "dataset_id": ds, "result_set": "plant_foreground_only_pred"}
                    for k in ["chamfer_symmetric_m", "fscore_5mm", "fscore_10mm", "fscore_20mm", "fscore_50mm",
                              "precision_10mm", "recall_10mm", "precision_20mm", "recall_20mm",
                              "fscore_1pctD", "fscore_2pctD", "fscore_5pctD",
                              "cov50mm_within_ratio_pred", "cov50mm_within_ratio_gt"]:
                        if k in mm_fore and mm_fore[k] is not None:
                            row[k] = round(float(mm_fore[k]), 6)
                    fg_rows.append(row)
                # phenotype (P0-8)
                if ds == "plant_view_3d":
                    pheno = phenotype_v3.phenotype_block(P_fore, Q_fore)
                    # 离群敏感 (P0-9)
                    sens = phenotype_v3.outlier_sensitivity(P_fore, Q_fore)
                    for regime, r in sens.items():
                        if r is None:
                            continue
                        pheno_sens_rows.append({
                            "sequence_id": sid, "regime": regime,
                            "height_robust_m": round(r["height_robust_m"], 4),
                            "bbox_width_x_robust_m": round(r["bbox_width_x_robust_m"], 4),
                            "bbox_xy_diagonal_robust_m": round(r["bbox_xy_diagonal_robust_m"], 4),
                            "pca_major_width_m": (round(r["pca_major_width_m"], 4) if r.get("pca_major_width_m") is not None else None),
                            "pca_minor_width_m": (round(r["pca_minor_width_m"], 4) if r.get("pca_minor_width_m") is not None else None),
                        })

        # ---- 深度审计 (plant_view) ----
        depth_audit = None
        if ds == "plant_view_3d":
            depth_audit = depth_audit_v3.depth_audit_sequence(seqjson, depth, rgb_paths,
                                                             max_frames=args.max_frames_depth)

        # ---- 汇总 ----
        pose_gate = load_pose_gate(sid)
        base_row = {
            "dataset_id": ds, "sequence_id": sid, "pose_gate": pose_gate, "frames_S": S,
            "reference_frame_auditable": comparable,
            "alignment_scale_s": round(al["scale_s"], 4),
            "align_residual_rmse_m": round(al["residual_rmse_centers"], 4),
            "D_bbox_diag_m_robust": (round(D, 4) if D is not None else None),
            "geometry_gate": "not_yet_established",
            "reference_type": "3dgs_pseudo_reference" if comparable else "non_metric_or_geo_offset",
            "is_physical_ground_truth": False if comparable else None,
        }
        fr = {**base_row, **_flat(mb_full), "result_set": "full_scene"}
        table_rows.append(fr)
        if block_fore is not None:
            table_rows.append({**base_row, **_flat(block_fore), "result_set": "plant_foreground_only_pred"})

        # per-seq JSON
        seq_out = {"sequence_id": sid, "dataset_id": ds, "pose_gate": pose_gate,
                   "reference_comparable": comparable,
                   "reference_type": "3dgs_pseudo_reference" if comparable else "non_metric_reference",
                   "is_physical_ground_truth": False if comparable else None,
                   "alignment": {"scale_s": al["scale_s"], "residual_rmse_centers": al["residual_rmse_centers"],
                                 "method": "umeyama_sim3_on_camera_centers"},
                   "metrics_full_scene": mb_full, "metrics_plant_foreground": block_fore,
                   "metrics_plant_foreground_mm": (gm.full_metric_block_mm(P_fore, Q_fore) if (comparable and P_fore is not None) else None),
                   "phenotype": pheno, "depth_audit": depth_audit}
        json.dump(seq_out, open(os.path.join(ROOT, "per_seq", f"{sid}.geo_v31.json"), "w"), indent=2, ensure_ascii=False)

        # 控制台摘要
        print(f"  pose_gate={pose_gate} auditable={comparable} scale_s={al['scale_s']:.3f} center_res={al['residual_rmse_centers']:.3f}")
        if block_fore is not None and mm_fore is not None:
            print(f"  FOREG: F@5mm={mm_fore.get('fscore_5mm')} F@10mm={mm_fore.get('fscore_10mm')} "
                  f"F@20mm={mm_fore.get('fscore_20mm')} F@50mm={mm_fore.get('fscore_50mm')} "
                  f"F@5%_D={block_fore.get('fscore_5pctD')}")
        if depth_audit:
            a = depth_audit["aggregate"]
            print(f"  DEPTH: raw AbsRel={a['mean_raw_absrel']:.3f} RMSE_m={a['mean_raw_rmse']:.3f} "
                  f"aligned AbsRel={a['mean_aligned_absrel']:.3f} scale={a['mean_median_scale']:.3f}")
        if pheno:
            print(f"  PHENO: Eh pred/ref={pheno.get('pred_height_robust_m')}/{pheno.get('ref_height_robust_m')} "
                  f"canopy_err={pheno.get('bbox_xy_diagonal_robust_error_m')}")

        # ---- 可视化 (P0-1/2/3/4): 传入 metric 用过的同一些点 ----
        if args.figures:
            if P_fore is not None and Q_fore is not None:
                figures_v31.overlay_fg_3d(figdir, sid, P_fore, Q_fore)
                figures_v31.overlay_fg_ortho(figdir, sid, P_fore, Q_fore)
                figures_v31.foreground_error_maps(figdir, sid, P_fore, Q_fore)
            if ds == "plant_view_3d":
                depth_dir = json.load(open(seqjson)).get("extra", {}).get("depth_dir")
                if depth_dir:
                    figures_v31.depth_montage_real(figdir, sid, rgb_paths, depth_dir, depth, depth)

    # 写 CSV
    _write_csv(os.path.join(ROOT, "FOREGROUND_METRICS_V31.csv"), fg_rows,
               ["sequence_id", "dataset_id", "result_set", "chamfer_symmetric_m",
                "fscore_5mm", "fscore_10mm", "fscore_20mm", "fscore_50mm",
                "precision_10mm", "recall_10mm", "precision_20mm", "recall_20mm",
                "fscore_1pctD", "fscore_2pctD", "fscore_5pctD",
                "cov50mm_within_ratio_pred", "cov50mm_within_ratio_gt"])
    _write_csv(os.path.join(ROOT, "PHENOTYPE_OUTLIER_SENSITIVITY.csv"), pheno_sens_rows,
               ["sequence_id", "regime", "height_robust_m", "bbox_width_x_robust_m",
                "bbox_xy_diagonal_robust_m", "pca_major_width_m", "pca_minor_width_m"])

    cols = sorted({k for r in table_rows for k in r.keys()})
    with open(os.path.join(ROOT, "GEOMETRY_AUDIT_V31_TABLE.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in table_rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    print(f"\n-> FOREGROUND_METRICS_V31.csv ({len(fg_rows)} rows), "
          f"PHENOTYPE_OUTLIER_SENSITIVITY.csv ({len(pheno_sens_rows)} rows)")


def _flat(d):
    flat = {}
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[k] = round(float(v), 6)
        elif isinstance(v, (list, dict)):
            flat[k] = json.dumps(v)
        else:
            flat[k] = v
    return flat


def _write_csv(path, rows, cols):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
