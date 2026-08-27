"""阶段 2.2 Geometry Audit v3 — 主驱动 (P0 全流程).

对 6 个代表序列:
  1. 加载 clean-rerun NPY (depth/ext/intr) -> 修正反投影 unproject_v3 -> pred 世界点云
  2. align_v3: 参考云加载 + 相机中心 Umeyama (保存 before/after)
  3. foreground_v3: 逐帧前景掩膜 -> FULL-SCENE 与 PLANT-FOREGROUND-ONLY 两套点集
  4. geometry_metrics_v3: 双向指标 (Chamfer/F/P/R@τ/D) -> 指标 block
  5. depth_audit_v3 (plant_view 仅): 单帧深度审计
  6. phenotype_v3 (plant_view 仅): 表型指标
  7. 可视化: 6 类图 (同坐标/同轴范围/同视角 overlay)
  8. 汇总: GEOMETRY_AUDIT_TABLE.csv + 每序列 JSON

严格分离:
  - Pose Gate 14/17 (冻结, 从 meta.pose_eval_v2 读取, 不重算)
  - Geometry Gate: 不反向设计, 先报分布 (not_yet_established)

输出目录: 02_vggt/geometry_audit_v3/
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

BASE = "/fj/VGGT+head+lora实验/阶段2"
ROOT = os.path.join(BASE, "02_vggt", "geometry_audit_v3")

# 6 代表序列: (vggt_dir_sid, seq_json_rel, 是否 plant_foreground 可用)
REPRESENTATIVES = [
    ("plant_view_3d/plantview__langdon_4__05-03-24", "plant_view/langdon_4__05-03-24.json"),
    ("plant_view_3d/plantview__langdon_4__12-03-24", "plant_view/langdon_4__12-03-24.json"),
    ("plant_view_3d/plantview__langdon_4__13-02-24", "plant_view/langdon_4__13-02-24.json"),
    ("plant_view_3d/plantview__langdon_4__20-02-24", "plant_view/langdon_4__20-02-24.json"),
    ("wheat3dgs/wheat3dgs__plot_463", "wheat3dgs/plot_463.json"),
    ("mustc/mustc__plot198__230613__ugv__pos00", "mustc/plot198__230613__ugv__pos00.json"),
]

SUBSAMPLE = 300000   # 指标计算点云下采样上限


def load_seq(sid_dir):
    d = os.path.join(BASE, "02_vggt", "v2_clean_rerun", sid_dir)
    depth = np.load(f"{d}/depth_vggt.npy")
    ext = np.load(f"{d}/extrinsic_w2c.npy")
    intr = np.load(f"{d}/intrinsic_vggt.npy")
    meta = json.load(open(f"{d}/prediction_meta.json"))
    return depth, ext, intr, meta


def subsample(P, n=SUBSAMPLE):
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    if len(P) > n:
        idx = np.linspace(0, len(P) - 1, n).astype(int)
        return P[idx]
    return P


def robust_bbox_diag(Q, frac=0.95):
    """对 Q 做 95% 中心分位 bbox 对角线, 抑制离群点/坐标系偏移导致的虚假大尺度."""
    Qc = np.asarray(Q, dtype=np.float64)
    Qc = Qc - Qc.mean(0)
    lo = np.quantile(Qc, (1 - frac) / 2, axis=0)
    hi = np.quantile(Qc, 1 - (1 - frac) / 2, axis=0)
    return float(np.linalg.norm(hi - lo))


def load_pose_gate(sid):
    """从 gate_stats_clean_rerun.json 读取冻结的 Pose Gate (14/17). 字段为 gate_pass (bool)."""
    gs = json.load(open(os.path.join(BASE, "02_vggt", "v2_clean_rerun_eval",
                                     "gate_stats_clean_rerun.json")))
    seqs = gs.get("sequences", [])
    if isinstance(seqs, list):
        for v in seqs:
            if isinstance(v, dict) and v.get("sequence_id") == sid:
                return bool(v.get("gate_pass"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-frames-depth", type=int, default=30)
    ap.add_argument("--figures", action="store_true", help="生成 6 类可视化图")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, ROOT)
    from unproject_v3 import unproject_v3
    import align_v3, foreground_v3, geometry_metrics_v3, depth_audit_v3, phenotype_v3

    figdir = os.path.join(ROOT, "figures_v3")
    if args.figures:
        os.makedirs(figdir, exist_ok=True)
    os.makedirs(os.path.join(ROOT, "per_seq"), exist_ok=True)

    table_rows = []
    for sid_dir, seqjson_rel in REPRESENTATIVES:
        seqjson = os.path.join(BASE, "01_sequences", "sequences", seqjson_rel)
        sid = os.path.basename(sid_dir)
        ds = json.load(open(seqjson))["dataset_id"]
        print(f"\n=== {sid} ({ds}) ===")

        depth, ext, intr, meta = load_seq(sid_dir)
        S, Hout, W = depth.shape
        valid = depth > 0
        pred_world = unproject_v3(depth, ext, intr)   # (S,Hout,W,3)
        pred_points = pred_world[valid]                 # 全部有效点 (世界系)

        # --- 对齐 ---
        al = align_v3.align_sequence(seqjson, ext, pred_world)
        ref = al["ref_cloud"]
        pred_aligned = al["pred_aligned"]

        # 参考系可比性判断:
        #  plant_view_3d -> VGGT metric 与 GS 参考 (meter) 同系, 直接可比
        #  wheat3dgs     -> COLMAP points3D 为任意重建单位 (非 meter), 与 VGGT metric 不可直接比
        #  mustc         -> LAS 为 plot-local 系但带 ~3.57e5 地理偏移且尺度未知, 不可直接比
        # 非可比数据集: 记为 comparable=False, 仅报告 scale-normalized 指标 (F@%D), 不报绝对米制指标.
        ref_center_spread = float(np.linalg.norm((ref - ref.mean(0)).max(0) - (ref - ref.mean(0)).min(0)))
        comparable = (ds == "plant_view_3d")
        if not comparable:
            # 把两云都中心化到质心, 使对齐度量仅反映相对形状/尺度, 而非绝对坐标偏移
            pred_aligned = pred_aligned - pred_aligned.mean(0)
            ref = ref - ref.mean(0)

        # 保存 before/after (下采样)
        np.save(os.path.join(ROOT, "per_seq", f"{sid}_pred_before.npy"), subsample(al["pred_before"], 200000))
        np.save(os.path.join(ROOT, "per_seq", f"{sid}_pred_aligned.npy"), subsample(pred_aligned, 200000))
        np.save(os.path.join(ROOT, "per_seq", f"{sid}_ref.npy"), subsample(ref, 200000))
        # 对齐 sim3 元信息
        sim3 = {"scale_s": al["scale_s"],
                "residual_rmse_centers": al["residual_rmse_centers"],
                "n_centers": al["n_centers"],
                "method": "umeyama_sim3_on_camera_centers",
                "reference_frame_comparable": comparable,
                "ref_center_spread_m": round(ref_center_spread, 3),
                "note": "plant_view: metric-compare OK; wheat/mustc: reference not in VGGT metric frame -> scale-normalized only" if not comparable else ""}
        json.dump(sim3, open(os.path.join(ROOT, "per_seq", f"{sid}_align_sim3.json"), "w"), indent=2)

        # 鲁棒 D (95% 中心分位 bbox 对角线), 抑制离群点/坐标系偏移
        D = robust_bbox_diag(ref, frac=0.95) if comparable else None

        # --- 前景 ---
        rgb_paths = json.load(open(seqjson))["rgb_paths"]
        fg_masks = foreground_v3.frame_foreground_for_sequence(seqjson, rgb_paths, Hout)
        full_rows = {}
        fore_rows = {}

        # FULL-SCENE
        P_full = subsample(pred_aligned)
        Q_full = subsample(ref)
        mb_full = geometry_metrics_v3.full_metric_block(P_full, Q_full, D)
        if not comparable:
            # wheat/mustc 参考云本身非 metric 系 (COLMAP 任意单位 / LAS 地理偏移+未知尺度):
            # D 与全部阈值指标均不可信 -> 清空所有指标, 标记 reference_frame_auditable=False.
            for k in list(mb_full.keys()):
                if k in ("result_set", "truncated_inlier_diagnostic_only"):
                    continue
                mb_full[k] = None
        # 非可比数据集: 绝对米制指标无效 -> 清空 (保留 scale-normalized F@%D)
        if not comparable:
            for k in list(mb_full.keys()):
                if ("_m" in k or k.startswith("cov5pctD")) and not k.startswith("fscore_") and "D_bbox" not in k:
                    if any(x in k for x in ["chamfer", "median", "p90", "p95", "precision_", "recall_", "within_ratio", "N_", "cov5pctD_within_ratio"]):
                        mb_full[k] = None
                # 绝对米制 F/P/R 也置空
                for tau in ("0.010m", "0.020m", "0.050m"):
                    if k.startswith(("precision_", "recall_", "fscore_")) and k.endswith(tau) and ("pctD" not in k):
                        mb_full[k] = None
        full_rows = {**mb_full, "result_set": "full_scene"}

        # PLANT-FOREGROUND-ONLY
        block_fore = None
        pheno = None
        if fg_masks is not None:
            fg_pts_world = foreground_v3.apply_foreground_to_points(pred_world, valid, fg_masks)
            if fg_pts_world is not None and len(fg_pts_world) > 0:
                fg_pred_aligned = align_v3.apply_sim3(al["sim3"], fg_pts_world)
                if not comparable:
                    fg_pred_aligned = fg_pred_aligned - fg_pred_aligned.mean(0)
                P_fore = subsample(fg_pred_aligned, SUBSAMPLE)
                Q_fore = subsample(ref, SUBSAMPLE)
                block_fore = geometry_metrics_v3.full_metric_block(P_fore, Q_fore, D)
                if not comparable:
                    for k in list(block_fore.keys()):
                        if k == "result_set" or k == "truncated_inlier_diagnostic_only":
                            continue
                        block_fore[k] = None
                block_fore["result_set"] = "plant_foreground_only_pred"
                if ds == "plant_view_3d":
                    pheno = phenotype_v3.phenotype_block(P_fore, Q_fore)

        # --- 深度审计 (plant_view 仅) ---
        depth_audit = None
        if ds == "plant_view_3d":
            depth_audit = depth_audit_v3.depth_audit_sequence(seqjson, depth, rgb_paths,
                                                             max_frames=args.max_frames_depth)

        # --- 汇总行 (full + foreground) ---
        pose_gate = load_pose_gate(sid)
        base_row = {
            "dataset_id": ds,
            "sequence_id": sid,
            "pose_gate": pose_gate,
            "frames_S": S,
            "reference_frame_auditable": comparable,
            "alignment_scale_s": round(al["scale_s"], 4),
            "align_residual_rmse_m": round(al["residual_rmse_centers"], 4),
            "D_bbox_diag_m_robust": (round(D, 4) if D is not None else None),
            "geometry_gate": "not_yet_established",
        }
        fr = {**base_row, **_flat(full_rows)}
        table_rows.append(fr)
        if block_fore is not None:
            fr2 = {**base_row, **_flat(block_fore),
                   "result_set": "plant_foreground_only_pred"}
            table_rows.append(fr2)

        # 每序列 JSON
        seq_out = {"sequence_id": sid, "dataset_id": ds, "pose_gate": pose_gate,
                   "reference_comparable": comparable,
                   "alignment": sim3, "metrics_full_scene": full_rows,
                   "metrics_plant_foreground": block_fore, "phenotype": pheno,
                   "depth_audit": depth_audit}
        json.dump(seq_out, open(os.path.join(ROOT, "per_seq", f"{sid}.geo_v3.json"), "w"), indent=2, ensure_ascii=False)

        # 控制台摘要
        print(f"  pose_gate={pose_gate} auditable={comparable} scale_s={al['scale_s']:.3f} center_res={al['residual_rmse_centers']:.3f} D_robust={('%.2f'%D if D is not None else 'N/A')}")
        ch = mb_full.get('chamfer_symmetric_m')
        print(f"  FULL:  Chamfer_sym={('%.4f'%ch if ch is not None else 'N/A')}  "
              f"F@5%_D={mb_full.get('fscore_5pctD')}  within5%_pred={mb_full.get('cov5pctD_within_ratio_pred')}")
        if block_fore is not None:
            chf = block_fore.get('chamfer_symmetric_m')
            f5 = block_fore.get('fscore_5pctD')
            print(f"  FOREG: Chamfer_sym={('%.4f'%chf if chf is not None else 'N/A')}m  "
                  f"F@5%_D={f5}")
        if depth_audit:
            agg = depth_audit["aggregate"]
            print(f"  DEPTH: raw AbsRel={agg.get('mean_raw_absrel',float('nan')):.3f}  "
                  f"aligned AbsRel={agg.get('mean_aligned_absrel',float('nan')):.3f}  "
                  f"n_frames={agg['n_frames_audited']}")
        if pheno:
            print(f"  PHENO: Eh pred={pheno.get('pred_Eh_m')} ref={pheno.get('ref_Eh_m')} "
                  f"canopy pred={pheno.get('pred_canopy_diam_m')} ref={pheno.get('ref_canopy_diam_m')}")

        # --- 可视化 ---
        if args.figures:
            _make_figures(sid, pred_aligned, ref, fg_masks, pred_world, valid, Hout, figdir, ds)

    # --- 写 CSV ---
    csv_path = os.path.join(ROOT, "GEOMETRY_AUDIT_TABLE.csv")
    cols = sorted({k for r in table_rows for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in table_rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
    print(f"\n-> {csv_path} ({len(table_rows)} rows)")


def _flat(d):
    """把嵌套 metric dict 拍平为可直接写入 CSV 的键值 (metric dict 已是一层)."""
    flat = {}
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            flat[k] = round(float(v), 6)
        elif isinstance(v, (list, dict)):
            flat[k] = json.dumps(v)
        else:
            flat[k] = v
    return flat


def _make_figures(sid, pred_aligned, ref, fg_masks, pred_world, valid, Hout, figdir, ds):
    """6 类图: 同坐标/同轴范围/同视角 overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # 下采样用于绘图
    def dsamp(P, n=60000):
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        if len(P) > n:
            return P[np.linspace(0, len(P) - 1, n).astype(int)]
        return P

    pa = dsamp(pred_aligned)
    rf = dsamp(ref)

    # 统一轴范围 (两云并集)
    allp = np.concatenate([pa, rf], 0)
    lo, hi = allp.min(0), allp.max(0)
    lim = [ (lo[i], hi[i]) for i in range(3) ]

    # 1. 全场景 overlay
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(rf[:, 0], rf[:, 1], rf[:, 2], s=0.3, c="green", alpha=0.4, label="reference")
    ax.scatter(pa[:, 0], pa[:, 1], pa[:, 2], s=0.3, c="red", alpha=0.4, label="VGGT pred")
    for i in range(3):
        ax.set_xlim(lim[i]) if i == 0 else (ax.set_ylim(lim[i]) if i == 1 else ax.set_zlim(lim[i]))
    ax.set_title(f"{sid}: full-scene overlay (same axes)")
    ax.legend()
    fig.savefig(f"{figdir}/{sid}_overlay_full.png", dpi=100); plt.close(fig)

    # 2. 前景 overlay (若有)
    if fg_masks is not None:
        from unproject_v3 import camera_centers
        import foreground_v3
        fg_world = foreground_v3.apply_foreground_to_points(pred_world, valid, fg_masks)
        if fg_world is not None and len(fg_world) > 0:
            fga = dsamp(fg_world)
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(rf[:, 0], rf[:, 1], rf[:, 2], s=0.3, c="green", alpha=0.4, label="reference")
            ax.scatter(fga[:, 0], fga[:, 1], fga[:, 2], s=0.3, c="orange", alpha=0.5, label="VGGT fg")
            for i in range(3):
                ax.set_xlim(lim[i]) if i == 0 else (ax.set_ylim(lim[i]) if i == 1 else ax.set_zlim(lim[i]))
            ax.set_title(f"{sid}: plant-foreground overlay (same axes)")
            ax.legend()
            fig.savefig(f"{figdir}/{sid}_overlay_foreground.png", dpi=100); plt.close(fig)

    # 3. 误差着色云 (pred->ref nn 距离热图)
    from scipy.spatial import cKDTree
    tQ = cKDTree(rf)
    d_p2g, _ = tQ.query(pa, k=1, workers=-1)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(pa[:, 0], pa[:, 1], pa[:, 2], s=0.3, c=d_p2g, cmap="jet")
    for i in range(3):
        ax.set_xlim(lim[i]) if i == 0 else (ax.set_ylim(lim[i]) if i == 1 else ax.set_zlim(lim[i]))
    plt.colorbar(sc, ax=ax, label="pred→ref nn dist (m)")
    ax.set_title(f"{sid}: error-colored pred cloud")
    fig.savefig(f"{figdir}/{sid}_error_colored.png", dpi=100); plt.close(fig)

    # 4. 深度 montage
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for j, ax in enumerate(axes.ravel()):
        if j < valid.shape[0]:
            dv = valid[j].astype(float)
            ax.imshow(dv, cmap="viridis"); ax.set_title(f"f{j}"); ax.axis("off")
        else:
            ax.axis("off")
    fig.suptitle(f"{sid}: depth validity montage")
    fig.savefig(f"{figdir}/{sid}_depth_montage.png", dpi=100); plt.close(fig)

    # 5. 双向 nn 直方图 (用 pred_aligned 与 ref 计算)
    tP = cKDTree(pa); d_g2p, _ = tP.query(rf, k=1, workers=-1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(d_p2g, bins=60); ax[0].set_title("pred→ref nn")
    ax[1].hist(d_g2p, bins=60); ax[1].set_title("ref→pred nn")
    fig.suptitle(f"{sid}: bidirectional nn distance")
    fig.savefig(f"{figdir}/{sid}_nn_hist.png", dpi=100); plt.close(fig)

    # 6. F-score vs τ 曲线
    import geometry_metrics_v3 as gm
    taus = np.linspace(0.005, 0.3, 40)
    Fs = [gm.precision_recall_fscore(pa, rf, t)[2] for t in taus]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(taus, Fs); ax.set_xlabel("τ (m)"); ax.set_ylabel("F-score")
    ax.set_title(f"{sid}: F-score vs threshold")
    fig.savefig(f"{figdir}/{sid}_fscore_curve.png", dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
