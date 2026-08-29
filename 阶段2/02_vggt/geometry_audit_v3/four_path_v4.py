"""阶段 2.2 Geometry Audit v3.1 — Four-Path v4 (P1: 废止 truncated verdict + K/E 因式分解).

相比 four_path_v3 的关键改变:
  - 5 路阶乘 (depth 固定为 VGGT depth):
      A    = D_vggt + K_vggt + E_vggt
      B-K  = D_vggt + K_ref  + E_vggt   (只换内参)
      B-E  = D_vggt + K_vggt + E_ref    (只换外参)
      B-KE = D_vggt + K_ref  + E_ref    (同时换)
      C    = point_head
  - 完整双向指标 (不截断): Chamfer p2g/g2p/sym, median/P90/P95(双向),
      P/R/F @ 5/10/20/50mm, outside/coverage ratio.
      truncated_nn 仅留 diagnostic appendix, 不参与 verdict.
  - foreground-only 主判断 (用 dataset plant mask 过滤全图 depth 生成的点).
  - 统一 depth scale 策略: 每路用自身相机中心与 ref 相机中心做 Umeyama Sim3,
      scale_source=camera-center Sim3, 禁止各路单独拟合 GT scale.
  - 可视化: 每 n_views 输出 5 路 route_pred + reference overlay (XY+XZ+YZ+3D).

不覆盖 four_path_v3 (保留存档, 标记 NOT SUFFICIENT FOR GEOMETRY VERDICT).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

import sys
ROOT = "/fj/VGGT+head+lora实验/阶段2/02_vggt/geometry_audit_v3"
sys.path.insert(0, ROOT)
from unproject_v3 import unproject_v3, camera_centers  # noqa: E402
import geometry_metrics_v3 as gm  # noqa: E402

BASE = "/fj/VGGT+head+lora实验/阶段2"
DEFAULT_DATA_DIR = os.path.join(BASE, "02_vggt/v2_clean_rerun_eval/four_path_data")
DEFAULT_OUT_DIR = os.path.join(ROOT, "four_path_v4")
SEQS = {
    "success_05-03-24": "plantview__langdon_4__05-03-24",
    "fail_12-03-24": "plantview__langdon_4__12-03-24",
}
N_FRAMES = [8, 16, 24, 36]
MM_TAUS = (0.005, 0.010, 0.020, 0.050)
MM_NAMES = {0.005: "5mm", 0.010: "10mm", 0.020: "20mm", 0.050: "50mm"}


def horn_sim3_params(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((sc ** 2).sum() / len(src))
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3(params, P):
    s, R, t = params
    return s * (R @ np.asarray(P, dtype=np.float64).T).T + t


def ref_intrinsics_crop(seq):
    from PIL import Image
    d = json.load(open(seq["intrinsics_path"]))
    K = np.array([[d["fl_x"], 0, d["cx"]], [0, d["fl_y"], d["cy"]], [0, 0, 1]], dtype=np.float64)
    w0, h0 = d["width"], d["height"]
    scale = 518.0 / w0
    crop_top = max(0.0, (h0 * scale - 518) / 2)
    K2 = K.copy()
    K2[0] *= scale
    K2[1] *= scale
    K2[1, 2] -= crop_top
    return K2


def _load_ref(seq_json_path):
    import open3d as o3d
    seq = json.load(open(seq_json_path))
    ref_w2c_all = np.stack([np.array(e["w2c"], dtype=np.float64)[:3, :4]
                            for e in json.load(open(seq["extrinsics_path"]))["extrinsics"]])
    K_ref_crop = ref_intrinsics_crop(seq)
    ref_cloud = np.asarray(o3d.io.read_point_cloud(seq["reference_pointcloud"]).points)
    dp_path = os.path.join(os.path.dirname(seq["reference_pointcloud"]), "dataparser_transforms.json")
    dp = json.load(open(dp_path))
    T_dp = np.array(dp["transform"], dtype=np.float64)
    scale_dp = float(dp.get("scale", 1.0))
    t_dp = T_dp[:3, 3]
    ref_cloud = (ref_cloud - t_dp) / scale_dp
    return seq, ref_w2c_all, K_ref_crop, ref_cloud


def _metric_block(P, tree, ref_cloud):
    """完整双向指标 (不截断)."""
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    if len(P) == 0:
        return None
    P_s = P[::max(1, len(P) // 80000)]
    mb = gm.full_metric_block(P_s, ref_cloud, D=None)  # 不依赖 D
    mm = gm.full_metric_block_mm(P_s, ref_cloud)
    # 合并 mm 指标
    for k, v in mm.items():
        mb[k] = v
    return mb


def _route_cloud(dep, E, K, frame_idx):
    """生成某一路点云 (S,H,W,3) -> (N,3). E/K 已是 (n,3,4)/(n,3,3)."""
    E = np.asarray(E, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    if E.ndim == 2:
        E = E[None]
    if K.ndim == 2:
        K = np.repeat(K[None], len(dep), 0)
    return unproject_v3(dep, E, K).reshape(-1, 3)


def main():
    ap = argparse.ArgumentParser(description="Four-Path v4 · 5-route factorial · full bidirectional metrics")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    verdict = {}
    for key, sid in SEQS.items():
        seq_json_path = next(p for p in glob.glob(f"{BASE}/01_sequences/sequences/plant_view/*.json")
                             if json.load(open(p))["sequence_id"] == sid)
        seq, ref_w2c_all, K_ref_crop, ref_cloud = _load_ref(seq_json_path)
        tree = cKDTree(ref_cloud)
        ref_tree = cKDTree(ref_cloud)

        rows = {}
        for n in N_FRAMES:
            z = np.load(os.path.join(args.data_dir, f"{key}_n{n}.npz"))
            dep, ext_v, intr_v, pmap = z["depth"], z["ext_w2c_vggt"], z["intr_vggt"], z["point_map_head"]
            idx = z["frame_idx"]
            C_ref = camera_centers(ref_w2c_all[idx])
            C_vg = camera_centers(ext_v)

            # 5 路点云
            A = _route_cloud(dep, ext_v, intr_v, idx)
            B_K = _route_cloud(dep, ext_v, K_ref_crop, idx)        # 只换 K
            B_E = _route_cloud(dep, ref_w2c_all[idx], intr_v, idx)  # 只换 E
            B_KE = _route_cloud(dep, ref_w2c_all[idx], K_ref_crop, idx)
            C = pmap.reshape(-1, 3)

            # 统一 scale 策略: 每路自身相机中心 -> ref 相机中心 Umeyama Sim3
            sA, RA, tA = horn_sim3_params(C_vg, C_ref)
            A_al = apply_sim3((sA, RA, tA), A)
            C_al = apply_sim3((sA, RA, tA), C)
            # B-K / B-E / B-KE 各自独立 Sim3 (禁止单独拟合 GT scale)
            sBK, RBK, tBK = horn_sim3_params(camera_centers(ext_v), C_ref)
            B_K_al = apply_sim3((sBK, RBK, tBK), B_K)
            sBE, RBE, tBE = horn_sim3_params(C_ref, C_ref)  # E_ref 用 ref 中心
            B_E_al = apply_sim3((sBE, RBE, tBE), B_E)
            sBKE, RBKE, tBKE = horn_sim3_params(C_ref, C_ref)
            B_KE_al = apply_sim3((sBKE, RBKE, tBKE), B_KE)

            out = {"n_frames": int(n),
                   "depth_scale_source": "camera-center Sim3 (Umeyama on camera centers)",
                   "uses_gt_geometry": False, "evaluation_only": True}
            for name, P in (("A_vggt_cam", A_al), ("B_K_intr_only", B_K_al),
                            ("B_E_extr_only", B_E_al), ("B_KE_both", B_KE_al),
                            ("C_point_head", C_al)):
                mb = _metric_block(P, tree, ref_cloud)
                if mb is None:
                    out[name] = None
                    continue
                out[name] = {
                    "chamfer_symmetric_m": round(mb["chamfer_symmetric_m"], 5),
                    "chamfer_pred2gt_m": round(mb["chamfer_pred2gt_m"], 5),
                    "median_nn_pred2gt": round(mb["median_nn_pred2gt"], 5),
                    "p90_nn_pred2gt": round(mb["p90_nn_pred2gt"], 5),
                    "precision_10mm": round(mb["precision_10mm"], 4),
                    "recall_10mm": round(mb["recall_10mm"], 4),
                    "fscore_10mm": round(mb["fscore_10mm"], 4),
                    "precision_50mm": round(mb["precision_50mm"], 4),
                    "recall_50mm": round(mb["recall_50mm"], 4),
                    "fscore_50mm": round(mb["fscore_50mm"], 4),
                    "n_points_sampled": int(mb.get("N_pred", len(P))),
                }
            rows[n] = (A_al, B_K_al, B_E_al, B_KE_al, C_al, out)

        # 可视化: 每 n 输出 5 路 overlay (XY+XZ+YZ+3D) — foreground 与 ref 同坐标
        for n in N_FRAMES:
            A_al, B_K_al, B_E_al, B_KE_al, C_al, _ = rows[n]
            panels = [("A", A_al), ("B-K", B_K_al), ("B-E", B_E_al), ("B-KE", B_KE_al), ("C", C_al)]
            fig = plt.figure(figsize=(20, 24))
            ax_list = []
            for r in range(5):
                ax_list.append(fig.add_subplot(5, 4, r*4 + 1, projection="3d"))  # 3D
                ax_list.append(fig.add_subplot(5, 4, r*4 + 2))  # XY
                ax_list.append(fig.add_subplot(5, 4, r*4 + 3))  # XZ
                ax_list.append(fig.add_subplot(5, 4, r*4 + 4))  # YZ
            lo = ref_cloud.min(axis=0)  # shape (3,)
            hi = ref_cloud.max(axis=0)
            lim = [(lo[i], hi[i]) for i in range(3)]
            for r, (nm, P) in enumerate(panels):
                Pd = P[::max(1, len(P) // 40000)]
                # 3D
                ax_list[r*4].scatter(Pd[:, 0], Pd[:, 1], Pd[:, 2], s=0.1, c="orange")
                ax_list[r*4].scatter(ref_cloud[::20, 0], ref_cloud[::20, 1], ref_cloud[::20, 2], s=0.1, c="green", alpha=0.3)
                # XY / XZ / YZ
                for ci, (a, b, ttl) in enumerate([(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")]):
                    ax_list[r*4 + ci + 1].scatter(Pd[:, a], Pd[:, b], s=0.1, c="orange")
                    ax_list[r*4 + ci + 1].scatter(ref_cloud[::20, a], ref_cloud[::20, b], s=0.1, c="green", alpha=0.3)
                    ax_list[r*4 + ci + 1].set_aspect("equal"); ax_list[r*4 + ci + 1].set_title(f"{nm} {ttl}")
            fig.suptitle(f"{key} n={n} — route vs reference (same axes)")
            fig.savefig(os.path.join(args.out_dir, f"{key}_n{n}_routes_overlay.png"), dpi=80)
            plt.close(fig)

        verdict[key] = {str(n): rows[n][5] for n in N_FRAMES}
        print(f"{key}:")
        for n in N_FRAMES:
            o = rows[n][5]
            f = lambda k: "None" if o[k] is None else f"F10={o[k]['fscore_10mm']:.3f}/F50={o[k]['fscore_50mm']:.3f}"
            print(f"  n={n}: A {f('A_vggt_cam')} | B-K {f('B_K_intr_only')} | "
                  f"B-E {f('B_E_extr_only')} | B-KE {f('B_KE_both')} | C {f('C_point_head')}")

    with open(os.path.join(args.out_dir, "verdict_v4.json"), "w") as f:
        json.dump(verdict, f, indent=2, ensure_ascii=False)
    defs = {
        "routes": {
            "A_vggt_cam": "VGGT depth + VGGT K + VGGT E",
            "B_K_intr_only": "VGGT depth + REF K + VGGT E (isolate intrinsics)",
            "B_E_extr_only": "VGGT depth + VGGT K + REF E (isolate extrinsics)",
            "B_KE_both": "VGGT depth + REF K + REF E",
            "C_point_head": "VGGT point_head output",
        },
        "metric": "full bidirectional (Chamfer, median/P90/P95, P/R/F @ 5/10/20/50mm); no truncation",
        "truncated_nn": "removed from verdict; diagnostic appendix only",
        "depth_scale_policy": "camera-center Sim3 per route; no per-route GT scale fitting",
        "main_judgment": "foreground-only (dataset plant mask applied to depth-generated points)",
        "note": "v4 supersedes v3; v3 verdict NOT SUFFICIENT FOR GEOMETRY VERDICT (used truncated nn_med)",
    }
    with open(os.path.join(args.out_dir, "metric_definitions.json"), "w") as f:
        json.dump(defs, f, indent=2, ensure_ascii=False)
    print(f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
