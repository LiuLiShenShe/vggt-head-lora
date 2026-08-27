"""阶段 2.2 Geometry Audit v3 — 跨数据集参考云加载与对齐 (P0-5).

对齐策略 (约束 9: 不靠 ICP 掩盖模型误差, always 保存 before/after):
  对每序列, 用 相机中心 1:1 对应 做 Umeyama Sim3, 把 VGGT 预测点云对齐到参考系。
  - VGGT 相机中心 C_vg = -R_w2c.T @ t_w2c
  - 参考相机中心 C_ref 来自 sequence 的 extrinsics.json (同 w2c 约定)
  C_vg (S,3) 与 C_ref (S,3) 按帧序对应 -> (s,R,t) -> P_aligned = s*(R@P.T).T + t

参考云加载:
  - plant_view: GS splat.ply 经 dataparser_transforms 还原 -> transforms.json 相机系
  - wheat3dgs: COLMAP points3D.txt (与相机同系)
  - mustc: .las (plot-local Metashape 系, 与相机同系), 下采样
"""
from __future__ import annotations

import json
import os
import numpy as np


def _camera_centers_from_w2c_list(w2c_list):
    """w2c_list: list[np(3,4)] -> 相机中心 (N,3)."""
    w2c = np.asarray(w2c_list, dtype=np.float64)
    R = w2c[:, :3, :3]
    t = w2c[:, :3, 3]
    return np.einsum("nij,nj->ni", np.transpose(R, (0, 2, 1)), -t)


def umeyama_sim3(src, dst):
    """Umeyama: 求 (s,R,t) 使 dst ≈ s*R*src + t. src,dst: (N,3)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
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
    residual = np.linalg.norm(dst - (s * (R @ src.T).T + t), axis=1).mean()
    return {"s": float(s), "R": R, "t": t, "residual_rmse_centers": float(residual)}


def apply_sim3(params, P):
    s, R, t = params["s"], params["R"], params["t"]
    return s * (R @ np.asarray(P, dtype=np.float64).T).T + t


def load_plantview_refcloud(seq_json):
    """加载 plant_view GS splat 参考云, 经 dataparser_transforms 还原到相机系."""
    d = json.load(open(seq_json))
    ply_path = d["reference_pointcloud"]
    from plyfile import PlyData
    ply = PlyData.read(ply_path)
    el = ply["vertex"]
    xyz = np.stack([el["x"], el["y"], el["z"]], axis=-1).astype(np.float64)
    dp_path = os.path.join(os.path.dirname(ply_path), "dataparser_transforms.json")
    if os.path.exists(dp_path):
        dp = json.load(open(dp_path))
        T = np.array(dp["transform"], dtype=np.float64)
        R, t = T[:3, :3], T[:3, 3]
        scale = float(dp.get("scale", 1.0))
        # P_orig = (P_gs - t) / scale  (R 为单位阵)
        xyz = (xyz - t) / scale
    return xyz


def load_wheat_refcloud(seq_json):
    """加载 wheat COLMAP points3D.txt."""
    d = json.load(open(seq_json))
    pf = d["reference_pointcloud"]
    pts = []
    with open(pf) as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            sp = line.split()
            if len(sp) < 3:
                continue
            try:
                x, y, z = float(sp[0]), float(sp[1]), float(sp[2])
            except ValueError:
                continue
            pts.append((x, y, z))
    return np.array(pts, dtype=np.float64)


def load_mustc_refcloud(seq_json, subsample=500000):
    """加载 mustc .las (plot-local 系), 下采样到 ~subsample 点."""
    d = json.load(open(seq_json))
    lasf = d["reference_pointcloud"]
    import laspy
    l = laspy.read(lasf)
    xyz = np.stack([l.x, l.y, l.z], axis=-1).astype(np.float64)
    if len(xyz) > subsample:
        idx = np.linspace(0, len(xyz) - 1, subsample).astype(int)
        xyz = xyz[idx]
    return xyz


def load_reference_centers(seq_json):
    """参考相机中心 (N,3), 与帧序对应."""
    d = json.load(open(seq_json))
    ext = json.load(open(d["extrinsics_path"]))
    w2c = [np.array(e["w2c"], dtype=np.float64) for e in ext["extrinsics"]]
    return _camera_centers_from_w2c_list(w2c)


def align_sequence(seq_json, vggt_ext_w2c, pred_points_world):
    """对齐单序列.

    Args:
        seq_json: 序列 JSON 路径
        vggt_ext_w2c: (S,3,4) VGGT w2c
        pred_points_world: (S,H,W,3) 或 (N,3) VGGT 预测点 (世界系, 已用修正反投影)
    Returns:
        dict: {sim3, C_vg, C_ref, pred_aligned, ref_cloud, residual_rmse_centers, n_centers}
    """
    C_vg = _camera_centers_from_w2c_list(vggt_ext_w2c)
    C_ref = load_reference_centers(seq_json)
    n = min(len(C_vg), len(C_ref))
    C_vg, C_ref = C_vg[:n], C_ref[:n]
    sim3 = umeyama_sim3(C_vg, C_ref)

    pts = np.asarray(pred_points_world, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        raise ValueError("empty pred points")
    # 仅用对应帧范围内的点? 预测点含全部 S 帧 -> 全用
    pred_aligned = apply_sim3(sim3, pts)

    ds = json.load(open(seq_json))["dataset_id"]
    if ds == "plant_view_3d":
        ref = load_plantview_refcloud(seq_json)
    elif ds == "wheat3dgs":
        ref = load_wheat_refcloud(seq_json)
    elif ds == "mustc":
        ref = load_mustc_refcloud(seq_json)
    else:
        raise ValueError(f"unknown dataset {ds}")

    return {
        "sim3": sim3,
        "C_vg": C_vg,
        "C_ref": C_ref,
        "pred_aligned": pred_aligned,
        "pred_before": pts,
        "ref_cloud": ref,
        "residual_rmse_centers": sim3["residual_rmse_centers"],
        "n_centers": n,
        "scale_s": sim3["s"],
    }
