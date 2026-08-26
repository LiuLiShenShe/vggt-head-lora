"""阶段2.2 位姿评估 v2(da3 环境,CPU)——审计修正版。

v1 问题(审计确认):
- Procrustes 解取了转置(Rg = V Uᵀ 应为 U Vᵀ 形式的正确支),残差公式组合也错;
- "相对旋转误差消除相机系常值 Q"表述不严谨(只消除左乘全局旋转);
- 相机坍缩指标无尺度归一化,与报告文字矛盾。

v2 修正:
- 全局旋转:目标 Rg @ Rv_i ≈ Rr_i,令 H = Σ Rv_i Rr_iᵀ = U Σ Vᵀ,则 Rg = U Vᵀ(det 修正);
- 残差:rot_angle_deg((Rg @ Rv_i)ᵀ @ Rr_i);
- 新增尺度归一化相机分布指标(中心协方差特征值占比、轨迹长/径向跨度);
- 结果写 prediction_meta.json 新字段 pose_eval_v2(旧 pose_eval 保留),汇总 pose_eval_summary_v2.json。

用法: python eval_pose_v2.py <seq_dir> [<seq_dir2> ...]
"""
import glob
import json
import os
import sys

import numpy as np

SEQ_BASE = "/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences"
OUT_BASE = "/fj/VGGT+head+lora实验/阶段2/02_vggt"
SUMMARY = os.path.join(OUT_BASE, "pose_eval_summary_v2.json")


def horn_sim3(src, dst):
    """dst ≈ s R src + t。返回 (s, R, t)。"""
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


def global_rotation_procrustes(R_vg_c2w, R_ref_c2w):
    """求 Rg 最小化 Σ ||Rg @ Rv_i - Rr_i||_F(列向量约定)。
    H = Σ Rv_i Rr_iᵀ = U Σ Vᵀ → Rg = V Uᵀ(det 修正)。
    (合成自检验证:v1 的错误在于 einsum 顺序为 ΣRr Rvᵀ 却取 Rg = V Uᵀ,即转置解。)"""
    H = np.einsum("sij,skj->ik", R_vg_c2w, R_ref_c2w)
    U, _, Vt = np.linalg.svd(H)
    Rg = Vt.T @ U.T
    if np.linalg.det(Rg) < 0:
        U[:, -1] *= -1
        Rg = Vt.T @ U.T
    return Rg


def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def w2c_centers(ext_w2c):
    R = ext_w2c[:, :3, :3]
    t = ext_w2c[:, :3, 3]
    return np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)


def _selftest():
    rng = np.random.RandomState(3)
    # 构造已知 G:S 个随机旋转,全部左乘同一 G
    G = np.linalg.qr(rng.randn(3, 3))[0]
    if np.linalg.det(G) < 0:
        G[:, 0] *= -1
    Rv = np.stack([np.linalg.qr(rng.randn(3, 3))[0] for _ in range(30)])
    Rv = np.stack([np.linalg.qr(m)[0] if np.linalg.det(m) > 0 else np.linalg.qr(m)[0] * np.diag([1, 1, -1.0]) for m in Rv])
    Rr = np.stack([G @ m for m in Rv])
    Rg = global_rotation_procrustes(Rv, Rr)
    assert rot_angle_deg(Rg.T @ G) < 1e-6, rot_angle_deg(Rg.T @ G)
    # 残差公式自检:完美情形残差应为 0
    res = [rot_angle_deg((Rg @ v).T @ r) for v, r in zip(Rv, Rr)]
    assert max(res) < 1e-5, max(res)


_selftest()


def camera_shape_metrics(C):
    """尺度归一化相机分布指标(C: (S,3) 相机中心)。"""
    if len(C) < 3:
        return {}
    span = float(np.linalg.norm(C - C.mean(0), axis=1).mean())  # 平均径向跨度
    cov = np.cov(C.T)
    ev = np.linalg.eigvalsh(cov)
    ev_ratio = float(ev[0] / (ev[-1] + 1e-12))
    traj_len = float(np.linalg.norm(np.diff(C, axis=0), axis=1).sum())
    return {
        "center_radial_span": span,
        "cov_eigen_min_max_ratio": round(ev_ratio, 5),
        "trajectory_length": round(traj_len, 4),
        "traj_over_span": round(traj_len / max(span, 1e-9), 3),
    }


def eval_sequence(seq_dir):
    meta_path = os.path.join(seq_dir, "prediction_meta.json")
    meta = json.load(open(meta_path))
    sid, ds = meta["sequence_id"], meta["dataset_id"]
    ds_dir = os.path.join(SEQ_BASE,
                          {"plant_view_3d": "plant_view", "wheat3dgs": "wheat3dgs",
                           "mustc": "mustc", "terra_ref": "terraref"}[ds])
    seq = next(json.load(open(p)) for p in glob.glob(os.path.join(ds_dir, "*.json"))
               if json.load(open(p))["sequence_id"] == sid)
    p = seq.get("extrinsics_path")
    if not p or not os.path.exists(p):
        print(f"  {sid}: 无参考相机,跳过")
        return None
    ref = json.load(open(p))["extrinsics"]
    ref_w2c = np.stack([np.array(e["w2c"], dtype=np.float64)[:3, :4] for e in ref])
    assert len(ref_w2c) == len(seq["rgb_paths"])

    vg_w2c = np.load(os.path.join(seq_dir, "extrinsic_w2c.npy")).astype(np.float64)
    S = len(vg_w2c)

    # --- 中心 Sim3 对齐(逐帧对应) ---
    C_vg = w2c_centers(vg_w2c)
    C_ref = w2c_centers(ref_w2c)
    s, R, t = horn_sim3(C_vg, C_ref)
    C_al = s * (R @ C_vg.T).T + t
    tr = np.linalg.norm(C_al - C_ref, axis=1)
    scale_ref = float(np.linalg.norm(C_ref - C_ref.mean(0), axis=1).mean())

    # --- 全局旋转 Procrustes + 逐相机残差(修正公式) ---
    R_vg_c2w = vg_w2c[:, :3, :3].transpose(0, 2, 1)
    R_ref_c2w = ref_w2c[:, :3, :3].transpose(0, 2, 1)
    Rg = global_rotation_procrustes(R_vg_c2w, R_ref_c2w)
    rot_err = np.array([rot_angle_deg((Rg @ Rv).T @ Rr)
                        for Rv, Rr in zip(R_vg_c2w, R_ref_c2w)])

    # --- 相对旋转误差(消除左乘全局旋转;右乘相机系差异不在其消除范围) ---
    rel_err = np.array([rot_angle_deg((R_ref_c2w[i].T @ R_ref_c2w[j]).T
                                      @ (R_vg_c2w[i].T @ R_vg_c2w[j]))
                        for i in range(S) for j in range(i + 1, S)])

    # --- 轨迹方向一致性 ---
    d_vg = np.diff(C_al, axis=0)
    d_ref = np.diff(C_ref, axis=0)
    cosines = []
    for a, b in zip(d_vg, d_ref):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-9 and nb > 1e-9:
            cosines.append(float(a @ b / (na * nb)))
    traj = {
        "mean_cosine": round(float(np.mean(cosines)), 4) if cosines else None,
        "median_cosine": round(float(np.median(cosines)), 4) if cosines else None,
        "n_pairs": len(cosines),
    }

    pe = {
        "sim3_scale": round(float(s), 6),
        "ref_center_spread": round(scale_ref, 4),
        "center_error_aligned": {
            "median": round(float(np.median(tr)), 5),
            "p90": round(float(np.percentile(tr, 90)), 5),
            "max": round(float(tr.max()), 5),
            "relative_to_spread": round(float(np.median(tr)) / max(scale_ref, 1e-9), 4),
        },
        "rotation_error_deg": {
            "median": round(float(np.median(rot_err)), 3),
            "mean": round(float(rot_err.mean()), 3),
            "p90": round(float(np.percentile(rot_err, 90)), 3),
            "max": round(float(rot_err.max()), 3),
            "note": "全局 Procrustes 左乘对齐后逐相机角度差(v2 修正公式)",
        },
        "relative_rotation_error_deg": {
            "median": round(float(np.median(rel_err)), 3),
            "mean": round(float(rel_err.mean()), 3),
            "p90": round(float(np.percentile(rel_err, 90)), 3),
            "max": round(float(rel_err.max()), 3),
            "note": "相机间相对朝向误差,消除左乘全局世界系旋转;右乘相机系差异不在其消除范围",
        },
        "global_alignment_rotation_deg": round(float(rot_angle_deg(Rg)), 3),
        "camera_shape_vggt": camera_shape_metrics(C_vg),
        "camera_shape_ref": camera_shape_metrics(C_ref),
        "trajectory_direction": traj,
    }

    if "pose_eval_v2" in meta:
        raise FileExistsError(f"{meta_path} 已含 pose_eval_v2,禁止覆盖")
    meta["pose_eval_v2"] = pe
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  {sid}: rot(v2) median {pe['rotation_error_deg']['median']}° "
          f"p90 {pe['rotation_error_deg']['p90']}° | rel {pe['relative_rotation_error_deg']['median']}° | "
          f"center rel {pe['center_error_aligned']['relative_to_spread']} | traj {traj['mean_cosine']}")
    return {"sequence_id": sid, "dataset_id": ds, **{
        k: v for k, v in pe.items() if k != "per_camera_rotation_deg"}}


def main():
    results = []
    for sd in sys.argv[1:]:
        print(f"== eval {sd}")
        r = eval_sequence(sd)
        if r:
            results.append(r)
    assert not os.path.exists(SUMMARY), f"{SUMMARY} 已存在,禁止覆盖"
    with open(SUMMARY, "w") as f:
        json.dump({"sequences": results}, f, indent=2, ensure_ascii=False)
    if results:
        med = [r["rotation_error_deg"]["median"] for r in results]
        p90 = [r["rotation_error_deg"]["p90"] for r in results]
        print(f"\n汇总 {len(results)} 序列(修正后绝对旋转误差): median 范围 "
              f"{min(med):.2f}–{max(med):.2f}°, p90 范围 {min(p90):.2f}–{max(p90):.2f}°")
    print(f"-> {SUMMARY}")


if __name__ == "__main__":
    main()
