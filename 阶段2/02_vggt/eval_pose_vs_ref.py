"""阶段2.2:VGGT 位姿 vs 参考相机评估(da3 环境,CPU)。

对有参考 extrinsics 的序列:
  1. Umeyama-Horn Sim(3) 对齐 VGGT 相机中心 → 参考相机中心(VGGT 世界系任意);
  2. 对齐后逐相机旋转误差:参考 c2w 旋转 vs VGGT c2w 旋转(先做全局旋转
     对齐 Kabsch,再算逐相机角度差);
  3. 轨迹方向一致性:对齐后相机中心序列的相邻位移向量余弦相似度。
结果写各序列 prediction_meta.json 的 pose_eval 字段 + 02_vggt/pose_eval_summary.json。

用法: python eval_pose_vs_ref.py <seq_dir> [<seq_dir2> ...]
"""
import glob
import json
import os
import sys

import numpy as np

SEQ_BASE = "/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences"
OUT_BASE = "/fj/VGGT+head+lora实验/阶段2/02_vggt"
SUMMARY = os.path.join(OUT_BASE, "pose_eval_summary.json")


def horn_sim3(src, dst):
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
    """求全局旋转 Rg 最小化 sum ||Rg @ Rv_i - Rr_i||_F。

    正交 Procrustes:令 M = sum(Rr_i @ Rv_i^T),M = U S V^T,
    则 Rg = V @ U^T(最大化 tr(Rg^T M))。"""
    M = np.einsum("sij,skj->ik", R_ref_c2w, R_vg_c2w)   # sum Rr Rv^T
    U, _, Vt = np.linalg.svd(M)
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


def load_ref_w2c(seq):
    """参考 extrinsics json: extrinsics[].w2c (4x4, opencv_w2c)。返回 (w2c(S,3,4), order)。"""
    p = seq.get("extrinsics_path")
    if not p or not os.path.exists(p):
        return None
    d = json.load(open(p))
    ext = d["extrinsics"]
    return np.stack([np.array(e["w2c"], dtype=np.float64)[:3, :4] for e in ext])


def eval_sequence(seq_dir):
    meta_path = os.path.join(seq_dir, "prediction_meta.json")
    meta = json.load(open(meta_path))
    sid, ds = meta["sequence_id"], meta["dataset_id"]
    ds_dir = os.path.join(SEQ_BASE,
                          {"plant_view_3d": "plant_view", "wheat3dgs": "wheat3dgs",
                           "mustc": "mustc", "terra_ref": "terraref"}[ds])
    seq = next(json.load(open(p)) for p in glob.glob(os.path.join(ds_dir, "*.json"))
               if json.load(open(p))["sequence_id"] == sid)
    ref_w2c = load_ref_w2c(seq)
    if ref_w2c is None:
        print(f"  {sid}: 无参考相机,跳过")
        return None
    assert len(ref_w2c) == len(seq["rgb_paths"])

    vg_w2c = np.load(os.path.join(seq_dir, "extrinsic_w2c.npy")).astype(np.float64)
    S = len(vg_w2c)

    # --- 中心 Sim(3) 对齐 ---
    C_vg = w2c_centers(vg_w2c)
    C_ref = w2c_centers(ref_w2c)
    s, R, t = horn_sim3(C_vg, C_ref)
    C_al = (s * (R @ C_vg.T).T + t)
    tr = np.linalg.norm(C_al - C_ref, axis=1)
    scale_ref = float(np.linalg.norm(C_ref - C_ref.mean(0), axis=1).mean())

    # --- 旋转:全局 Procrustes 对齐 c2w 旋转矩阵,再逐相机角度差 ---
    # 注意:VGGT 世界系任意,且相机系可能与参考差常值右乘旋转 Q,
    # 因此"全局对齐后逐相机角度差"受全局模糊性污染。真正与全局系无关的
    # 精度指标是"相对旋转误差"(见 rel_rot_err),其值即模型真实位姿精度。
    R_vg_c2w = vg_w2c[:, :3, :3].transpose(0, 2, 1)     # (S,3,3)
    R_ref_c2w = ref_w2c[:, :3, :3].transpose(0, 2, 1)
    Rg = global_rotation_procrustes(R_vg_c2w, R_ref_c2w)
    rot_err = np.array([rot_angle_deg(Rg @ Rv.T @ Rr)
                        for Rv, Rr in zip(R_vg_c2w, R_ref_c2w)])

    # --- 相对旋转误差(全局系无关):相机 i→j 相对朝向 R_i^T R_j ---
    all_pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
    rel_err = np.array([rot_angle_deg((R_ref_c2w[i].T @ R_ref_c2w[j]).T
                                      @ (R_vg_c2w[i].T @ R_vg_c2w[j]))
                        for i, j in all_pairs])

    # --- 轨迹方向一致性:相邻相机中心位移向量余弦 ---
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
        },
        "relative_rotation_error_deg": {
            "median": round(float(np.median(rel_err)), 3),
            "mean": round(float(rel_err.mean()), 3),
            "p90": round(float(np.percentile(rel_err, 90)), 3),
            "max": round(float(rel_err.max()), 3),
            "note": "相机间相对朝向误差,消除全局世界系/相机系模糊性,反映模型真实位姿精度",
        },
        "global_alignment_rotation_deg": round(float(rot_angle_deg(Rg)), 3),
        "trajectory_direction": traj,
        "per_camera_rotation_deg": [round(float(x), 3) for x in rot_err],
    }

    # 禁止覆盖:meta 已含 pose_eval 则报错
    if "pose_eval" in meta:
        raise FileExistsError(f"{meta_path} 已含 pose_eval,禁止覆盖")
    meta["pose_eval"] = pe
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  {sid}: rel_rot median {pe['relative_rotation_error_deg']['median']}° "
          f"p90 {pe['relative_rotation_error_deg']['p90']}° | "
          f"global_rot {pe['global_alignment_rotation_deg']}° | center rel "
          f"{pe['center_error_aligned']['relative_to_spread']} | traj cos "
          f"{traj['mean_cosine']}")
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
        med = [r["relative_rotation_error_deg"]["median"] for r in results]
        p90 = [r["relative_rotation_error_deg"]["p90"] for r in results]
        print(f"\n汇总 {len(results)} 序列(相对旋转误差): median 范围 "
              f"{min(med):.2f}–{max(med):.2f}°, p90 范围 {min(p90):.2f}–{max(p90):.2f}°")
    print(f"-> {SUMMARY}")


if __name__ == "__main__":
    main()
