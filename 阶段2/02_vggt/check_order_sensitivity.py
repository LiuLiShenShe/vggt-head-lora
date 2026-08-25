"""阶段2.2:输入顺序敏感性抽查(vggt_lora 环境,GPU)。

把 wheat3dgs plot_463 与 mustc pos00 的输入图像顺序反序后重新推理,
将反序结果按索引还原(反序输出的第 j 帧 = 原序第 S-1-j 帧),
与已保存的原序 extrinsic_w2c.npy 比较:
  - 旋转误差:VGGT 每次推理世界系任意,先用 Kabsch 估计全局旋转 R_g
    (最小化 sum ||R_g @ R_rev - R_orig||),再算逐相机角度差(度);
  - 平移误差:相机中心做 Sim(3)(Umeyama-Horn)对齐后统计欧氏距离。
结果写 10_failures/order_sensitivity.json(禁止覆盖,已存在则报错)。

用法: python check_order_sensitivity.py
"""
import json
import os
import sys
import time

import numpy as np
import torch

VGGT_ROOT = "/fj/VGGT+head+lora实验/vggt"
sys.path.insert(0, VGGT_ROOT)
sys.path.insert(0, "/fj/VGGT+head+lora实验/阶段2/00_environment")

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

OUT_BASE = "/fj/VGGT+head+lora实验/阶段2/02_vggt"
SEQ_BASE = "/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences"
RESULT = "/fj/VGGT+head+lora实验/阶段2/10_failures/order_sensitivity.json"

TARGETS = [
    os.path.join(SEQ_BASE, "wheat3dgs", "plot_463.json"),
    os.path.join(SEQ_BASE, "mustc", "plot198__230613__ugv__pos00.json"),
]


def kabsch_rotation(A, B):
    """求 R 使 B ≈ R @ A(列向量约定:R @ a ≈ b)。A,B:(N,3)。"""
    H = (B - B.mean(0)).T @ (A - A.mean(0))
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def horn_sim3(src, dst):
    """dst ≈ s R src + t,返回对齐后的 src。"""
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
    return (s * (R @ src.T).T + t)


def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def run_reversed(seq_path, model, device, dtype):
    seq = json.load(open(seq_path))
    sid = seq["sequence_id"]
    orig = np.load(os.path.join(OUT_BASE, seq["dataset_id"], sid, "extrinsic_w2c.npy"))
    S = len(seq["rgb_paths"])
    assert orig.shape[0] == S

    rgb_rev = seq["rgb_paths"][::-1]
    images = load_and_preprocess_images(rgb_rev, mode="crop").to(device)
    H, W = images.shape[-2:]
    torch.manual_seed(42)
    t0 = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            pose_enc = model.camera_head(model.aggregator(images.unsqueeze(0))[0])[-1]
            ext_w2c_rev, _ = pose_encoding_to_extri_intri(pose_enc, (H, W))
    ext_rev = ext_w2c_rev.squeeze(0).float().cpu().numpy()   # (S,3,4) 反序
    dt = time.time() - t0

    # 还原到原序:orig_idx = S-1-rev_idx
    perm = np.arange(S)[::-1]
    ext_restored = ext_rev[perm]

    # 全局旋转对齐(Kabsch on rotation matrices)
    R_o = ext_restored[:, :3, :3].transpose(0, 2, 1)   # c2w 旋转(行向量约定存 w2c)
    R_r = ext_rev[perm][:, :3, :3].transpose(0, 2, 1)
    Rg = kabsch_rotation(R_r.reshape(-1, 3), R_o.reshape(-1, 3))
    rot_err = np.array([rot_angle_deg(Rg @ Rr.T @ Ro) for Rr, Ro in zip(R_r, R_o)])

    # 相机中心 Sim(3) 对齐后比较
    def centers(e):
        R = e[:, :3, :3]
        t = e[:, :3, 3]
        return np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)
    C_o = centers(ext_restored)
    C_r = centers(ext_rev[perm])
    C_al = horn_sim3(C_r, C_o)
    tr_err = np.linalg.norm(C_al - C_o, axis=1)

    return {
        "sequence_id": sid,
        "S": S,
        "forward_seconds": round(dt, 2),
        "rotation_error_deg": {
            "median": round(float(np.median(rot_err)), 3),
            "p90": round(float(np.percentile(rot_err, 90)), 3),
            "max": round(float(rot_err.max()), 3),
        },
        "center_error_aligned": {
            "median": round(float(np.median(tr_err)), 5),
            "p90": round(float(np.percentile(tr_err, 90)), 5),
            "max": round(float(tr_err.max()), 5),
            "scale_ref": round(float(np.linalg.norm(C_o - C_o.mean(0), axis=1).mean()), 4),
        },
        "per_camera_rotation_deg": [round(float(x), 3) for x in rot_err],
    }


def main():
    assert not os.path.exists(RESULT), f"{RESULT} 已存在,禁止覆盖"
    device, dtype = "cuda", torch.bfloat16
    print("loading VGGT-1B ...", flush=True)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    results = [run_reversed(p, model, device, dtype) for p in TARGETS]
    out = {
        "description": "输入图像顺序反序后 VGGT 位姿一致性抽查(还原索引后与原序比较)",
        "global_rotation_alignment": "Kabsch on c2w rotation matrices",
        "center_alignment": "Umeyama-Horn Sim(3) on camera centers",
        "results": results,
    }
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for r in results:
        print(f"{r['sequence_id']}: rot median {r['rotation_error_deg']['median']}° "
              f"p90 {r['rotation_error_deg']['p90']}° | center median "
              f"{r['center_error_aligned']['median']}", flush=True)
    print(f"-> {RESULT}")


if __name__ == "__main__":
    main()
