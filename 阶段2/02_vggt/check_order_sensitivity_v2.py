"""阶段2.2:输入顺序敏感性检查 v2(vggt_lora 环境,GPU)——审计修正版。

v1 问题(审计确认):R_o/R_r 均来自反序推理数组(自比),orig 加载后未使用,
结论无效。v2:R_o/C_o 用已保存的原序 extrinsic_w2c.npy,R_r/C_r 用反序推理
结果按索引还原,再做全局旋转 Procrustes + 中心 Sim3 对齐后比较。
结果写 10_failures/order_sensitivity_v2.json(旧文件保留)。

用法: python check_order_sensitivity_v2.py
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
RESULT = "/fj/VGGT+head+lora实验/阶段2/10_failures/order_sensitivity_v2.json"

TARGETS = [
    os.path.join(SEQ_BASE, "wheat3dgs", "plot_463.json"),
    os.path.join(SEQ_BASE, "mustc", "plot198__230613__ugv__pos00.json"),
]


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


def global_rotation_procrustes(R_a, R_b):
    """Rg: Rg @ Ra_i ≈ Rb_i。H = Σ Ra_i Rb_iᵀ = UΣVᵀ → Rg = V Uᵀ。"""
    H = np.einsum("sij,skj->ik", R_a, R_b)
    U, _, Vt = np.linalg.svd(H)
    Rg = Vt.T @ U.T
    if np.linalg.det(Rg) < 0:
        U[:, -1] *= -1
        Rg = Vt.T @ U.T
    return Rg


def rot_angle_deg(R):
    cos = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cos)))


def centers(e):
    return np.einsum("sij,sj->si", e[:, :3, :3].transpose(0, 2, 1), -e[:, :3, 3])


def run_reversed(seq_path, model, device, dtype):
    seq = json.load(open(seq_path))
    sid = seq["sequence_id"]
    orig = np.load(os.path.join(OUT_BASE, seq["dataset_id"], sid, "extrinsic_w2c.npy")).astype(np.float64)
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
    ext_rev = ext_w2c_rev.squeeze(0).float().cpu().numpy().astype(np.float64)
    dt = time.time() - t0

    # 还原到原序索引:反序输出的第 j 帧 = 原序第 S-1-j 帧
    perm = np.arange(S)[::-1]
    ext_restored = ext_rev[perm]

    # 旋转:orig vs 反序还原,全局 Procrustes 对齐后逐相机角度差
    R_o = orig[:, :3, :3].transpose(0, 2, 1)
    R_r = ext_restored[:, :3, :3].transpose(0, 2, 1)
    Rg = global_rotation_procrustes(R_r, R_o)
    rot_err = np.array([rot_angle_deg((Rg @ Rr_i).T @ Ro_i) for Rr_i, Ro_i in zip(R_r, R_o)])

    # 相机中心 Sim3 对齐后比较
    C_o = centers(orig)
    C_r = centers(ext_restored)
    s, R, t = horn_sim3_params(C_r, C_o)
    C_al = s * (R @ C_r.T).T + t
    tr_err = np.linalg.norm(C_al - C_o, axis=1)
    span = float(np.linalg.norm(C_o - C_o.mean(0), axis=1).mean())

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
            "relative_to_spread": round(float(np.median(tr_err)) / max(span, 1e-9), 4),
        },
        "global_alignment_rotation_deg": round(float(rot_angle_deg(Rg)), 3),
    }


def main():
    assert not os.path.exists(RESULT), f"{RESULT} 已存在,禁止覆盖"
    device, dtype = "cuda", torch.bfloat16
    print("loading VGGT-1B ...", flush=True)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    results = [run_reversed(p, model, device, dtype) for p in TARGETS]
    out = {
        "description": "输入图像顺序反序后 VGGT 位姿一致性(v2 修正:v1 存在自比 bug,原序与反序均来自同一数组)",
        "original_source": "已保存的 extrinsic_w2c.npy(原序推理)",
        "reversed_source": "反序输入重新前向,按索引还原",
        "alignment": "旋转:全局 Procrustes 左乘对齐;中心:Umeyama-Horn Sim3",
        "results": results,
    }
    os.makedirs(os.path.dirname(RESULT), exist_ok=True)
    with open(RESULT, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    for r in results:
        print(f"{r['sequence_id']}: rot median {r['rotation_error_deg']['median']}° "
              f"p90 {r['rotation_error_deg']['p90']}° | center rel "
              f"{r['center_error_aligned']['relative_to_spread']}", flush=True)
    print(f"-> {RESULT}")


if __name__ == "__main__":
    main()
