"""阶段2.2:VGGT 几何推理主脚本(vggt_lora 环境,GPU)。

用法:
  python run_vggt_inference.py <sequence.json> [<sequence2.json> ...]

对每个 ready 序列保存规范要求的全部输出(禁止覆盖)。
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
from vggt.utils.geometry import unproject_depth_map_to_point_map
from provenance import make_provenance

OUT_BASE = "/fj/VGGT+head+lora实验/阶段2/02_vggt"
TOKEN_LAYERS = (4, 11, 17, 23)
SAVE_TOKEN_MAX_S = 200   # 超过此长度不落盘 tokens(meta 记录)


def sanity_checks(depth, depth_conf, pts_direct, pts_unproj,
                  extrinsic, intrinsic):
    """工程检查,返回 dict。全部 numpy,S 维在最前。"""
    d = depth.reshape(-1).astype(np.float64)
    valid_ratio = float(((d > 1e-8) & np.isfinite(d)).mean())
    nan_inf = {
        "depth": int((~np.isfinite(depth)).sum()),
        "depth_conf": int((~np.isfinite(depth_conf)).sum()),
        "points_direct": int((~np.isfinite(pts_direct)).sum(axis=(1, 2, 3)).clip(None).sum() if np.isnan(pts_direct).any() else (~np.isfinite(pts_direct)).sum()),
        "points_unprojected": int((~np.isfinite(pts_unproj)).sum()),
    }
    fx, fy = intrinsic[..., 0, 0], intrinsic[..., 1, 1]
    focal_pos_ratio = float(((fx > 0) & (fy > 0)).mean())

    # 相机中心(w2c 的逆平移): C = -R^T t
    R = extrinsic[..., :3, :3]
    t = extrinsic[..., :3, 3]
    centers = np.einsum("sij,sj->si", R.transpose(0, 2, 1), -t)
    if len(centers) > 1:
        dm = np.linalg.norm(centers[:, None] - centers[None], axis=-1)
        np.fill_diagonal(dm, np.inf)
        cam_min_dist = float(dm.min())
        collapse_cam = bool(cam_min_dist < 1e-3)
    else:
        cam_min_dist, collapse_cam = None, False

    # 点云平面度:协方差最小特征值 / 最大特征值
    sub = pts_unproj[::max(1, len(pts_unproj) // 8)].reshape(-1, 3)[::37]
    cov = np.cov(sub.T)
    ev = np.linalg.eigvalsh(cov)
    flatness = float(ev[0] / (ev[-1] + 1e-12))

    return {
        "depth_valid_positive_ratio": valid_ratio,
        "nan_inf_counts": nan_inf,
        "focal_positive_ratio": focal_pos_ratio,
        "camera_center_min_pairwise_dist": cam_min_dist,
        "camera_collapse": collapse_cam,
        "pointcloud_flatness_eigratio": flatness,
        "pointcloud_flattened": bool(flatness < 1e-4),
    }


def run_sequence(seq_path: str, model, device, dtype):
    seq = json.load(open(seq_path))
    assert seq["status"] == "ready", f"skip non-ready: {seq['sequence_id']}"
    sid = seq["sequence_id"]
    dataset_id = seq["dataset_id"]
    out_dir = os.path.join(OUT_BASE, dataset_id, sid)

    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} 已存在,禁止覆盖。如需重跑请先删除。")
    os.makedirs(os.path.join(out_dir, "checks"), exist_ok=True)
    print(f"\n=== {sid} (S={len(seq['rgb_paths'])}) ===", flush=True)

    rgb_paths = seq["rgb_paths"]
    S = len(rgb_paths)

    # 预处理:crop 模式(官方默认;宽 518 居中裁剪)
    images = load_and_preprocess_images(rgb_paths, mode="crop").to(device)
    H, W = images.shape[-2:]

    torch.manual_seed(42)
    t0 = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            aggregated_tokens_list, ps_idx = model.aggregator(images.unsqueeze(0))
            # tokens: layer 4/11/17/23
            tokens_fp16 = {f"layer_{i:02d}": aggregated_tokens_list[i].squeeze(0).half().cpu()
                           for i in TOKEN_LAYERS}
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (H, W))
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images.unsqueeze(0), ps_idx)
            pts3d, pts3d_conf = model.point_head(aggregated_tokens_list, images.unsqueeze(0), ps_idx)
    dt = time.time() - t0
    print(f"  forward done in {dt:.1f}s | mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB", flush=True)
    torch.cuda.reset_peak_memory_stats()

    # -> numpy
    pose_enc_np = pose_enc.squeeze(0).float().cpu().numpy()                # (S,9)
    ext_w2c = extrinsic.squeeze(0).float().cpu().numpy()                   # (S,3,4)
    intr = intrinsic.squeeze(0).float().cpu().numpy()                      # (S,3,3)
    depth_np = depth_map.squeeze(0).squeeze(-1).float().cpu().numpy()      # (S,H,W)
    dconf_np = depth_conf.squeeze(0).float().cpu().numpy()
    pdir_np = pts3d.squeeze(0).squeeze(-2).float().cpu().numpy()           # (S,H,W,3)
    pconf_np = pts3d_conf.squeeze(0).float().cpu().numpy()

    # c2w = inv(T_cw)
    ext_c2w = np.stack([np.linalg.inv(np.vstack([e, [0, 0, 0, 1.0]])) for e in ext_w2c])

    # 反投影点图(第一候选)—— 官方函数接受 numpy/torch
    punj_np = unproject_depth_map_to_point_map(depth_np[..., None], ext_w2c, intr)

    # ---- 保存 ----
    np.save(os.path.join(out_dir, "pose_enc.npy"), pose_enc_np)
    np.save(os.path.join(out_dir, "extrinsic_w2c.npy"), ext_w2c)
    np.save(os.path.join(out_dir, "extrinsic_c2w.npy"), ext_c2w)
    np.save(os.path.join(out_dir, "intrinsic_vggt.npy"), intr)
    np.save(os.path.join(out_dir, "depth_vggt.npy"), depth_np.astype(np.float32))
    np.save(os.path.join(out_dir, "depth_conf_vggt.npy"), dconf_np.astype(np.float32))
    np.save(os.path.join(out_dir, "point_map_direct.npy"), pdir_np.astype(np.float32))
    np.save(os.path.join(out_dir, "point_conf_direct.npy"), pconf_np.astype(np.float32))
    np.save(os.path.join(out_dir, "point_map_unprojected.npy"), punj_np.astype(np.float32))

    tokens_saved = S <= SAVE_TOKEN_MAX_S
    if tokens_saved:
        for k, v in tokens_fp16.items():
            np.save(os.path.join(out_dir, f"tokens_{k}.npy"), v.numpy())

    # ---- sanity ----    (tokens NaN 检查在 GPU 上做)
    tok_nan = sum(int((~torch.isfinite(t)).sum()) for t in tokens_fp16.values())
    sanity = sanity_checks(depth_np, dconf_np, pdir_np, punj_np, ext_w2c, intr)
    sanity["token_nan_inf_count"] = tok_nan

    # ---- meta(provenance 8 项 + 工程信息)----
    prov = make_provenance(
        "vggt", rgb_paths,
        resize_crop_params=f"load_and_preprocess_images(mode='crop'), target 518x{H}",
        intrinsics_transform="pose_encoding_to_extri_intri; cx,cy=W/2,H/2; fov→fx,fy",
        precision_mode=f"{dtype} autocast (official demo mode)",
        extra={"S": S},
    )
    meta = {
        **prov,
        "sequence_id": sid,
        "dataset_id": dataset_id,
        "output_shapes": {"pose_enc": list(pose_enc_np.shape),
                          "extrinsic_w2c": list(ext_w2c.shape),
                          "intrinsic": list(intr.shape),
                          "depth": list(depth_np.shape),
                          "point_map_unprojected": list(punj_np.shape)},
        "forward_seconds": round(dt, 2),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "tokens_saved": tokens_saved,
        "interface_verified": True,   # aggregator 直调已验证可取任意层 token
        "sanity": sanity,
        "primary_pointmap_candidate": "point_map_unprojected (per official recommendation)",
    }
    with open(os.path.join(out_dir, "prediction_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  saved outputs to {out_dir} | sanity: valid={sanity['depth_valid_positive_ratio']:.4f}, "
          f"flat={sanity['pointcloud_flatness_eigratio']:.2e}", flush=True)
    return out_dir


def main():
    seq_paths = sys.argv[1:]
    assert seq_paths, "usage: run_vggt_inference.py <seq.json> ..."
    device = "cuda"
    dtype = torch.bfloat16
    print("loading VGGT-1B ...", flush=True)
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    for sp in seq_paths:
        try:
            run_sequence(sp, model, device, dtype)
        except FileExistsError as e:
            print(f"SKIP: {e}")
    print("\nALL DONE")


if __name__ == "__main__":
    main()
