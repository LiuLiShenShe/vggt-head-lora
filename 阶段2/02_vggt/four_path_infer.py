"""四路判别实验 · 步骤1:GPU 推理落盘(vggt_lora 环境,无 matplotlib 依赖)。

对成功(05-03-24)与失败(12-03-24)序列各取均匀 8/16/24/36 帧,
保存路径 A/B/C 所需的全部中间量 → four_path_data/{key}_n{n}.npz:
  depth (n,518,518), ext_w2c_vggt (n,3,4), intr_vggt (n,3,3),
  point_map_head (n,518,518,3) [路径 C]
参考量(参考相机 w2c/crop 内参/参考云)由步骤2(da3)从 sequence.json 读取。

用法: python four_path_infer.py
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

VGGT_ROOT = "/fj/VGGT+head+lora实验/vggt"
sys.path.insert(0, VGGT_ROOT)

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

BASE = "/fj/VGGT+head+lora实验/阶段2"
DEFAULT_DATA_DIR = os.path.join(BASE, "10_failures", "four_path_data")
SEQS = {
    "success_05-03-24": "plantview__langdon_4__05-03-24",
    "fail_12-03-24": "plantview__langdon_4__12-03-24",
}
N_FRAMES = [8, 16, 24, 36]


def main():
    ap = argparse.ArgumentParser(description="四路判别 · 步骤1 GPU 推理(禁止覆盖)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                    help="npz 输出目录(默认 %(default)s;clean rerun 指向 v2_clean_rerun_eval)")
    args = ap.parse_args()
    assert not os.path.exists(args.data_dir), f"{args.data_dir} 已存在"
    os.makedirs(args.data_dir)
    device, dtype = "cuda", torch.bfloat16
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()

    for key, sid in SEQS.items():
        seq = next(json.load(open(p)) for p in glob.glob(f"{BASE}/01_sequences/sequences/plant_view/*.json")
                   if json.load(open(p))["sequence_id"] == sid)
        for n in N_FRAMES:
            idx = np.linspace(0, len(seq["rgb_paths"]) - 1, n).astype(int)
            rgb = [seq["rgb_paths"][i] for i in idx]
            images = load_and_preprocess_images(rgb, mode="crop").to(device)
            H, W = images.shape[-2:]
            torch.manual_seed(42)
            t0 = time.time()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
                tokens, ps_idx = model.aggregator(images.unsqueeze(0))
                pose_enc = model.camera_head(tokens)[-1]
                ext_w2c, intr = pose_encoding_to_extri_intri(pose_enc, (H, W))
                depth, dconf = model.depth_head(tokens, images.unsqueeze(0), ps_idx)
                pts3d, _ = model.point_head(tokens, images.unsqueeze(0), ps_idx)
            dt = time.time() - t0
            out = os.path.join(args.data_dir, f"{key}_n{n}.npz")
            np.savez_compressed(
                out,
                depth=depth.squeeze(0).squeeze(-1).float().cpu().numpy(),
                ext_w2c_vggt=ext_w2c.squeeze(0).float().cpu().numpy(),
                intr_vggt=intr.squeeze(0).float().cpu().numpy(),
                point_map_head=pts3d.squeeze(0).squeeze(-2).float().cpu().numpy(),
                frame_idx=idx,
            )
            print(f"{key} n={n}: {dt:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
