"""3D Plant View (langdon_4) 适配器。

每个日期目录生成一个 sequence:
- 相机来自 transforms/adjusted/transforms.json(nerfstudio/OpenGL c2w)
- 转为 OpenCV w2c 输出
- group_key = 植株 langdon_4(同一物理植株跨日期)
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (ns_c2w_to_opencv_w2c, dump_intrinsics_json,
                    dump_extrinsics_json, write_sequence)

ROOT = "/fj/VGGT+head+lora实验/阶段1-数据集/3D Plant View/langdon_4"


def adapt_date(date_dir: str):
    date = os.path.basename(date_dir)
    tf_path = os.path.join(date_dir, "transforms", "adjusted", "transforms.json")
    if not os.path.exists(tf_path):
        print(f"  SKIP {date}: no adjusted transforms.json")
        return
    with open(tf_path) as f:
        tf = json.load(f)

    frames = sorted(tf["frames"], key=lambda fr: fr["file_path"])
    # 只保留实际存在的 rgb 图像
    rgb_paths, kept = [], []
    for fr in frames:
        img_abs = os.path.normpath(os.path.join(os.path.dirname(tf_path), fr["file_path"]))
        if os.path.exists(img_abs):
            rgb_paths.append(img_abs)
            kept.append(fr)
    print(f"  {date}: {len(rgb_paths)} images (of {len(frames)} frames)")

    cam_dir = os.path.join("/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences",
                           "plant_view", f"langdon_4__{date}", "camera")

    # 内参:所有帧共享同一组(nerfstudio transforms.json 通常全局一致)
    f0 = kept[0]
    K = {
        "model": "OPENCV",
        "width": int(f0["w"]), "height": int(f0["h"]),
        "fl_x": f0["fl_x"], "fl_y": f0["fl_y"],
        "cx": f0["cx"], "cy": f0["cy"],
        "dist": {"k1": f0.get("k1", 0), "k2": f0.get("k2", 0),
                 "k3": f0.get("k3", 0), "k4": f0.get("k4", 0),
                 "p1": f0.get("p1", 0), "p2": f0.get("p2", 0)},
        "source": os.path.relpath(tf_path, ROOT),
    }
    kpath = dump_intrinsics_json(cam_dir, K)

    ext_list = []
    for i, fr in enumerate(kept):
        w2c = ns_c2w_to_opencv_w2c(fr["transform_matrix"])
        ext_list.append({"camera_id": f"view_{i:04d}",
                         "image_name": os.path.basename(fr["file_path"]),
                         "w2c": [[float(v) for v in row] for row in w2c]})
    epath = dump_extrinsics_json(cam_dir, ext_list)

    seq = {
        "dataset_id": "plant_view_3d",
        "sequence_id": f"plantview__langdon_4__{date}",
        "status": "ready",
        "rgb_paths": rgb_paths,
        "camera_ids": [e["camera_id"] for e in ext_list],
        "intrinsics_path": kpath,
        "extrinsics_path": epath,
        "reference_pointcloud": os.path.join(
            date_dir, "gaussian-splatting", "undistorted_segmented", "splatfacto", "1", "splat.ply"),
        "linear_unit": "meter",
        "camera_convention": "opencv_w2c",
        "group_key": "langdon_4",
        "dataset_root": ROOT,
        "source_files": {"transforms_json": tf_path},
        "extra": {
            "sort_key": "frame file_path natural order from adjusted/transforms.json",
            "depth_dir": os.path.join(date_dir, "images", "depth"),
            "mask_dir": os.path.join(date_dir, "images", "mask"),
            "intrinsics_shared_across_views": True,
            "note": "adjusted transforms preferred over original; "
                    "reference cloud is GS splat.ply (units assumed meter)",
            "capture_config": os.path.join(date_dir, "config.txt"),
        },
    }
    write_sequence(seq, "plant_view", f"langdon_4__{date}.json")


if __name__ == "__main__":
    for d in sorted(glob.glob(os.path.join(ROOT, "*-*-*"))):
        print(f"== adapting {os.path.basename(d)}")
        adapt_date(d)
