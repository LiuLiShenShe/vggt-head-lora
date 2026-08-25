"""Wheat3DGS 适配器:7 个 plot,每 plot 36 视图。

- 内参:sparse/0/cameras.txt(PINHOLE 单相机)
- 外参:sparse/0/images.txt(COLMAP qvec/tvec,已是 OpenCV w2c 约定,无需转换)
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (parse_cameras_txt, parse_images_txt, quat_qvec2rotmat,
                    rot_t_to_w2c, dump_intrinsics_json, dump_extrinsics_json,
                    write_sequence)

ROOT = "/fj/VGGT+head+lora实验/阶段1-数据集/Wheat3DGS/dataset"


def adapt_plot(plot_dir: str):
    plot_id = os.path.basename(plot_dir)
    sparse = os.path.join(plot_dir, "sparse", "0")
    cams = parse_cameras_txt(os.path.join(sparse, "cameras.txt"))
    imgs = parse_images_txt(os.path.join(sparse, "images.txt"))

    # 图像目录实际文件
    img_files = {os.path.basename(p): p for p in glob.glob(os.path.join(plot_dir, "images", "*"))}
    # pose 与磁盘图像一一对应校验
    pose_names = [e["name"] for e in imgs]
    missing_pose = [n for n in img_files if n not in set(pose_names)]
    missing_img = [n for n in pose_names if n not in img_files]
    if missing_pose or missing_img:
        print(f"  WARN {plot_id}: poses without image={missing_pose}, images without pose={missing_img}")

    assert len(cams) == 1, f"{plot_id}: expect single camera, got {len(cams)}"
    cam = list(cams.values())[0]
    assert cam["model"] == "PINHOLE", f"{plot_id}: model {cam['model']}"

    seq_dir = os.path.join("/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences",
                           "wheat3dgs", plot_id, "camera")
    fx, fy, cx, cy = cam["params"]
    K = {"model": "PINHOLE", "width": cam["width"], "height": cam["height"],
         "fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy,
         "source": os.path.join("sparse", "0", "cameras.txt")}
    kpath = dump_intrinsics_json(seq_dir, K)

    rgb_paths, ext_list, cam_ids = [], [], []
    for e in imgs:
        if e["name"] not in img_files:
            continue
        R = quat_qvec2rotmat(e["qvec"])
        T = rot_t_to_w2c(R, e["tvec"])  # COLMAP images.txt 本身就是 w2c
        cid = f"cam_{os.path.splitext(e['name'])[0].split('_cam_')[-1]}"
        ext_list.append({"camera_id": cid, "image_name": e["name"],
                         "w2c": [[float(v) for v in row] for row in T]})
        rgb_paths.append(img_files[e["name"]])
        cam_ids.append(cid)

    epath = dump_extrinsics_json(seq_dir, ext_list)

    seq = {
        "dataset_id": "wheat3dgs",
        "sequence_id": f"wheat3dgs__{plot_id}",
        "status": "ready",
        "rgb_paths": sorted(rgb_paths, key=lambda p: ext_list[rgb_paths.index(p)] and 0) or rgb_paths,
        "camera_ids": cam_ids,
        "intrinsics_path": kpath,
        "extrinsics_path": epath,
        "reference_pointcloud": os.path.join(sparse, "points3D.txt"),
        "linear_unit": "unknown_colmap_units",
        "camera_convention": "opencv_w2c",
        "group_key": plot_id,
        "dataset_root": ROOT,
        "source_files": {"cameras_txt": os.path.join(sparse, "cameras.txt"),
                         "images_txt": os.path.join(sparse, "images.txt")},
        "extra": {
            "sort_key": "COLMAP images.txt order (natural by filename)",
            "mask_dir": os.path.join(plot_dir, "masks"),
            "bboxes_dir": os.path.join(plot_dir, "bboxes"),
            "manual_label_note": "see dataset paper; masks are YOLO-SAM instance masks",
            "n_views_expected": 36,
            "poses_missing_image": missing_img,
            "images_missing_pose": missing_pose,
        },
    }
    # rgb_paths 按 extrinsics 顺序(即 COLMAP 自然排序)
    name2idx = {e["image_name"]: i for i, e in enumerate(ext_list)}
    seq["rgb_paths"] = [p for _, p in sorted(
        ((name2idx[os.path.basename(p)], p) for p in rgb_paths), key=lambda x: x[0])]
    write_sequence(seq, "wheat3dgs", f"{plot_id}.json")


if __name__ == "__main__":
    for d in sorted(glob.glob(os.path.join(ROOT, "plot_*"))):
        print(f"== adapting {os.path.basename(d)}")
        adapt_plot(d)
