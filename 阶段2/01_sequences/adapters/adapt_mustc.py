"""MuST-C 适配器:UGV plot198 / 230613,pos00–03 四个 20 机位同步子序列。

- 外参:cam_params.xml 中 camera(label=nikon_N) 的 transform(Metashape 4x4 c2w,
  plot 局部坐标系)→ 转 OpenCV w2c
- 内参:sensor 的 calibration(f, cx, cy, k1..p2;cx/cy 以像素为单位)
- 点云 GT:UGV-LMI 与 UGV-Ouster las
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dump_intrinsics_json, dump_extrinsics_json, write_sequence

ROOT = "/fj/VGGT+head+lora实验/阶段1-数据集/MuST-C"
PLOT_DIR = os.path.join(ROOT, "images", "UGV-RGB", "230613", "plot198")
XML_PATH = os.path.join(ROOT, "images", "UGV-RGB", "230613", "cam_params.xml")
SEQ_BASE = "/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences/mustc"


def parse_xml():
    """返回 (sensors{sid: K_dict}, cams{label: {"sensor_id","c2w"}})。"""
    root = ET.parse(XML_PATH).getroot()
    sensors = {}
    for s in root.iter("sensor"):
        cal = s.find("calibration")
        if cal is None:
            continue
        w = int(cal.find("resolution").get("width"))
        h = int(cal.find("resolution").get("height"))
        f = float(cal.find("f").text)
        sensors[int(s.get("id"))] = {
            "model": "OPENCV",
            "width": w, "height": h,
            "fl_x": f, "fl_y": f,
            "cx": float(cal.find("cx").text), "cy": float(cal.find("cy").text),
            "dist": {k: float(cal.find(k).text)
                     for k in ("k1", "k2", "k3", "p1", "p2") if cal.find(k) is not None},
            "source": "cam_params.xml sensor calibration (Metashape adjusted)",
        }
    cams = {}
    for c in root.iter("camera"):
        t = np.array([float(v) for v in c.find("transform").text.split()]).reshape(4, 4)
        cams[c.get("label")] = {"sensor_id": int(c.get("sensor_id")), "c2w": t}
    return sensors, cams


def metashape_c2w_to_opencv_w2c(c2w):
    """Metashape 变换为右手系 c2w(+Z 指向观察方向的反向即 +X 右 +Y 下 +Z 前
    取决于版本;官方 XML 的 rotation 列即相机轴在 world 中的朝向)。
    Metashape 默认相机坐标系:+X 右、+Y 下、+Z 向后(指向场景外),与 OpenCV 相同,
    因此无需翻转,直接取逆得 w2c。"""
    return np.linalg.inv(c2w)


def adapt_pos(pos_dir: str):
    pos = os.path.basename(pos_dir)  # e.g. pos00
    imgs = sorted((p for p in os.listdir(pos_dir) if p.endswith(".jpeg")),
                  key=lambda s: int(re.search(r"(\d+)", s).group(1)))
    if not imgs:
        print(f"  SKIP {pos}: no images")
        return

    sensors, cams = parse_xml()
    seq_dir = os.path.join(SEQ_BASE, f"plot198__230613__ugv__{pos}", "camera")

    # 内参:每机位独立 sensor → 写成 per-camera 数组
    kpath = os.path.join(seq_dir, "intrinsics.json")
    if os.path.exists(kpath):
        raise FileExistsError(f"{kpath} 已存在,禁止覆盖")
    per_cam = {}
    for im in imgs:
        label = os.path.splitext(im)[0]  # nikon_0
        cam = cams[label]
        K = dict(sensors[cam["sensor_id"]])
        per_cam[label] = K
    os.makedirs(seq_dir, exist_ok=True)
    with open(kpath, "w") as f:
        import json
        json.dump({"per_camera": True, "cameras": per_cam}, f, indent=2)

    ext_list = []
    for i, im in enumerate(imgs):
        label = os.path.splitext(im)[0]
        w2c = metashape_c2w_to_opencv_w2c(cams[label]["c2w"])
        ext_list.append({"camera_id": label, "image_name": im,
                         "w2c": [[float(v) for v in row] for row in w2c]})
    epath = dump_extrinsics_json(seq_dir, ext_list)

    seq = {
        "dataset_id": "mustc",
        "sequence_id": f"mustc__plot198__230613__ugv__{pos}",
        "status": "ready",
        "rgb_paths": [os.path.join(pos_dir, im) for im in imgs],
        "camera_ids": [e["camera_id"] for e in ext_list],
        "intrinsics_path": kpath,
        "extrinsics_path": epath,
        "reference_pointcloud": os.path.join(ROOT, "point_clouds", "UGV-LMI", "plot198", "230613.las"),
        "linear_unit": "meter",
        "camera_convention": "opencv_w2c",
        "group_key": "plot198",
        "dataset_root": ROOT,
        "source_files": {"cam_params_xml": XML_PATH},
        "extra": {
            "sort_key": "nikon index numeric order within each pos dir",
            "platform": "UGV", "sensor_run": "UGV-RGB",
            "date": "230613",
            "coordinate_frame": "plot-local (Metashape chunk coords), NOT UTM",
            "secondary_pointcloud": os.path.join(ROOT, "point_clouds", "UGV-Ouster", "plot198", "230613.las"),
            "n_views_expected": 20,
            "intrinsics_per_camera": True,
        },
    }
    write_sequence(seq, "mustc", f"plot198__230613__ugv__{pos}.json")


if __name__ == "__main__":
    for d in sorted(os.listdir(PLOT_DIR)):
        if d.startswith("pos"):
            print(f"== adapting {d}")
            adapt_pos(os.path.join(PLOT_DIR, d))
