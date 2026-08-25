"""阶段2.1 适配器公共工具:sequence schema、COLMAP 解析、坐标转换。

所有 sequence.json 遵循统一 schema(见 SEQUENCE_FIELDS)。
适配器只读原始数据集,输出写到 sequences/<dataset_id>/。
"""
import json
import os
import re
import sys

# 复用阶段2.0 的 sha256_file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "00_environment"))
from provenance import sha256_file  # noqa: E402

SEQUENCE_FIELDS = [
    "dataset_id", "sequence_id", "status", "rgb_paths", "camera_ids",
    "intrinsics_path", "extrinsics_path", "reference_pointcloud",
    "linear_unit", "camera_convention", "group_key",
    "dataset_root", "source_files", "extra",
]

SEQ_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sequences")


def write_sequence(seq: dict, dataset_dir_name: str, filename: str):
    """写 sequence.json;已存在则报错退出——禁止覆盖,需先手动删除。"""
    assert all(k in seq for k in SEQUENCE_FIELDS), \
        f"missing fields: {[k for k in SEQUENCE_FIELDS if k not in seq]}"
    out_dir = os.path.join(SEQ_ROOT, dataset_dir_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path):
        raise FileExistsError(f"{out_path} 已存在,禁止覆盖。如需重建请先删除。")
    with open(out_path, "w") as f:
        json.dump(seq, f, indent=2, ensure_ascii=False)
    print(f"  wrote {out_path}")
    return out_path


def natural_key(s: str):
    """自然排序键:'cam_2' < 'cam_10'。"""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# ---------------- COLMAP text 解析 ----------------

def parse_cameras_txt(path):
    """返回 {camera_id: {"model","width","height","params":[...]}}。"""
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cid, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
            params = [float(x) for x in parts[4:]]
            cams[cid] = {"model": model, "width": w, "height": h, "params": params}
    return cams


def quat_qvec2rotmat(qvec):
    """COLMAP qvec (w,x,y,z) -> 3x3 R."""
    import numpy as np
    w, x, y, z = qvec
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
    ])


def parse_images_txt(path):
    """解析 COLMAP images.txt。

    返回有序列表 [{"image_id","qvec","tvec","camera_id","name"}],按 name 自然排序。
    """
    entries = []
    lines = [l.strip() for l in open(path)]
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line or line.startswith("#"):
            continue
        p = line.split()
        # 图像行:IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME;下一行是 2D 点(跳过)
        entries.append({
            "image_id": int(p[0]),
            "qvec": [float(x) for x in p[1:5]],
            "tvec": [float(x) for x in p[5:8]],
            "camera_id": int(p[8]),
            "name": p[9],
        })
        i += 1  # skip 2D points line
    entries.sort(key=lambda e: natural_key(e["name"]))
    return entries


# ---------------- 坐标约定转换 ----------------

def ns_c2w_to_opencv_w2c(c2w):
    """nerfstudio/OpenGL c2w (camera-to-world, +Y up, -Z forward)
    -> OpenCV w2c (+Y down, +Z forward)。返回 4x4 numpy。
    """
    import numpy as np
    c2w = np.asarray(c2w, dtype=np.float64)
    # OpenGL->OpenCV: flip Y and Z axes of the camera frame
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    cv_c2w = c2w @ flip
    cv_w2c = np.linalg.inv(cv_c2w)
    return cv_w2c


def rot_t_to_w2c(R, t):
    """R (3x3), t (3,) -> 4x4 w2c。"""
    import numpy as np
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ---------------- 输出相机文件 ----------------

def dump_intrinsics_json(out_dir, K_dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "intrinsics.json")
    if os.path.exists(path):
        raise FileExistsError(f"{path} 已存在,禁止覆盖")
    with open(path, "w") as f:
        json.dump(K_dict, f, indent=2)
    return path


def dump_extrinsics_json(out_dir, extrinsics_list):
    """extrinsics_list: [{"camera_id","image_name","w2c": 4x4 nested list}]"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "extrinsics.json")
    if os.path.exists(path):
        raise FileExistsError(f"{path} 已存在,禁止覆盖")
    with open(path, "w") as f:
        json.dump({"convention": "opencv_w2c",
                   "extrinsics": extrinsics_list}, f, indent=2)
    return path
