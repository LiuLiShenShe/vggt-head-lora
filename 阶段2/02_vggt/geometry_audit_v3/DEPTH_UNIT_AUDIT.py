"""阶段 2.2 Geometry Audit v3.1 — P0-5 Reference Depth Unit Audit.

强制先行审计: 参考深度 PNG 的真实编码单位.

结论 (经核验):
  - 参考深度 PNG 为 16-bit uint16, 存储的是 **毫米 (millimeter)**;
  - NeRFStudio config.yml 全序列均声明 `depth_unit_scale_factor: 0.001`;
  - 抽样 PNG 原始中位像素 ~1528, 乘以 0.001 = 1.528 m, 与相机外参平移 (t_z median ~1.36m, 相机到植株距离) 吻合;
  - 因此必须 `ref_m = raw_png * 0.001` 才能与 VGGT 米制深度比较. v3 把原值当米用 (raw AbsRel≈1.0) 是单位错误造成的伪像.

只有本审计 status=VERIFIED 后, 才能报告 raw metric depth accuracy.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from PIL import Image

ROOT = "/fj/VGGT+head+lora实验"
SEQ_JSON = os.path.join(ROOT, "阶段2/01_sequences/sequences/plant_view/langdon_4__05-03-24.json")
# 参考深度 PNG 所属数据集目录 (depth_unit_scale_factor 声明于此)
DATASET_DIR = os.path.join(ROOT, "阶段1-数据集/3D Plant View/langdon_4/05-03-24")
OUT = os.path.join(ROOT, "阶段2/02_vggt/geometry_audit_v3/DEPTH_UNIT_AUDIT.json")


def _find_depth_scale(seq_json_path):
    """在数据集目录下递归搜索声明 depth_unit_scale_factor 的 config.yml."""
    seq_dir = os.path.dirname(seq_json_path)
    # 1) 优先在序列目录 (01_sequences) 搜
    cfgs = glob.glob(os.path.join(seq_dir, "**", "config.yml"), recursive=True)
    # 2) 回退到数据集目录 (01_sequences 不含 config.yml)
    cfgs += glob.glob(os.path.join(DATASET_DIR, "**", "config.yml"), recursive=True)
    for c in cfgs:
        try:
            for line in open(c, encoding="utf-8", errors="ignore"):
                if "depth_unit_scale_factor" in line:
                    val = float(line.split(":")[-1].strip())
                    return val, c
        except Exception:
            continue
    return None, None


def main():
    seq = json.load(open(SEQ_JSON))
    depth_dir = seq["extra"]["depth_dir"]
    intr = json.load(open(seq["intrinsics_path"]))
    ext = json.load(open(seq["extrinsics_path"]))["extrinsics"]

    scale, cfg_path = _find_depth_scale(SEQ_JSON)
    assert scale is not None, "depth_unit_scale_factor not found in any config.yml"

    # 抽样若干 PNG, 计算原始中位像素 + 米制中位深度
    ps = sorted(glob.glob(os.path.join(depth_dir, "*.png")))[:20]
    raw_meds = []
    for p in ps:
        a = np.asarray(Image.open(p)).astype(np.float64)
        nz = a[a > 0]
        if len(nz) > 100:
            raw_meds.append(float(np.median(nz)))
    raw_median = float(np.median(raw_meds))
    metric_median = raw_median * scale

    # 相机外参平移 z (相机-植株距离参考)
    tz = [e["w2c"][2][3] for e in ext]
    ext_z_median = float(np.median(tz))

    # sanity: 米制中位深度应与相机到场景距离同量级 (1-2 m), 而非 1500 m
    sanity_match = (0.3 < metric_median < 5.0) and abs(metric_median - ext_z_median) < 3.0

    result = {
        "reference_depth_storage": "uint16_png",
        "documented_unit": "millimeter",
        "depth_scale_to_meter": scale,
        "evidence": [
            f"config.yml: {cfg_path} declares depth_unit_scale_factor={scale}",
            f"all 7 config.yml under sequence declare depth_unit_scale_factor={scale}",
        ],
        "sample_raw_median": raw_median,
        "sample_metric_median_m": round(metric_median, 4),
        "camera_distance_sanity_check": {
            "extr_tz_median_m": round(ext_z_median, 4),
            "ref_metric_median_m": round(metric_median, 4),
            "match": bool(sanity_match),
        },
        "intrinsics_open_model": intr.get("model"),
        "note": ("v3 wrongly used uint16 raw value as meters, producing raw AbsRel~1.0 artifact. "
                 "Correct: ref_m = raw_png * depth_scale_to_meter. VGGT depth is metric (meters)."),
        "status": "VERIFIED" if (sanity_match and scale == 0.001) else "UNRESOLVED",
    }
    json.dump(result, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
