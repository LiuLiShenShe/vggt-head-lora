"""TERRA-REF 适配器(catalog 级,无本地影像 → status=pending_download)。

规则:同一 (plot, date±window) 在 rgb_geotiff_plots 与 laser3d_las_plots
中都有记录才生成目录级 sequence;RGB 需 _left/_right stereo 成对。
绝不默认配对——窗口与重叠都显式记录在 extra 中。
"""
import datetime
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import write_sequence

CAT_ROOT = "/fj/VGGT+head+lora实验/阶段1-数据集/TERRA-REF/sensor_data_catalogs/sensors"
OUT_BASE = "/fj/VGGT+head+lora实验/阶段2/01_sequences/sequences/terraref"

PLOT_RE = re.compile(r"Range (\d+) Column (\d+)")

# 实际文件名:file_catalog_season6_rgb_geotiff_plots_2018-05-05.json
def load_catalog(sensor_dir, date):
    """sensor_dir 形如 season_6_catalog/rgb_geotiff_plots;
    文件名形如 file_catalog_season6_rgb_geotiff_plots_2018-05-05.json。"""
    base, sub = sensor_dir.split("/")
    tag = base.replace("_catalog", "").replace("_", "")  # season6 / season4
    p = os.path.join(CAT_ROOT, sensor_dir,
                     f"file_catalog_{tag}_{sub}_{date}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def extract_plot_entries(catalog):
    """返回 {plot_key: [file_entry,...]},plot_key = "rangeR_colC"。"""
    out = {}
    for c in catalog["collections"].values():
        for dk, ds in c.get("datasets", {}).items():
            m = PLOT_RE.search(dk)
            if not m:
                continue
            key = f"range{m.group(1)}_col{m.group(2)}"
            out.setdefault(key, []).extend(ds.get("files", []))
    return out


def list_dates(sensor_dir):
    dates = []
    for p in glob.glob(os.path.join(CAT_ROOT, sensor_dir, "*.json")):
        m = re.search(r"(\d{4}-\d{2}-\d{2})\.json$", os.path.basename(p))
        if m:
            dates.append(m.group(1))
    return sorted(dates)


def adapt_season(season: str, max_seqs=None):
    rgb_dir = f"{season}_catalog/rgb_geotiff_plots"
    las_dir = f"{season}_catalog/laser3d_las_plots"
    if not os.path.isdir(os.path.join(CAT_ROOT, las_dir)):
        print(f"{season}: no laser3d_las_plots catalog (empty dir), skip")
        return
    WINDOW = datetime.timedelta(days=1)   # 短时间窗:±1 天

    rgb_dates = {datetime.date.fromisoformat(d) for d in list_dates(rgb_dir)}
    las_dates = {datetime.date.fromisoformat(d) for d in list_dates(las_dir)}
    print(f"{season}: rgb dates={len(rgb_dates)}, laser dates={len(las_dates)}")

    n = 0
    for rd in sorted(rgb_dates):
        near_laser = sorted(d for d in las_dates if abs((d - rd).days) <= WINDOW.days)
        if not near_laser:
            continue
        rgb_cat = load_catalog(rgb_dir, rd.isoformat())
        rgb_plots = extract_plot_entries(rgb_cat)
        for ld in near_laser:
            las_cat = load_catalog(las_dir, ld.isoformat())
            las_plots = extract_plot_entries(las_cat)
            both = set(rgb_plots) & set(las_plots)
            for plot in sorted(both):
                rgb_files = rgb_plots[plot]
                has_left = any("_left" in e["name"] for e in rgb_files)
                has_right = any("_right" in e["name"] for e in rgb_files)
                if not (has_left and has_right):
                    continue

                seq_id = f"terraref__{season}__{plot}__rgb{rd}__las{ld}"
                seq = {
                    "dataset_id": "terra_ref",
                    "sequence_id": seq_id,
                    "status": "pending_download",
                    "rgb_paths": [],
                    "camera_ids": ["cam_left", "cam_right"],
                    "intrinsics_path": None,
                    "extrinsics_path": None,
                    "reference_pointcloud": None,
                    "linear_unit": "meter",
                    "camera_convention": "opencv_w2c",
                    "group_key": plot,
                    "dataset_root": CAT_ROOT,
                    "source_files": {
                        "rgb_catalog": os.path.join(CAT_ROOT, rgb_dir,
                            f"file_catalog_{'season' + season.split('_')[1]}_{season}_catalog_{rd}.json".replace("season_" + season.split("_")[1] + "_catalog", season + "_catalog")),
                        "laser_catalog": os.path.join(CAT_ROOT, las_dir,
                            f"file_catalog_{'season' + season.split('_')[1]}_{season}_catalog_{ld}.json".replace("season_" + season.split("_")[1] + "_catalog", season + "_catalog")),
                    },
                    "extra": {
                        "sort_key": "left then right (stereo pair)",
                        "rgb_date": rd.isoformat(),
                        "laser_date": ld.isoformat(),
                        "date_window_days": WINDOW.days,
                        "remote_paths": [
                            {"role": "rgb_left" if "_left" in e["name"] else
                                     ("rgb_right" if "_right" in e["name"] else "rgb"),
                             "path": e["path"], "checksum_md5": e.get("checksum"),
                             "size": e.get("size")}
                            for e in rgb_files],
                        "laser_remote_paths": [{"path": e["path"],
                                                "checksum_md5": e.get("checksum"),
                                                "size": e.get("size")}
                                               for e in las_plots[plot]],
                        "download_via": "Globus (UA-MAC endpoint); paths relative to TERRA-REF Globus root",
                        "n_rgb_candidates": len(rgb_files),
                    },
                }
                write_sequence(seq, "terraref", f"{seq_id}.json")
                n += 1
                if max_seqs and n >= max_seqs:
                    print(f"  reached max_seqs={max_seqs}, stop.")
                    return
    print(f"{season}: generated {n} pending sequences")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["season_6", "season_4"])
    ap.add_argument("--max-seqs", type=int, default=5,
                    help="每个 season 最多生成的序列数;传 0 表示不限")
    a = ap.parse_args()
    for s in a.seasons:
        print(f"== adapting {s}")
        adapt_season(s, a.max_seqs or None)
