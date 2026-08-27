"""Clean rerun P0-3:为 v2_clean_rerun 每个序列写 run_provenance.json(da3)。

读 prediction_meta.json(make_provenance 已含 input_image_sha256/image_order),
计算全部输出文件 sha256,合并 RUN_MANIFEST 的版本信息。

用法: python make_seq_provenance.py
"""
import glob
import hashlib
import json
import os

RERUN = "/fj/VGGT+head+lora实验/阶段2/02_vggt/v2_clean_rerun"
MANIFEST = os.path.join(RERUN, "RUN_MANIFEST.json")


def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    man = json.load(open(MANIFEST))
    meta_files = sorted(glob.glob(os.path.join(RERUN, "*", "*", "prediction_meta.json")))
    assert len(meta_files) == 17, f"期望 17 个序列,实得 {len(meta_files)}"
    for mp in meta_files:
        d = os.path.dirname(mp)
        meta = json.load(open(mp))
        out_files = {}
        for p in sorted(glob.glob(os.path.join(d, "*.npy"))):
            out_files[os.path.basename(p)] = {"sha256": sha256_file(p),
                                              "bytes": os.path.getsize(p)}
        prov = {
            "run_id": man["run_id"],
            "project_git_commit": man["project_git_commit"],
            "vggt_commit": man["vggt_code"]["note"],
            "checkpoint": man["checkpoint"]["repo_id"],
            "checkpoint_sha256": man["checkpoint"]["checkpoint_sha256"],
            "input_images": meta["image_order"],
            "input_image_sha256": dict(zip(meta["image_order"],
                                           meta["input_image_sha256"])),
            "frame_order": list(range(len(meta["image_order"]))),
            "num_frames": len(meta["image_order"]),
            "dtype": man["inference_config"]["dtype"],
            "device": man["inference_config"]["device"],
            "inference_config": man["inference_config"],
            "output_files": {k: v["bytes"] for k, v in out_files.items()},
            "output_sha256": {k: v["sha256"] for k, v in out_files.items()},
        }
        out = os.path.join(d, "run_provenance.json")
        with open(out, "w") as f:
            json.dump(prov, f, indent=2, ensure_ascii=False)
        print(f"{os.path.basename(d)}: {len(out_files)} files hashed -> {out}")
    print("DONE")


if __name__ == "__main__":
    main()
