"""阶段2 推理结果溯源元数据工具。

所有模型推理输出(npz/npy/json)保存时,必须附带本模块生成的 provenance dict,
写入同名 *_provenance.json。禁止反复覆盖:文件已存在时 save_with_provenance
会自动加 _runN 后缀,绝不静默覆盖。

每条记录包含(规范要求 8 项):
  1. model_name          模型名称
  2. checkpoint_version  checkpoint 版本(repo id + sha256)
  3. input_image_sha256  输入图像 SHA256(按顺序列表)
  4. image_order         图像顺序(文件名列表,与 sha 一一对应)
  5. resize_crop_params  resize/crop 参数
  6. intrinsics_transform 相机内参变换说明
  7. precision_mode      推理精度模式(fp32/fp16/bf16)
  8. code_commit_hash    代码提交哈希
"""
import hashlib
import json
import os


# ---- 冻结的版本常量(与 00_environment/*.json 一致) ----
FROZEN = {
    "vggt": {
        "model_name": "facebook/VGGT-1B",
        "checkpoint_sha256": "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e",
        "code_commit": "a288dd0f14786c93483e45524328726ab7b1b4ce",
        "env": "/home/test/miniconda3/envs/vggt_lora",
    },
    "da3metric": {
        "model_name": "depth-anything/DA3METRIC-LARGE",
        "checkpoint_sha256": "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776",
        "code_commit": "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4",
        "env": "/home/test/miniconda3/envs/da3",
    },
    "unidepth_v2": {
        "model_name": "lpiccinelli/unidepth-v2-vitl14",
        "checkpoint_sha256": "ba73d3de735302ccc64a50f1e557122050c4b1893e6060b28dba05d6af3e67c6",
        "code_commit": "cc938d920e5defa388f400a753a1614ca98733cb",
        "env": "/home/test/miniconda3/envs/unidepth",
    },
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_provenance(model_key: str, image_paths: list, resize_crop_params: str,
                    intrinsics_transform: str, precision_mode: str,
                    extra: dict | None = None) -> dict:
    m = FROZEN[model_key]
    prov = {
        "model_name": m["model_name"],
        "checkpoint_version": {"repo_id": m["model_name"], "sha256": m["checkpoint_sha256"]},
        "input_image_sha256": [sha256_file(p) for p in image_paths],
        "image_order": [os.path.basename(p) for p in image_paths],
        "resize_crop_params": resize_crop_params,
        "intrinsics_transform": intrinsics_transform,
        "precision_mode": precision_mode,
        "code_commit_hash": m["code_commit"],
        "extra": extra or {},
    }
    return prov


def _next_free_path(path_no_ext: str, ext: str) -> str:
    """path_run0.ext, path_run1.ext ... 找到第一个不存在的。"""
    i = 0
    while os.path.exists(f"{path_no_ext}_run{i}{ext}"):
        i += 1
    return f"{path_no_ext}_run{i}{ext}"


def save_with_provenance(out_dir: str, stem: str, arrays: dict, provenance: dict,
                         ext: str = ".npz") -> str:
    """保存推理结果 + 溯源 json。已存在同名文件时不覆盖,自动 _runN 递增。

    返回实际写出的 npz 路径;provenance 写在 <stem>_runN_provenance.json。
    """
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, stem)
    if ext == ".npz":
        import numpy as np
        if not os.path.exists(base + ext):
            out_path = base + ext
            np.savez_compressed(out_path, **arrays)
        else:
            out_path = _next_free_path(base, ext)
            np.savez_compressed(out_path, **arrays)
    else:
        raise ValueError(f"unsupported ext {ext}")
    prov_path = out_path.replace(ext, "_provenance.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False)
    return out_path
