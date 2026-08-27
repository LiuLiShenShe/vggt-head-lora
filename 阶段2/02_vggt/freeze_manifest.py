"""Clean rerun P0-1:冻结本次运行版本,写 v2_clean_rerun/RUN_MANIFEST.json。

在 vggt_lora 环境运行(需要 torch 报告 CUDA/cuDNN 信息):
  python freeze_manifest.py
"""
import datetime
import json
import os
import subprocess

BASE = "/fj/VGGT+head+lora实验/阶段2/02_vggt"
OUT = os.path.join(BASE, "v2_clean_rerun", "RUN_MANIFEST.json")
REPO = "/fj/VGGT+head+lora实验"


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main():
    assert not os.path.exists(OUT), f"{OUT} 已存在"
    import torch
    git_commit = sh(f"git -C {REPO} rev-parse HEAD")
    dirty = [l for l in sh(f"git -C {REPO} status --short").splitlines()
             if "freeze_manifest.py" not in l]   # 本脚本自身未提交属预期
    assert git_commit == "208c2b194a5fddd9c9ff880f6b56c419fbc0671b", \
        f"commit 与计划冻结值不符: {git_commit}"
    assert not dirty, f"工作树不干净:\n{dirty}"
    ckpt_blob = ("f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e")
    run_id = "clean_rerun_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "run_id": run_id,
        "run_type": "clean_reproducibility_rerun",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_git_commit": git_commit,
        "vggt_code": {
            "path": "/fj/VGGT+head+lora实验/vggt",
            "note": "与主仓库同一 git 仓库的子目录(无独立 .git/commit),版本由 project_git_commit 锁定",
        },
        "checkpoint": {
            "repo_id": "facebook/VGGT-1B",
            "hf_snapshot": "860abec7937da0a4c03c41d3c269c366e82abdf9",
            "weights_file": "model.safetensors",
            "checkpoint_sha256": ckpt_blob,
            "sha256_note": "HF blob 文件名即内容 sha256(Hub 惯例),未重算大文件",
            "bytes": 5026367224,
        },
        "environment": {
            "conda_env": "vggt_lora",
            "python": "3.10.20",
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "cuda_version_torch": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_driver": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
            "gpu_name": sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
        },
        "inference_config": {
            "device": "cuda (CUDA_VISIBLE_DEVICES 由启动命令决定)",
            "dtype": "bfloat16 autocast(torch.cuda.amp.autocast); tokens 落盘 fp16",
            "seed_per_sequence": 42,
            "preprocessing": "load_and_preprocess_images(mode='crop'), 官方默认 518 宽居中裁剪",
            "token_layers": [4, 11, 17, 23],
            "save_token_max_S": 200,
            "determinism_flags": "未显式设置 cudnn.deterministic/benchmark/use_deterministic_algorithms(保持与原运行一致;BF16+CUDA 非确定性 → 不要求逐 bit 一致)",
        },
        "sequences_source": "阶段2/01_sequences/sequences/{wheat3dgs,mustc,plant_view}/*.json (status=ready 共 17 个)",
        "output_root": BASE + "/v2_clean_rerun",
        "no_reuse_of_previous_npys": True,
        "overwrite_policy": "FileExistsError 保护;旧目录(v2 主结果/checks_v2/four_path*)一律不动",
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"-> {OUT}")
    print("VGGT_RUN_ID=" + run_id)


if __name__ == "__main__":
    main()
