# 阶段2.0 Gate 报告 — 环境与模型版本冻结

日期:2026-08-25

## 目录结构

```
阶段2/
├── 00_environment/   ✅ software_versions.json / checkpoint_hashes.json / gpu_environment.json
│                     ✅ provenance.py(推理溯源标准,禁止覆盖,自动 _runN)
│                     ✅ check_determinism.py + determinism_check.json
├── 01_sequences/sequence_manifests/
├── 02_vggt/{dataset_id}/{sequence_id}/
├── 03_metric_depth/da3metric/  |  unidepth_v2/
├── 04_intrinsics/
├── 05_depth_alignment/
├── 06_msam/
├── 07_world_coordinate/
├── 08_geometry_evaluation/
├── 09_efficiency/
├── 10_failures/
└── phase2_gate_report.md
```

## 环境冻结(四个独立环境)

| 环境 | Python | PyTorch | CUDA runtime | 状态 |
|---|---|---|---|---|
| `vggt_lora` (=env_vggt) | 3.10.20 | 2.3.1+cu121 | 12.1 | ✅ |
| `da3` (=env_da3) | 3.10.20 | 2.3.1+cu121 | 12.1 | ✅ |
| `unidepth` (=env_unidepth) | 3.11.15 | 2.5.1+cu121 | 12.1 | ✅ |
| `env_geometry_eval` | — | — | — | ⏳ 阶段08 前创建 |

## 模型权重冻结(SHA256)

| 模型 | repo | sha256 |
|---|---|---|
| VGGT-1B | facebook/VGGT-1B | `f164acf6...e60467e` |
| DA3METRIC-LARGE | depth-anything/DA3METRIC-LARGE | `bbea5b0b...324776` |
| UniDepthV2 ViT-L | lpiccinelli/unidepth-v2-vitl14 | `ba73d3de...3e67c6` |

## Gate 通过条件核验

| 条件 | 结果 |
|---|---|
| 三个模型官方 demo 均可运行 | ✅ VGGT smoke(7 视图前向)/ DA3 smoke(5 图 metric depth)/ UniDepthV2 smoke(ViT-L 单目)全部通过 |
| 同一输入重复推理无明显随机差异 | ✅ determinism_check.json:VGGT(bf16 autocast)diff=0.0;DA3(fp32)diff=0.0;UniDepthV2(fp32)diff=0.0 |
| 权重和代码版本可复现 | ✅ 三仓库 commit 已记录;三权重 SHA256 已固化于 checkpoint_hashes.json |
| 无 CUDA/xFormers 版本错误 | ✅ 无报错。备注:DA3 未装 gsplat(仅影响 3DGS 渲染导出);UniDepth 可选 CUDA 算子未编译(仅影响评估速度),均不影响本阶段推理 |

## 推理溯源规范(强制)

所有后续推理结果必须经 `00_environment/provenance.py` 的
`make_provenance()` + `save_with_provenance()` 落盘:
- 结果文件已存在时自动 `_runN` 递增,**禁止覆盖**;
- 每条结果附 `*_provenance.json`,含规范要求的全部 8 项
  (模型名 / checkpoint 版本+SHA256 / 输入图像 SHA256 / 图像顺序 /
   resize-crop 参数 / 内参变换 / 精度模式 / 代码 commit)。

## 备注

- GPU:2× RTX A6000 (CC 8.6, 48GB),driver 590.48.01。
  **GPU 0 有其他进程占用(~15GB)**,批量推理建议 `CUDA_VISIBLE_DEVICES=1`。
- HF 下载需走镜像:`export HF_ENDPOINT=https://hf-mirror.com`(权重均已缓存,二次运行离线可用)。

**结论:阶段2.0 gate 通过。**
