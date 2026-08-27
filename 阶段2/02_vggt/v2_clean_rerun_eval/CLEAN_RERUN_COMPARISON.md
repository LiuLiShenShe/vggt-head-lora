# 阶段2.2 Clean Reproducibility Rerun — 对账报告 (P0-6)

**run_id**: `clean_rerun_20260826T065510Z`
**生成时间**: 2026-08-27
**冻结 commit**: `208c2b194a5fddd9c9ff880f6b56c419fbc0671b`
**复现判定**: **PASS**

---

## 1. 是否复用旧 NPY?

**完全没有。** 证据链：

- 新推理输出目录 `02_vggt/v2_clean_rerun/<dataset>/<sid>/` 为本次从原始 RGB 全新生成,文件 mtime 比旧 `02_vggt/<dataset>/<sid>/` 晚约 24 小时。
- 新 meta 含 `run_id = "clean_rerun_20260826T065510Z"`;旧 meta 无 `run_id` 字段(早于该字段支持),二者可明确区分。
- 推理脚本 `run_vggt_inference.py` 有 `FileExistsError` 覆盖保护,旧目录未被读取或改动。

---

## 2. 逐序列 pose_eval_v2 对账

| Sequence | old rot median | clean rot median | old P90 | clean P90 | Δ median | old gate | clean gate | status |
|---|---:|---:|---:|---:|---:|---|---|---|
| mustc__plot198__230613__ugv__pos00 | 1.84 | 1.84 | 3.39 | 3.39 | 0.000 | PASS | PASS | 一致 |
| mustc__plot198__230613__ugv__pos01 | 1.38 | 1.38 | 3.90 | 3.90 | 0.000 | PASS | PASS | 一致 |
| mustc__plot198__230613__ugv__pos02 | 1.90 | 1.90 | 4.62 | 4.62 | 0.000 | PASS | PASS | 一致 |
| mustc__plot198__230613__ugv__pos03 | 1.41 | 1.41 | 1.87 | 1.87 | 0.000 | PASS | PASS | 一致 |
| plantview__langdon_4__05-03-24 | 1.71 | 1.71 | 4.10 | 4.10 | 0.000 | PASS | PASS | 一致 |
| plantview__langdon_4__12-03-24 | 84.19 | 84.19 | 158.48 | 158.48 | 0.000 | FAIL | FAIL | 一致 |
| plantview__langdon_4__13-02-24 | 2.22 | 2.22 | 4.69 | 4.69 | 0.000 | PASS | PASS | 一致 |
| plantview__langdon_4__15-04-24 | 89.82 | 89.82 | 163.31 | 163.31 | 0.000 | FAIL | FAIL | 一致 |
| plantview__langdon_4__19-03-24 | 78.95 | 78.95 | 159.53 | 159.53 | 0.000 | FAIL | FAIL | 一致 |
| plantview__langdon_4__20-02-24 | 3.59 | 3.59 | 7.51 | 7.51 | 0.000 | PASS | PASS | 一致 |
| wheat3dgs__plot_461 | 3.60 | 3.60 | 5.43 | 5.43 | 0.000 | PASS | PASS | 一致 |
| wheat3dgs__plot_462 | 3.66 | 3.66 | 5.66 | 5.66 | 0.000 | PASS | PASS | 一致 |
| wheat3dgs__plot_463 | 3.45 | 3.45 | 5.64 | 5.64 | 0.000 | PASS | PASS | 一致 |
| wheat3dgs__plot_464 | 3.21 | 3.21 | 5.03 | 5.03 | 0.000 | PASS | PASS | 一致 |
| wheat3dgs__plot_465 | 2.95 | 2.95 | 4.58 | 4.58 | 0.000 | PASS | PASS | 一致 |
| wheat3dgs__plot_466 | 3.23 | 3.23 | 4.83 | 4.83 | 0.000 | PASS | PASS | 一致 |
| wheat3dgs__plot_467 | 3.33 | 3.33 | 4.62 | 4.62 | 0.000 | PASS | PASS | 一致 |

**gate 判定一致 17/17**,失败序列仍为 12-03-24 / 15-04-24 / 19-03-24。

---

## 3. 推理输出 NPY 数值对账 (P0-6)

- 对比对象:旧 `02_vggt/<ds>/<sid>/*.npy` (9 个核心 + 短序列 4 个 token) vs 新 `v2_clean_rerun/<ds>/<sid>/*.npy`
- 指标(以 float64 计算): `max_abs_diff`、`mean_abs_diff`、`relative_diff=‖new−old‖/‖old‖`、`allclose(atol=1e-3, rtol=1e-2)`
- **结果:全部 17 序列、全部 NPY 的 `max_abs_diff = 0.0`,`allclose = True`,无 shape mismatch**

逐位一致说明:在固定 checkpoint (`facebook/VGGT-1B`, blob `f164…67e`)、固定输入(帧序/图像 hash 与原运行一致)、固定预处理 (`mode="crop"`, 518)、固定 dtype (bfloat16 autocast)、固定 per-sequence seed (42),且 **`cudnn.benchmark=False`(默认)** 的条件下,VGGT BF16 前向在此 A6000/CUDA 12.1 环境是**确定性**的。计划预判的"BF16 可能非逐位一致"在本环境未发生。

> 注:原 v2 运行与本次 clean rerun 的环境(`vggt_lora`, torch 2.3.1+cu121)与 GPU 型号(A6000)相同,这是逐位复现的前提。若后续换卡/换 CUDA 版本,应预期 BF16 级数值漂移(本对账脚本的容差 atol=1e-3/rtol=1e-2 已为这种情况预留)。

**数值漂移是否影响 Gate**:否。所有序列 drift=0,Gate 无任何翻转风险。

---

## 4. 四路判别复现 (P0-5)

成功序列 `05-03-24`:A/B/C 三路均正常(A 0.029–0.038, B 0.031–0.036, C 0.020–0.037)。
失败序列 `12-03-24`:**B 正常 (0.033–0.040),A 与 C 失败(None,全截断)**。
视角数 8→36 无改善,排除"320 帧冗余视角稀释"。
→ 与旧 v2 结论一致:**VGGT 相机头失败、深度头输出正确**。

---

## 5. 结论

| 复现项 | 旧 v2 | clean rerun | 结论 |
|---|---|---|---|
| 前向成功率 | 17/17 | 17/17 | ✅ |
| 有效深度比例 | 1.0 | 1.0 | ✅ |
| NaN/Inf | 0 | 0 | ✅ |
| gate 通过 | 14/17 | 14/17 | ✅ 一致 |
| 失败序列 | 12/15/19-03-24 | 12/15/19-03-24 | ✅ 一致 |
| 成功序列 rot median | 低个位数度 | 低个位数度 | ✅ |
| 失败序列 rot median | 79–90° | 79–90° | ✅ |
| four-path 定性 | 相机头失败 | 相机头失败 | ✅ |

**reproducibility_closure = PASS**
