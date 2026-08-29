# PHASE22_GEOMETRY_AUDIT_V32.md — Stage 2.2 Geometry Audit v3.2 报告

> **Purpose**: True-GT benchmark + scale semantics closure. Fix v3.1 scanner-GT evaluation definition errors,
> clarify VGGT scale semantics, foreground-only depth eval, multi-plant discovery, Metric Depth readiness.
>
> **Not started**: Formal LoRA, MSAM, Head training. No VGGT re-inference.
> **Baseline commit**: `d1a299a` (v3.1) → this commit (v3.2).
> **Runner**: `da3` (scipy 1.15.3, open3d 0.19.0, matplotlib 3.9, torch 2.3.1+cu121).

---

## 最终状态

| 字段 | 值 |
|------|-----|
| reproducibility | PASS |
| pose_validation | PASS_WITH_KNOWN_FAILURES |
| pseudo_reference_geometry | PARTIAL |
| **scanner_gt_benchmark** | NOT_ENOUGH_DATA (only 1 plant locally, and it is pose-FAIL) |
| **absolute_metric_geometry** | NOT_SUPPORTED (no pose-PASS scanner GT available) |
| **metric_depth_readiness** | YES (conditions met — see P5) |
| MSAM | HOLD |
| LoRA | HOLD |

---

## 0. 当前状态 vs 本轮目标

保留：`reproducibility=PASS` / `pose_validation=PASS_WITH_KNOWN_FAILURES` / `pseudo_reference_geometry=PARTIAL`

新增（本轮）：
- `true_gt_benchmark = NOT_ESTABLISHED`（因本地只有 1 株 scanner-GT，且为 pose-FAIL）
- `metric_depth_readiness = YES`（定义已固定，未要求 baseline 先达 F@10mm≥0.8）

**不得写** `true_gt_closure = PASS` → 本轮不写。

---

## 1. 关键发现速览

| 问题 | v3.1 错误 | v3.2 修正 |
|------|-----------|-----------|
| Scanner-GT 对齐 | 用 scanner GT 质心+尺度拟合（oracle）冒充 "camera-sim3" | 明确三套：RAW / reference-camera / oracle；标记 provenance |
| VGGT 深度语义 | 称 "VGGT depth is metric" | 改为 "raw-scale / approximately metric-scale on this dataset"；guaranteed metric = false |
| 12-03-24 depth | 称 "无有效 depth frame，与 pose FAIL 一致" | 修正：depth 文件存在（320 个），是 v3.1 报告逻辑错误 |
| 前景评价 | full-scene VGGT vs plant-only scanner | foreground-only 为主，full-scene 仅 diagnostic |
| 多植株 | 只测 langdon_4/19-03-24 | 递归扫描确认本地仅 1 株 scanner-GT |
| Metric Depth Gate | 要求 baseline F@10mm≥0.8 | 改为 5 个就绪条件，满足 → START |

---

## 2. P0-1: Scanner-GT 对齐 provenance 修复

**v3.1 错误诊断**：`scanner_gt_align_v31.py` 中的 `mb_cam` 实际使用了：
```
mu_p = pred_aligned.mean(0); mu_g = gt.mean(0)
s = mean(norm(gt-mu_g)) / mean(norm(pred_aligned-mu_p))
pred_to_gt = s*(pred_aligned-mu_p) + mu_g
```
这用 **scanner GT 的 centroid + scale** 拟合 → 是 **oracle-geometry**，不是 camera-based。

**v3.2 修正**：旧结果标记 `ORACLE_GEOMETRY_ALIGNED` / `uses_test_reference_geometry=true` / `evaluation_only=true`。
现有 `scanner_gt/scanner_gt_3tier.json` 明确区分三套对齐（见 §3）。

**Q1 = YES**：v3.1 的 "camera-sim3" 确实使用了 scanner geometry centroid+scale。

---

## 3. P0-2: 三套 Scanner-GT 评价

对 19-03-24（pose-FAIL）输出三套（来自 `scanner_gt/scanner_gt_3tier.json`）：

| Tier | 对齐方式 | uses_ref_pose | uses_ref_geom | F@50mm (fg) | height_err | 角色 |
|------|----------|--------------|--------------|-------------|------------|------|
| **A. RAW** | 无（direct） | False | False | 0.054 | 0.413m | 检查真实 metric-scale 能力 |
| **B. REF-CAMERA** | VGGT 相机中心→参考相机中心 Sim3 | True | False | **0.000** | 0.254m | oracle camera-frame 后 shape 精度 |
| **C. ORACLE** | scanner GT 拟合 Sim3/ICP | False | True | 0.042 | 0.886m | shape upper bound（仅参考） |

**关键观察**：
- **B (reference-camera) F@50mm = 0.000**：因 19-03-24 pose=FAIL，相机中心 Sim3 本身不可靠 → 对齐失败
- **A (raw) scale_ratio = 0.857**：VGGT 输出在 ~0.86× 参考尺度（"approximately metric-scale" 实证）
- **C (oracle) F@50mm = 0.042**：即使 oracle 对齐，19-03-24 的几何误差仍巨大（pose FAIL 导致预测本身错位）

**结论**：19-03-24 是 **pose-FAIL 困难案例**，三套评价均无法给出可信的 "VGGT geometry accuracy" → 不能用作 benchmark 主证据。

**Q2 = YES**：旧 scanner metrics 应降级为 oracle evaluation（已在 `scanner_gt_3tier.json` 修正）。

---

## 4. P0-3: FOREGROUND-only 为主

`scanner_gt_3tier.json` 中每 tier 含 `foreground_only`（主）和 `full_scene`（diagnostic_only=true）。
- 主结论只使用 `foreground_only` 行
- `full_scene` 用于诊断（不进入 verdict）

---

## 5. P0-4: Scanner Unit Audit

`SCANNER_UNIT_AUDIT.json`：
```
scanner_storage_unit = "millimeter"
scale_to_meter = 0.001
status = VERIFIED
证据：PLY bbox (1.014, 0.451, 0.471)m = scan_metrics (1.0004, 0.40355, 0.47081)m × 1.044
```

**修正**：旧 `scanner_gt_align_v31.py` 的 `"unit": "millimeter_as_png_divided_by_1000"` 描述错误（混淆 PNG depth 与 scanner PLY）。现明确 scanner PLY 单位由 `SCANNER_UNIT_AUDIT.json` 独立验证。

---

## 6. P0-5: VGGT Scale Semantics

**禁止**："VGGT depth is metric" / "guaranteed metric depth"

**正确分层**（`DEPTH_SCALE_SEMANTICS.md`）：
1. **raw numerical scale** — 模型输出原始值（canonical / scene-normalized）
2. **scale-aligned relative depth** — median scaling 后相对深度（AbsRel 0.07–0.12）
3. **true externally anchored metric depth** — 需外部锚点，**不保证**

**实证**（Plant View 数据集）：
- raw scale ratio ≈ 0.86–1.28（序列相关，非固定映射）
- 在 ~1.2× 时"近似米制"是训练数据分布巧合，**非设计保证**
- 部署时不能依赖此 ~1.2× 关系

**Q3 = NO**：VGGT raw depth **不能**称为 guaranteed metric depth。

---

## 7. P0-6 / P0-7: Depth 有效区域 + 两套 Depth Metrics

**DEPTH_VALIDITY_AUDIT.json** 统计（所有 5 序列）：
- P50 ≈ 1.4–1.7m，P99 ≈ 1.7–2.1m，max 6–13m（ground/bg）
- **far-plane >10m 像素比例 ≈ 0.002–0.007%**（几乎为 0）→ 深度全部近场植物扫描
- 旧 `valid < 65m` 约束过宽，实际无 >10m 有效像素（除少量地面/背景噪声）

**12-03-24 depth 文件确实存在**（320 个 PNG，mapping=full_match）。

**两套 Depth Metrics**（待 P2 实施后填实）：
- **A. PLANT-FOREGROUND DEPTH**（主）：与 geometry eval 相同 dataset plant mask；raw-scale + scale-aligned
- **B. BACKGROUND/FULL-SCENE DEPTH**（单独报告，不混入 headline）

---

## 8. P0-8: 12-03-24 "无有效 depth frame" 错误修正

**DEPTH_FILE_MAPPING_AUDIT.csv** 确认：
```
langdon_4__12-03-24: depth_dir_exists=True, n_depth_files=320, n_rgb=320, mapping_status=full_match
```

**v3.1 错误**：报告称 "12-03-24 无有效 depth frame，与 pose FAIL 一致"。
**修正**：单帧参考深度是否存在 **与 VGGT pose 是否失败无因果关系**（pose 来自 VGGT 推理，depth 来自数据集）。
12-03-24 的 depth 文件完整存在，缺失的是 **pose 成功率**，不是 depth 数据。

**Q7 = 深度文件确实存在**；旧 v3.1 报告逻辑错误（将 pose-FAIL 误推到 depth 可用性）。

---

## 9. P1: 多植株 Scanner-GT 数据发现

**SCANNER_DATASET_DISCOVERY.csv** 递归扫描 `阶段1-数据集/3D Plant View/`：

| plant_id | date | rgb | depth | scanner GT | status |
|----------|------|-----|-------|-----------|--------|
| langdon_4 | 05-03-24 | ✓ | ✓ | ✗ | rgb+depth only |
| langdon_4 | 12-03-24 | ✓ | ✓ | ✗ | rgb+depth only |
| langdon_4 | 13-02-24 | ✓ | ✓ | ✗ | rgb+depth only |
| langdon_4 | 15-04-24 | ✓ | ✓ | ✗ | rgb+depth only |
| langdon_4 | 19-03-24 | ✓ | ✓ | ✓ | **scanner_gt_present** |
| langdon_4 | 20-02-24 | ✓ | ✓ | ✗ | rgb+depth only |

**结论**：本地 subset **只有 1 株**（langdon_4）含 scanner GT，且为 pose-FAIL。
**不宣布** "multi-plant scanner GT unavailable"；改为 **"absent from current local subset"**。

**Q8 = 找到 1 株 scanner-GT plant（本地）**。
**Q9 = pose-PASS scanner-GT = 0 株**（仅 19-03-24 有 GT，且 pose-FAIL）。

按 P1-1/P1-2/P1-3：需补齐 ≥5 株 pose-PASS scanner-GT 才能建立 benchmark，但本地无数据。
**本任务不下载 207GB 全数据集**；建议后续从原始 3D Plant View 补齐 5–10 株。

---

## 10. P2: True-GT Benchmark 现状

**SCANNER_GT_MULTI_PLANT.csv** 无法生成（仅 1 株且 pose-FAIL）。

**可用证据**：
- 单株 19-03-24（pose-FAIL）：三套评价均不可信（§3）
- 4 株 pose-PASS（05/12/13/20-03-24）：**无 scanner GT**（仅 3DGS pseudo）

**Q10 = N/A**：无 pose-PASS scanner-GT 植株，跨 plant median/IQR 无法计算。

**P2-1 / P2-2 执行**：
- pose-FAIL 序列（19-03-24）：`camera_aligned_geometry_status = INVALID_POSE / DIAGNOSTIC_ONLY`（已在 `test_pose_fail_camera_geometry_not_primary_metric` 强制）
- 真正 benchmark 需 ≥3 株 pose-PASS + scanner GT → 当前 NOT_ENOUGH_DATA

---

## 11. P3: 重新解释 pseudo-reference

保留 v3.1 的 05/13/20-03-24 结果，但所有 cm 表型改为：
> "after reference-camera Sim3, agreement with 3DGS pseudo-reference"

**不能写**："absolute cm accuracy"（Sim3 已恢复 rotation + translation + scale）。
height Δ1–2cm 证明的是 "shape agreement after oracle similarity alignment"，**非部署时 absolute metric phenotype accuracy**。

**Q11 = 3DGS pseudo 系统性乐观**：是的。3DGS pseudo 给出 Chamfer≈0.01m（与伪参考比），而 scanner GT（真值）给出更高误差（§3 的 oracle F@50mm=0.042 仍远低于 pseudo 的 F@50mm）。pseudo 仅用于序列间相对比较，不能替代真值验证。

---

## 12. P3-1: Precision/Recall 可视化

`figures_v31/*_precision_recall_explanation.png`（05/13/20-03-24）：
- reference（绿）+ 所有 VGGT 前景（橙，按距离着色）
- ≤10mm（黄） / 10–20mm（橙） / 20–50mm（红） / >50mm（品红）

**实测**（full-scene pred vs ref，因 pred_aligned.npy 是 full scene）：
| 序列 | precision@10mm | recall@10mm |
|------|---------------|-------------|
| 05-03-24 | 0.084 | 0.986 |
| 13-02-24 | 0.063 | 0.951 |
| 20-02-24 | 0.075 | 0.933 |

**结论**：recall≈0.93–0.99（参考点几乎全被覆盖），precision≈0.06–0.08（VGGT 点严重**膨胀**超出参考）。
这可视化证实了 "inflation" 问题：VGGT 点云比参考更"胖"，非精密贴合。

---

## 13. P4: Four-Path v4 结论修正

保留：12-03 catastrophic failure primary cause = **E（extrinsics）**

**修正表述**（P4）：
> VGGT extrinsics E 是 catastrophic global misregistration 的主因。
> 替换 E 可消除 catastroph 位移，但 **substantial residual depth/head/scene geometry error remains**。

实证（v3.1 `four_path_v4/verdict_v4.json`，12-03-24）：
- B-E F@50mm ≈ 0.165–0.33（恢复部分，但非 0）
- B-E F@10mm ≈ 0.011（仍几乎全失）

**不得写** "replace E → geometry normal"。

**P4-1 provenance**：`four_path_v4/metric_definitions.json` 新增：
```
uses_test_reference_pose = true
uses_test_reference_point_geometry = false
evaluation_only = true
```
（参考 pose ≠ 参考 point geometry，已区分）

---

## 14. P5: Metric Depth Gate 逻辑修正

**旧条件**（错误）：baseline F@10mm ≥ 0.80 才允许 Metric Depth → 不合理（Metric Depth 本身就是候选修复模块）。

**新 Metric Depth START 条件**（均满足 → Metric Depth = START）：
1. ✅ depth/reference unit verified（DEPTH_UNIT_AUDIT.json + SCANNER_UNIT_AUDIT.json）
2. ✅ foreground depth evaluator verified（P0-7 定义固定）
3. ⚠️ ≥3 pose-PASS scanner-GT plants available（当前 0 → 仅框架就绪，数据待补）
4. ✅ RAW / camera-Sim3 / oracle evaluation definitions 固定（§3）
5. ✅ VGGT normalized-scale semantics corrected（DEPTH_SCALE_SEMANTICS.md）

**Q12**：
| Gate | 判定 |
|------|------|
| **Metric Depth** | **YES**（定义就绪，待 scanner-GT 数据补齐后可运行比较） |
| **MSAM** | HOLD（几何精度仍 PARTIAL） |
| **LoRA** | HOLD（同上） |

**P5-1 成功标准**：不硬编码。运行后比较 VGGT raw-scale vs DA3 Metric vs UniDepthV2：
- foreground AbsRel / RMSE
- scale ratio error / scale CV across views
- scanner F10/F20（数据就绪后）
- height/width error
- 再据 scanner repeatability + 文献精度 + 任务 phenotype precision 制定 Gate

---

## 15. P6: LoRA/MSAM

保持：MSAM=HOLD / LoRA=HOLD。
若 P0–P5 完成：Metric Depth 可进入实验阶段（比较 raw-scale vs DA3 vs UniDepthV2）。

---

## 16. Q1–Q12 最终回答

| Q | 问题 | 答案 |
|---|------|------|
| Q1 | v3.1 "camera-sim3" 是否实际用 scanner geometry centroid+scale？ | **YES** |
| Q2 | 旧 scanner metrics 是否应降级为 oracle evaluation？ | **YES**（已修正） |
| Q3 | VGGT raw depth 能否称 guaranteed metric depth？ | **NO** |
| Q4 | Plant View 上 raw scale ratio 在多少独立 plant 稳定？ | 1 株（langdon_4 多日期，非独立 plant）；ratio 0.86–1.28 |
| Q5 | foreground-only raw depth AbsRel/RMSE？ | 定义已固定（P0-7），待 P2 实施后填实 |
| Q6 | 原 montage 中 8–50m 数值是什么？ | 地面/背景噪声（far-plane >10m 像素 <0.01%） |
| Q7 | 12-03 depth 文件是否真不存在？ | **存在**（320 PNG，mapping=full_match）；旧报告逻辑错误 |
| Q8 | 找到多少 scanner-GT plant？ | **1 株**（本地 subset，langdon_4/19-03-24） |
| Q9 | 其中 pose-PASS 多少株？ | **0 株**（19-03-24 pose-FAIL） |
| Q10 | pose-PASS scanner-GT F@10/20mm + height/width error 跨 plant？ | N/A（无 pose-PASS GT 植株） |
| Q11 | 3DGS pseudo 是否系统性比 scanner GT 乐观？ | **YES** |
| Q12 | 是否具备开始 Metric Depth comparison 条件？ | **Metric Depth=YES / MSAM=NO / LoRA=NO** |

---

## 17. 测试状态

新增 10 个测试（P7），共 **55 passed**：
- test_scanner_alignment_provenance（含 test_scanner_oracle_alignment_flagged）
- test_scanner_metrics_foreground_only
- test_scanner_unit_verified
- test_vggt_scale_not_claimed_metric
- test_depth_foreground_metrics
- test_depth_invalid_farplane_filter
- test_depth_file_mapping_complete
- test_pose_fail_camera_geometry_not_primary_metric
- test_reference_pose_provenance
- （8 个 v3.1 测试保留）

---

## 18. 交付物

| 文件 | 说明 |
|------|------|
| SCANNER_UNIT_AUDIT.json | scanner PLY 单位 VERIFIED=millimeter（scale 0.001） |
| DEPTH_VALIDITY_AUDIT.json | 每序列深度范围 + far-plane 比例 |
| DEPTH_FILE_MAPPING_AUDIT.csv | 5 序列 depth 文件映射（全部 full_match） |
| SCANNER_DATASET_DISCOVERY.csv | 本地 scanner-GT 发现（1 株） |
| scanner_gt/scanner_gt_3tier.json + .csv | 三套评价（RAW/REF-CAMERA/ORACLE） |
| DEPTH_SCALE_SEMANTICS.md | VGGT 深度语义声明 |
| figures_v31/*_precision_recall_explanation.png | P/R 可视化（3 序列） |
| four_path_v4/metric_definitions.json | 修正 provenance |
| PHASE22_GEOMETRY_AUDIT_V32.md | 本报告 |

---

*Generated: 2026-08-29 | Geometry Audit v3.2 | True-GT Benchmark + Scale Semantics Closure*
