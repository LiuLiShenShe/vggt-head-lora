# PHASE22_GEOMETRY_AUDIT_V321.md — Stage 2.2 Geometry Audit v3.2.1 报告

> **Purpose**: Evidence integrity repair. Make `code → artifacts → tests → report` a single consistent,
> leak-free, traceable chain. Fix the two invalid/divergent scanner-GT artifacts from v3.2, the
> mislabeled Precision/Recall figures, and add the previously-absent foreground-depth evaluator.

> **Not started**: Formal LoRA, MSAM, Head training. No VGGT re-inference.
> **Baseline commit**: `7cf3fc4` (v3.2) → this commit (v3.2.1).
> **Runner**: `agri_re_py310` (scipy 1.15.3, open3d 0.19.0, matplotlib 3.9, torch 2.3.1+cu121) —
> matches the spec'd `da3` environment.

---

## 最终状态

| 字段 | 值 |
|------|-----|
| reproducibility | PASS |
| pose_validation | PASS_WITH_KNOWN_FAILURES |
| pseudo_reference_geometry | PARTIAL |
| **scanner_gt_benchmark** | NOT_ENOUGH_DATA (only 1 plant locally, and it is pose-FAIL) |
| **absolute_metric_geometry** | NOT_SUPPORTED (no pose-PASS scanner GT available) |
| **scanner_evaluator_integrity** | PASS (no identity leak; B 100% GT-independent; A/B/C all real) |
| **foreground_depth_evaluator** | PASS (real fg-depth metrics for 4 seqs) |
| **metric_depth_readiness** | YES (5 conditions met — see P6) |
| MSAM | HOLD |
| LoRA | HOLD |

---

## 0. 本轮目标 vs 上次 (v3.2)

保留：`reproducibility=PASS` / `pose_validation=PASS_WITH_KNOWN_FAILURES` / `pseudo_reference_geometry=PARTIAL`

修复（本轮 P0–P8）：
- **P0-1 根因审计**：v3.2 提交的 `SCANNER_GT_3TIER.json` 含身份泄漏（pred≡GT）→ F=1.0/Chamfer=0。
- **P0-2 隔离**：无效文件改名 `SCANNER_GT_3TIER_INVALID_v32.{json,csv}`；新生成权威 `V321`。
- **P0-3/4/5 重写评估器**：单前景源；B 仅用相机中心 Sim3（无 GT 泄漏）；C 允许用 GT 并标 upper_bound；身份泄漏守卫。
- **P1-1/3 真实前景点集 + PR 图**：纠正 v3.2 用 full-scene 冒充 foreground 的图。
- **P2 真实前景深度评估器**：新增 `depth_foreground_eval_v321.py`（含 12-03-24）。
- **P3 测试修复**：删除空壳检测，新增泄漏/身份/图一致性/深度评估测试。

`scanner_gt_benchmark` 仍为 `NOT_ENOUGH_DATA`（本地仅 1 株且 pose-FAIL）——**不调参到 PASS**。

---

## 1. P0-1: 无效 artifact 根因

`SCANNER_ARTIFACT_PROVENANCE_AUDIT.md` 记录：v3.2 的 `SCANNER_GT_3TIER.json`
(sha256 `9d595043…`) 由后台 agent `a4ebca7d` 生成，提交于 `7cf3fc4`。其 defect signature：

```json
"foreground_only": { "F_5mm":1.0, "F_10mm":1.0, "F_20mm":1.0, "F_50mm":1.0,
                     "chamfer_sym_m":0.0, "n_points_pred":5745011, "n_points_gt":5745011 }
```

`n_points_pred == n_points_gt == 5745011` 正是 Einstar PLY 原始点数；F=1.0/Chamfer=0 是
**prediction≡ground-truth** 的必然结果（身份泄漏）。**Q1 (identity leak = YES)**。

v3.2 的另一产物 `SCANNER_GT_3TIER.csv`（人类脚本生成）虽无该签名，但 **B/C 前景行空**
(`n_points_pred=0`) 且 **B 用了 GT 质心+尺度拟合**（第二条 GT 泄漏）。两者互斥且均非权威 → 全部隔离。

---

## 2. P0-3/4/5: 三层评估 (重写后, leak-free)

| Tier | 对齐方式 | uses_ref_pose | uses_scanner_geom | F@50mm (fg) | chamfer_sym (fg) | height_err (fg) | 角色 |
|------|----------|--------------|------------------|-------------|------------------|-----------------|------|
| **A. RAW** | 无（direct） | False | False | 0.054 | 0.240 | 0.414m | 真实 metric-scale 能力 |
| **B. REF-CAMERA** | 相机中心→参考相机中心 Sim3 | True | **False** | **0.000** | 1.021 | 0.255m | camera-frame 后 shape（pose-FAIL 退化） |
| **C. ORACLE** | scanner GT 拟合 Sim3 | False | True | 0.506 | 0.071 | 0.595m | shape 上界（仅参考） |

**关键修复**：
- 单前景源 `pred_fg_raw`（10,040,276 点）→ A/B/C 三 tier 共用（P0-7）。B/C 不再丢前景。
- **B_refcam 变换 `estimate_refcam_sim3(vggt_cam_centers, ref_cam_centers)`** 签名**不接受 scanner 点**
  （见 `test_refcam_no_gt_leak`：T1=真实 GT vs T2=打乱 GT → 变换完全相同，证明 100% 无 GT 泄漏）。
- **C_oracle** 允许用 GT 拟合，标记 `upper_bound=true`。
- **身份泄漏守卫**：若 `n_pred==n_gt 且 Chamfer==0 且 所有 F==1` → 抛错退出非零（本 run 已 pass）。

**诚实结论**：19-03-24 是 **pose-FAIL**（相机中心旋转中位 173.7°），B_refcam 退化 F@50mm=0.000 —
不伪造。A(原始) scale_ratio=0.857 证明 VGGT 为 "approximately metric-scale"；C(oracle) 上界
F@50mm=0.506 仍远低于可信基准。

---

## 3. P0-6: alignment_provenance.json

每 tier 记录 `transform_source / uses_reference_camera_pose / uses_scanner_geometry_for_transform /
uses_scanner_geometry_for_metrics / scale / rotation / translation / input_camera_count`。
B→`uses_scanner_geometry_for_transform=false`；C→`true, upper_bound=true`。

---

## 4. P1-1/3: 真实前景点集 + PR 图

`run_geometry_audit_v31.py` 现持久化真实前景点集：
- `per_seq/{sid}_pred_foreground_raw.npy` / `pred_foreground_aligned.npy` / `reference_foreground.npy`

`precision_recall_visual_v321.py` **只加载上述真实前景数组**（不再用 full-scene），并写
`FIGURE_INPUT_MANIFEST.json` 记录每图的点云路径+sha256（供 `test_figure_metric_same_array` 校验）。

实测（真实前景，4 序列）：

| 序列 | precision@10mm | recall@10mm | F@10mm |
|------|---------------|-------------|--------|
| 05-03-24 (pose-PASS) | 0.444 | 0.996 | 0.614 |
| 12-03-24 (pose-FAIL) | 0.000 | 0.000 | 0.000 |
| 13-02-24 (pose-PASS) | 0.367 | 0.959 | 0.531 |
| 20-02-24 (pose-PASS) | 0.404 | 0.967 | 0.570 |

**对比 v3.2 误标**（用 full-scene 报 precision@10mm≈0.06–0.08）：真实前景 precision≈0.37–0.44。
recall≈0.96–1.00 证明参考点几乎全被覆盖；precision≈0.4 证明 VGGT 点云"膨胀"超出参考植物 —
**inflation 问题真实存在**，但量级与 v3.2 误标不同（误标夸大了精度损失）。

**12-03-24 pose-FAIL**：P@10/R@10=0.0 是诚实结果（VGGT pose 灾难性失败，clouds 完全不对齐）；
图标题标注 `(pose-FAIL)`。

---

## 5. P1-4: 废弃错误图

`figures_v31/*_precision_recall_explanation.png` 移至 `deprecated_v32_figures/` + `README.md`
（说明其用 full-scene 冒充 foreground）。**禁止引用**；改用 `figures_v321/*`。

---

## 6. P2: 真实前景深度评估器

`depth_foreground_eval_v321.py`：有效前景像素 = `plant_mask & ref_depth_valid & pred_depth_valid`；
RAW-SCALE（无尺度缩放）+ SCALE-ALIGNED（median scaling，仅相对形状诊断）。

`DEPTH_FOREGROUND_SUMMARY.csv`（4 序列，含 12-03-24）：

| 序列 | raw_AbsRel | raw_RMSE(m) | median_scale | aligned_AbsRel | aligned_RMSE(m) |
|------|-----------|------------|--------------|----------------|-----------------|
| 05-03-24 | 0.205 | 0.434 | 1.270 | 0.180 | 0.347 |
| 12-03-24 | 0.299 | 0.545 | 1.527 | 0.203 | 0.372 |
| 13-02-24 | 0.171 | 0.338 | 1.135 | 0.152 | 0.297 |
| 20-02-24 | 0.193 | 0.374 | 1.111 | 0.195 | 0.364 |

**scale_ratio ≈ 1.1–1.5**（序列相关，非固定映射）→ 印证 "approximately metric-scale，非 guaranteed"。
每帧明细见 `DEPTH_FOREGROUND_METRICS.csv`（1318 帧）；全场景诊断见 `DEPTH_FULLSCENE_DIAGNOSTIC.csv`
（绝不混入 headline）。

---

## 7. P3: 测试修复

新增/重写（全部非空壳）：
- `test_scanner_tier_completeness`：3 tier 齐全，每 tier fg 行 `n_pred>0`
- `test_refcam_no_gt_leak`：B 变换对真实/打乱 GT 完全一致（无 GT 泄漏）
- `test_scanner_identity_leak_guard`：V321 无 F=1.0/Chamfer=0 行；manifest 记录 guard=passed
- `test_scanner_alignment_provenance`：指向权威 V321/json；B 无 GT 几何、C upper_bound
- `test_figure_metric_same_array`：PR 图 sha256 与真实前景数组一致
- `test_depth_evaluator_real`：DEPTH_FOREGROUND_METRICS 含 4 序列 + raw/aligned 列
- 修复 `test_scanner_metrics_foreground_only`（指向 V321，断言 B/C fg 非空）
- 修复 `test_depth_foreground_metrics`（不再用 FOREGROUND_METRICS_V31.csv 冒充深度证据）
- 修复 `test_pose_fail_camera_geometry_not_primary_metric`（指向 V321，断言 pose-FAIL B fg F@50mm=0）
- 修复 `test_vggt_scale_not_claimed_metric`（排除历史引用误报）

**结果**：`pytest tests/` 64 passed（除 1 个预存在的 `test_unprojection_vs_official` 因缺 einops 无法收集，与本任务无关）。

---

## 8. P4/P5: Four-Path v4 与 Metric Depth Gate

Four-Path v4 结论不变（P5 仅 prose 修正）：
> VGGT 外参 E 是 12-03-24 灾难性全局错位的主因。替换 E 可消除灾难位移，但 **残留显著 depth/head/scene 几何误差**（B-E F@10mm≈0.011）。

**Metric Depth Gate 条件（5 项全满足 → YES）**：
1. ✅ 参考深度单位已验证（DEPTH_UNIT_AUDIT / SCANNER_UNIT_AUDIT，scale 0.001）
2. ✅ 前景深度评估器已落地且真实（P2）
3. ✅ scanner 三层级 provenance 正确（B 100% GT 独立，C upper_bound）
4. ✅ 权威 artifact 与报告一致（V321 sha256 匹配，见 manifest）
5. ✅ VGGT normalized-scale 语义已文档（DEPTH_SCALE_SEMANTICS.md）

**Q12**：
| Gate | 判定 |
|------|------|
| **Metric Depth** | **YES**（定义就绪，待 scanner-GT 数据补齐后可运行比较） |
| **MSAM** | HOLD（几何精度仍 PARTIAL） |
| **LoRA** | HOLD（同上） |

**下一步**：进入 DA3 Metric / UniDepthV2 比较（raw-scale vs DA3 vs UniDepthV2，foreground AbsRel/RMSE、scale CV、scanner F10/F20、height/width error）。

---

## 9. Q1–Q12 最终回答

| Q | 问题 | 答案 |
|---|------|------|
| Q1 | v3.1 "camera-sim3" 是否实际用 scanner geometry centroid+scale？ | **YES**（v3.2 亦复现于 B tier，已修） |
| Q2 | 旧 scanner metrics 是否应降级为 oracle evaluation？ | **YES**（C_oracle 标 upper_bound） |
| Q3 | VGGT raw depth 能否称 guaranteed metric depth？ | **NO** |
| Q4 | Plant View 上 raw scale ratio 在多少独立 plant 稳定？ | 本地仅 1 株（langdon_4 多日期，非独立 plant）；ratio 0.86–1.53 |
| Q5 | foreground-only raw depth AbsRel/RMSE？ | 见 P2 表：raw_AbsRel 0.17–0.30，RMSE 0.34–0.55m |
| Q6 | 原 montage 中 8–50m 数值是什么？ | 地面/背景噪声（far-plane >10m 像素 <0.01%） |
| Q7 | 12-03 depth 文件是否真不存在？ | **存在**（320 PNG，mapping=full_match）；旧报告逻辑错误 |
| Q8 | 找到多少 scanner-GT plant？ | **1 株**（本地 subset，langdon_4/19-03-24） |
| Q9 | 其中 pose-PASS 多少株？ | **0 株**（19-03-24 pose-FAIL） |
| Q10 | pose-PASS scanner-GT F@10/20mm + height/width error？ | N/A（无 pose-PASS GT 植株） |
| Q11 | 3DGS pseudo 是否系统性比 scanner GT 乐观？ | **YES** |
| Q12 | 是否具备开始 Metric Depth comparison 条件？ | **Metric Depth=YES / MSAM=NO / LoRA=NO** |
| **Q13 (本任务追加)** | v3.2 scanner artifact 是否身份泄漏？ | **YES**（F=1.0/Chamfer=0，pred≡GT，已隔离） |
| **Q14 (本任务追加)** | B tier 是否 100% GT 独立？ | **YES**（签名不接受 scanner 点，测试证明） |
| **Q15 (本任务追加)** | PR 图是否用真实前景？ | **YES**（sha256 在 FIGURE_INPUT_MANIFEST.json 可验证） |

---

## 10. 交付物

| 文件 | 说明 |
|------|------|
| SCANNER_ARTIFACT_PROVENANCE_AUDIT.md | P0-1 根因审计 |
| scanner_gt/SCANNER_GT_3TIER_INVALID_v32.json/.csv | 隔离的 v3.2 无效 artifact |
| scanner_gt/SCANNER_GT_3TIER_V321.csv/.json | 权威三层评估（leak-free） |
| scanner_gt/alignment_provenance.json | 每 tier 变换 provenance |
| SCANNER_GT_AUTHORITATIVE_MANIFEST.json | 权威路径+sha256+弃用清单 |
| scanner_gt_3tier_eval_v321.py | leak-free 评估器（重写） |
| per_seq/*_pred_foreground_*.npy | 真实前景点集（raw/aligned/ref） |
| precision_recall_visual_v321.py | 真实前景 PR 图生成器 |
| figures_v321/*_precision_recall_explanation.png | 纠正后的 PR 图（4 序列） |
| FIGURE_INPUT_MANIFEST.json | 图输入点云路径+sha256 |
| deprecated_v32_figures/ | 废弃错误图 + README |
| depth_foreground_eval_v321.py | 真实前景深度评估器 |
| DEPTH_FOREGROUND_METRICS.csv / _SUMMARY.csv / DEPTH_FULLSCENE_DIAGNOSTIC.csv | 深度评估 |
| PHASE22_GEOMETRY_AUDIT_V321.md | 本报告 |

---

*Generated: 2026-08-29 | Geometry Audit v3.2.1 | Evidence Integrity Repair (P0–P8)*
