# 阶段2.2 报告 — VGGT 几何推理(v2 审计修订版)

日期:2026-08-25(v2 修订;v1 报告内容已被本版取代,v1 数字仅存档于各序列 meta 的 pose_eval 字段与 git 历史)

## 总结论(三层)

1. **推理工程通过**:17/17 ready 序列完成前向推理,全部输出落盘,无 NaN/Inf,深度有效比例 1.0。
2. **几何评价 v1 未通过审计,v2 已修复并重算(以 v2 数字为准)**:v1 的点云对齐(随机配对 Horn)、相机可视化(未对齐同系)、绝对旋转评价(Procrustes 转置错误)、顺序敏感性(自比 bug)、坍缩指标(无尺度归一化)共 5 处实现错误,均已在 v2 修正(修正内容见下节)。
3. **plant_view 3 序列(12-03-24 / 15-04-24 / 19-03-24)确认为 VGGT 相机估计真实失败**,已由四路判别实验定性(见"四路判别"节):深度头输出正确,相机头/point head 失败。归因不是笼统"农业域不匹配"——同一株小麦的其他 3 个日期通过;暂定假设为高密度重复细叶场景的多视角可观测性失败,留待 LoRA 阶段针对性验证。

## v1 → v2 修正明细

| # | v1 问题 | v2 修正 | 影响的 v1 结论 |
|---|---|---|---|
| 1 | `align_refcloud_vs_vggt.png` 用无对应关系的随机配对做 Horn,尺度趋零 → "点云塌陷"为可视化假象 | 相机中心逐帧对应 Umeyama Sim3,同一变换作用于完整 VGGT 云;mustc 参考云为 UTM 系,改用 FPFH-RANSAC+ICP(fitness 0.63–0.77) | v1 塌陷结论撤回;v2 图显示 VGGT 云呈植株形态 |
| 2 | 相机图红(VGGT 原系)/蓝(参考系)不同世界系,朝向不可比 | VGGT 相机中心+朝向均经 Sim3 变换后与参考同系绘制 | v1 "朝向相反"图不可作证据;v2 图红蓝逐帧贴合(成功序列) |
| 3 | Procrustes einsum 顺序与 Rg 公式不匹配(取了转置解),残差组合公式错误 → 绝对旋转误差虚高至 14–99° | 修正为 H=ΣRvRrᵀ, Rg=V Uᵀ;残差 ang((Rg@Rv)ᵀ@Rr);脚本内置合成自检(恢复误差<1e-5) | 修正后绝对旋转误差与相对误差同量级(mustc 1.4–1.9°, wheat 3.3°, plant_view 成功序列 1.7–3.6°) |
| 4 | 顺序敏感性脚本 R_o/R_r 同源(自比),"0–0.5°"无效 | R_o 用已存原序 NPY,R_r 用反序推理还原索引,再做全局 Procrustes+Sim3 | v2 真实值:旋转中位 0.99°/1.55°,中心误差相对跨度 0.12/0.03 —— 顺序鲁棒结论成立但以 v2 数字为准 |
| 5 | 相机坍缩阈值 1e-3 无尺度归一化(gate_stats 全 false 与报告"坍缩"文字矛盾);flatness 高比值≠结构正确 | 新增尺度归一化相机分布指标(径向跨度/协方差特征值占比/轨迹长/深度跨度);坍缩判据改为"中心跨度 <1% 深度跨度";flatness 注明"仅排除单一平面" | 矛盾消除:失败序列的"相机坍缩"由 pose_eval_v2 的 center rel 0.87–0.95 与 camera_shape 指标承担 |

另:v1 失败记录中 subset_test 文案误复制到 3 个序列(实际仅 12-03-24 做过,且为连续前缀帧非均匀采样),已在 `plantview_pose_failures_v2.json` 更正。

## 推理输出(17 序列,不变)

| 数据集 | 序列数 | S | forward | 峰值显存 |
|---|---|---|---|---|
| wheat3dgs | 7 | 36 | 3.8–4.5s | 10.1GB |
| mustc | 4 | 20 | 1.7–1.9s | 9.4GB |
| plant_view_3d | 6 | 318–340 整序列 | 246–281s | 36.8–38.8GB |

每序列:`pose_enc/extrinsic_w2c/extrinsic_c2w/intrinsic_vggt/depth_vggt/depth_conf_vggt/point_map_direct/point_conf_direct/point_map_unprojected(第一候选)/prediction_meta.json` + tokens(S≤200)+ `checks_v2/` 8 文件(17/17 齐全;旧 checks/ 保留作审计痕迹)。

## 位姿评估 v2(pose_eval_summary_v2.json)

| 序列组 | 绝对旋转误差 median/P90 | 中心误差(相对参考跨度) | 轨迹余弦 |
|---|---|---|---|
| wheat3dgs(7) | 3.3° / 4.6° | 0.10–0.15 | 0.99 |
| mustc(4) | 1.4–1.9° / 1.9–4.6° | 0.04–0.06 | 1.00 |
| plant_view 成功(3) | 1.7–3.6° / 4.1–7.5° | 0.03–0.06 | 0.94–0.99 |
| plant_view 失败(3) | 78.9–89.8° / 158–163° | 0.87–0.95 | 0.44–0.53 |

**14/17 序列满足"median ≤10°, P90 ≤20°"门槛;3/17 相机估计失败。**

## 四路判别实验(10_failures/four_path_discrimination/)

成功(05-03-24)+ 失败(12-03-24)各取均匀 8/16/24/36 帧,4 条路径到参考 GS 云的截断 NN 距离(截断=5% 场景对角线;参考云已按 dataparser_transforms.json 还原平移 Z≈−1.09/−1.16 至 transforms.json 相机系——**该平移差为本次实验新发现,GS 云与相机标注系不重合,下游使用 GS 云时必须做此修正**):

| 序列 | A: VGGT深度+VGGT相机 | B: VGGT深度+参考相机 | C: point_head |
|---|---|---|---|
| 成功 05-03-24 | 0.029–0.038(正常) | 0.031–0.035(正常) | 0.020–0.037(正常) |
| 失败 12-03-24 | **None(>7cm 全截断,失败)** | **0.033–0.040(正常!)** | **None(失败)** |

**判定:B 正常、A/C 失败 → VGGT 相机头失败,深度头输出正确**(point head 与 camera head 共享聚合特征故同败)。视角数 8→36 无改善,排除"320 帧冗余视角稀释"为主因。失败序列的深度/点云/token 输出仍可用于下游(配合参考相机即可得到正确 3D)。

## 顺序敏感性 v2(10_failures/order_sensitivity_v2.json)

反序输入还原索引后与原序比较:plot_463 旋转中位 0.99°/P90 1.40°;mustc pos00 中位 1.55°/P90 2.95°。VGGT 对输入顺序鲁棒(v2 修正自比 bug 后的结论)。

## Gate 判定

| 指标 | 门槛 | v2 实测 | 结果 |
|---|---|---|---|
| 推理成功率 | ≥90% | 17/17 前向成功 | ✅ |
| 有效深度比例 | ≥95% | 17/17 = 1.0 | ✅ |
| NaN/Inf | 无 | 0 | ✅ |
| 点云/相机坍缩 | 无 | 成功序列无;3 失败序列相机坍缩(尺度归一化判定) | ⚠️ |
| 3D Plant View 结构 | 完整植株 | checks_v2:成功序列完整;失败序列深度正确但相机失败 | ✅(成功序列) |
| 旋转误差 | median≤10°, P90≤20° | 14/17 满足;3 序列 79–90° | ⚠️ 14/17=82.4% |

**结论:推理工程通过;几何质量 14/17 通过、3 个 plant_view 序列相机估计失败(已定性为相机头问题,深度无损)。** 对 LoRA 阶段的含义:这 3 个序列是天然的"相机头困难样本",可作为微调效果的验证集;其深度监督信号仍然有效。评价管线 v2 已修复审计发现的全部实现错误,后续阶段以 v2 指标为准。

---

# 阶段2.2 Clean Reproducibility Rerun 复现闭环 (2026-08-27)

## 目的
在不复用任何旧 NPY 的前提下,从原始 RGB 重跑全部 17 序列的 VGGT 推理与 v2 评价,验证现有结论可复现,建立完整 provenance,并完成评价版本字段清理。

## 冻结版本 (RUN_MANIFEST.json)
- **run_id**: `clean_rerun_20260826T065510Z`
- **project_git_commit**: `208c2b194a5fddd9c9ff880f6b56c419fbc0671b`(脚本参数化 commit 后冻结)
- **checkpoint**: `facebook/VGGT-1B`, blob sha256 `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`, 5.03 GiB
- **环境**: `vggt_lora` Python 3.10.20 / torch 2.3.1+cu121 / CUDA 12.1 / cuDNN 8.9.2 / RTX A6000
- **推理配置**: mode="crop" 518、bfloat16 autocast、每序列 seed=42、TOKEN_LAYERS=(4,11,17,23)、`cudnn.benchmark=False`(默认)
- 所有旧结果目录(v2 主结果 / checks_v2 / four_path*) 一律未改动。

## 交付物
1. `v2_clean_rerun/RUN_MANIFEST.json` — 版本冻结
2. `v2_clean_rerun/<ds>/<sid>/` — 17 序列全新推理输出(9+4 npy + meta + run_provenance.json + checks_v2/)
3. `v2_clean_rerun_eval/pose_eval_summary_v2.json` — 17 序列 pose_eval_v2
4. `v2_clean_rerun/<ds>/<sid>/checks_v2/` — 17×8 检查图
5. `v2_clean_rerun_eval/gate_stats_clean_rerun.json` — Gate 14/17
6. `v2_clean_rerun_eval/four_path_data/` + `four_path_discrimination/` — 四路判别(8 npz + grid + verdict + metric_definitions)
7. `v2_clean_rerun_eval/CLEAN_RERUN_COMPARISON.md` + `npy_diff_stats.json` — 对账
8. `00_environment/eval_version.py` — 版本字段强制读取器(读取 v1 必报错)
9. 全部 34 个 meta(17 旧 + 17 新)迁移到 `active_evaluation_version="v2"` 字段格式(旧 `pose_eval` 重命名为 `pose_eval_v1`),备份存 `prediction_meta.pre_migration.json`

## 复现结果
| 指标 | 旧 v2 | clean rerun | 一致? |
|---|---|---|---|
| 前向成功率 | 17/17 | 17/17 | ✅ |
| gate 通过 | 14/17 | 14/17 | ✅ |
| 失败序列 | 12/15/19-03-24 | 12/15/19-03-24 | ✅ |
| 成功序列 rot median | 1.7–3.7° | 1.7–3.7° | ✅ |
| 失败序列 rot median | 78.9–89.8° | 78.9–89.8° | ✅ |
| 四路定性 | 相机头失败/深度正确 | 相机头失败/深度正确 | ✅ |
| NPY 数值差异 | — | max_abs_diff = 0.0 (全部) | ✅ 逐位一致 |

## 四个核心问题
1. **是否完全没有复用旧 VGGT NPY?** 是。两次运行 mtime 相隔 24h、run_id 不同、无旧目录读取,且旧 meta 无 run_id 字段可佐证。
2. **clean rerun 是否仍为 14/17 Gate 通过?** 是,14/17,失败者与旧 v2 完全相同。
3. **原先 3 个 Plant View 失败序列是否仍然失败?** 是(12-03-24 / 15-04-24 / 19-03-24,rot median 84.2° / 89.8° / 78.9°)。
4. **基于 clean rerun,阶段2.2 是否已具备冻结并进入下一阶段的条件?** 是。

## 复现结论
```
reproducibility_closure = PASS
```
冻结阶段2.2 v2 → 多植株 pose validation → Metric Depth → MSAM → Frozen VGGT + Head baseline → 正式 LoRA。

## 附注:确定性复现
本环境(同 GPU 型号 A6000、同 CUDA 12.1、同 `vggt_lora` 环境、`cudnn.benchmark=False`)下 BF16 前向具确定性,两次运行逐位一致(max_abs_diff=0)。若后续换 GPU 型号或 CUDA 版本,预期出现 BF16 级漂移;对账脚本已按 atol=1e-3/rtol=1e-2 预留容差,届时只需确认 Gate 不翻转即可。
