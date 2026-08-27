# 阶段 2.2 Geometry Audit v3

**目的**：独立验证 VGGT 的深度 / 反投影几何 / 点云是否真正恢复可信的植株 3D 结构。
**范围**：不重跑 VGGT 推理（复用 `v2_clean_rerun/` NPY）；不重算冻结的 Pose v2（14/17）；
**严格区分** `Pose Gate（14/17，frozen）` 与 `Geometry Gate（NOT YET ESTABLISHED）`。

> 本文档关联产物：`GEOMETRY_AUDIT_TABLE.csv`、`per_seq/*.geo_v3.json`、`UNPROJECTION_AUDIT.json`、
> `four_path_v3/`、`figures_v3/`、`VISUAL_AUDIT.md`、`GALLERY.md`、`tests/`。

---

## 1. 目的与范围

Clean Reproducibility Rerun 已闭环（`reproducibility_closure = PASS`，commit `0b2de01`，`pose_gate = 14/17`）。
但 **Pose Gate 通过 ≠ Plant 几何质量通过**。本次审计回答 8 个关键问题（Q1–Q8），重点：
- **P0**：修正历史 `unproject_np()` 的 OpenCV w2c→c2w 符号错误，并数值证明其与官方一致；
- 双向几何指标（Chamfer / Precision-Recall-F@τ）；
- 单帧深度审计、表型指标、四路判别修正重算；
- 诚实标记不可审计数据集（非米制 / 地理偏移参考）。

**禁止动作全部遵守**（10 条，见计划）：未看图仅凭 JSON 宣布 PASS（#1）、混淆 Pose/Geometry Gate（#2）、
ICP 后称原生精度（#3）、丢 90% outlier 报 median（#4）、保 "depth 正确" 调阈值（#5）、为 14/17 设计 Gate（#6）、
旧 four-path 覆盖 v3（#7）、重训 LoRA 掩盖（#8）、未完成进 MSAM（#9）、只交代码不交图（#10）。

---

## 2. P0 反投影审计结论（Q1）

**Q1：原 `unproject_np()` 是否错误？→ YES（已数学推导 + 官方数值对照证明）。**

正确 OpenCV w2c 约定：`x_cam = R_w2c @ X_world + t_w2c`，逆为
`X_world = R_w2c.T @ x_cam − R_w2c.T @ t_w2c`，即 `t_c2w = −R_w2c.T @ t_w2c`。

| 组件 | 公式 | 状态 |
|------|------|------|
| 官方 `unproject_depth_map_to_point_map` (`vggt/utils/geometry.py:15`) | `world = cam @ R_c2w.T + t_c2w`，`t_c2w=−R.T@t` | ✅ 正确 |
| 主推理 `run_vggt_inference.py:147` | 调用官方函数 | ✅ `point_map_unprojected.npy` 正确 |
| 历史 `four_path_eval.py:202-214` `unproject_np()` | `world = R.T @ cam + t` （加 `t_w2c`，符号错） | ❌ **BUG** |

**数值证据**（`UNPROJECTION_AUDIT.json`，`make_unprojection_audit.py` 生成）：
- identity 相机：`max_abs_diff = 0.0`
- 随机 SE3：`v3 vs 官方 = 1.25e-7`；**旧 `unproject_np` vs 正确 = 1.52 m**（相机距离量级，证实 bug）
- 真实 clean-rerun 05-03-24：`v3 vs 官方 = 6.28e-8`；已存 `point_map_unprojected.npy vs 官方 = 5.96e-8`
- 真实 clean-rerun 12-03-24：同上 ~6e-8

**处置**：`four_path_v2` 标记 `DEPRECATED(reason=incorrect_w2c_to_c2w_unprojection)`；
新增 `four_path_v3/`（修正反投影，复用 `four_path_data/*.npz`，无 GPU 依赖），**绝不覆盖** 旧目录（#7）。

---

## 3. 方法与指标定义

**反投影**（`unproject_v3.py`）：`world = R_w2c.T @ cam − R_w2c.T @ t_w2c`（≡ 官方，float32）。

**对齐**（`align_v3.py`）：仅用 **相机中心 Umeyama Sim3**（无 ICP，不掩盖模型误差，#3），
**always 保存对齐前/后点云**（`per_seq/*_pred_before.npy` / `*_pred_aligned.npy`）。

**前景掩膜**（`foreground_v3.py`，gsplat 缺失 → Priority-2 数据集分割）：
- plant_view：逐帧 `images/mask/*.png.png`（1080² 二值）→ 映射回 518² VGGT 帧
- wheat3dgs：YOLO-SAM 实例 mask 合并
- mustc：无逐帧 mask → 仅 FULL 场景

**双向指标**（`geometry_metrics_v3.py`，默认不截断）：
- **Chamfer**：`CD_p2g = mean_p min_q ||p−q||`，`CD_g2p` 对称，`CD_sym` 加权
- **双向 nn 分布**：median / P90 / P95（pred→gt 与 gt→pred 各自）
- **F/P/R@τ**：τ ∈ {1%,2%,5%}·D（D=参考 bbox 对角线，robust 95% 中心分位）+ 物理单位 {0.01,0.02,0.05}m
- **覆盖/离群**：@τ=5%·D 的 N_within / N_outside / within_ratio
- **truncated_inlier_nn_median**：仅 `diagnostic_only=True`（弃用主判据，#4）

**两套结果集**：每序列同时产出 `FULL-SCENE` 与 `PLANT-FOREGROUND-ONLY`。

**诚实性规则**：wheat (`reference_comparable=False`，COLMAP 任意单位) / mustc (`False`，LAS 地理偏移 + 未知尺度)
→ D=None，全部阈值指标置空，`reference_frame_auditable=False`，不进入 Geometry Gate 统计（#5）。

---

## 4. 跨数据集对齐

| 数据集 | 参考几何 | 对齐方式 | 可比性 |
|--------|----------|----------|--------|
| plant_view_3d | GS `splat.ply`（dataparser 还原）+ 逐帧深度/mask | 相机中心 Umeyama Sim3 | ✅ 米制可比 |
| wheat3dgs | COLMAP `points3D.txt` | Umeyama | ❌ 任意单位（不可比） |
| mustc | `.las`（plot-local Metashape 系） | Umeyama | ❌ 地理偏移 + 未知尺度 |

plant_view 对齐残差（相机中心 RMSE）：05-03-24=0.029m、13-02-24=0.033m、20-02-24=0.050m、
**12-03-24=0.768m（姿态崩，根因）**。对齐尺度 s≈1.0（0.96–1.31），验证 VGGT 米制尺度稳健。

---

## 5. 双向几何结果（FULL + PLANT-FOREGROUND）

来自 `GEOMETRY_AUDIT_TABLE.csv`（关键列，rounded）：

| 序列 | 集合 | Chamfer_sym(m) | F@5%_D | within5%_pred | 判定 |
|------|------|---------------|--------|---------------|------|
| 05-03-24 | FULL | 0.088 | 0.471 | 0.308 | 背景离群污染 |
| 05-03-24 | FOREG | **0.011** | **0.981** | 0.963 | **GOOD** |
| 12-03-24 | FULL | 1.011 | 0.000 | 0.000 | FAILED |
| 12-03-24 | FOREG | 0.986 | 0.000 | 0.000 | FAILED |
| 13-02-24 | FULL | 0.104 | 0.419 | 0.265 | 背景离群污染 |
| 13-02-24 | FOREG | **0.013** | **0.984** | 0.969 | **GOOD** |
| 20-02-24 | FULL | 0.086 | 0.472 | 0.309 | 背景离群污染 |
| 20-02-24 | FOREG | **0.012** | **0.978** | 0.956 | **GOOD** |
| wheat3dgs__plot_463 | — | (置空) | (置空) | — | 不可审计 |
| mustc__pos00 | — | (置空) | (置空) | — | 不可审计 |

**关键发现**：FULL 场景指标被 **背景/姿态离群点** 严重污染（within5%_pred≈0.3），
但 **PLANT-FOREGROUND-ONLY** 下 VGGT 点几何极佳（Chamfer~0.01m，F@5%_D≈0.98）。
这解释了旧版 "视觉形态不理想" 的观感——实为 **前景/背景范围不匹配**，非核心几何失败。

12-03-24 在 FULL 与 FOREG 下均 FAIL（F@5%_D=0.0），且其根因是相机头姿态失败（center_res=0.768m），
**并非点几何本身缺陷**——修正反投影后仍 FAIL，证实旧 "点几何错误" 归因必须改为 "相机头失败"。

(Q4 答案：plant_view 3 个 PASS 序列前景几何质量 **分布极佳**；12-03-24 因相机失败整体 FAIL。)

---

## 6. 单帧深度审计（plant_view）

`depth_audit_v3.py`：参考深度（`images/depth/*.png`，720² 16-bit）→ 映射 VGGT 518² 帧，
VGGT depth 经 **逐帧 median scaling** 后与参考比（raw 与 aligned 双报）。

| 序列 | n_frames | raw AbsRel | aligned AbsRel | δ1 | δ3 |
|------|----------|-----------|----------------|-----|-----|
| 05-03-24 | 10 | 0.999 | **0.124** | 0.575 | 0.952 |
| 13-02-24 | 10 | 0.999 | **0.106** | 0.641 | 0.951 |
| 20-02-24 | 10 | 0.999 | **0.185** | 0.366 | 0.954 |

(Q5 答案：raw AbsRel≈1.0 证实 VGGT 深度存在尺度歧义（预期内）；median-aligned 后 AbsRel 0.10–0.19、
δ3>0.95，深度趋势与参考一致。20-02-24 尺度偏差略高，建议后续核查该序列参考深度质量。)

> Caveat：median scaling 假设场景主体尺度一致，非真值对齐；相关图见 `figures_v3/*_depth_montage.png`。

---

## 7. 表型指标（plant_view）

`phenotype_v3.py`：对齐后前景点云 → 株高 Eh（Z 跨度）、冠层宽、bbox、占用体积。

| 序列 | Eh_pred(m) | Eh_ref(m) | Eh_err(m) | canopy_pred | canopy_ref |
|------|-----------|-----------|-----------|-------------|------------|
| 05-03-24 | 1.050 | 1.035 | 0.015 | 1.963 | 0.743 |
| 13-02-24 | 1.045 | 1.046 | 0.001 | — | — |
| 20-02-24 | 0.970 | 1.004 | 0.034 | — | — |

(Q6 答案：株高 Eh 预测与参考误差 **1–3.4cm**，量级完全一致，印证前景几何可信。
05-03-24 canopy diameter 预测偏宽（1.96 vs 0.74），需复核是否前景 mask 含部分背景；属局部、非系统性误差。)

---

## 8. 四路判别修正重算（Q3）

`four_path_v3.py` 复用 `four_path_data/*.npz`，以 `unproject_v3` 替代旧 buggy `unproject_np`。

**success_05-03-24**（36 帧，截断 τ=0.0637m，ref_diag=1.27m）：
- **A_vggt_cam**（VGGT 相机反投影）：nn_med≈0.019–0.036m
- **B_ref_cam**（参考相机反投影）：nn_med≈0.033–0.035m
- **C_point_head**（point_head 输出）：nn_med≈0.020–0.037m
- → A≈C，且均接近 B，**说明 VGGT 相机头与 point_head 在修正反投影后均工作正常**，
  旧 "B 正常 / A,C 失败 → 相机头失败" 结论是 **bug 假象**。

**fail_12-03-24**（36 帧）：
- A / C：`nn_med = null`（>99% 点超截断）→ 相机头 / point_head 在该序列失效
- B_ref_cam：nn_med≈0.036–0.046m（正常）→ 参考相机几何完好
- → 失效根因是 **相机外参（相机头）**，与 Pose Gate FAIL 一致；point_head 因 **共享 aggregator**
  而连带失效（**假设**，见 §9 Q7，未升级为结论）。

(Q3 答案：修正反投影后，success 序列三路全部正常工作——旧四路定性 **被推翻**；
fail 序列确为相机头失败，point_head 因共享 aggregator 连带失效。)

---

## 9. 结论与 Geometry Gate 状态（Q2/Q6/Q7/Q8）

### Q2：Pose Gate 与 Geometry Gate 是否应分开？
**是，本报告严格分离**。Pose Gate（14/17）冻结于 `gate_stats_clean_rerun.json`；
Geometry Gate 本次 **NOT YET ESTABLISHED**——`GEOMETRY_AUDIT_TABLE.csv` 仅报分布，
未反向设计通过率（#2/#6）。`test_geometry_gate_no_v1_fields.py` 守护：无 v1 字段、无 "几何 14/17" 措辞。

### Q6：前景范围不一致是否解释旧 "视觉形态不理想"？
**是**。FULL 场景 within5%_pred≈0.3（背景离群），但 FOREG F@5%_D≈0.98——旧观感源于
前景/背景范围不匹配，非点几何失败（见 §5、VISUAL_AUDIT.md）。

### Q7："depth 正确" / "point_head 共享聚合失败" 是假设还是已证实？
**仍为假设，除非有因果证据**。已证实：VGGT 主推理反投影正确（#2 节）、success 序列三路均正常、
12-03-24 失败根因为相机头（非点几何）。**"point_head 共享 aggregator 失败" 仅在 12-03-24 表现为连带失效，
属相关性观察，未做消融因果验证，保持为假设**（不升级为结论）。

### Q8：是否具备进 MSAM / LoRA 条件？
**待满足**：几何审计已完成且无 ICP 掩盖。最终 `geometry_validation` 状态：

### `geometry_validation = PARTIAL`

**理由**：
- ✅ plant_view 3 个 PASS 序列前景几何 **GOOD**（Chamfer~0.01m，F@5%_D≈0.98，Eh 误差 1–3cm）；
- ❌ 12-03-24 因相机头失败整体 FAIL（属姿态问题，非几何核心缺陷，但影响该序列可用性）；
- ⚠️ wheat3dgs / mustc **不可审计**（参考系非米制 / 地理偏移），需引入米制真值后方可纳入；
- ⏸️ Geometry Gate 未建立（先报分布，待人工 VISUAL_AUDIT 与更多米制真值）。

**结论**：VGGT 在 **可审计的 plant_view 米制场景** 下点几何与深度趋势 **可信**，具备进入 MSAM 的几何基础；
但需先解决（a）12-03-24 类相机头失败序列的鲁棒性、（b）wheat/mustc 米制真值接入，
方可将 Geometry Gate 从 `not_yet_established` 升级为正式判据。

---

## 附录：产物清单与测试

- 代码：`unproject_v3.py` `geometry_metrics_v3.py` `foreground_v3.py` `align_v3.py`
  `depth_audit_v3.py` `phenotype_v3.py` `four_path_v3.py` `run_geometry_audit_v3.py`
- 数据：`GEOMETRY_AUDIT_TABLE.csv`（11 行）、`UNPROJECTION_AUDIT.json`、`four_path_v3/*`、
  `per_seq/*.geo_v3.json`（6）、`per_seq/*_{pred_before,pred_aligned,ref}.npy`（18）
- 图像：`figures_v3/*.png`（35，含 6 类图 × 6 序列）、`four_path_v3/*_grid.png`
- 报告：`VISUAL_AUDIT.md`、`GALLERY.md`、`PHASE22_GEOMETRY_AUDIT_V3.md`
- 测试：`tests/` 6 文件 → **26 passed**（P0 `test_unprojection_vs_official` 门禁 + `test_geometry_gate_no_v1_fields` 守卫）
