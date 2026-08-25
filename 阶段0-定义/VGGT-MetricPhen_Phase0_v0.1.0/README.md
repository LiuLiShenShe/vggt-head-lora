# VGGT-MetricPhen Phase 0 冻结规范包

版本：`v0.1.0-phase0`  
冻结日期：`2026-08-24`  
状态：`FROZEN_FOR_PHASE1_AUDIT`  

## 1. 本包解决什么

本包完成正式方案的阶段0工作，冻结以下内容：

1. 株高、冠幅、冠层面积、可见功能叶长宽的操作性定义；
2. `plant_id / plant_date_id / sequence_id / leaf_uid / leaf_obs_id`身份规则；
3. 地面、重力方向和作物行方向的坐标定义；
4. 训练、验证、测试的分组划分与防泄漏规则；
5. 数据等级（L1–L5）和样本质量（QA–QR）规范；
6. MuST-C、Wheat3DGS、3D Plant View和TERRA-REF的准入角色；
7. 数据目录、样本清单、叶片清单、文件审计和许可证据模板；
8. 进入阶段1、阶段2和自采试验前必须满足的门禁。

本包不包含模型训练，不下载公开数据，也不宣称四个公开数据集已经完成逐文件审计。

## 2. 当前四个公开数据集的冻结角色

| 数据集 | 阶段0准入 | 正式角色 | 当前限制 |
|---|---|---|---|
| MuST-C | 通过 | 田间多传感器几何、跨平台、plot级验证 | 无统一株高/冠幅/叶长/叶宽真值 |
| Wheat3DGS | 有条件 | 密植小麦plot、多视角相机、麦穗实例基线 | 数据许可证需逐文件固定；TLS仍待官方发布/确认 |
| 3D Plant View | 通过 | 受控小麦几何、视角数、相机与扫描消融 | 非大田；无统一四表型真值 |
| TERRA-REF | 通过 | 第二田间数据源、plot级外部验证、RGB/扫描对齐审计 | 体量和对齐成本高；不是逐株/逐叶四表型数据 |

详细证据见 `templates/dataset_admission_audit.csv`。

## 3. 文件说明

- `00_FREEZE_MANIFEST.yaml`：冻结版本、变更规则与阶段门禁；
- `01_phenotype_protocol.md`：表型、坐标系、解析量和评价目标定义；
- `02_identity_leaf_split_protocol.md`：身份、leaf ID、分组划分和泄漏规则；
- `03_directory_and_manifest_spec.md`：统一目录、文件命名和清单规范；
- `04_quality_and_grading.md`：L1–L5数据等级与QA–QR质量等级；
- `05_phase0_completion_report.md`：阶段0结论、未完成事项和阶段1入口；
- `config/dataset_registry.yaml`：四个公开数据集的机器可读角色配置；
- `config/controlled_vocabularies.yaml`：受控词表；
- `schemas/sample_manifest.schema.json`：单个样本JSON Schema；
- `schemas/leaf_measurement.schema.json`：叶片观测JSON Schema；
- `templates/*.csv`：可直接填写的项目模板；
- `scripts/validate_phase0_package.py`：规范包静态校验脚本。

## 4. 使用顺序

1. 项目负责人签署 `templates/phase0_freeze_checklist.csv`；
2. 阶段1每下载一个最小样，填写 `dataset_file_audit_template.csv`；
3. 只有许可状态为`VERIFIED`、文件映射为`PASS`的数据才能进入正式训练；
4. 自采试验按 `sample_manifest_template.csv` 和 `leaf_measurement_template.csv`登记；
5. 任何定义变更必须增加版本号并记录在冻结清单中，不能静默修改；
6. 运行 `python scripts/validate_phase0_package.py` 检查规范包结构。

## 5. 核心硬规则

- 主结论是植株/冠层级表型；叶长宽只针对有稳定`leaf_uid`且满足可见性要求的功能叶；
- 同一物理植株、plot或连续序列的所有图像和模态必须处于同一数据划分；
- 测试植株人工株高、冠幅或扫描尺寸不得用于拟合该测试样本尺度；
- 图像派生标签必须标为`image_derived`，不得冒充人工或TLS真值；
- 代码许可证不能替代数据许可证；
- 许可、单位、坐标系或身份映射不清的数据不得进入正式训练与评价。

## 6. 官方证据入口

- MuST-C：https://www.ipb.uni-bonn.de/data/MuST-C/
- Wheat3DGS：https://github.com/zdwww/Wheat-3DGS
- 3D Plant View数据：https://doi.org/10.5524/102661
- 3D Plant View代码：https://github.com/Lewis-Stuart-11/3D-Plant-View-Synthesis
- TERRA-REF：https://www.terraref.org/data/access-data.html
- TERRA-REF Dryad：https://doi.org/10.5061/dryad.4b8gtht99

