# 数据等级与样本质量规范

协议版本：`v0.1.0-phase0`

## 1. 两条轴必须分开

- 数据等级`L1–L5`：描述有哪些模态和真值；
- 质量等级`QA–QR`：描述该具体样本是否清晰、同步、可标定和可测。

例如，具有TLS和人工四表型的样本可以是`L1-QC`；它仍然是金标准模态配置，但图像运动严重，只能用于失败分析。

## 2. 数据等级

### L1 金标准精测数据

必须同时包含：

- 多视角RGB；
- 真实相机内参；
- 物理尺度来源；
- LiDAR/TLS/高精度扫描；
- 人工株高、双向冠幅和抽样功能叶长宽；
- leaf ID、可见性和重复测量；
- 完整元数据与清晰许可。

用途：主监督、尺度模型选择、最终精度评价。

### L2 表型监督数据

包含多视角RGB、相机内参、人工株高和冠幅、已知尺度或系统标定；可能只有部分叶片标签，无独立高精度点云。

用途：Head与LoRA训练、跨plot表型评价。

### L3 几何数据

包含多视角RGB以及相机位姿、点云或LiDAR之一，但缺少统一四表型。

用途：VGGT几何、尺度、相机和跨域评价。

### L4 无标签多视角农业视觉数据

包含可分组的多视角RGB或连续视频，无可靠三维或表型真值。

用途：域适配、多视角一致性、质量模型和吞吐量测试。

### L5 辅助数据

纯点云、单视图实例分割、合成数据或没有配对RGB的器官数据。

用途：分割、点云编码器和器官先验预训练。不能作为VGGT主输入或最终四表型主评价。

## 3. 样本质量等级

### QA 主要评价可用

- 有效视角数达到协议；
- 模糊、过曝和风动均在可接受范围；
- 相机与RGB一一对应；
- 尺度、地面和坐标系明确；
- 目标实例身份明确；
- 标签定义与单位完整；
- 无数据泄漏风险。

可用于训练和主评价。

### QB 训练/次要评价可用

存在轻度缺视角、局部遮挡或少量相机元数据缺失，但不会改变目标定义；必须保存缺陷码。

可用于训练和分层评价，不应作为最严格几何金标。

### QC 仅失败分析或弱监督

明显风动、严重遮挡、地面不可见、尺度不稳定或实例粘连，但仍可追溯身份和失败原因。

不能进入主精度指标，可用于失败检测和可测性训练。

### QR 拒收

身份、许可、单位、映射关系或数据完整性不可恢复；测试泄漏无法消除；原始数据损坏。

不得进入正式训练与评价。

## 4. 关键自动QC指标

每个sequence至少计算：

- `valid_view_count`；
- `blur_fraction`；
- `overexposed_fraction`；
- `underexposed_fraction`；
- `duplicate_frame_fraction`；
- `camera_mapping_success`；
- `view_coverage_score`；
- `motion_severity`；
- `metric_scale_cv`；
- `ground_inlier_ratio`；
- `plant_mask_coverage`；
- `leaf_id_completeness`。

阈值在30–60株试采的训练/校准部分确定，不能使用正式test调节。

## 5. 缺陷码

推荐缺陷码：

- `Q01_BLUR`；
- `Q02_OVEREXPOSURE`；
- `Q03_UNDEREXPOSURE`；
- `Q04_WIND_MOTION`；
- `Q05_INSUFFICIENT_VIEWS`；
- `Q06_LOW_OVERLAP`；
- `Q07_CAMERA_MISMATCH`；
- `Q08_SCALE_UNSTABLE`；
- `Q09_GROUND_UNOBSERVED`；
- `Q10_INSTANCE_MERGE`；
- `Q11_LEAF_ID_UNCERTAIN`；
- `Q12_LABEL_DEFINITION_UNKNOWN`；
- `Q13_LICENSE_UNVERIFIED`；
- `Q14_COORDINATE_UNKNOWN`；
- `Q15_DUPLICATE_LEAKAGE_RISK`；
- `Q16_CORRUPT_FILE`。

## 6. 当前公开数据的预期等级

| 数据集 | 预期最高等级 | 说明 |
|---|---:|---|
| MuST-C | L3 | 多视角/多传感器几何强，但无统一四表型 |
| Wheat3DGS | L3（有条件） | RGB与相机强；数据许可和TLS发布需审计 |
| 3D Plant View | L3 | 受控几何与扫描强，无统一四表型 |
| TERRA-REF | L3 | plot级RGB/扫描/表型强，逐株逐叶闭环不足 |

最终等级必须根据阶段1逐文件审计下调或确认，不能仅依据论文描述。

