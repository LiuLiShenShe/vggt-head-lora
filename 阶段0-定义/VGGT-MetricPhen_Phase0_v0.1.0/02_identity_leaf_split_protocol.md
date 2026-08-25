# 身份、叶片ID与数据划分冻结协议

协议版本：`v0.1.0-phase0`

## 1. 身份层级

统一身份层级：

```text
site_id
└── year
    └── plot_id
        └── plant_id
            └── plant_date_id
                └── sequence_id
                    └── view_id / frame_id
```

其中：

- `plant_id`：物理植株永久身份，不随日期改变；
- `plant_date_id`：某物理植株在某次观测日期的状态；
- `sequence_id`：一次连续多视角采集或一个固定多相机同步拍摄；
- `view_id`：多相机或人工选取视角；
- `frame_id`：视频原始帧。

推荐格式：

```text
site_id        = SC01
plot_id        = SC01-2026-P012
plant_id       = SC01-2026-P012-PL007
plant_date_id  = SC01-2026-P012-PL007-20260824
sequence_id    = SC01-2026-P012-PL007-20260824-S01
view_id        = ...-V012
```

不允许使用文件名顺序隐式表示身份。

## 2. 叶片身份

### 2.1 两级叶片ID

- `leaf_uid`：物理叶片在同一植株时间序列中的稳定身份；
- `leaf_obs_id`：该叶片在某个`plant_date_id`中的观测身份。

推荐格式：

```text
leaf_uid    = {plant_id}-L{emergence_order:02d}
leaf_obs_id = {plant_date_id}-{leaf_uid_basename}
```

例如：

```text
SC01-2026-P012-PL007-L05
SC01-2026-P012-PL007-20260824-L05
```

### 2.2 leaf ID建立规则

1. 优先使用自下而上叶序或出叶顺序建立ID；
2. 采集现场保存叶位示意、近景图和至少两个可识别视角；
3. 如果使用物理标签，应在正式RGB采集后放置，或确保标签不进入训练ROI，防止模型依赖标签外观；
4. 跨日期匹配需由叶序、基部位置、形态和前一日期记录联合确认；
5. 不能确认时标记`leaf_match_status=uncertain`，不得强制复用旧ID；
6. 不同植株之间的`L05`没有共享类别语义。

### 2.3 叶片金标准入

进入叶长宽主评价集必须同时满足：

- `leaf_match_status=confirmed`；
- `visibility_ratio>=0.60`；
- 基部与叶尖可识别；
- 测量协议明确；
- 非严重破损；
- 人工值不是零填充或纯图像弱标签。

## 3. 公开数据身份映射

### 3.1 MuST-C

```text
site_id      = MUSTC_CKA
plot_id      = 官方plot编号
plant_id     = NA，除非文件级证据明确提供逐株身份
plant_date_id= NA
sequence_id  = crop + plot + date + platform + sensor_run
group_key    = site + plot
```

同一plot的不同日期和不同传感器在主泛化划分中保持同组。

### 3.2 Wheat3DGS

```text
site_id      = W3DGS_ETH
plot_id      = plot_461 ... plot_467
plant_id     = NA
sequence_id  = plot_id + capture_batch
group_key    = plot_id
```

同一plot的36张图像、COLMAP相机、mask及后续点云不得拆分到不同集合。

### 3.3 3D Plant View

```text
site_id      = 3DPVS_UON
plot_id      = controlled_environment
plant_id     = 官方plant/scene目录
plant_date_id= plant_id + 官方date目录
sequence_id  = plant_date_id + capture_run
group_key    = plant_id
```

主划分按物理`plant_id`，同一植株的不同日期不得跨train/validation/test。视角留出只用于几何/NVS子实验，不能当作表型泛化测试。

### 3.4 TERRA-REF

```text
site_id      = TERRA_MARICOPA
plot_id      = 官方range/column或plot名称
plant_id     = NA，除非明确提供逐株对应
sequence_id  = season + plot + date + sensor
group_key    = season + plot
```

相同plot跨传感器、跨相邻日期保持同组。跨season测试作为额外泛化设置。

## 4. 正式数据划分

### 4.1 自采主划分

最低层级：`plot_id`分组。推荐：

- train：约60%独立plot；
- validation：约20%独立plot；
- test：约20%独立plot。

在样本规模允许时，额外设置一个完全独立日期、地点或平台测试集。

比例可以因plot数量调整，但以下规则不可改变：

- 同一物理植株所有日期同组；
- 同一sequence所有帧同组；
- 相邻且空间重叠的行段窗口同组；
- 同一原始扫描的裁剪块同组；
- 派生mask、深度、点图和伪标签继承原样本split。

### 4.2 公开数据划分

公开数据主要用于几何与外部验证，不与自采表型主测试集合并计算一个总分。

- MuST-C：按plot或field block划分；
- Wheat3DGS：按plot划分；7个plot样本较少时使用leave-one-plot-out；
- 3D Plant View：按plant划分；
- TERRA-REF：按plot划分，跨season为额外测试。

### 4.3 测试集冻结

测试集一经生成，应保存：

- split配置版本；
- group key列表；
- 生成脚本版本；
- 样本清单SHA256；
- 冻结日期。

测试集不得用于：

- 选择Metric Depth模型；
- 调整MSAM阈值；
- 选择LoRA rank；
- 选择可见性门槛；
- 早停或学习率选择；
- 根据人工表型拟合测试序列尺度。

## 5. 时间与空间去重

满足任一条件视为潜在近重复样本：

- 同一视频相隔少于规定时间且视场重叠高；
- 相机中心距离小于一个冠层宽度且观察同一对象；
- 图像感知哈希或局部特征高度相似；
- 同一plot同一日期被不同派生流程重复导出；
- 同一原始点云被不同裁剪边界切出高度重叠块。

潜在重复样本必须共享`duplicate_group_id`并进入同一split。

## 6. 标签与尺度泄漏禁止清单

禁止：

1. 使用测试植株人工株高拟合该植株的尺度；
2. 使用测试plot TLS尺寸拟合后再评价同一plot几何；
3. 将同一植株不同视角随机拆分；
4. 通过文件名中的品种、处理或真值编码向模型泄漏标签；
5. 在全数据上计算标准化均值和方差；
6. 在测试集上选择P95/P99或过滤阈值；
7. 将公开数据官方测试视角加入训练后继续报告其官方测试结果。

