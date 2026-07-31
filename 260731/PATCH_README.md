# RA-TCGF 时间语义、范围控制与平铺输出补丁 v4

本版本建立在“严格时间对齐 + 原绘图风格”版本上，解决以下问题：

1. 图标题同时出现 `deployment` 与 `prediction target`，容易误解图的时刻；
2. 派生视图只有“最后 N 步”，缺少显式起止时间；
3. 部署缺少显式起止时间；
4. `deployment_steps` 为每个时间步创建子目录，不便按时间浏览；
5. 输出数量改变后，固定目录名 `views_97steps` 容易造成误解。

## 时间语义

- 网络输入窗口：`[t-L+1, ..., t]`；
- 融合矩阵：`S_D^t`，对应时刻 `t`；
- 部署子图：`G_D^t`，对应时刻 `t`；
- 预测标签：`X^{t+1}`，仅用于训练损失和结果审计，不是图的时刻。

因此，融合图和部署图标题只显示 `time t` / `deployment time t`。
`prediction_target_time` 仍保留在 CSV、JSON 和 NPZ 中，但不再显示在图上。

## 派生视图设置

在 `config.py -> ViewExportConfig` 中设置：

```python
num_steps = 97
start_time = None
end_time = None
step_stride = 1
selection = "latest"
```

命令行参数：

```text
--views-steps / --show-steps
--views-start-time
--views-end-time
--views-stride
--views-selection latest|earliest|uniform
```

## 部署设置

在 `config.py -> DeployConfig` 中设置：

```python
num_steps = 24
start_time = None
end_time = None
step_stride = 1
selection = "latest"
source_split = "val"
```

命令行参数：

```text
--deploy-steps
--deploy-start-time
--deploy-end-time
--deploy-stride
--deploy-selection latest|earliest|uniform
--deploy-split train|val|all
```

筛选顺序：

```text
split 候选 -> 起止时间闭区间 -> stride -> selection -> num_steps
```

## 平铺部署输出

所有部署时间步的图片直接保存到：

```text
figures/deployment_steps/
```

不再创建时间步子目录。文件名以原始时间位置和真实时间开头，例如：

```text
pos000144_20250107T000000_c0001_seq000_deployment_subgraph_<run_ts>.png
pos000144_20250107T000000_c0001_seq000_fused_SD_<run_ts>.png
```

因此按文件名排序即可按时间浏览。

## 派生视图目录

固定名称 `views_97steps` 已改为更准确的：

```text
views_steps/
```

实际导出数量由 `cfg.views.num_steps` 控制。
各视图仍按原有类别子目录保存，时间信息放在文件名开头。
