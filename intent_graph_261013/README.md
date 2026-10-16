# 意图图实验框架：目录、文档与运行手册

本目录用于在统一数据、预测目标和评估体系下，比较交通流预测模型及其边级意图图。默认历史窗口为12个五分钟观测，预测未来5、10、15分钟。所有正式对照都需要核对协议、训练设置和评估设置，而不只是模型名称。

**当前主要使用两个入口：增量构建用 run_construction.ps1；严格消融用 run_ablations.ps1。run.ps1 默认运行早期六模型，但它对应的 run_experiment.py 仍是整个项目的公共命令行入口，不是废弃代码。**

## 导航

- [1. 三个训练入口与默认实验](#1-三个训练入口与默认实验)
- [2. 目录结构与源文件职责](#2-目录结构与源文件职责)
- [3. 文档阅读地图](#3-文档阅读地图)
- [4. 配置文件及优先级](#4-配置文件及优先级)
- [5. 可直接复制的训练命令](#5-可直接复制的训练命令)
- [6. 模型名称与别名](#6-模型名称与别名)
- [7. 运行参数与训练日志](#7-运行参数与训练日志)
- [8. 输出目录和结果文件](#8-输出目录和结果文件)
- [9. 离线补绘与跨实验汇总](#9-离线补绘与跨实验汇总)
- [10. 历史文件、兼容文件与不再需要的步骤](#10-历史文件兼容文件与不再需要的步骤)
- [11. 常见问题与验证](#11-常见问题与验证)

## 1. 三个训练入口与默认实验

| Python 入口 | PowerShell 入口 | 默认数据协议 | 默认模型配置 | 不传 models 时运行什么 |
|---|---|---|---|---|
| run_experiment.py | run.ps1 | protocol_dev.json | experiments.json | 旧六模型：Persistence、Ridge、Local、FixedGraph、StaticGate、DynamicGate |
| run_construction.py | run_construction.ps1 | protocol_construction_dev.json | experiments_construction.json | B1–B6、M1–M7、M5_ATTENTION，共14个不同模型 |
| run_ablations.py | run_ablations.ps1 | protocol_construction_dev.json | experiments_ablation.json | 以M7-L0为母模型的严格结构/意图机制消融，共11个不同模型 |

上表中的 JSON 均位于 configs/。按当前默认种子0、1、2计算：旧六模型共14次拟合/评估；构建组共38次；严格消融共33次。Persistence和Ridge仅运行一次，神经模型运行三个种子。

三者不是三套独立训练代码，实际调用关系是：

~~~text
run.ps1 ──────────────→ run_experiment.py ───────────────┐
run_construction.ps1 ─→ run_construction.py ─→ 公共CLI ──┤
run_ablations.ps1 ────→ run_ablations.py ────→ 公共CLI ──┤
                                                       ↓
                                        experiment/runner.py
                                                       ↓
                              数据 → 模型 → 训练 → 三分区评估 → 保存/汇总
~~~

run_construction.py 的主要作用是替换默认 protocol 和 experiments；run_ablations.py 除了替换默认文件，还向公共入口传入 strict_ablation=True，检查母模型配置一致、图正则为零等条件。**单纯给 run_experiment.py 换成消融 JSON，并不等于启用了严格入口的配置检查。**

当前配置默认值：

| 设置 | 旧六模型 | 增量构建 | 严格消融 |
|---|---:|---:|---:|
| 最大epoch | 20 | 100 | 100 |
| patience | 5 | 15 | 15 |
| 训练batch | 8 | 32 | 4 |
| 评估batch | 128 | 8 | 8 |
| 默认数据 | 前30天，X1协议 | 前30天，准备X7后按模型选择 | 前30天，X7协议 |
| 连续窗口图 | 默认未开启，可手动开启 | 默认测试集前10个窗口 | 默认适用模型测试集前10个窗口 |

旧六模型与新构建协议虽都可能使用前30天，但特征准备及协议版本不同。为了与现有M系列联合对比，**补跑B1–B6应优先使用 run_construction.ps1 并选择B模型**，不要直接把旧入口默认结果混入新协议表格。

## 2. 目录结构与源文件职责

~~~text
intent_graph_261013/
├── README.md                       本手册
├── run_experiment.py / run.ps1      公共CLI；默认旧六模型
├── run_construction.py / .ps1       增量构建入口
├── run_ablations.py / .ps1          严格消融入口
├── main.py                         旧调用兼容入口
├── compare_runs.py                 现有结果联合比较
├── plot_intent_sequences.py         独立连续意图图补绘
├── requirements.txt                Python依赖声明
├── configs/                        数据、模型/训练/评估、绘图配置
├── docs/                           数学定义、设计、审查及实现说明
├── experiment/                     实验核心代码
│   ├── models/                     基线和DEIG模型
│   └── 其它模块                    数据、训练、评估、保存等
├── integration/                    已集成补丁的历史档案
├── tests/                          行为及回归测试
└── outputs/                        时间戳suite及各模型结果
~~~

### 2.1 核心实验模块

| 文件 | 负责内容 | 要修改什么时看 |
|---|---|---|
| experiment/protocol.py | 协议读取、路径解析、参数校验、身份哈希 | 数据协议规则，不是模型参数 |
| experiment/data.py | 读取数据、缺失处理、训练统计量、标准化、特征、分区窗口 | 数据预处理；避免未来信息泄漏 |
| experiment/dataset.py | 按窗口取batch、目标mask、相邻历史窗口 | 采样、batch和时间平滑输入 |
| experiment/graph.py | 候选有向图及静态边属性 | 候选边、距离/关联等先验 |
| experiment/model_inputs.py | 同一已准备数据上的X1/X7视图 | 不重做目标/切分的模型输入选择 |
| experiment/runner.py | 模型×种子调度、三分区评估、保存与增量汇总 | 实验流程和输出时机 |
| experiment/ablation.py | 配置继承解析、严格M7-L0检查、消融角色 | 母模型共享参数和严格性校验 |
| experiment/trainer.py | 优化器、逐epoch日志、梯度裁剪、早停、恢复最佳模型 | 训练过程；常规epoch修改只改JSON/CLI即可 |
| experiment/losses.py | 掩码Huber/MAE和可选稀疏、方向、时间正则 | 损失的数学实现 |
| experiment/evaluation.py | 完整目标预测指标、图诊断、验证阈值、删边/留边 | 指标计算与干预范围 |
| experiment/artifacts.py | 模型结果、NPZ/JSON/CSV及三分区保存 | 结果文件布局和保存内容 |
| experiment/reporting.py | 预测、horizon、图、诊断、效率等suite汇总表 | 报表字段和统计输出 |
| experiment/visualization.py | 常规预测/损失/门控等图 | 仍在使用，不是废弃绘图代码 |
| experiment/sequence_visualization.py | 独立连续意图图加载、筛边、渲染和清单 | Top-K/阈值、连续窗口、样式与命名 |
| experiment/sequence_plot_hook.py | 训练保存完成后的连续图调用 | 自动绘图开关与接入 |
| experiment/__init__.py | 包入口 | 通常不需要改 |

### 2.2 模型模块

| 文件 | 内容 |
|---|---|
| experiment/models/__init__.py | 模型注册、大小写/别名归一化和模型创建 |
| experiment/models/persistence.py | 持续性基线，重复最近观测，无训练 |
| experiment/models/ridge_ar.py | Ridge自回归预测、闭式拟合和验证集选择正则参数 |
| experiment/models/gated_forecaster.py | B3–B6共享骨架；也提供DEIG复用的训练、预测、设备与保存接口，不能视为旧代码删除 |
| experiment/models/deig_tcn.py | M1–M7及严格变体的GAT、TCN、时间注意力、评分与后置消息实现 |
| experiment/models/deig_variants.py | M7母结构和各去组件/替换的规范定义，不依赖PyTorch |

tests/ 中 test_data、test_model_inputs 负责数据；test_models、test_construction_models、test_strict_ablations 负责模型；test_evaluation、test_split_evaluation、test_compare、test_reporting 负责评价与输出；test_construction_runner、test_ablation_runner 负责集成；test_sequence_visualization、test_optional_sequence_integration 负责绘图与接入。“optional”指开关可选，不表示尚未集成。

## 3. 文档阅读地图

| 文档 | 内容和用途 | 当前定位 |
|---|---|---|
| [01 数据协议](docs/01_data_protocol.md) | 切分、缺失、标准化、候选图与共享目标 | 基础规则 |
| [02 基线与初级模型](docs/02_baselines_and_minimal_model.md) | 六模型的数学形式、门控、残差预测 | B1–B6背景 |
| [03 指标与边干预](docs/03_metrics_and_interventions.md) | MAE/RMSE/Skill、Top/Low/Random删边/留边 | 指标定义 |
| [04 参数与逐级实验](docs/04_parameters_and_experiment_ladder.md) | 初期参数和逐级丰富路线 | 基础/历史阶段说明；默认数值以当前JSON为准 |
| [05 架构与产物](docs/05_architecture_and_outputs.md) | 模块职责、结果关系和可比性 | 与本README配合 |
| [06 GPU与验证](docs/06_gpu_and_verification.md) | 设备、复现、资源与验证边界 | 环境/运行背景 |
| [07 实际验证记录](docs/07_verification_results.md) | 早期六模型的小型验证 | 历史证据，不是当前正式消融结论 |
| [08 三分区与汇总](docs/08_result_summaries.md) | train/val/test、字段、种子统计 | 读输出重点 |
| [09 连续补绘](docs/09_consecutive_plotting.md) | 独立绘图、筛边、命名、历史接入方案 | 接入已完成，不再应用旧patch |
| [10 方案审查](docs/10_baseline_and_ablation_review.md) | 公平性、双空间路径、损失设计 | 审查背景，不代表所有建议均已实现 |
| [11 指标与旧论文](docs/11_metrics_review_and_legacy_mapping.md) | 忠实度、动态诊断和旧论文指标对应 | 解释边界 |
| [12 构建实现](docs/12_construction_implementation_and_running.md) | M1–M7实现、训练输出、epoch/早停 | 增量构建主说明 |
| [13 严格消融](docs/13_strict_m7_ablations.md) | M5_ATTENTION、M7变体、数学定义、共享参数与验证 | 当前消融主说明 |
| [完整统一设计](docs/DEIG-TCN_基线与消融实验设计_完整统一版.md) | 总体模型矩阵、第6节消融、第7节损失、第8节指标 | 原始设计参考，含计划内容/历史状态字样 |

建议顺序：**运行看本README → 理解当前模型看12/13 → 数据看01 → 结果和指标看03/08/11 → 全局研究设计看统一版。**

数学文档继续用公式和文字描述；本README是操作手册，专门提供命令。设计文档中“拟新增”不一定表示当前未实现；第7节L1–L4整套批处理、可选TCN→GAT顺序实验等也不能因出现在设计里就假定已经实现。

## 4. 配置文件及优先级

### 4.1 数据协议：决定使用什么数据和目标

| 文件（configs/下） | 范围 | 特征/适用入口 |
|---|---|---|
| protocol_smoke.json | 前576点，2天 | 早期X1基线烟测 |
| protocol_dev.json | 前8640点，30天 | 早期X1基线开发 |
| protocol_full.json | CSV全部时刻 | 早期X1基线全量 |
| protocol_construction_smoke.json | 前576点，2天 | 新构建/消融，准备完整X7 |
| protocol_construction_dev.json | 前8640点，30天 | 当前构建/消融默认 |
| protocol_construction_full.json | CSV全部时刻 | 当前构建/消融全量，**不是30天** |

30天指截取源序列前30天，然后按时间切成train/val/test，不是训练集本身有30天。默认切分比例0.6/0.2/0.2，W=12、H=3，训练stride=3、验证/测试stride=1；边界窗口和目标有效性由代码处理。

协议中的 flow_path、coord_path 相对 intent_graph_261013 根目录解析。用户传入的相对 --protocol/--experiments/--out 路径则要注意当前Shell工作目录。因此下面命令统一先进入项目目录。

### 4.2 模型、训练、评价和图片

| 文件 | 内容 |
|---|---|
| experiments.json | 旧六模型和旧默认训练/评估参数 |
| experiments_construction.json | B/M增量构建及M5_ATTENTION，含自动连续图配置 |
| experiments_ablation.json | 严格消融11个方案，变体通过inherits共享M7数值参数 |
| sequence_plots.json | 独立绘图样式配置，供训练入口的 --sequence-plots 使用 |

experiments文件的三个顶层部分：models决定模型名单/结构数值参数；training决定训练、种子、损失；evaluation决定评估及自动图设置。

**覆盖规则：CLI显式参数优先于所选JSON；JSON优先于代码缺省值。** 两个包装入口先填默认 --protocol/--experiments，再接收用户参数，用户仍可显式覆盖。没有 --models 就运行该配置的全部模型；指定 --models 则只跑列出的模型，且必须存在于该配置中。

例如，只给 --experiments 不会自动换数据协议；只给 --protocol 也不会自动换模型组。严格入口不允许混入B模型、M5等非M7严格变体，也不允许非零图正则或不一致的变体数值配置。

## 5. 可直接复制的训练命令

以下是PowerShell命令。先进入目录：

~~~powershell
Set-Location 'C:\Work-lcr\Tjnu-p\Mp\Intention-Recognition\intent_graph_261013'
~~~

三个.ps1均优先使用已有Python 3.10解释器；可通过环境变量INTENT_GRAPH_PYTHON指定其它解释器，不会修改系统全局Python。使用裸python之前，要确认它和.ps1使用的是同一个环境，否则可能没有CUDA。

### 5.1 当前最常用：M7/M6已完成，只补其余9个严格消融

~~~powershell
.\run_ablations.ps1 --models M7_NO_GAT M7_NO_TCN M7_NO_EXTRA_NODE_FEATURES M7_NO_GAT_EDGE M7_NO_SCORE_EDGE M5_ATTENTION M7_NO_INTENT M7_FIXED_INTENT M7_STATIC_INTENT --device cuda --batch-size 4 --epochs 100 --patience 15
~~~

默认前30天、三个种子，共27次训练。该suite的completed只覆盖这9个选中模型，不会自动将旧M7/M6结果复制进来；后续可联合比较。

### 5.2 运行全部严格消融

~~~powershell
.\run_ablations.ps1 --device cuda --batch-size 4 --epochs 100 --patience 15
~~~

默认11个方案×3种子，共33次。已有同名结果不会自动跳过；每次新建suite并从头训练。

### 5.3 只运行M5＋时间注意力

~~~powershell
.\run_ablations.ps1 --models M5_ATTENTION --device cuda --batch-size 4 --epochs 100 --patience 15
~~~

也可以通过构建入口选择M5_ATTENTION。它等价于M7去掉显式动态边特征；不是M6。

### 5.4 运行M0–M7增量构建链

~~~powershell
.\run_construction.ps1 --models M0 M1 M2 M3 M4 M5 M6 M7 --device cuda --batch-size 4 --epochs 100 --patience 15
~~~

8个神经模型×3种子，共24次，不包含额外的M5_ATTENTION。构建入口不传models则运行配置全部14个模型：

~~~powershell
.\run_construction.ps1 --device cuda --batch-size 4 --epochs 100 --patience 15
~~~

### 5.5 在新协议下补跑B1–B6基线

~~~powershell
.\run_construction.ps1 --models B1 B2 B3 B4 B5 B6 --device cuda --batch-size 4 --epochs 100 --patience 15
~~~

Persistence无需训练；Ridge闭式拟合仍在CPU上，不打印epoch。其它神经模型在指定GPU上训练。

### 5.6 使用旧六模型入口

~~~powershell
.\run.ps1 --device cuda --batch-size 4 --epochs 100 --patience 15
~~~

对应Python调用是run_experiment.py。其默认旧协议与新构建协议不是同一个协议身份，不宜与当前M系列不加检查地合并。若只是要补当前基线，优先使用5.5。

### 5.7 只补某个模型的某个种子

~~~powershell
.\run_ablations.ps1 --models M7_NO_TCN --seeds 2 --device cuda --batch-size 4 --epochs 100 --patience 15
~~~

这是一个新run，不是断点续训。之后联合结果时不要将同一模型同配置的同一种子重复计数。

### 5.8 小型功能验证和全量训练

~~~powershell
# 96个时间点、一个种子、一轮；保留1次随机干预检查
.\run_ablations.ps1 --protocol configs/protocol_construction_smoke.json --max-steps 96 --seeds 0 --device cuda --batch-size 4 --epochs 1 --patience 1 --random-seeds 1

# 全部源数据，资源开销可能明显增加；不是30天配置
.\run_ablations.ps1 --protocol configs/protocol_construction_full.json --device cuda --batch-size 4 --epochs 100 --patience 15
~~~

构建入口也接受同样的protocol参数。batch=4主要减少训练时显存，不保证全量预处理、评估和结果数组的CPU内存开销安全。

### 5.9 Python直接调用及帮助

~~~powershell
# 指定与.ps1一致的解释器；仅查看帮助，不启动训练
& 'C:\Users\123\AppData\Local\Programs\Python\Python310\python.exe' -B .\run_experiment.py --help
& 'C:\Users\123\AppData\Local\Programs\Python\Python310\python.exe' -B .\run_construction.py --help
& 'C:\Users\123\AppData\Local\Programs\Python\Python310\python.exe' -B .\run_ablations.py --help
~~~

把--help换成前面对应的训练参数即可直接用Python入口。--help列出的是公共注册模型，**不是所选JSON一定配置了所有模型**；例如旧JSON没有M7。

## 6. 模型名称与别名

### 6.1 基线及增量构建

| 输入名称 | 规范输出目录名 | 含义 |
|---|---|---|
| B1 | persistence | 持续性 |
| B2 | ridge_ar | Ridge自回归 |
| B3 | local | 本地历史编码，无图消息 |
| B4 | fixed_graph | 固定候选图消息 |
| B5 | static_gate | 静态可学习门控 |
| B6、M0 | dynamic_gate | 简化动态门控，同一模型 |
| M1 | m1 | 普通GAT＋历史MLP |
| M2 | m2 | 普通GAT＋TCN末端读取，X1 |
| M3 | m3 | 增加X7节点特征 |
| M4 | m4 | 静态边特征进入评分器 |
| M5 | m5 | 静态边特征也进入GAT |
| M6 | m6 | 增加显式动态边特征，仍用末端读取 |
| M7 | m7 | 再增加时间注意力 |
| M5_ATTENTION | m5_attention | M5＋时间注意力，无显式动态边特征 |

### 6.2 严格消融的11个唯一方案

| 输入名称 | 相对M7的变化 |
|---|---|
| M7 / M7_L0 / M7_DYNAMIC_INTENT | 完整母模型 |
| M7_NO_GAT | 节点局部逐时刻映射替代前置GAT |
| M7_NO_TCN | 逐时刻映射替代TCN，保留时间注意力 |
| M7_NO_EXTRA_NODE_FEATURES | GAT只接收流量，完整历史仍供动态边特征使用 |
| M7_NO_GAT_EDGE | GAT不接收静态边特征 |
| M7_NO_SCORE_EDGE | 评分器不接收静态边特征，GAT仍接收 |
| M5_ATTENTION / M7_NO_DYNAMIC_EDGE | 评分器去掉显式动态边特征 |
| M6 / M7_NO_TEMPORAL_ATTENTION | 时间注意力替换为末端读取 |
| M7_NO_INTENT | 删除后置消息及无监督的评分器 |
| M7_FIXED_INTENT | 后置边权恒为1 |
| M7_STATIC_INTENT | 每条后置边一个可学习静态参数 |

同一行的名称是别名，不要同时放进一条命令。训练入口接受大小写并归一化；离线绘图的--models使用真实输出目录名，如m5_attention，不负责解析M模型别名。

严格消融中的M6角色标识只是结果元数据，表示“去掉时间注意力”，不改变M6计算。母模型相同指结构规范、数据和训练条件一致，各模型独立从头训练；不表示复用训练好的M7检查点，也不保证不同形状网络的共有参数初值逐位相同。

## 7. 运行参数与训练日志

| 参数/配置 | 控制内容 | 注意事项 |
|---|---|---|
| --epochs | 最大训练轮数 | 构建/消融默认100；实际可能早停 |
| --patience | 连续未达到验证改善的轮数 | 默认15；0不是关闭早停，需为正 |
| --min-delta | 最小标准化验证MAE改善 | 默认0 |
| --batch-size | 训练及训练过程中的验证批大小 | 正式三分区评估另用evaluation.batch_size |
| --seeds | 独立训练种子 | 不同于随机删边的种子 |
| --random-seeds | 每个预算随机删边/留边的次数 | 默认配置20个；改变它会改变评估签名 |
| --device | auto/cpu/cuda | cuda不可用时报错，不会悄悄改CPU |
| --max-steps | 覆盖读取数据点数 | 缩短数据改变协议，不可混作30天结果 |
| --skip-interventions | 跳过昂贵删边/留边 | 不等于指标为0；正式忠实度比较一般不跳过 |
| --graph-intervention-splits | 哪些分区执行干预 | 默认test；其它分区仍保存预测与门控 |
| --no-plots | 关闭本次训练后图片 | 数值仍保存，以后可离线补绘 |
| --sequence-plots | 指定训练自动连续图的JSON | 见configs/sequence_plots.json |
| --prepare-only | 仅准备数据/协议产物，不训练 | suite状态为prepared |
| --out | suite输出根目录 | 其下仍新建时间戳目录，不是指定现成suite续跑 |
| --encoder | 旧B3–B6的MLP/TCN替换 | M构建固定结构，严格入口拒绝此覆盖 |
| --lambda-sparse/direction/temporal | 三个图正则权重 | 严格结构消融必须均为0 |

学习率、权重衰减、梯度裁剪、评估batch和模型宽度在JSON修改，没有对应的所有CLI选项。当前神经训练默认Adam、学习率0.001、weight_decay=0、gradient_clip=5。

当前构建和严格消融都使用**标准化有效目标上的Huber损失，delta=1**，三个图正则权重为0；未来三个预测步按有效目标等权参与。早停/最佳模型选择依据是完整验证集标准化MAE，不是测试集，也不是Huber数值本身。

每轮打印prediction、sparse、direction、temporal、total、验证预测损失、验证MAE、best_epoch、未改善轮数与耗时。epoch0是持续性初始化评估；训练结束恢复最佳检查点再评价全部分区。best_epoch=0不等于未执行训练。Ridge和Persistence不使用这套反向传播损失，也没有神经epoch。

## 8. 输出目录和结果文件

~~~text
outputs/suite_时间戳/
├── protocol.json / data_metadata.json / suite_data.npz
├── suite_summary.json
├── experiment_design.json          严格消融入口特有
├── comparison.json / comparison.csv
├── prediction_summary.csv / horizon_summary.csv
├── graph_summary.csv / diagnostics_summary.csv / efficiency_summary.csv
├── report_manifest.json
└── 模型名/seed_种子/
    ├── model.pt / model.npz / model.json
    ├── resolved_config.json / run_summary.json
    ├── training_history.json / split_metrics.json / per_seed_summary.csv
    └── splits/
        ├── train/
        ├── val/
        └── test/
            ├── predictions.npz / edge_scores.npz（适用时）
            ├── graph_evaluation.json / graph_evaluation.npz（适用时）
            ├── resolved_config.json / run_summary.json / training_history.json
            ├── per_window_metrics.csv
            └── figures/
                └── consecutive/绘图时间戳/筛选方法/
~~~

| 结果文件 | 内容 |
|---|---|
| suite_summary.json | 已完成run列表、suite状态；全部选中的模型种子完成后为completed |
| protocol.json / data_metadata.json | 实际协议、时间范围、节点/边/窗口数和数据来源 |
| suite_data.npz | 共享标准化统计、节点、候选边、时间、split窗口等 |
| experiment_design.json | 严格消融母模型、模型角色、共同超参数和训练/评估设置 |
| comparison.json / comparison.csv | 比较详情及按模型/配置签名分组的均值、训练种子标准差 |
| prediction_summary.csv | 每个run、每个分区的整体预测指标 |
| horizon_summary.csv | 未来5/10/15分钟等分预测步指标 |
| graph_summary.csv | 删边/留边、随机对照、全消息关闭等适用指标 |
| diagnostics_summary.csv | 门控分布、密度、阈值、相邻变化等诊断 |
| efficiency_summary.csv | 参数量、epoch、早停、耗时、吞吐、显存等 |
| report_manifest.json | 报表行数、状态说明、空白与标准差含义 |
| run_summary.json | 单模型种子完整摘要，含fit、结构/能力和三分区结果 |
| resolved_config.json | 当次实际配置；已展开inherits，不受后续编辑JSON影响 |
| model.pt / .npz / .json | 分别对应神经、闭式及无训练模型产物，不要混用 |
| training_history.json | 每轮训练/验证记录；split中的副本不是对该split重新训练 |
| split_metrics.json / per_seed_summary.csv | 单模型种子的train/val/test汇总 |
| predictions.npz | 完整预测、真实目标、mask、窗口ID、时间和误差等 |
| edge_scores.npz | 未阈值的候选边分数、有效传播权重及绘图所需图信息 |
| graph_evaluation.json/.npz | 图指标及适用的窗口级数值证据 |
| per_window_metrics.csv | 每个窗口整体误差、持续性误差、目标数量和时间 |

同一种子下面多个摘要文件不是多次训练：根目录描述整个run；train/val/test描述同一最佳模型在不同分区的评估。当前默认仅test绘图；三个分区都保存数值，适用时都保存门控。

M7_NO_INTENT/Persistence/Ridge/Local没有有意义的意图边分数，不保存edge_scores.npz；Fixed Intent全1，不做伪造的高低边排序；Static Intent分数跨窗口不变。空白指标可能“不适用/未计算/未定义”，不能填成0。

每个run完成保存后，suite级报告会更新。程序意外退出时，已完成run的结果通常仍可读取；suite的running不代表仍有进程存活。当前并无通用自动断点续跑；模型训练、评估或绘图异常都可能阻止最终completed，须核对日志和run_summary，不只看是否有model.pt。

## 9. 离线补绘与跨实验汇总

### 9.1 连续图补绘，不需要重新训练

独立脚本只读取已经保存的edge_scores.npz，不做模型推理。**必须显式选择suite：脚本默认仍指向历史suite_20261013_221703_583502，不会自动找最新结果。**

下面使用已有GPU烟测suite作为可运行例子；正式分析时替换--suite路径。

~~~powershell
# 所有模型/种子的测试集前10个相邻窗口，Top10%，只输出PNG
& 'C:\Users\123\AppData\Local\Programs\Python\Python310\python.exe' -B .\plot_intent_sequences.py --suite outputs/suite_20261015_181406_727224 --splits test --viz-count 10 --mode topk --fraction 0.1 --formats png

# 只绘制m5_attention的seed0，从第2个保存位置开始取10帧
& 'C:\Users\123\AppData\Local\Programs\Python\Python310\python.exe' -B .\plot_intent_sequences.py --suite outputs/suite_20261015_181406_727224 --models m5_attention --seeds 0 --start 2 --viz-count 10 --mode topk --fraction 0.1 --formats png

# 每窗口90%分位数筛选
& 'C:\Users\123\AppData\Local\Programs\Python\Python310\python.exe' -B .\plot_intent_sequences.py --suite outputs/suite_20261015_181406_727224 --mode window_quantile --q 0.9 --viz-count 10 --formats png

# 使用保存的验证集冻结阈值
& 'C:\Users\123\AppData\Local\Programs\Python\Python310\python.exe' -B .\plot_intent_sequences.py --suite outputs/suite_20261015_181406_727224 --mode saved_threshold --viz-count 10 --formats png

# 只检查哪些run可绘制，不生成图片
& 'C:\Users\123\AppData\Local\Programs\Python\Python310\python.exe' -B .\plot_intent_sequences.py --suite outputs/suite_20261015_181406_727224 --list-only
~~~

start是保存序列的零基位置，不是实际时间戳；count不要超过剩余窗口数量。train若stride=3，连续保存窗口对应15分钟间隔，不应误称逐5分钟。固定图自动画全候选边；无图模型跳过。

--formats可选png/svg/pdf，--panel-count控制拼图帧数，--dpi控制分辨率，--norm选择log/linear，--node-labels选择index/id/none，--out可将新图写到独立目录。图像产生于新的绘图时间戳目录，不修改原始门控。

**训练自动图和离线图参数名称并非完全相同：** JSON是count、panel_count、norm_scale；离线CLI是--viz-count、--panel-count、--norm。--sequence-plots读取样式JSON只适用于训练入口；离线补绘通过自己的CLI选项设置。

### 9.2 合并已有结果，不重新训练

compare_runs.py接收的是**模型/seed目录**，不是suite根目录；要求存在可用的run_summary及预测数组。它校验协议/评估签名和实际目标指纹，对不同训练配置分组，拒绝同一配置下重复神经训练种子。它不会自动把旧结果“认证”为严格对照，也不会补出历史未保存的分区。

~~~powershell
# 可运行例子：比较同一已有30天suite中的M6/M7各seed0
& 'C:\Users\123\AppData\Local\Programs\Python\Python310\python.exe' -B .\compare_runs.py --runs outputs/suite_20261015_090833_339433/m6/seed_0 outputs/suite_20261015_090833_339433/m7/seed_0 --out outputs/combined_m6_m7_seed0.json
~~~

把更多实际run路径追加到--runs后即可跨suite合并；--out结尾使用.json或.csv。重复执行会覆盖这个指定文件，建议为新的比较取新名字。

当前该CLI仅导出指定JSON或分组CSV，**不会自动生成全部五类详细报告或新的completed suite**；完整详细报表由训练runner调用reporting模块生成。比较多个模型时还应人工核对实际epoch预算、batch、损失、种子及代码变更，不能只看协议检查通过。

## 10. 历史文件、兼容文件与不再需要的步骤

本次仅说明状态，不删除或停用任何文件。保留历史文件有助于追溯，不代表应继续把它们作为当前正式入口。

| 文件/目录 | 当前状态 | 应怎样处理 |
|---|---|---|
| main.py | 兼容包装，调用run_experiment.main | 不必作为新入口；旧调用仍可使用 |
| run_experiment.py | **仍是公共CLI**，默认实验较早 | 不可当成废弃文件删除 |
| run.ps1、experiments.json | 早期六模型默认入口/配置 | 需要旧协议复现才直接使用；当前基线对比优先走construction |
| protocol_smoke/dev/full.json | 旧X1协议，仍受支持 | 当前M7消融改用construction系列，不能仅凭同为30天混用 |
| integration/consecutive_training.patch | **已集成的历史补丁档案** | 不要再次应用，不是当前安装步骤 |
| integration/README.md | 补丁迁移背景 | 阅读即可，接入已在artifacts/evaluation等模块中 |
| sequence_plots.json | 可选公共图片样式 | 仍可通过--sequence-plots使用；默认构建/消融取自身JSON的evaluation.sequence_plots |
| visualization.py | 常规图代码，仍调用 | 与sequence_visualization.py分工，不是被后者替代 |
| docs/07及早期审查文档 | 历史验证/设计背景 | 不把早期epoch2烟测当正式实验；不假定所有审查建议都已实施 |
| outputs中的旧suite | 历史数值证据 | 只读对比/补绘，注意数据范围、代码和协议差异 |
| 项目外260628等旧目录 | 原框架/参考来源 | 不属于本入口默认执行路径，本项目不修改其代码 |

当前不需要重新应用绘图补丁，也不需要重建PyTorch环境才能使用新模型。若使用其它解释器，先验证依赖/CUDA。requirements.txt声明依赖范围，不等于当前环境的精确版本锁文件；不要为更新文档盲目重装。

## 11. 常见问题与验证

**为什么命令写100，结果actual_epochs不是100？**

100是上限，patience允许早停；best_epoch表示最佳检查点。训练记录是当时保存的实际配置，不随之后编辑JSON变化。

**为什么模型已经跑过又训练？**

三个入口都不会扫描历史suite自动跳过，也没有--resume选项。使用--models和--seeds只选未完成部分。

**为什么只改batch仍可能内存不足？**

训练batch影响显存，但准备全量X7、保存全部预测/门控、图干预仍有CPU内存和评估开销。full协议读取所有CSV行，并非30天。

**为什么没有意图图/某些图指标空白？**

检查模型是否支持边分数、是否--no-plots/--skip-interventions、选择的split、阈值和剩余窗口。无后置意图消息模型不输出随机未训练图。

**能否把本次9个消融的completed理解为完整11个都重新训练了？**

不能。本次completed只表示选中9个模型和指定种子全部结束；M7/M6旧结果需要另外联合，并核对实验条件。

**如何只准备数据？**

~~~powershell
.\run_ablations.ps1 --prepare-only
~~~

这会新建prepared状态的suite，不训练。仍会读取/预处理所选协议的数据，不是零开销的帮助命令。

**如何运行测试？**

~~~powershell
& 'C:\Users\123\AppData\Local\Programs\Python\Python310\python.exe' -B -m unittest discover -s tests
~~~

截至2026-10-15，严格消融代码验证记录为93项通过；实际CUDA小跑suite_20261015_181406_727224包含11个模型各1种子、1轮、真实数据前96个时刻，已完成三分区保存、适用的边干预和10帧连续图。该记录只证明功能可运行，不是正式30天实验效果结论。

最后，预测改善、边排序忠实度和真实结构恢复是不同证据。当前干预只针对后置消息，GAT仍可能传递空间信息；没有真实边标签时，不应把预测误差或意图图直接解释为已识别真实因果关系。
