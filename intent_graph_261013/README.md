# 意图图基本单元验证框架

本目录用于回答一个具体问题：在相同数据、目标窗口与评价方法下，空间消息、静态边门控和动态边门控分别提供多少增益？首先建立可信的预测基线，再验证意图图的基本单元，随后逐项增加模型部件与损失项。

M1–M7 构建实验现已实现，独立入口为 run_construction.py／run_construction.ps1，配置为 configs/experiments_construction.json。原入口仍默认六模型配置；新入口默认 B1–B6 加 M1–M7，M0 与 B6 共用 DynamicGate。每个神经训练 epoch 都打印损失、验证指标和早停状态，构建实验默认自动保存相邻测试窗口图片。正式运行与参数位置见下方构建文档。

当前模型组包括 Persistence、Ridge-AR、Local、FixedGraph、StaticGate 和 DynamicGate。Persistence 无需拟合；Ridge-AR 用带正则的线性历史模型建立主要预测基线；后四项共用节点历史编码器、消息投影和预测头，以边门控方式区分模型。

默认试验使用最近 12 个五分钟观测，直接预测未来 3 步。默认节点特征只有流量历史，默认损失只有预测损失。动态图分数首先称为“边门控分数”；是否能够反映预测贡献，需要通过删边、留边与随机对照检查。

## 文档导航

| 文档 | 主要内容 |
|---|---|
| [数据与实验协议](docs/01_data_protocol.md) | 时间切分、缺失与因果填补、标准化、窗口、候选图、协议一致性 |
| [基线与初级模型](docs/02_baselines_and_minimal_model.md) | 六个模型的数学定义、消息与门控、残差预测、可选 TCN |
| [指标与边干预](docs/03_metrics_and_interventions.md) | 完整目标评价、MAE/RMSE/Skill、Top/Low/Random、动态诊断 |
| [参数与逐级实验](docs/04_parameters_and_experiment_ladder.md) | 默认参数、损失项、种子、调参和模型丰富顺序 |
| [代码组织与产物](docs/05_architecture_and_outputs.md) | 模块责任、可复用来源、统一接口、运行产物与可比较性 |
| [GPU 与验证约定](docs/06_gpu_and_verification.md) | 自动选择设备、显存、可复现边界与验证证据 |
| [实际验证结果](docs/07_verification_results.md) | 已完成的小型六模型运行、误差与检查范围 |
| [三分区保存与结果汇总](docs/08_result_summaries.md) | 训练／验证／测试完整产物、六模型指标汇总与适用范围 |
| [连续窗口补绘与训练接入](docs/09_consecutive_plotting.md) | 独立批量补绘、筛边、命名与参数、可选训练接入补丁 |
| [基线与消融方案审查](docs/10_baseline_and_ablation_review.md) | B1–B6／M0–M7 的公平性、双空间路径及损失设计 |
| [指标审查与旧论文对应](docs/11_metrics_review_and_legacy_mapping.md) | 忠实度、动态图稳定性、旧指标适用条件及定义差异 |
| [构建实验实现与正式运行](docs/12_construction_implementation_and_running.md) | B1–B6／M0–M7 实现、入口、epoch、早停、逐轮日志及验证范围 |

这些文档只用公式与文字描述方法，不含程序代码。入口的参数说明由 `run_experiment.py` 自身提供；数据协议和实验配置分别放在 `configs` 中，结果放在 `outputs` 中。

新运行在恢复最佳模型后，分别保存训练、验证、测试的完整预测、有效目标、门控与诊断。组级结果包含预测、各 horizon、边干预、门控诊断和效率汇总，不局限于 MAE、RMSE 两项。默认只在测试集执行昂贵的 Top／Low／20 次 Random 删除与保留；训练和验证仍保存完整预测与门控，未执行的干预明确记为未计算。

阈值始终由验证集或预先指定参数确定，同一模型的三个分区共用冻结阈值。训练集结果由最佳模型重新推断得到，区别于训练过程中各批次的损失。历史运行没有保存的训练、验证数组不会由新汇总自动补出来；[实际验证结果](docs/07_verification_results.md) 保留当时测试集运行的真实记录。

基础数据、线性模型和图表依赖 NumPy、Pandas、scikit-learn 与 Matplotlib；四个神经模型另外需要 PyTorch。PyTorch 按所选模型加载，因此 Persistence 和 Ridge-AR 可以在未安装 PyTorch 的环境中独立运行。神经模型默认自动选择设备：CUDA 可用时使用 GPU，否则使用 CPU；当前目标设备是 NVIDIA RTX 4060 Ti，显存 8 GB。数据预处理、Persistence 和 Ridge-AR 仍在 CPU 上运行。

运行环境是否具备依赖应由入口检查，不能把缺少库误判为模型算法失败。GPU 改变计算位置和速度，不改变预测目标、模型公式或评价标准；CPU 与 CUDA 的浮点结果不保证逐位相同。

已核验的现有环境是 `C:/Users/123/AppData/Local/Programs/Python/Python310/python.exe`，Python 3.10，PyTorch 2.7.1+cu118，CUDA 运行库 11.8，可使用 RTX 4060 Ti。无需为本框架重新安装 PyTorch。六模型、两轮神经训练的小型端到端运行已完成，结果见实际验证文档；该运行用于验证流程，不能作为动态门控有效的正式结论。

## 判断结果的顺序

1. Persistence 和 Ridge-AR 确定同一测试范围上的基本误差水平。
2. Local 与 Ridge-AR 比较，检查非线性时间编码是否有价值。
3. FixedGraph 与 Local 比较，检查候选图邻居消息是否有价值。
4. StaticGate 与 FixedGraph 比较，检查学习边权是否有价值。
5. DynamicGate 与 StaticGate 比较，检查窗口相关边权是否有价值。
6. 对门控模型检查高分边是否比同预算随机边更影响模型预测。

如果某一步没有稳定增益，结论应定位到这一基本单元，不应通过继续增加复杂损失来掩盖。预测增益、排序忠实度与真实结构识别是不同证据；本版真实交通数据没有边级真值，因此不能凭预测误差或删边效果宣称已经恢复真实因果结构。

## 与原框架的关系

新实验在 `intent_graph_261013` 内组织，沿用 `260628` 的候选图和交通图展示思路，数据与评价改为统一实验协议。历史运行目录仍可作为诊断材料；不同时间范围、窗口或缺失规则的结果不能直接与新结果混合排名。

此前旧运行的持续性误差存在只统计最后一个批次的问题。因此，以往模型误差与该数值的比较不能作为新的基线结论；本框架在全部相同的有效目标上重算持续性预测。

固定候选入度归一化让门控减小时消息确实减小，消除了原“门控和归一化分母同比缩小”的直接退化。但消息投影与预测头仍可能补偿门控尺度，所以门控不是天然校准的绝对贡献量。第一阶段优先检验预测和边排序，绝对阈值图只作探索性产物。
