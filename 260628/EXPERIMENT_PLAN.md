# 交通边级意图图实验方案

建议最小实验由三部分组成：真实 PeMS 数据上的预测效果与图贡献、已知动态依赖的合成数据上的结构识别和跟踪、一个可靠的外部图学习基线。现有定义是“当前历史窗口中，对未来交通状态具有预测贡献的有向边强度”；它不是驾驶员行为类别，也不是因果效应。该定义见根目录 `New Text Document.md`。真实流量 CSV 没有边级意图真值，因此不能直接计算真实意图识别准确率。

## 工程与重跑

使用 `260628/main.py` 和 `260628/deig`。`pems_spatial_kmeans_topK` 只存数据；`260731` 是后续融合与部署模块，不作为当前意图图训练入口。

本次保留原模型和损失，补充显式数据路径、步长、窗口、预测长度、stride、短程数据上限；时间戳按分钟生成，5 分钟周期残差使用星期与日内 5 分钟位置。CSV 列顺序必须与坐标 node_id 顺序一致。默认数据为现有的 1 小时文件，避免旧入口引用已不存在的文件名。

在 PowerShell 中先进入项目根目录：

```powershell
Set-Location 'D:\Tjnu-p\Mp\Intention Recognition'
```

新数据短程验证命令，读取前两天，不能用于论文结论：

```powershell
.\.venv\Scripts\python.exe -u .\260628\main.py --flow-csv .\pems_spatial_kmeans_topK\id_standardize_Env_A_timeseries_5m.csv --step-minutes 5 --window 3 --horizon 1 --stride 1 --max-steps 576 --epochs 2 --viz-count 10 --out .\output\intent_5m_smoke
```

覆盖完整 212 天数据的旧模型重跑命令，每小时取一个窗口，历史仍是 3 个 5 分钟采样值，预测下一步。stride 只改变窗口抽样，不能解释为小时聚合。图覆盖所有被抽样的有效窗口，不覆盖每个 5 分钟时刻。

```powershell
.\.venv\Scripts\python.exe -u .\260628\main.py --flow-csv .\pems_spatial_kmeans_topK\id_standardize_Env_A_timeseries_5m.csv --step-minutes 5 --window 3 --horizon 1 --stride 12 --epochs 2 --viz-count 10 --out .\output\intent_5m_full_stride12
```

旧 1 小时数据复跑：

```powershell
.\.venv\Scripts\python.exe -u .\260628\main.py --step-minutes 60 --window 3 --horizon 1 --stride 1 --epochs 2 --viz-count 10 --out .\output\intent_1h_rerun
```

正式 5 分钟设置建议 W=12（12 个历史采样值），H=3（预测未来 5、10、15 分钟），辅助 H=12（60 分钟）；先统一 stride=12 做开发，最终测试每 5 分钟窗口。若严格保持旧实验的 3 小时历史、1 小时未来，则采用 W=36、H=12。H=12 预测完整未来序列，不等价于只预测第 60 分钟一个值。

**正式长窗口全量运行前，应先把验证、测试及特征构建改成批处理或惰性窗口。** 旧入口把整个验证/测试集一次拼接，长窗口全量运行内存风险明显。上面已经运行的设置用于旧流程验证，不是充分训练的最终比较；当前默认仅两轮，也没有验证集最佳模型选择。

## 当前实现需要先明确的问题

本次已完成完整 212 天、stride=12 的重跑，目录 `output/intent_5m_full_stride12/20261011_221736`。读取 61,056 步、100 节点，候选边 488，训练/验证/测试窗口为 3053/1018/1017；保存全部 5088 个抽样窗口得分与阈值图，10 个测试窗口图。已验证得分有限、导出形状正确、相邻图时间间隔为 60 分钟。结果如下，单位为 Z：

| 检查 | MAE |
|---|---:|
| 完整旧模型 | 0.0848643354 |
| 持续性预测 | 0.0779975028 |
| 删除全测试排序最高的 10% 分数 | 0.0872401589 |
| 删除相同数量随机分数 | 0.0849388286 |
| 删除最低分数 | 0.0848643354 |

模型比持续性 MAE 高约 8.8%。删高分边有贡献迹象，但不足以认定设计有效。首张测试图色条达到 1e-10 量级；所有导出分数范围为 0 至 0.03109，提示部分窗口绝对分数明显收缩，需检查入边归一化与稀疏正则的耦合。训练曾出现 sigmoid 的 exp 溢出警告，最终导出有限；正式训练应使用稳定 sigmoid 并监控梯度与分数分布。两轮训练结果不能作为最终论文结论。短程两天验证另保存于 `output/intent_5m_smoke/20261011_221629`。

1. W=3 时 `_windowed_lag_corr` 的最小有效长度条件使正反方向滞后相关均为零。正式 W 至少为 5，建议 12；候选图的全训练段滞后相关仍正常计算。
2. 代码只实现预测 Huber、稀疏项、双向乘积惩罚。文档中的时间平滑和空间正则没有实际进入当前损失，不能声称已经完成这些消融。
3. 每窗口 90% 分位数阈值通常保留约 10% 候选边，因此图密度近乎恒定是构造结果，不能验证热点发现；并列分数甚至可能保留更多边。主评价使用固定预算 Top K，辅助扫描 5%、10%、20%。若需判断无事件时刻，另在验证集设定绝对阈值，允许空图。
4. 预测器按入边归一化分数，缩放一组入边可能几乎不改变预测，而稀疏项会推动绝对分数变小。sigmoid 分数不是校准概率，不宜使用 Brier/ECE 或将分数大小直接解释为实际效应。
5. 删边目前只关掉意图消息传播的权重，GAT 编码器仍使用候选图。结果应称为“意图消息通道贡献”，不能称为完全移除网络连接的效果。需要另做编码器和消息通道同时屏蔽的结构干预作为辅助验证。
6. 现有删边实验把所有测试窗口的边分数合并排序，预算不按窗口控制；正式评价应每个窗口取相同 K，随机对照重复至少 20 次。
7. 正式评估需要原始缺失 mask，排除被插值填补的预测目标；历史插值不得用未来观测。当前 loader 不能还原已预处理 CSV 的原始缺失情况。

## 真实数据上的主要指标

设历史输入截止 t，目标 Y_t=(Z_{t+1},...,Z_{t+H})。定义窗口损失

$
L_t(\hat Y)=\frac{1}{NH}\sum_{j=1}^N\sum_{h=1}^H|Z_{j,t+h}-\hat Z_{j,t+h}|.
$

所有公式都是本实验对交通回归任务的明确适配；参考文献提供评价思路，不表示文献原文采用同一公式。真实数据指标应命名为预测贡献或忠实度。

### 预测误差与相对改进

$
MAE=\frac1M\sum_t L_t,\qquad RMSE=\sqrt{\frac{\sum_{t,j,h}(Z_{j,t+h}-\hat Z_{j,t+h})^2}{MNH}}.
$

$
Skill=1-\frac{MAE_{model}}{MAE_{persistence}+\epsilon}.
$

持续性预测为 $\hat Z_{j,t+h}=Z_{j,t}$。Skill>0 表示胜过持续性。另报告各 horizon MAE。当前 CSV 已标准化，结果单位为再次训练段标准化后的 Z 单位；没有原始尺度与转换链不能称为车辆数 MAE。数据接近零或包含负标准化值，MAPE 不适合作为主指标。

### 删除贡献与保留充分性

固定训练完成的模型，不再训练。对每个窗口选择 Top K 边集合 Q_t，令 S_t 为原始边分数。

$
D_K=\frac1M\sum_t\left[L_t(f(X_t,S_t\odot1_{e\notin Q_t}))-L_t(f(X_t,S_t))\right].
$

D_K 越大，删除这些边损失越明显。负值必须保留，说明删除后改善预测。比较同预算高分边、低分边、随机边；主要报告随机校正删除贡献

$
D_K^{excess}=D_K^{top}-\frac1B\sum_{b=1}^B D_K^{random,b},\quad B=20.
$

$
S_K=\frac1M\sum_t\left[L_t(f(X_t,S_t\odot1_{e\in Q_t}))-L_t(f(X_t,S_t))\right].
$

S_K 越接近零说明少量边能保留全图预测能力；同时保留负值。报告 K 占候选边 5%、10%、20% 时的曲线，避免用一次选阈值后的单点证明合理性。改动分数后重新按接收节点归一化，不对结构干预模型重训。此设计借鉴图解释评价中 necessity/sufficiency 与 fidelity 的区分：[Amara 等，GraphFramEx，LoG 2022](https://proceedings.mlr.press/v198/amara22a.html)。本方案是面向 MAE 的适配，不是原论文分类 fidelity 的直接复刻。

### 单边排序一致性

在预先抽取的测试窗口上，计算每条候选边的模型内删除贡献

$
c_e^t=L_t(f(X_t,S_t^{(-e)}))-L_t(f(X_t,S_t)),\qquad
\rho_t=Spearman(s_e^t,c_e^t).
$

报告窗口均值及按天重采样置信区间。对原始分数与实际入边归一化权重分别计算 rho；遇到常量向量记 NA。该指标衡量模型内一致性，不是独立真值，也不具有因果识别含义。单边交互和替代路径可能降低 rho，即使 Top K 整体有用，因此作为辅助指标。

## 有真值时的识别和跟踪指标

### 动态合成数据

构造稳定的切换有向 VAR 或可控传播过程：

$
x_t=\sum_{\ell=1}^{L}A_{t,\ell}x_{t-\ell}+b_{slot(t)}+\eta_t.
$

仅用过去数据生成未来，控制系数与噪声保证序列稳定；指定若干时段的边产生、消退、反向与传播延迟。需要把外生共同周期与真实传播区分。静态全天存在的链不能作为动态跟踪真值。首先 L=1、H=1，定义真值有向边集 $E_t^*=support(A_{t+1,1})$，对应输入截止 t 和预测目标 t+1；H>1 时路径和未来机制变化会影响真值，应另明确直接边或多步有效依赖，不混用。

由于模型的候选图会限制可恢复边，同时报告候选覆盖率

$
Coverage=\frac{\sum_t|E_t^*\cap E_0|}{\sum_t|E_t^*|}.
$

主结果在全部 N(N-1) 有向边上评价，非候选边分数置零并保留并列；辅助结果在候选边内部评价。否则候选图漏掉的正边会被隐藏。H=1 的机制支持真值是可控依赖代理，仍不表示真实交通意图标签。

### 边级 AP 与 Top K

对每个有正例窗口，以边分数排序：

$
P@K_t=\frac{|Q_t\cap E_t^*|}{K},\quad
R@K_t=\frac{|Q_t\cap E_t^*|}{|E_t^*|},\quad
F1@K_t=\frac{2P@K_tR@K_t}{P@K_t+R@K_t}.
$

$
AP_t=\sum_r(R_r-R_{r-1})P_r.
$

AP 按唯一阈值计算，避免人为打破并列导致偏差；AP 与梯形 PR-AUC 不混称。主报 macro AP 和 F1@K；micro AP 辅助报告，注明正例比例。无正例窗口的 AP 记 NA，另报告空场景误激活率；Top K 必然在空场景激活，需绝对阈值协议验证空场景。稀疏正例场景优先 PR 评价的依据：[Saito 与 Rehmsmeier，PLOS ONE 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)。AUROC 可辅助，但不单独据此判断稀疏边识别质量。

### 新增和消退的跟踪

$
\Delta_t^+=E_t^*\setminus E_{t-1}^*,\quad
\Delta_t^-=E_{t-1}^*\setminus E_t^*,
$

预测新增、消退集合以同样差集从预测边集生成，分别汇总 TP、FP、FN，计算 $F1=2TP/(2TP+FP+FN)$。主报新增 F1 和消退 F1，使用固定 5 分钟相邻预测时刻；stride=12 的运行只能评价每小时采样跟踪，不能与逐 5 分钟结果直接比较。

对真值新增边事件起点 a，预先设等待上限 D（建议 3 步即 15 分钟），首次被识别为活动边的时刻 $\hat a$ 定义 delay=$(\hat a-a)\times5$ 分钟。若 a 时已被误激活，则不算新增命中。超过 D 记漏检，单独报告漏检率，不只对成功案例报平均延迟。严格时刻 F1 主报，允许 1 步容差的结果单独列。

若需要评价连续活动区间，以每条有向边的活动时间段作为事件，计算时间 IoU：$IoU(A,B)=|A\cap B|/|A\cup B|$。在同一有向边内按最大总 IoU 一对一匹配，预设 IoU>=0.5 为命中；以匹配事件数 TP、未匹配预测 FP、未匹配真值 FN 计算事件 F1。这是本实验的简化事件协议，灵感来自区间评价，**不称为 Tatbul 原公式**：[Tatbul 等，Precision and Recall for Time Series，NeurIPS 2018](https://papers.nips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)。禁止把一次命中扩展为整个真值区间全部命中的 point adjustment。

## 外部基线选择

| 方法 | 学习结构 | 与本任务的关系 | 优先级 |
|---|---|---|---|
| Graph WaveNet | 节点嵌入学习共享自适应有向依赖矩阵，配合时序卷积 | 可靠的静态任务驱动图基线，验证动态图是否值得引入 | 第一阶段选择 |
| DGCRN | 根据节点属性与动态过滤器逐步生成动态图，并结合静态图 | 对动态交通依赖建模更直接的对照 | 若论文重点声称动态跟踪优势，替代或追加 |
| GTS | 联合预测学习离散图结构 | 可作为学习图结构基线，改造与训练成本较高 | 后续备选 |
| GNNExplainer | 为已有预测模型提取解释子图 | 后验解释而非同等预测驱动建图，适合辅助解释对照 | 不作为首个主基线 |

建议先选 **Graph WaveNet 原始实现**：有交通数据实验、有作者公开代码，静态自适应图加时序卷积与当前目标较贴近，短期成本可控。它是经典可靠基线，不宣称当前最新 SOTA；动态跟踪指标对静态模型会天然不利，所以不能用“动态变化 F1 胜过 Graph WaveNet”单独证明优于先进动态建模方法。若只允许一个基线且核心主张就是动态追踪，优先选 DGCRN。

Graph WaveNet 的自适应矩阵为

$
\tilde A=softmax(ReLU(E_1E_2^T)).
$

不要把该矩阵当作边标签概率。统一源节点与目标节点方向：对官方传播算子检查单边脉冲从 i 是否影响 j，按需要转置，再导出边分数。图比较时选同预算 K；原始预测模型保留其原设计。若将矩阵限制在本项目候选边并接入同一预测器，应命名为“Graph WaveNet 启发的静态图学习变体”，不能冒充原模型复现。

公平比较分两层：外部完整预测模型统一训练/验证/测试时段、标准化、W/H、可用特征和调参预算，报告 MAE/RMSE；图模块比较将各方法提取的图接到相同下游预测器并按相同预算独立训练，报告图的可用性。各模型内部删边忠实度不能仅比绝对 D_K 大小，需要结合原误差、随机校正和固定预算。所有图估计只读历史和训练段，禁止未来目标参与候选图构建。

参考来源：

- Wu 等，Graph WaveNet for Deep Spatial Temporal Graph Modeling，IJCAI 2019：[论文](https://arxiv.org/abs/1906.00121)，[作者实现](https://github.com/nnzhan/Graph-WaveNet)。
- Li 等，Dynamic Graph Convolutional Recurrent Network for Traffic Prediction Benchmark and Solution，预印本 2021，TKDD 2023：[论文](https://arxiv.org/abs/2104.14917)，[作者实现](https://github.com/LakeChuan/DGCRN)。
- Shang 等，Discrete Graph Structure Learning for Forecasting Multiple Time Series，ICLR 2021：[论文](https://arxiv.org/abs/2101.06861)，[作者实现](https://github.com/chaoshangcs/GTS)。
- Ying 等，GNNExplainer Generating Explanations for Graph Neural Networks，NeurIPS 2019：[论文](https://papers.nips.cc/paper_files/paper/2019/file/d80b7040b773199015de6d3b4293c8ff-Paper.pdf)。

## 最小实验与局部消融

第一阶段真实数据比较完整模型、Graph WaveNet、持续性预测；主表报告各 horizon MAE/RMSE/Skill，图贡献图报告 3 个 K 的随机校正删除贡献和保留充分性。第二阶段小型切换合成网络报告 candidate coverage、AP、F1@K、新增/消退 F1 和延迟漏检率。合成数据只验证可控依赖恢复能力，不替代真实流量结果。

优先消融两项：把训练段平均边分数冻结为静态图，验证动态建图贡献；去掉动态边特征，验证显式方向与趋势信息。两者均重训，保持其余训练预算一致。其次去掉双向惩罚、去掉稀疏项，分别重训；因为入边归一化与分位数阈值会掩盖绝对缩放，额外报告归一化权重集中度，不能只看密度。删边属于固定模型的干预检查，与重训消融分开。

至少 3 个随机种子，验证集选择最佳 checkpoint，给相同训练上限和早停规则。测试指标按天配对 block bootstrap 计算 95% 区间，避免把重叠窗口当独立样本。峰时、非峰时采用预先定义时间段，不能用模型高分时段挑选效果。每个实验保存配置、数据范围、种子、候选边、原始分数、预测目标时间及预测值。

验收条件：模型稳定胜过持续性；高分删边贡献在多个预算下稳定超过随机；少量高分边能保留较多预测能力；已知动态网络能识别有向边变化；动态版优于静态版且不是由更多参数或预算造成。若仅部分成立，应缩小方法结论，不用漂亮图替代验证。
