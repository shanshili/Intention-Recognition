# 意图图评价指标与基线方法报告

## 1. 实验对象与统一符号

本项目把交通意图定义为：在当前可获得的历史交通观测下，某条有向边对未来交通状态预测的任务相关贡献。它不是驾驶员意图类别，也不是已经证明的物理因果效应。

设：

| 符号 | 含义 |
|---|---|
| $N$ | 传感器数量，本数据为 100 |
| $T$ | 时间步总数 |
| $\Delta$ | 采样间隔，本数据为 5 分钟 |
| $x_i^t$ | 节点 (i) 在时间步 (t) 的交通流量 |
| $W$ | 历史窗口长度 |
| $H$ | 预测步数 |
| $X_t$ | 当前输入窗口，包含 (t-W+1) 到 (t) 的观测 |
| $Y_t$ | 真实未来目标，包含 (t+1) 到 (t+H) |
| $\hat Y_t$ | 模型预测的未来目标 |
| $M$ | 评价窗口数量 |
| $E_0$ | 固定候选有向边集合 |
| $E_I^t$ | 时间 (t) 激活的意图边集合 |
| $s_{ij}^t$ | 模型对有向边 $i	o j$ 的连续意图分数 |
| $Q_t$ | 当前窗口选出的边集合，例如 Top-K 集合 |
| $L_t$ | 单个窗口的预测损失 |

当前旧代码使用 (W=3,H=1)。对于 5 分钟数据，这表示 15 分钟范围内的 3 个采样点预测下一个 5 分钟值。正式实验建议使用验证集确定 (W,H)，例如 (W=12,H=3)，表示用最近 60 分钟预测未来 15 分钟。

所有标准化、候选边统计和模型训练必须只使用训练期可用信息。原始缺失位置应由 `missing_mask.csv` 保留；含有缺失预测目标的窗口不能直接作为普通测试样本计算误差。

## 2. Top-K 与每窗口 90% 分位数的区别

### 2.1 固定预算 Top-K

将当前窗口所有候选边按分数降序排列，严格保留指定数量：

$
Q_{t,K}=\operatorname{TopK}\left(\{s_{ij}^t:(i,j)\in E_0\},K\right).
$

如果候选边数为$|E_0|=488$：

| 预算 | 保留数量 |
|---:|---:|
| 5% | 约 24 条 |
| 10% | 约 49 条 |
| 20% | 约 98 条 |

Top-K 的主要作用是控制资源或部署预算。每个窗口边数相同，便于比较不同方法，也适合后续传感器选择和成本约束实验。

它回答的问题是：

> 在只能保留固定数量边的情况下，哪些边最值得保留？

Top-K 的限制是：即使当前所有边分数都很小，也会强行保留 K 条边；因此它不能判断当前是否真的存在强意图结构。

### 2.2 每窗口 90% 分位数阈值

对每个窗口单独计算阈值：

$
\delta_t=Q_{0.90}\left(\{s_{ij}^t:(i,j)\in E_0\}\right),
$

然后保留：

$
E_I^t=\{(i,j)\in E_0:s_{ij}^t\ge \delta_t\}.
$

如果所有分数都不同，通常接近保留 10% 的边。但如果大量分数相等，特别是大量分数为 0，则所有等于阈值的边都会被保留。例如：

```text
400 条边分数为 0，88 条边分数大于 0
90% 分位数 = 0
所有 488 条边都满足 score >= 0
```

因此分位数阈值不能保证固定边数。它回答的问题是：

> 在当前窗口相对排序中，哪些边位于高分区域？

它适合观察相对热点，但不适合作为固定预算规则，也不适合在分数整体收缩时直接解释为图变稠密。

### 2.3 绝对分数阈值

使用固定阈值：

$
E_{I,\delta}^t=\{(i,j)\in E_0:s_{ij}^t\ge\delta_{abs}\}.
$

它允许不同窗口保留不同数量的边，也允许在没有足够强边时输出空图。阈值必须只用训练集或验证集确定，例如根据验证集的预测损失、稀疏度和删边贡献选择。不能用测试集调阈值。

### 2.4 建议同时报告三类结果

1. 正式实验建议同时报告：

   1. Top-5%、Top-10%、Top-20%：评价固定预算下的边排序；
   2. 验证集确定的绝对阈值：评价强意图边是否真的达到稳定强度；
   3. 90% 分位数：仅作为相对热点可视化，不作为唯一主结果。

## 3. 预测性能指标

### 3.1 MAE

定义：

$
MAE=\frac{1}{MNH}
\sum_{t=1}^{M}\sum_{h=1}^{H}\sum_{i=1}^{N}
\left|y_{i,t+h}-\hat y_{i,t+h}\right|.
$

其中：

- $M$：测试窗口数量；
- $N$：节点数量；
- $H$：预测步数；
- $y_{i,t+h}$：真实值；
- $\hat y_{i,t+h}$：预测值。

MAE 的作用是衡量平均绝对预测误差。它与交通流量误差的实际大小具有直接关系，较少受极少数大误差影响，适合作为主指标。

MAE 主要验证：

> 模型整体预测是否准确。

局限：它不会特别强调极端拥堵或突发错误；不同节点尺度不同的时候，应使用统一标准化或原始单位下的加权方案。                                 

### 3.2 RMSE

定义：

$
RMSE=\sqrt{
\frac{1}{MNH}
\sum_{t=1}^{M}\sum_{h=1}^{H}\sum_{i=1}^{N}
\left(y_{i,t+h}-\hat y_{i,t+h}\right)^2
}.
$

RMSE 对大误差使用平方惩罚，因此适合发现模型是否在少数时间段或少数传感器上出现严重失准。

RMSE 主要验证：

> 模型是否存在严重的局部失败或峰值预测错误。

MAE 与 RMSE 应一起报告：

- MAE 低、RMSE 高：平均表现不错，但存在少数大错误；
- MAE 和 RMSE 都低：整体和极端误差都较好；
- RMSE 改善而 MAE 不明显：主要改善了少数峰值窗口。

### 3.3 持续性基线

持续性预测直接复制最近观测：

$
\hat y_{i,t+h}^{pers}=x_i^t.
$

如果数据已经标准化，则使用标准化后的最后值；如果评价使用原始单位，则使用原始最后值。

持续性基线的作用不是提供先进算法，而是提供一个非常强的最低参考：交通流量通常具有短时连续性，复杂模型必须至少超过“未来等于现在”的预测。

定义相对技能：

$
Skill=1-
\frac{MAE_{model}}
{MAE_{pers}+\epsilon}.
$

解释：

- (Skill>0)：模型优于持续性；
- (Skill=0)：模型与持续性相同；
- (Skill<0)：模型不如持续性。

持续性基线主要验证：

> 复杂模型是否真的从空间关系、趋势或图结构中学到了额外信息。

本项目之前的 full 运行中，模型 MAE 高于持续性 MAE，因此当前模型尚未通过这一基本检查。

### 3.4 分预测步 MAE

对于 (H>1)，分别报告：

$
MAE_h=\frac{1}{MN}
\sum_{t=1}^{M}\sum_{i=1}^{N}
\left|y_{i,t+h}-\hat y_{i,t+h}\right|.
$

这样可以区分：模型是只擅长最近 5 分钟，还是能够保持 15、30、60 分钟的预测能力。

MAPE 不建议作为主指标，因为标准化值可能接近零甚至为负；原始交通流量也可能存在接近零的分母。

## 4. 意图图忠实度与边贡献指标

真实交通数据没有边级意图真值，因此下面指标评价的是：模型自己给出的边分数是否真正影响预测。

### 4.1 删除必要性

对每个窗口选择高分边集合 (Q_t)，屏蔽这些边后重新推断：

$
D_K=\frac1M\sum_{t=1}^{M}
\left[
L_t(f(X_t,S_t\odot1_{e\notin Q_t}))
-L_t(f(X_t,S_t))
\right].
$

其中 $L_t$ 是当前窗口的 MAE。

解释：

- $D_K>0$：删除高分边后误差变大；
- $D_K$ 越大：高分边对模型预测越必要；
- $D_K\approx0$：高分边可能没有实际预测作用；
- $D_K<0$：删除后反而改善，说明边分数排序可能不可靠。

### 4.2 保留充分性

只保留高分边：

$
S_K=\frac1M\sum_{t=1}^{M}
\left[
L_t(f(X_t,S_t\odot1_{e\in Q_t}))
-L_t(f(X_t,S_t))
\right].
$

如果 $S_K$ 接近零，说明少量高分边可以保留完整图的大部分预测能力。

### 4.3 随机校正

随机选择同样数量的边，重复 (B) 次：

$
D_K^{excess}=D_K^{top}
-\frac1B\sum_{b=1}^{B}D_K^{random,b}.
$

它检验高分边是否比“任意删边”更有贡献。

### 4.4 参考出处

直接删除图解释边可能制造分布外输入，导致忠实度评价偏差。Zheng 等在 ICLR 2024 的 *Towards Robust Fidelity for Evaluating Explainability of Graph Neural Networks* 中分析了这一问题，并提出通过部分删除和部分保留降低分布偏移。[论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/34293d684b1012ed45c3274b4a7edc00-Paper-Conference.pdf)

本项目的$D_K$和 $S_K$是将其思想改写到交通回归任务的版本，不能称为论文原始分类指标。

## 5. 动态稳定性指标

### 5.1 边集合 Jaccard 稳定性

相邻窗口的边集合重合度：

$
J_t=\frac{|E_I^t\cap E_I^{t-1}|}
{|E_I^t\cup E_I^{t-1}|+\epsilon}.
$

作用：衡量意图图是否频繁跳变。

- $J_t$ 高：相邻图结构稳定；
- $J_t$ 低：边集合变化剧烈。

它本身没有“越高越好”的绝对结论。过低可能是不稳定，过高也可能说明模型只输出固定图。

### 5.2 边权变化量

在候选边全集上定义：

$
V_t=\frac{1}{|E_0|}
\sum_{e\in E_0}|s_e^t-s_e^{t-1}|.
$

作用：衡量连续分数的变化强度。

应同时报告 $J_t$ 和 $V_t$，因为边集合可能不变但权重明显变化，也可能边集合变化来自阈值并列。

### 5.3 热点持续时间

对边 $e$，若 $e\in E_I^t$，定义激活指示：

$
a_e^t=1[e\in E_I^t].
$

连续激活区间长度为：

$
Duration=(t_{end}-t_{start}+1)\Delta.
$

作用：判断意图边是短暂噪声，还是持续的局部交通关系。

## 6. 有真值的结构识别指标

真实数据没有真实意图边标签，因此需要使用已知动态传播机制的合成数据：

$
x_t=\sum_{\ell=1}^{L}A_{t,\ell}x_{t-\ell}+b_t+\eta_t.
$

真实边集：

$
E_t^*=support(A_{t+1,1}).
$

### 6.1 候选图覆盖率

$
Coverage=
\frac{\sum_t|E_t^*\cap E_0|}
{\sum_t|E_t^*|}.
$

它验证候选图是否已经包含真实关系。覆盖率低时，后续识别指标不能公平评价意图模型。

### 6.2 Precision@K、Recall@K、F1@K

$
P@K=\frac{|Q_{t,K}\cap E_t^*|}{K}
$

$
R@K=\frac{|Q_{t,K}\cap E_t^*|}{|E_t^*|}
$

$
F1@K=\frac{2P@K R@K}{P@K+R@K+\epsilon}.
$

作用：直接验证固定预算下是否选中了真实动态边。

### 6.3 Average Precision

按边分数排序，逐步计算 precision 和 recall：

$
AP=\sum_r(R_r-R_{r-1})P_r.
$

AP 不需要事先固定一个唯一阈值，适合稀疏正例。Saito 与 Rehmsmeier 的研究说明，在正负样本严重不平衡时，PR 曲线通常比 ROC 曲线更能反映正例识别质量。[论文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)

## 7. 新增与消退跟踪指标

真实新增边：

$
\Delta_t^+=E_t^*\setminus E_{t-1}^*.
$

真实消退边：

$
\Delta_t^-=E_{t-1}^*\setminus E_t^*.
$

预测新增和消退边同样用预测集合的差集计算。

新增 F1：

$
F1_{new}=\frac{2TP_{new}}
{2TP_{new}+FP_{new}+FN_{new}}.
$

消退 F1：

$
F1_{decay}=\frac{2TP_{decay}}
{2TP_{decay}+FP_{decay}+FN_{decay}}.
$

检测延迟：

$
Delay=(\hat t_{detect}-t_{true})\Delta.
$

应同时报告漏检率，不能只对成功检测的边计算平均延迟。

连续活动区间还可以使用时间 IoU：

$
IoU(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$

可预先规定 $IoU\ge0.5$ 为事件命中，再计算事件级 precision、recall 和 F1。Tatbul 等 NeurIPS 2018 的区间 precision/recall 工作可作为区间评价的参考；本项目的交通边事件版本需要明确说明是适配协议。[论文](https://papers.nips.cc/paper_files/paper/2018/hash/8f468c873a32bb061e9eaeb2050ba45d1-Abstract.html)

## 8. 基线方法

### 8.1 持续性基线

$
\hat y_{i,t+h}=x_i^t.
$

流程：读取最后一个观测值，复制到所有未来步。作用是检验复杂模型是否超过短时交通的自然连续性。

### 8.2 空间邻居平均

固定选择节点 $j$ 的空间邻居 $\mathcal N(j)$：

$
\hat y_{j,t+1}
=\alpha x_j^t
+\frac{1-\alpha}{|\mathcal N(j)|}
\sum_{i\in\mathcal N(j)}x_i^t.
$

流程：根据坐标建立固定 KNN 图，每个目标节点使用自身值和邻居均值预测。它检验当前方法是否只是利用空间平滑。

### 8.3 固定图时空预测器

固定图邻居聚合：

$
H_t=\phi(X_t,A_0),
$

再使用 TCN、GRU 或 MLP 预测：

$
\hat Y_t=g_\theta(H_t).
$

流程：候选图固定，预测器训练方式与当前模型一致。它用于检验动态意图图相对于固定空间图是否提供增量。

### 8.4 Graph WaveNet

Graph WaveNet 学习自适应邻接矩阵：

$
A_{adapt}=softmax(ReLU(E_1E_2^\top)).
$

然后进行图卷积和扩张时间卷积：

$
H^{(l+1)}=GraphConv(H^{(l)},A_{adapt}),
$

$
\hat Y_t=TCN(H^{(L)}).
$

它适合作为经典任务驱动图基线。其自适应图通常是全局共享的，不能直接等同于每个窗口变化的意图图。

参考：Wu 等，*Graph WaveNet for Deep Spatial-Temporal Graph Modeling*，IJCAI 2019。[论文](https://arxiv.org/abs/1906.00121)，[代码](https://github.com/nnzhan/Graph-WaveNet)。

### 8.5 DGCRN

DGCRN 根据节点状态生成动态图：

$
A_t=g_\theta(X_t,H_{t-1}).
$

然后：

$
H_t=GRU(H_{t-1},GraphConv(X_t,A_t)).
$

它与当前动态意图边的目标比较接近，适合验证动态关系是否提高预测性能。

参考：Li 等，*Dynamic Graph Convolutional Recurrent Network for Traffic Prediction*，2021；TKDD 2023。[论文](https://arxiv.org/abs/2104.14917)，[代码](https://github.com/LakeChuan/DGCRN)。

### 8.6 DLF / DSTG

DLF 将动态图学习和趋势、周期结构分开建模，并持续更新动态模式：

$
H_t^{graph}=f_{graph}(X_t),
\qquad
H_t^{season}=f_{season}(X_{t-L:t}).
$

它适合比较长期动态图分布变化，但工程成本高于 Graph WaveNet。

参考：Wang 等，*Towards Dynamic Spatial-Temporal Graph Learning: A Decoupled Perspective*，AAAI 2024。[论文](https://ojs.aaai.org/index.php/AAAI/article/view/28759)，[代码](https://github.com/UnderReview24/DLF)。

### 8.7 STDN

STDN 同时学习动态关系图，并分解趋势、周期成分：

$
X_t=X_t^{trend}+X_t^{season}+X_t^{residual}.
$

动态图由时空表示产生：

$
A_t=g_\theta(H_t^{spatial},H_t^{temporal}).
$

它适合当前意图图的外部动态图基线。

参考：Cao 等，*Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting*，AAAI 2025。[论文](https://ojs.aaai.org/index.php/AAAI/article/view/33247)，[代码](https://github.com/roarer008/STDN)。

### 8.8 MoGERNN

MoGERNN 面向部分观测和未观测节点预测。它使用多个图专家：

$
H_t=\sum_{k=1}^{K}\pi_k(X_t)f_k(X_t,A_k),
\qquad
\sum_k\pi_k=1.
$

它适合后续有限观测、传感器失效和节点重新配置实验，不是当前全节点意图图训练的最小基线。

参考：Zhou 等，*MoGERNN: An Inductive Traffic Predictor for Unobserved Locations*，Transportation Research Part C 2025。[论文](https://www.sciencedirect.com/science/article/abs/pii/S0968090X25000841)，[代码](https://github.com/ZJU-TSELab/MoGERNN)。

## 9. 推荐的最小比较表

| 类别 | 方法 | 主要回答的问题 |
|---|---|---|
| 无学习基线 | 持续性 | 复杂模型是否超过短时连续性 |
| 空间基线 | 空间邻居平均 | 是否只是空间平滑 |
| 固定图 | 固定 KNN + 时空预测器 | 动态图是否有增量 |
| 经典自适应图 | Graph WaveNet | 全局任务图是否足够 |
| 动态图 | DGCRN 或 STDN | 动态边关系是否提高预测 |
| 有限观测 | MoGERNN | 部分节点观测时能否预测全网 |

所有基线必须使用相同的训练、验证、测试时间范围、输入窗口、预测长度、标准化规则、缺失处理和调参预算。

## 10. 需要优先修正的实现问题

1. 当前入口使用按全期标准化的 CSV，并再次标准化；正式流程应从原始 CSV 开始，只用训练段计算统计量。
2. `missing_mask.csv` 尚未接入当前意图图入口；插值值可能被当作真实输入和预测目标。
3. `stride=12` 只每 60 分钟产生一张图；如果需要每 5 分钟图，应使用 `stride=1`。
4. 当前 `W=3,H=1` 的历史跨度过短，窗口内滞后相关特征基本无法稳定计算。
5. 当前每窗口 90% 分位数筛选在大量零分时会把全部候选边保留。
6. 应增加验证集确定的绝对阈值，并记录阈值类型、阈值值和边数。
7. `edge_mask_ablation` 当前把所有测试窗口的窗口—边分数混合排序，应改为逐窗口 Top/Low/Random。
8. 当前随机删边只重复一次，应重复多个随机种子。
9. 当前只训练 2 个 epoch，没有验证集最佳 checkpoint。
10. 当前训练损失图只有两个点，不能据此判断收敛。
11. 当前测试结果没有按原始缺失目标进行 mask。
12. 意图图可视化色标对极小分数不稳定，应使用稳健范围或对数尺度。
13. 输出应支持连续 1–2 周的 5 分钟窗口统计，而不是只查看 10 张图。
14. 需要保存每个窗口的分数统计、零分比例、边数、输入时间、目标时间和缺失状态。

## 11. 主要参考文献

1. Zheng, X. et al. *Towards Robust Fidelity for Evaluating Explainability of Graph Neural Networks*. ICLR 2024. [论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/34293d684b1012ed45c3274b4a7edc00-Paper-Conference.pdf)
2. Li, J. et al. *Graph Neural Network Explanations are Fragile*. ICML 2024. [论文](https://proceedings.mlr.press/v235/li24bd.html)
3. Wang, B. et al. *Towards Dynamic Spatial-Temporal Graph Learning: A Decoupled Perspective*. AAAI 2024. [论文](https://ojs.aaai.org/index.php/AAAI/article/view/28759)
4. Cao, L. et al. *Spatiotemporal-aware Trend-Seasonality Decomposition Network for Traffic Flow Forecasting*. AAAI 2025. [论文](https://ojs.aaai.org/index.php/AAAI/article/view/33247)
5. Zhou, Q. et al. *MoGERNN: An Inductive Traffic Predictor for Unobserved Locations*. Transportation Research Part C, 2025. [论文](https://www.sciencedirect.com/science/article/abs/pii/S0968090X25000841)
6. Boniol, P. et al. *VUS: Effective and Efficient Accuracy Measures for Time-Series Anomaly Detection*. The VLDB Journal, 2025. [论文](https://link.springer.com/article/10.1007/s00778-025-00907-x)
7. Wu, Z. et al. *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*. IJCAI 2019. [论文](https://arxiv.org/abs/1906.00121)
8. Li, F. et al. *Dynamic Graph Convolutional Recurrent Network for Traffic Prediction*. 2021/2023. [论文](https://arxiv.org/abs/2104.14917)
9. Saito, T. and Rehmsmeier, M. *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE, 2015. [论文](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)
10. Tatbul, N. et al. *Precision and Recall for Time Series*. NeurIPS 2018. [论文](https://papers.nips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html)
