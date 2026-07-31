**DEIG-TCN：Dynamic Edge-level Intent Graph Temporal Convolutional Network**
即：**预测驱动的动态边级交通意图图构建算法**。

核心思想是：

$$
\boxed{
\text{不再把意图定义为低/中/高类别，而是定义为当前窗口中对未来交通状态有预测贡献的有向边强度。}
}
$$

也就是先学习统一的边级意图强度：

$$
s_{ij}^{t,H}\in[0,1]
$$

再由强意图边构建动态意图图：

$$
G_I^t=(V,E_I^t,A_I^t)
$$

必要时，再从 $G_I^t$ 中分解多个局部热点意图子图。

---

# 1. 符号定义

设交通传感器网络共有 $N$ 个节点：

$$
V=\{v_1,v_2,\ldots,v_N\}
$$

每个节点有二维坐标：

$$
\mathbf{c}_i=(lon_i,lat_i)\in \mathbb{R}^2
$$

所有节点坐标矩阵为：

$$
\mathbf{C}=[\mathbf{c}_1,\mathbf{c}_2,\ldots,\mathbf{c}_N]^\top
\in \mathbb{R}^{N\times 2}
$$

交通流量时间序列为：

$$
\mathbf{X}=\{x_i^t\}\in \mathbb{R}^{N\times T}
$$

其中：

- $x_i^t$：节点 $v_i$ 在时间步 $t$ 的交通流量；
- $T$：总时间步数；
- 数据间隔为 1 小时；
- 数据长度约为 7 个月。

历史窗口长度为 $W$，预测步长为 $H$。以时间 $t$ 为当前窗口末端，则历史输入窗口为：

$$
\mathbf{X}_{t-W+1:t}
$$

未来预测目标为：

$$
\mathbf{Y}_t
=
\mathbf{X}_{:,t+1:t+H}
\in \mathbb{R}^{N\times H}
$$

模型输出为：

$$
\hat{\mathbf{Y}}_t
=
\hat{\mathbf{X}}_{:,t+1:t+H}
\in \mathbb{R}^{N\times H}
$$

以及边级意图矩阵：

$$
\mathbf{S}^t=
[s_{ij}^{t,H}]
\in[0,1]^{N\times N}
$$

其中：

$$
s_{ij}^{t,H}
$$

表示在当前窗口 $[t-W+1,t]$ 内，有向边：

$$
v_i\rightarrow v_j
$$

对未来 $H$ 步交通状态的意图强度。

这里的“意图强度”不是类别概率，而是：

当前窗口中，节点 $i$ 到节点 $j$ 的有向关系对未来交通状态的预测贡献、传播趋势和任务相关性强度。

---

# 2. 整体算法框架

整体流程为：

$$
\mathbf{X},\mathbf{C}
\rightarrow
\text{数据预处理}
\rightarrow
\text{候选有向图构建}
\rightarrow
\text{滑窗样本构造}
\rightarrow
\text{GAT-TCN 时空编码}
\rightarrow
\text{边级意图识别}
\rightarrow
\text{未来交通预测}
\rightarrow
\text{动态意图图构建}
\rightarrow
\text{局部热点意图子图提取}
$$

模型结构可以写成：

$$
\mathbf{X}_{t-W+1:t}
\rightarrow
\text{ST-Encoder}
\rightarrow
\begin{cases}
\mathbf{S}^t & \text{边级意图矩阵}\\
\hat{\mathbf{Y}}_t & \text{全图未来预测}
\end{cases}
$$

其中 $\mathbf{S}^t$ 进一步用于构建：

$$
G_I^t=(V,E_I^t,A_I^t)
$$

并可选地分解为多个局部热点意图：

$$
\mathcal{G}_I^t
=
\{g_1^t,g_2^t,\ldots,g_{M_t}^t\}
$$

---

# 3. 数据处理

## 3.1 时间顺序划分

先按照时间顺序划分训练集、验证集和测试集，而不是随机划分。

设所有可用窗口样本数为 $M$，则：

$$
\mathcal{D}_{train}
=
\{\mathcal{D}_1,\ldots,\mathcal{D}_{\lfloor 0.6M\rfloor}\}
$$

$$
\mathcal{D}_{val}
=
\{\mathcal{D}_{\lfloor 0.6M\rfloor+1},\ldots,\mathcal{D}_{\lfloor 0.8M\rfloor}\}
$$

$$
\mathcal{D}_{test}
=
\{\mathcal{D}_{\lfloor 0.8M\rfloor+1},\ldots,\mathcal{D}_M\}
$$

所有标准化参数、周期均值、相关性、候选图结构都只用训练集历史区间计算，避免未来泄露。

---

## 3.2 缺失值与异常值处理

对每个节点 $v_i$ 的时间序列：

$$
x_i^1,x_i^2,\ldots,x_i^T
$$

先检查：

$$
x_i^t \neq NaN,\quad x_i^t \neq Inf
$$

若有缺失，可以采用：

1. 短缺失：线性插值；
2. 长缺失：同小时历史均值填补；
3. 极端异常：按训练集分位数截断。

例如，用训练集的 1% 和 99% 分位数进行截断：

$$
x_i^t
\leftarrow
\min
\left(
\max(x_i^t,Q_{0.01}^i),
Q_{0.99}^i
\right)
$$

其中 $Q_{0.01}^i,Q_{0.99}^i$ 只在训练集上计算。

---

## 3.3 标准化

对每个节点单独标准化。训练集均值和标准差为：

$$
\mu_i
=
\frac{1}{T_{train}}
\sum_{t\in train}
x_i^t
$$

$$
\sigma_i
=
\sqrt{
\frac{1}{T_{train}}
\sum_{t\in train}
(x_i^t-\mu_i)^2
}
$$

标准化流量为：

$$
z_i^t
=
\frac{x_i^t-\mu_i}{\sigma_i+\varepsilon}
$$

得到标准化矩阵：

$$
\mathbf{Z}
=
\{z_i^t\}
\in \mathbb{R}^{N\times T}
$$

---

## 3.4 变化率特征

交通意图往往体现在变化趋势中，因此构造一阶变化：

$$
\Delta z_i^t
=
z_i^t-z_i^{t-1}
$$

对于 $t=1$，可设：

$$
\Delta z_i^1=0
$$

---

## 3.5 周期残差特征

交通流有明显小时周期和星期周期。设时间 $t$ 对应的周期索引为：

$$
q(t)=(hour(t),weekday(t))
$$

在训练集上计算节点 $i$ 在周期 $q$ 下的平均流量：

$$
\bar{z}_{i,q}^{train}
=
\frac{1}{|\mathcal{T}_q^{train}|}
\sum_{t\in \mathcal{T}_q^{train}}
z_i^t
$$

其中：

$$
\mathcal{T}_q^{train}
=
\{t\in train\mid q(t)=q\}
$$

周期残差为：

$$
r_i^t
=
z_i^t-\bar{z}_{i,q(t)}^{train}
$$

该残差表示当前流量相对于“正常周期模式”的偏离程度。

---

## 3.6 时间编码特征

构造小时和星期的周期编码：==(不直接用数字表示时间，而是映射到单位圆上，23点离0点的数字很远，但是正余弦函数很近)==

$$
h_{sin}^t=\sin\left(\frac{2\pi hour(t)}{24}\right)
$$

$$
h_{cos}^t=\cos\left(\frac{2\pi hour(t)}{24}\right)
$$

$$
w_{sin}^t=\sin\left(\frac{2\pi weekday(t)}{7}\right)
$$

$$
w_{cos}^t=\cos\left(\frac{2\pi weekday(t)}{7}\right)
$$

最终节点 $i$ 在时间 $t$ 的输入特征为：

$$
\mathbf{f}_i^t
=
[
z_i^t,
\Delta z_i^t,
r_i^t,
h_{sin}^t,
h_{cos}^t,
w_{sin}^t,
w_{cos}^t
]
\in \mathbb{R}^{F_0}
$$

其中：

$$
F_0=7
$$

如果有节假日、天气、道路等级等外部变量，也可以继续拼接进去。

---

# 4. 候选有向图构建

边级意图识别不能在所有 $N(N-1)$ 条边上无约束学习，否则容易产生大量虚假边。因此先构建候选有向图：

$$
G_0=(V,E_0)
$$

最终意图图 $G_I^t$ 是 $G_0$ 上的动态激活子图。

---

## 4.1 空间距离

先对坐标归一化：

$$
\tilde{\mathbf{c}}_i
=
\frac{
\mathbf{c}_i-\mathbf{c}_{min}
}{
\mathbf{c}_{max}-\mathbf{c}_{min}+\varepsilon
}
$$

节点间距离为：

$$
d_{ij}
=
\|\tilde{\mathbf{c}}_i-\tilde{\mathbf{c}}_j\|_2
$$

方向向量为：

$$
\mathbf{u}_{ij}
=
\tilde{\mathbf{c}}_j-\tilde{\mathbf{c}}_i
$$

方向角为：

$$
\theta_{ij}
=
\arctan2
(
\tilde{y}_j-\tilde{y}_i,
\tilde{x}_j-\tilde{x}_i
)
$$

---

## 4.2 训练集相关性

只使用训练集历史区间计算 Pearson 相关性：

$$
\rho_{ij}
=
\frac{
\sum_{t\in train}
(z_i^t-\bar{z}_i)(z_j^t-\bar{z}_j)
}{
\sqrt{\sum_{t\in train}(z_i^t-\bar{z}_i)^2}
\sqrt{\sum_{t\in train}(z_j^t-\bar{z}_j)^2}
+\varepsilon
}
$$

---

## 4.3 时滞相关性

为了捕捉潜在方向 $i\rightarrow j$，计算时滞相关性：

$$
\rho_{ij}^{lag}
=
\max_{\tau\in\{1,\ldots,\tau_{max}\}}
Corr
(
z_i^{t-\tau},
z_j^t
)
$$

更展开地写：

$$
\rho_{ij}^{lag}
=
\max_{\tau\in\{1,\ldots,\tau_{max}\}}
\frac{
\sum_{t\in train}
(z_i^{t-\tau}-\bar{z}_{i,\tau})
(z_j^t-\bar{z}_j)
}{
\sqrt{\sum_{t\in train}(z_i^{t-\tau}-\bar{z}_{i,\tau})^2}
\sqrt{\sum_{t\in train}(z_j^t-\bar{z}_j)^2}
+\varepsilon
}
$$

最优时滞为：

$$
\tau_{ij}^{*}
=
\arg\max_{\tau\in\{1,\ldots,\tau_{max}\}}
Corr
(
z_i^{t-\tau},
z_j^t
)
$$

如果：

$$
\rho_{ij}^{lag}\rho_{ji}^{lag}
$$

则说明 $i\rightarrow j$ 的方向性强于 $j\rightarrow i$。

注意，这里只是候选方向特征，不直接等价于最终意图。

---

## 4.4 候选边集合

可以采用 KNN + 相关性混合方式：

$$
(i,j)\in E_0
$$

当满足以下任一条件：

$$
j\in KNN(i)
$$

或：

$$
d_{ij}<\epsilon_d
$$

或：

$$
|\rho_{ij}|\epsilon_{\rho}
$$

或：

$$
\rho_{ij}^{lag}\epsilon_{lag}
$$

实际建议第一版用：

$$
(i,j)\in E_0
\quad \text{if} \quad
j\in KNN(i)
$$

并为每个节点保留双向候选边：

$$
(i,j)\in E_0,\quad (j,i)\in E_0
$$

因为真实交通方向未知时，不应过早删除反向边。

定义候选边掩码矩阵：

$$
M_{ij}
=
\begin{cases}
1, & (i,j)\in E_0,\\
0, & \text{otherwise}.
\end{cases}
$$

---

## 4.5 边特征

每条候选有向边 $i\rightarrow j$ 的静态边特征为：

$$
\mathbf{e}_{ij}
=
[
d_{ij},
\exp(-d_{ij}/\sigma_d),
\cos\theta_{ij},
\sin\theta_{ij},
\rho_{ij},
\rho_{ij}^{lag},
\tau_{ij}^{*}/\tau_{max},
\rho_{ij}^{lag}-\rho_{ji}^{lag}
]
$$

其中：

$$
\mathbf{e}_{ij}\in \mathbb{R}^{F_e}
$$

这些边特征只表示候选边的空间和历史统计属性，不是最终意图强度。

---

# 5. 滑动窗口样本构造

以时间 $t$ 为当前窗口末端，其中：

$$
W\leq t\leq T-H
$$

输入样本为：

$$
\mathcal{X}_t
=
\{\mathbf{F}^{t-W+1},\mathbf{F}^{t-W+2},\ldots,\mathbf{F}^{t}\}
\in \mathbb{R}^{W\times N\times F_0}
$$

其中：

$$
\mathbf{F}^{\tau}
=
[
\mathbf{f}_1^\tau,\ldots,\mathbf{f}_N^\tau
]^\top
\in \mathbb{R}^{N\times F_0}
$$

预测目标为：

$$
\mathcal{Y}_t
=
\mathbf{Z}_{:,t+1:t+H}
\in \mathbb{R}^{N\times H}
$$

样本集合为：

$$
\mathcal{D}
=
\{(\mathcal{X}_t,\mathcal{Y}_t)\}_{t=W}^{T-H}
$$

若滑动步长为 $r$，则：

$$
t_k=W+k\cdot r
$$

---

# 6. 模型结构

模型由四个核心模块组成：

$$
\text{GAT 空间编码器}
+
\text{TCN 时间编码器}
+
\text{边级意图识别器}
+
\text{动态图预测器}
$$

---

## 6.1 GAT 空间编码器

对历史窗口中的每个时间步 $\tau$，输入节点特征：

$$
\mathbf{H}^{\tau,0}
=
\mathbf{F}^{\tau}
\in \mathbb{R}^{N\times F_0}
$$

在候选图 $G_0$ 上做 GAT 传播。

对于第 $l$ 层，节点 $i$ 的表示为：

$$
\mathbf{h}_i^{\tau,l}
$$

先做线性映射：

$$
\mathbf{z}_i^{\tau,l}
=
\mathbf{W}^{l}
\mathbf{h}_i^{\tau,l}
$$

对于候选有向边 $i\rightarrow j$，计算边注意力打分：

$$
e_{ij}^{\tau,l}
=
\text{LeakyReLU}
\left(
\mathbf{a}^{l\top}
[
\mathbf{z}_i^{\tau,l}
\Vert
\mathbf{z}_j^{\tau,l}
\Vert
\mathbf{U}^{l}\mathbf{e}_{ij}
]
\right)
$$

其中：

- $\Vert$：向量拼接；
- $\mathbf{e}_{ij}$：静态边特征；
- $\mathbf{U}^l$：边特征映射矩阵。

对所有指向节点 $j$ 的入边做 softmax：

$$
\alpha_{ij}^{\tau,l}
=
\frac{
\exp(e_{ij}^{\tau,l})
}{
\sum_{p:(p,j)\in E_0}
\exp(e_{pj}^{\tau,l})
}
$$

节点 $j$ 的下一层表示为：

$$
\mathbf{h}_j^{\tau,l+1}
=
\sigma
\left(
\sum_{i:(i,j)\in E_0}
\alpha_{ij}^{\tau,l}
\mathbf{z}_i^{\tau,l}
\right)
$$

经过 $L_g$ 层后得到：

$$
\mathbf{H}^{\tau,G}
=
[
\mathbf{h}_1^{\tau,G},
\ldots,
\mathbf{h}_N^{\tau,G}
]^\top
\in \mathbb{R}^{N\times F_g}
$$

这里的 GAT 注意力 $\alpha_{ij}$ 只是空间编码过程中的注意力，不直接等价于最终意图强度。

---

## 6.2 TCN 时间编码器

对每个节点 $i$，取其在历史窗口内的 GAT 表示序列：

$$
\mathbf{H}_i^t
=
[
\mathbf{h}_i^{t-W+1,G},
\mathbf{h}_i^{t-W+2,G},
\ldots,
\mathbf{h}_i^{t,G}
]
\in \mathbb{R}^{W\times F_g}
$$

输入 TCN：

$$
\mathbf{Q}_i^t
=
\text{TCN}
(
\mathbf{H}_i^t
)
\in \mathbb{R}^{W\times F_t}
$$

TCN 第 $l$ 层可表示为：

$$
\mathbf{Q}_i^{l}
=
\sigma
\left(
\text{BN}
\left(
\text{Conv1D}_{d_l}
(
\mathbf{Q}_i^{l-1}
)
\right)
\right)
$$

其中：

- $d_l$：扩张卷积 dilation；
- $\text{BN}$：批归一化；
- $\sigma$：ReLU 或 GELU。

为得到**节点级窗口**表示，对时间维度做注意力池化：

$$
u_{i,\tau}^t
=
\mathbf{v}_q^\top
\tanh
(
\mathbf{W}_q\mathbf{q}_{i,\tau}^t
)
$$

$$
\beta_{i,\tau}^t
=
\frac{
\exp(u_{i,\tau}^t)
}{
\sum_{\ell=t-W+1}^{t}
\exp(u_{i,\ell}^t)
}
$$

$$
\mathbf{z}_i^t
=
\sum_{\tau=t-W+1}^{t}
\beta_{i,\tau}^t
\mathbf{q}_{i,\tau}^t
\in \mathbb{R}^{F_z}
$$

所有节点的窗口表示为：

$$
\mathbf{Z}^t
=
[
\mathbf{z}_1^t,\ldots,\mathbf{z}_N^t
]^\top
\in \mathbb{R}^{N\times F_z}
$$

---

# 7. 边级意图识别

对于每条候选有向边：

$$
(i,j)\in E_0
$$

构造**边级动态表示**：

$$
\mathbf{m}_{ij}^t
=
[
\mathbf{z}_i^t
\Vert
\mathbf{z}_j^t
\Vert
\mathbf{z}_i^t-\mathbf{z}_j^t
\Vert
\mathbf{z}_i^t\odot \mathbf{z}_j^t
\Vert
\mathbf{e}_{ij}
\Vert
\mathbf{d}_{ij}^t
]
$$

其中：

$$
\odot
$$

表示逐元素乘法，$\mathbf{d}_{ij}^t$ 是窗口内的显式动态边特征。

---

## 7.1 显式动态边特征

可以定义：

$$
\mathbf{d}_{ij}^t
=
[
D_{ij}^t,
D_{ij}^t-D_{ji}^t,
A_i^t,
A_j^t,
G_i^t,
G_j^t
]
$$

其中：

### 方向传播特征

$$
D_{ij}^t
=
\max_{\tau\in\{1,\ldots,\tau_{max}\}}
Corr
(
r_i^{t-W+1:t-\tau},
r_j^{t-W+1+\tau:t}
)
$$

如果 $D_{ij}^tD_{ji}^t$，说明当前窗口中 $i\rightarrow j$ 的时滞传播关系更强。

### 节点异常强度

$$
A_i^t
=
\frac{1}{W}
\sum_{\ell=t-W+1}^{t}
|r_i^\ell|
$$

### 节点近期趋势

$$
G_i^t
=
\frac{1}{W-1}
\sum_{\ell=t-W+2}^{t}
\Delta z_i^\ell
$$

这些显式特征有助于增强可解释性。

---

## 7.2 边级意图分数

边级意图识别器为一个 MLP：

$$
u_{ij}^t
=
MLP_{edge}
(
\mathbf{m}_{ij}^t
)
$$

$$
s_{ij}^{t,H}
=
M_{ij}
\cdot
\sigma(u_{ij}^t)
$$

其中：

- $M_{ij}$：候选边掩码；
- $\sigma(\cdot)$：Sigmoid；
- $s_{ij}^{t,H}\in[0,1]$。

对非候选边：

$$
M_{ij}=0
\Rightarrow
s_{ij}^{t,H}=0
$$

得到边级意图矩阵：

$$
\mathbf{S}^t
=
[s_{ij}^{t,H}]
\in[0,1]^{N\times N}
$$

注意：

$$
s_{ij}^{t,H}
\neq
s_{ji}^{t,H}
$$

因此该方法天然支持有向边级意图识别。

---

# 8. 基于意图图的未来预测

为了避免边级意图分数变成“事后解释”，必须让它参与预测任务。

对每个接收节点 $j$，归一化其入边意图权重：

$$
\tilde{s}_{ij}^{t,H}
=
\frac{
s_{ij}^{t,H}
}{
\sum_{p:(p,j)\in E_0}
s_{pj}^{t,H}
+
\varepsilon
}
$$

基于意图强度进行动态图消息传播：

$$
\mathbf{g}_j^t
=
\sum_{i:(i,j)\in E_0}
\tilde{s}_{ij}^{t,H}
\mathbf{W}_m
\mathbf{z}_i^t
$$

融合自身表示和意图邻居表示：

$$
\bar{\mathbf{z}}_j^t
=
[
\mathbf{z}_j^t
\Vert
\mathbf{g}_j^t
]
$$

预测节点 $j$ 的未来 $H$ 步流量：

$$
\hat{\mathbf{y}}_j^t
=
MLP_{pred}
(
\bar{\mathbf{z}}_j^t
)
\in \mathbb{R}^{H}
$$

所有节点预测结果为：

$$
\hat{\mathbf{Y}}_t
=
[
\hat{\mathbf{y}}_1^t,
\hat{\mathbf{y}}_2^t,
\ldots,
\hat{\mathbf{y}}_N^t
]^\top
\in \mathbb{R}^{N\times H}
$$

也可以采用残差预测形式：

$$
\hat{x}_j^{t+h}
=
z_j^t
+
\Delta \hat{x}_j^{t+h}
$$

其中：

$$
[\Delta \hat{x}_j^{t+1},\ldots,\Delta \hat{x}_j^{t+H}]
=
MLP_{pred}
(
\bar{\mathbf{z}}_j^t
)
$$

残差预测通常更适合短期交通流预测。

---

# 9. 无边级标签下的训练目标

由于没有真实边级意图标签，训练不能依赖：

$$
y_{ij}^{intent}
$$

而是采用：

$$
\text{预测监督}
+
\text{结构正则}
+
\text{弱监督辅助}
+
\text{边遮蔽贡献蒸馏}
$$

---

## 9.1 预测损失

主损失为全图未来预测误差。

推荐使用 MAE 或 Huber Loss。Huber Loss 定义为：

$$
\rho_{\kappa}(e)
=
\begin{cases}
\frac{1}{2}e^2, & |e|\leq \kappa,\\
\kappa(|e|-\frac{1}{2}\kappa), & |e|\kappa.
\end{cases}
$$

预测损失为：

$$
\mathcal{L}_{pred}
=
\frac{1}{BNH}
\sum_{b=1}^{B}
\sum_{i=1}^{N}
\sum_{h=1}^{H}
\rho_{\kappa}
\left(
\hat{x}_{b,i}^{t+h}
-
x_{b,i}^{t+h}
\right)
$$

其中 $B$ 是批大小。

---

## 9.2 稀疏约束

意图应是局部热点结构，不应所有候选边都被激活。

$$
\mathcal{L}_{sparse}
=
\frac{1}{B|E_0|}
\sum_{b=1}^{B}
\sum_{(i,j)\in E_0}
s_{b,ij}^{t,H}
$$

该项促使模型只保留少量关键意图边。

---

## 9.3 方向约束

为避免 $i\rightarrow j$ 和 $j\rightarrow i$ 同时无意义地被拉高，加入方向反对称约束：

$$
\mathcal{L}_{dir}
=
\frac{1}{B|\mathcal{P}|}
\sum_{b=1}^{B}
\sum_{(i,j)\in \mathcal{P}}
s_{b,ij}^{t,H}
s_{b,ji}^{t,H}
$$

其中：

$$
\mathcal{P}
=
\{(i,j)\mid (i,j)\in E_0,(j,i)\in E_0,i<j\}
$$

这个约束权重不能过大，因为真实交通中可能存在双向流动。

---

## 9.4 时间平滑约束

意图是动态的，但不应在正常状态下剧烈随机跳变。

$$
\mathcal{L}_{temp}
=
\frac{1}{B|E_0|}
\sum_{b=1}^{B}
\sum_{(i,j)\in E_0}
\omega_t
\left|
s_{b,ij}^{t,H}
-
s_{b,ij}^{t-r,H}
\right|
$$

其中 $r$ 是滑窗步长。

$\omega_t$ 是突变门控。当前窗口突变强时，降低平滑约束。

定义全局异常强度：

$$
A^t
=
\frac{1}{N}
\sum_{i=1}^{N}
|r_i^t|
$$

定义突变门控：

$$
\eta_t
=
\sigma
\left(
\frac{A^t-\mu_A}{\sigma_A+\varepsilon}
\right)
$$

则：

$$
\omega_t=1-\eta_t
$$

当当前窗口异常明显时：

$$
\eta_t \uparrow,\quad \omega_t \downarrow
$$

模型允许意图快速变化。

---

## 9.5 空间先验约束

如果候选图较稠密，可以加入距离惩罚：

$$
\mathcal{L}_{dist}
=
\frac{1}{B|E_0|}
\sum_{b=1}^{B}
\sum_{(i,j)\in E_0}
s_{b,ij}^{t,H}
d_{ij}
$$

该项鼓励高意图边优先出现在空间上合理的邻近区域。

如果候选图已经由 KNN 限制得很稀疏，这一项可以不用。

# 以下项目实现的工程代码目前有输入文件：

## 因果图

该最终核心因果子图文件为 **Pickle 序列化字典**，包含 3 个一级键：`graph`、`val_matrix` 和 `meta`，分别存储因果图结构、因果连接强度及子图元信息。

---

### 1. 数据总体结构
根据 `visualize_mask_subgraph.py` 第 171-191 行的保存逻辑，该 `.pkl` 文件解包后为一个字典，结构如下：
```python
{
 'graph': np.ndarray,        # 因果连接关系
 'val_matrix': np.ndarray,   # 因果连接强度
 'meta': dict                # 子图元数据与节点筛选信息
}
```

### 2. 核心字段内涵详解

| 键名         | 数据结构                           | 维度                | 内涵说明                                                     |
| :----------- | :--------------------------------- | :------------------ | :----------------------------------------------------------- |
| `graph`      | `numpy.ndarray` (dtype=object/str) | `[N, N, tau_max+1]` | **因果图邻接张量**。元素值为 `'--'` 表示存在因果边，否则为空。维度分别对应：目标节点、源节点、时间滞后 $\tau \in \{0, 1, ..., \text{tau\_max}\}$。 |
| `val_matrix` | `numpy.ndarray` (dtype=float)      | `[N, N, tau_max+1]` | **因果效应强度矩阵**。存储 PCMCI 算法计算出的偏相关系数等数值，与 `graph` 一一对应，量化了对应因果边的强弱与方向（正负）。 |
| `meta`       | `dict`                             | -                   | **子图元信息**，包含阈值设定及核心节点筛选结果，详见下表。   |

### 3. `meta` 字典内涵详解
`meta` 键存储了基于 Mask 阈值和 Top 节点策略的过滤信息（第 175-191 行）：

| 键名                   | 数据类型     | 内涵说明                                                     |
| :--------------------- | :----------- | :----------------------------------------------------------- |
| `mask_threshold`       | `float`      | 生成此子图使用的掩码阈值（此文件为 `0.5`）。                 |
| `selected_edges_count` | `int`        | 被掩码和 Top 节点策略双重筛选后保留的因果边总数。            |
| `total_edges_count`    | `int`        | 原始完整 PCMCI 图中的因果边总数。                            |
| `top_nodes`            | `list[int]`  | **核心节点列表**。基于 Mask 得分选出的前 30% 关键节点全局 ID（0-99）。 |
| `top_nodes_count`      | `int`        | 核心节点数量。                                               |
| `top_ratio`            | `float`      | 核心节点占比（默认 `0.30`）。                                |
| `node_score_mode`      | `str`        | 节点评分模式（默认 `"inout"`，综合入度出度计算得分）。       |
| `edge_keep_mode`       | `str`        | 边保留模式（默认 `"or"`，只要边两端任一节点在 `top_nodes` 中即保留）。 |
| `node_scores`          | `array-like` | 全局所有节点的因果重要性得分，用于排序筛选 Top 节点。        |

### 4. 数据生成逻辑链
1. **掩码还原**：加载 `full_mask_matrix_*.npy` (400×400)，按时间块重塑为 `[100, 100, 4]` 格式。
2. **阈值过滤**：将重塑后的 Mask 大于 `0.5` 的位置提取为二值化掩码。
3. **节点筛选**：按 `inout` 模式计算节点得分，取前 30% 为 `top_nodes`。
4. **诱导子图**：从原始 PCMCI 结果中，仅保留 `top_nodes` 间（`or` 模式）且被二值化掩码激活的 `'--'` 边及对应 `val_matrix`，最终保存为该 `.pkl` 文件。

## 意图图npz 

该代码块保存了**所有展示窗口（时间步）的意图图动态拓扑与权重信息**，记录了从候选图中激活的意图子图 $G_I^t$ 的核心结构。 

### 1. 整体数据结构
保存的 `.npz` 文件包含 3 个数组，它们通过索引一一对应，共同描述了 $N$ 个时间步的意图图演化：
- `edges_list`: 各时间步保留的意图边索引
- `strength_list`: 对应边的意图强度得分
- `delta_list`: 各时间步的意图激活阈值

---

### 2. 各字段详细内涵

#### 📌 `edges_list` (形状: `[n_show]` 的对象数组)
- **数据类型**: 每个元素是一个 `int` 型的 2D 数组，形状为 `[|E_I^t|, 2]`。
- **内涵**: 记录了该时间步 $t$ 被激活的意图边。数组的两列分别代表候选图 `cand` 中有向边的起点索引 `src` 和终点索引 `dst`。
- **来源**: 由 `build_intent_graph` 函数中 `np.stack([src[keep], dst[keep]], axis=1)` 生成（见 `260628/deig/intent_graph.py:14`）。

#### 📌 `strength_list` (形状: `[n_show]` 的对象数组)
- **数据类型**: 每个元素是一个 `float` 型的 1D 数组，形状为 `[|E_I^t|]`。
- **内涵**: 与 `edges_list` 中的边顺序严格对齐，记录了每条有向边的**意图强度** $s_{ij}$。该得分越高，表示节点 $i$ 对节点 $j$ 的意图驱动作用越强。
- **来源**: 由模型推理得分 `s_vec` 经分位数阈值过滤后保留的部分 `s_vec[keep]` 生成（见 `260628/deig/intent_graph.py:13`）。

#### 📌 `delta_list` (形状: `[n_show]` 的 1D 数组)
- **数据类型**: `float` 型标量组成的 1D 数组。
- **内涵**: 记录了该时间步生成意图图时使用的**激活阈值** $\delta$。它是该时间步所有边得分 `s_vec` 的 $q=0.90$ 分位数（见 `260628/deig/intent_graph.py:10`）。只有 $s_{ij} \ge \delta$ 的边才会出现在 `edges_list` 中。

---

### 3. 数据关联与使用逻辑
这三个数组完整刻画了动态意图图 $G_I^t$：
1. **筛选机制**: `delta_list[t]` 决定了当前时间步的意图稀疏度。
2. **拓扑与权重**: 遍历 `edges_list[t]` 和 `strength_list[t]`，即可重构该时间步的带权有向意图图邻接矩阵 $A$（如 `node_intent_attributes` 函数所消费的数据）。
3. **时空演化**: 通过对比不同时间步 $t$ 的这三个变量，可以分析意图热点的产生、消亡与转移。


算法方案整体输入为：  

$$
G_C,\quad G_I^{t-r},\quad G_I^t,\quad G_I^{t+r},\quad X_{t-L+1:t}
$$

整体输出为：

$$
G_D^t
$$

其中 $G_D^t$ 是用于弹性传感网部署优化的动态核心子图。

整体流程为：

$$
\boxed{
(G_C,G_I^{t-r},G_I^t,G_I^{t+r},X_{t-L+1:t})
\rightarrow
\mathcal{A}^{t-r:t+r}
\rightarrow
\{H_m^t,\mathbf{g}_m^t\}_{m\in\mathcal{M}}
\rightarrow
S_D^t
\rightarrow
G_D^t
}
$$

其中：

- $\mathcal{A}^{t-r:t+r}$：前后窗口构造得到的因果-意图联合多视图；
- $H_m^t$：经过 GNN + TCN 后得到的视图 $m$ 的节点级时空表示；
- $\mathbf{g}_m^t$：经过 GNN + TCN 后得到的视图 $m$ 的图级时空表示；
- $S_D^t$：最终融合边分数矩阵；
- $G_D^t$：最终部署子图。

---

# 2. 模块一：残差图构造

模块一负责构造因果-意图联合多视图结构。

对任意时间窗口：

$$
\tau\in\{t-r,t,t+r\}
$$

给定归一化因果图：

$$
\tilde{A}_C
$$

和归一化意图图：

$$
\tilde{A}_I^\tau
$$

先构造因果掩码：

$$
M_C[i,j]
=
\sigma
\left(
\frac{\tilde{A}_C[i,j]-\delta_C}{\eta}
\right)
$$

构造意图掩码：

$$
M_I^\tau[i,j]
=
\sigma
\left(
\frac{\tilde{A}_I^\tau[i,j]-\delta_I}{\eta}
\right)
$$

其中：

- $\delta_C$：因果边激活阈值；
- $\delta_I$：意图边激活阈值；
- $\eta$：温度系数；
- $\sigma(\cdot)$：Sigmoid 函数。

---

## 2.1 因果视图

因果视图为：

$$
A_C^\tau
=
\tilde{A}_C
$$

虽然因果图结构本身不随时间变化，但由于节点状态 $X_\tau$ 随时间变化，因此因果视图的编码表示仍然是动态的。

---

## 2.2 当前意图视图

意图视图为：

$$
A_I^\tau
=
\tilde{A}_I^\tau
$$

该视图表示当前窗口任务目标所激活的意图结构。

---

## 2.3 因果-意图共识视图

因果-意图共识图为：

$$
A_{CI}^{\tau}
=
\tilde{A}_C
\odot
\tilde{A}_I^\tau
$$

即：

$$
A_{CI}^{\tau}[i,j]
=
\tilde{A}_C[i,j]\cdot \tilde{A}_I^\tau[i,j]
$$

该视图表示：

$$
\text{既属于因果骨架，又被当前意图激活的结构}
$$

---

## 2.4 意图残差视图

意图残差图为：

$$
R_I^\tau
=
(1-M_C)\odot \tilde{A}_I^\tau
$$

即：

$$
R_I^\tau[i,j]
=
(1-M_C[i,j])\tilde{A}_I^\tau[i,j]
$$

该视图表示：

$$
\text{当前意图图中不在因果骨架内的部分}
$$

它可能是任务热点，也可能是噪声，需要后续门控判断。

---

## 2.5 因果未激活视图

因果未激活图为：

$$
R_C^\tau
=
\tilde{A}_C\odot(1-M_I^\tau)
$$

即：

$$
R_C^\tau[i,j]
=
\tilde{A}_C[i,j](1-M_I^\tau[i,j])
$$

该视图表示：

$$
\text{因果骨架中当前未被意图激活的部分}
$$

它主要对应抗扰兜底结构。

---

## 2.6 意图持续、新增与消退结构

为了显式保留意图图前后关系，在模块一中直接构造前后关系图，不再单独设置时序模块。

意图持续图为：

$$
A_P^t
=
\min
(
\tilde{A}_I^t,
\tilde{A}_I^{t-r}
)
$$

意图新增图为：

$$
A_N^t
=
[
\tilde{A}_I^t-\tilde{A}_I^{t-r}
]_+
$$

意图消退图为：

$$
A_E^t
=
[
\tilde{A}_I^{t-r}-\tilde{A}_I^t
]_+
$$

其中：

$$
[x]_+=\max(x,0)
$$

---

## 2.7 多视图集合

最终用于编码和融合的候选视图集合为：

$$
\mathcal{M}
=
\{
C,I,CI,R,C\setminus I,P,N
\}
$$

对应邻接矩阵集合为：

$$
\mathcal{A}^t
=
\{
A_C^t,
A_I^t,
A_{CI}^t,
R_I^t,
R_C^t,
A_P^t,
A_N^t
\}
$$

其中：

$$
A_C^t=\tilde{A}_C
$$

$$
A_I^t=\tilde{A}_I^t
$$

$$
R_I^t=A_R^t
$$

$$
R_C^t=A_{C\setminus I}^t
$$

意图消退图 $A_E^t$ 不直接作为部署候选边来源，而是作为门控状态特征输入，用于判断当前意图结构是否正在消退。

---

# 3. 模块二：多视图时空图编码

模块二现在同时完成两件事：

$$
\boxed{
\text{空间图编码}
+
\text{时间序列编码}
}
$$

即：

$$
\boxed{
\text{GNN/GAT/GraphSAGE}
+
\text{TCN}
}
$$

其作用是：

$$
\text{对每个因果-意图联合视图，提取节点级和图级的时空表示}
$$

---

## 3.1 空间图编码

对每个时间窗口：

$$
\tau\in \mathcal{T}_t
$$

其中：

$$
\mathcal{T}_t=\{t-L+1,t-L+2,\dots,t\}
$$

或者在三窗口写法下：

$$
\mathcal{T}_t=\{t-r,t,t+r\}
$$

对每个视图：

$$
m\in\mathcal{M}
$$

使用 GNN 编码：

$$
Z_m^\tau
=
\text{GNN}_m(X_\tau,A_m^\tau)
$$

其中：

$$
Z_m^\tau
=
[
\mathbf{z}_{1,m}^\tau,
\mathbf{z}_{2,m}^\tau,
\dots,
\mathbf{z}_{N,m}^\tau
]^\top
$$

$$
Z_m^\tau\in \mathbb{R}^{N\times d}
$$

第 $i$ 个节点在时间 $\tau$、视图 $m$ 下的空间表示为：

$$
\mathbf{z}_{i,m}^\tau\in\mathbb{R}^{d}
$$

如果使用 GAT，则可写为：

$$
\mathbf{z}_{i,m}^{\tau,l}
=
W_m^l\mathbf{h}_{i,m}^{\tau,l-1}
$$

$$
e_{ij,m}^{\tau,l}
=
\text{LeakyReLU}
\left(
\mathbf{a}_m^{l\top}
[
\mathbf{z}_{i,m}^{\tau,l}
\Vert
\mathbf{z}_{j,m}^{\tau,l}
\Vert
A_m^\tau[i,j]
]
\right)
$$

$$
\alpha_{ij,m}^{\tau,l}
=
\frac{
\exp(e_{ij,m}^{\tau,l})
}{
\sum_{k\in\mathcal{N}_m^\tau(i)}
\exp(e_{ik,m}^{\tau,l})+\varepsilon
}
$$

$$
\mathbf{h}_{i,m}^{\tau,l}
=
\sigma
\left(
\sum_{j\in\mathcal{N}_m^\tau(i)}
\alpha_{ij,m}^{\tau,l}
\mathbf{z}_{j,m}^{\tau,l}
\right)
$$

经过 $L_g$ 层后：

$$
Z_m^\tau
=
H_m^{\tau,L_g}
$$

---

## 3.2 节点级 TCN 时序编码

对节点 $i$ 和视图 $m$，将其在多个时间窗口下的空间表示组成序列：

$$
\mathcal{Z}_{i,m}^{t}
=
[
\mathbf{z}_{i,m}^{t-L+1},
\mathbf{z}_{i,m}^{t-L+2},
\dots,
\mathbf{z}_{i,m}^{t}
]
$$

输入节点级 TCN：

$$
\mathbf{h}_{i,m}^{t}
=
\text{TCN}_{node,m}
(
\mathcal{Z}_{i,m}^{t}
)
$$

其中：

$$
\mathbf{h}_{i,m}^{t}\in\mathbb{R}^{d_h}
$$

表示节点 $i$ 在视图 $m$ 下的时空表示。

对所有节点堆叠得到：

$$
H_m^t
=
[
\mathbf{h}_{1,m}^{t},
\mathbf{h}_{2,m}^{t},
\dots,
\mathbf{h}_{N,m}^{t}
]^\top
$$

其中：

$$
H_m^t\in\mathbb{R}^{N\times d_h}
$$

因此，$H_m^t$ 已经不是单纯的当前空间表示，而是包含历史变化的时空表示。

---

## 3.3 图级 TCN 时序编码

先对每个时间窗口和视图做 READOUT：

$$
\mathbf{u}_m^\tau
=
\text{READOUT}(Z_m^\tau)
$$

例如：

$$
\beta_{i,m}^{\tau}
=
\frac{
\exp
(
\mathbf{w}_r^\top
\tanh(W_r\mathbf{z}_{i,m}^{\tau})
)
}{
\sum_{k=1}^{N}
\exp
(
\mathbf{w}_r^\top
\tanh(W_r\mathbf{z}_{k,m}^{\tau})
)
+\varepsilon
}
$$

$$
\mathbf{u}_m^\tau
=
\sum_{i=1}^{N}
\beta_{i,m}^{\tau}
\mathbf{z}_{i,m}^{\tau}
$$

然后形成图级时间序列：

$$
\mathcal{U}_m^t
=
[
\mathbf{u}_m^{t-L+1},
\mathbf{u}_m^{t-L+2},
\dots,
\mathbf{u}_m^{t}
]
$$

输入图级 TCN：

$$
\mathbf{g}_m^t
=
\text{TCN}_{graph,m}
(
\mathcal{U}_m^t
)
$$

其中：

$$
\mathbf{g}_m^t\in\mathbb{R}^{d_g}
$$

表示视图 $m$ 在当前窗口下的图级时空表示。

---

## 3.4 模块二输出

模块二最终输出为：

$$
\boxed{
\{H_m^t,\mathbf{g}_m^t\}_{m\in\mathcal{M}}
}
$$

也就是：

$$
\{
H_C^t,\mathbf{g}_C^t,
H_I^t,\mathbf{g}_I^t,
H_{CI}^t,\mathbf{g}_{CI}^t,
H_R^t,\mathbf{g}_R^t,
H_{C\setminus I}^t,\mathbf{g}_{C\setminus I}^t,
H_P^t,\mathbf{g}_P^t,
H_N^t,\mathbf{g}_N^t
\}
$$

这些表示将直接输入后续动态图主导门控。

因此，模块二的作用非常明确：

$$
\boxed{
\text{为动态图主导门控提供多视图、时空化、可学习的节点级和图级特征}
}
$$

---

# 4. 模块三：动态图主导门控

删掉独立时序模块后，动态图主导门控成为第三个模型模块。

它的输入包括两类信息：

1. 模块一得到的结构统计特征；
2. 模块二得到的多视图时空表示。

也就是：

$$
\boxed{
\mathbf{q}_{struct}^t,
\quad
\{H_m^t,\mathbf{g}_m^t\}_{m\in\mathcal{M}}
}
$$

---

## 4.1 图级结构统计特征

定义图级结构统计向量：

$$
\mathbf{q}_{struct}^t
=
[
o_{CI}^t,
r_{I\setminus C}^t,
r_{C\setminus I}^t,
p_I^{t,-},
p_I^{t,+},
d_I^{t,-},
d_I^{t,+},
n_I^t,
e_I^t
]
$$

其中，因果-意图重合度为：

$$
o_{CI}^t
=
\frac{
\sum_{i,j}
\min(\tilde{A}_C[i,j],\tilde{A}_I^t[i,j])
}{
\sum_{i,j}
\max(\tilde{A}_C[i,j],\tilde{A}_I^t[i,j])
+\varepsilon
}
$$

意图残差比例为：

$$
r_{I\setminus C}^t
=
\frac{
\sum_{i,j}R_I^t[i,j]
}{
\sum_{i,j}\tilde{A}_I^t[i,j]+\varepsilon
}
$$

因果未激活比例为：

$$
r_{C\setminus I}^t
=
\frac{
\sum_{i,j}R_C^t[i,j]
}{
\sum_{i,j}\tilde{A}_C[i,j]+\varepsilon
}
$$

过去意图持续性为：

$$
p_I^{t,-}
=
J(\tilde{A}_I^t,\tilde{A}_I^{t-r})
$$

未来意图延续性为：

$$
p_I^{t,+}
=
J(\tilde{A}_I^t,\tilde{A}_I^{t+r})
$$

过去意图变化强度为：

$$
d_I^{t,-}
=
\frac{
\|\tilde{A}_I^t-\tilde{A}_I^{t-r}\|_1
}{
\|\tilde{A}_I^{t-r}\|_1+\varepsilon
}
$$

未来意图变化强度为：

$$
d_I^{t,+}
=
\frac{
\|\tilde{A}_I^{t+r}-\tilde{A}_I^t\|_1
}{
\|\tilde{A}_I^{t}\|_1+\varepsilon
}
$$

意图新增比例为：

$$
n_I^t
=
\frac{
\sum_{i,j}A_N^t[i,j]
}{
\sum_{i,j}\tilde{A}_I^t[i,j]+\varepsilon
}
$$

意图消退比例为：

$$
e_I^t
=
\frac{
\sum_{i,j}A_E^t[i,j]
}{
\sum_{i,j}\tilde{A}_I^{t-r}[i,j]+\varepsilon
}
$$

这些前后关系不再由独立时序模块处理，而是作为门控结构特征直接输入。

---

## 4.2 图级主导门控

先拼接所有视图的图级时空表示：

$$
\mathbf{g}_{all}^t
=
[
\mathbf{g}_C^t
\Vert
\mathbf{g}_I^t
\Vert
\mathbf{g}_{CI}^t
\Vert
\mathbf{g}_R^t
\Vert
\mathbf{g}_{C\setminus I}^t
\Vert
\mathbf{g}_P^t
\Vert
\mathbf{g}_N^t
]
$$

图级门控输入为：

$$
\mathbf{x}_g^t
=
[
\mathbf{q}_{struct}^t
\Vert
\mathbf{g}_{all}^t
]
$$

图级主导权重为：

$$
\boldsymbol{\pi}^t
=
\text{softmax}
(
\text{MLP}_g(\mathbf{x}_g^t)
)
$$

即：

$$
\boldsymbol{\pi}^t
=
[
\pi_C^t,
\pi_I^t,
\pi_{CI}^t,
\pi_R^t,
\pi_{C\setminus I}^t,
\pi_P^t,
\pi_N^t
]
$$

其中：

- $\pi_C^t$：因果骨架主导权重；
- $\pi_I^t$：当前意图主导权重；
- $\pi_{CI}^t$：因果-意图共识主导权重；
- $\pi_R^t$：意图残差主导权重；
- $\pi_{C\setminus I}^t$：因果未激活兜底权重；
- $\pi_P^t$：意图持续结构权重；
- $\pi_N^t$：意图新增结构权重。

满足：

$$
\sum_{m\in\mathcal{M}}\pi_m^t=1
$$

$$
\pi_m^t\ge 0
$$

---

## 4.3 节点级 ego-graph 门控

对节点 $i$，构造其多视图节点时空表示：

$$
\mathbf{h}_{i,all}^t
=
[
\mathbf{h}_{i,C}^t
\Vert
\mathbf{h}_{i,I}^t
\Vert
\mathbf{h}_{i,CI}^t
\Vert
\mathbf{h}_{i,R}^t
\Vert
\mathbf{h}_{i,C\setminus I}^t
\Vert
\mathbf{h}_{i,P}^t
\Vert
\mathbf{h}_{i,N}^t
]
$$

构造节点 $i$ 的局部结构统计量：

$$
\mathbf{q}_{i,struct}^t
=
[
o_{CI,i}^t,
r_{I\setminus C,i}^t,
r_{C\setminus I,i}^t,
p_{I,i}^{t,-},
n_{I,i}^t
]
$$

其中这些局部指标均在节点 $i$ 的 $k$-跳 ego-graph 内计算。

节点级门控输入为：

$$
\mathbf{x}_i^t
=
[
\mathbf{q}_{i,struct}^t
\Vert
\mathbf{h}_{i,all}^t
\Vert
\boldsymbol{\pi}^t
]
$$

节点级局部门控为：

$$
\boldsymbol{\gamma}_i^t
=
\text{softmax}
(
\text{MLP}_{node}(\mathbf{x}_i^t)
)
$$

其中：

$$
\boldsymbol{\gamma}_i^t
=
[
\gamma_{i,C}^t,
\gamma_{i,I}^t,
\gamma_{i,CI}^t,
\gamma_{i,R}^t,
\gamma_{i,C\setminus I}^t,
\gamma_{i,P}^t,
\gamma_{i,N}^t
]
$$

该权重表示节点 $i$ 所在局部区域更倾向由哪类图结构主导。

---

## 4.4 边级门控

对边 $(i,j)$，构造边级结构特征：

$$
\mathbf{b}_{ij}^t
=
[
A_C^t[i,j],
A_I^t[i,j],
A_{CI}^t[i,j],
R_I^t[i,j],
R_C^t[i,j],
A_P^t[i,j],
A_N^t[i,j],
A_E^t[i,j]
]
$$

构造两端节点的融合表示：

$$
\mathbf{h}_{i,D}^t
=
\sum_{m\in\mathcal{M}}
\pi_m^t
\mathbf{h}_{i,m}^t
$$

$$
\mathbf{h}_{j,D}^t
=
\sum_{m\in\mathcal{M}}
\pi_m^t
\mathbf{h}_{j,m}^t
$$

边级门控输入为：

$$
\mathbf{x}_{ij}^t
=
[
\mathbf{b}_{ij}^t
\Vert
\boldsymbol{\gamma}_i^t
\Vert
\boldsymbol{\gamma}_j^t
\Vert
\mathbf{h}_{i,D}^t
\Vert
\mathbf{h}_{j,D}^t
\Vert
\boldsymbol{\pi}^t
]
$$

边级局部门控为：

$$
\boldsymbol{\omega}_{ij}^t
=
\text{softmax}
(
\text{MLP}_{edge}(\mathbf{x}_{ij}^t)
)
$$

其中：

$$
\boldsymbol{\omega}_{ij}^t
=
[
\omega_{ij,C}^t,
\omega_{ij,I}^t,
\omega_{ij,CI}^t,
\omega_{ij,R}^t,
\omega_{ij,C\setminus I}^t,
\omega_{ij,P}^t,
\omega_{ij,N}^t
]
$$

---

## 4.5 全局-局部联合门控

图级门控 $\pi_m^t$ 判断当前窗口整体哪类结构占主导。

边级门控 $\omega_{ij,m}^t$ 判断边 $(i,j)$ 所在局部区域哪类结构占主导。

最终联合门控权重为：

$$
\alpha_{ij,m}^t
=
\frac{
\pi_m^t\omega_{ij,m}^t
}{
\sum_{n\in\mathcal{M}}
\pi_n^t\omega_{ij,n}^t
+\varepsilon
}
$$

最终动态融合边分数为：

$$
S_D^t[i,j]
=
\sum_{m\in\mathcal{M}}
\alpha_{ij,m}^tA_m^t[i,j]
$$

展开为：

$$
\begin{aligned}
S_D^t[i,j]
=&
\alpha_{ij,C}^tA_C^t[i,j]
+
\alpha_{ij,I}^tA_I^t[i,j]
+
\alpha_{ij,CI}^tA_{CI}^t[i,j]
\\
&
+
\alpha_{ij,R}^tR_I^t[i,j]
+
\alpha_{ij,C\setminus I}^tR_C^t[i,j]
+
\alpha_{ij,P}^tA_P^t[i,j]
+
\alpha_{ij,N}^tA_N^t[i,j]
\end{aligned}
$$

这一步输出：

$$
S_D^t
$$

即动态部署融合图的边分数矩阵。

---

# 5. 模块四：部署子图生成与验证

根据融合边分数 $S_D^t$ 生成部署子图。

节点分数为：

$$
Score_i^t
=
\sum_{j=1}^{N}S_D^t[i,j]
+
\sum_{j=1}^{N}S_D^t[j,i]
+
\text{MLP}_{score}(\mathbf{h}_{i,D}^t)
$$

边分数为：

$$
Score_{ij}^t
=
S_D^t[i,j]
$$

节点部署变量为：

$$
p_i^t\in\{0,1\}
$$

边部署变量为：

$$
q_{ij}^t\in\{0,1\}
$$

预算约束为：

$$
\sum_i c_ip_i^t
+
\sum_{i,j}c_{ij}q_{ij}^t
\le B
$$

边选择依赖节点选择：

$$
q_{ij}^t\le p_i^t
$$

$$
q_{ij}^t\le p_j^t
$$

部署优化目标为：

$$
G_D^t
=
\arg\max_{p,q}
\left[
U_{pred}(p,q,t)
+
\lambda_{el}U_{elastic}(p,q,t)
+
\lambda_{tr}U_{track}(p,q,t)
-
\lambda_cCost(p,q)
\right]
$$

其中：

$$
Cost(p,q)
=
\sum_i c_ip_i^t
+
\sum_{i,j}c_{ij}q_{ij}^t
$$

预测效用为：

$$
U_{pred}(p,q,t)
=
-
\ell
(
\widehat{Y}_t^{G(p,q)},
Y_t
)
$$

弹性效用为：

$$
U_{elastic}(p,q,t)
=
-
\mathbb{E}_{\delta\sim\mathcal{D}_{fail}}
\left[
\ell
(
\widehat{Y}_t^{G(p,q)\setminus\delta},
Y_t
)
\right]
$$

任务追踪效用为：

$$
U_{track}(p,q,t)
=
J(A_{G(p,q)},A_I^t)
+
J(A_{G(p,q)},A_P^t)
$$

第一版实现时，可以不求解复杂整数规划，而采用 top-$K$ 或贪心策略生成部署图。

---

# 6. 修正后的模块输入输出关系

## 模块一：残差图构造

输入：

$$
G_C,\quad G_I^{t-r},\quad G_I^t,\quad G_I^{t+r}
$$

输出：

$$
\mathcal{A}^t,
\quad
\mathbf{q}_{struct}^t,
\quad
\mathbf{b}_{ij}^t
$$

---

## 模块二：多视图时空图编码

输入：

$$
X_{t-L+1:t},
\quad
\mathcal{A}^{t-L+1:t}
$$

输出：

$$
\{H_m^t,\mathbf{g}_m^t\}_{m\in\mathcal{M}}
$$

其中：

$$
H_m^t
=
\text{TCN}_{node,m}
(
[
\text{GNN}_m(X_\tau,A_m^\tau)
]_{\tau=t-L+1}^{t}
)
$$

$$
\mathbf{g}_m^t
=
\text{TCN}_{graph,m}
(
[
\text{READOUT}(\text{GNN}_m(X_\tau,A_m^\tau))
]_{\tau=t-L+1}^{t}
)
$$

---

## 模块三：动态图主导门控

输入：

$$
\mathbf{q}_{struct}^t,
\quad
\mathbf{b}_{ij}^t,
\quad
\{H_m^t,\mathbf{g}_m^t\}_{m\in\mathcal{M}}
$$

输出：

$$
\boldsymbol{\pi}^t,
\quad
\boldsymbol{\gamma}_i^t,
\quad
\boldsymbol{\omega}_{ij}^t,
\quad
S_D^t
$$

---

## 模块四：部署子图生成与验证

输入：

$$
S_D^t,
\quad
\mathbf{h}_{i,D}^t,
\quad
B
$$

输出：

$$
G_D^t
$$

---

# 7. 最终简洁流程

最终可以写成：

$$
\begin{aligned}
&\textbf{Step 1：因果-意图残差多视图构造} \\
&
(G_C,G_I^{t-r},G_I^t,G_I^{t+r})
\rightarrow
\mathcal{A}^t
\\[6pt]
&\textbf{Step 2：多视图时空图编码} \\
&
(X_{t-L+1:t},\mathcal{A}^{t-L+1:t})
\rightarrow
\{H_m^t,\mathbf{g}_m^t\}_{m\in\mathcal{M}}
\\[6pt]
&\textbf{Step 3：动态图主导门控} \\
&
(\mathbf{q}_{struct}^t,\mathbf{b}_{ij}^t,\{H_m^t,\mathbf{g}_m^t\})
\rightarrow
S_D^t
\\[6pt]
&\textbf{Step 4：部署子图生成与验证} \\
&
S_D^t
\rightarrow
G_D^t
\end{aligned}
$$

---

# 8. 这样修改后的逻辑优势

修改后，流程更顺：

第一，模块一负责结构分解：

$$
\text{因果}
+
\text{意图}
+
\text{共识}
+
\text{残差}
+
\text{兜底}
+
\text{持续}
+
\text{新增}
$$

第二，模块二负责统一编码：

$$
\text{GNN 提取空间结构}
$$

$$
\text{TCN 提取时序变化}
$$

第三，模块三直接用模块二输出的节点级、图级时空表示做门控：

$$
\{H_m^t,\mathbf{g}_m^t\}
\rightarrow
\boldsymbol{\pi}^t,
\boldsymbol{\gamma}_i^t,
\boldsymbol{\omega}_{ij}^t
$$

第四，模块四生成部署图：

$$
S_D^t\rightarrow G_D^t
$$

因此，模块二不再是“算了但没用”，而是动态图主导门控的表示基础。

---

# 9. 最终一句话表述

修正后的 RA-TCGF 可以概括为：

$$
\boxed{
\text{RA-TCGF 先将因果图与意图图分解为多种联合残差视图，再通过 GNN-TCN 编码每个视图的时空表示，随后利用图级与 ego-graph 级动态门控判断当前部署应由因果骨架、意图热点、共识结构、残差结构或兜底结构主导，最终生成满足预算约束的弹性传感网动态部署子图。}
}
$$