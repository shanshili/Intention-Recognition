# 已保存门控的连续窗口补绘与训练接入

## 1. 本轮实现范围

独立入口为 plot_intent_sequences.py；绘制实现为 experiment/sequence_visualization.py。入口只读取已保存的 edge_scores.npz 以及同一运行的 resolved_config.json、graph_evaluation.json，不读取原始流量、不加载模型、不重新推断、不训练。默认实验目录是 outputs/suite_20261013_221703_583502，默认测试分区、全部模型和全部种子。

现有训练入口与评估计算保持原状。integration/consecutive_training.patch 是可审阅、尚未应用的局部接入补丁；experiment/sequence_plot_hook.py 和 configs/sequence_plots.json 为补丁提供公共调用与参数。

## 2. 与 260628 的对应

旧版 main.py 选择指定起点之后的连续测试窗口，逐帧调用 plot_single_intent_window，前若干帧调用 plot_intent_windows，并保存候选图及密度曲线。本次恢复这些产物及时间标签，继续使用有向箭头、时间戳命名和 PNG、SVG、PDF 保存。

旧版 main.py 的实际 viz-count 默认值为 10，文件中的“97个”注释不是当前默认。本入口默认同样为 10，可显式设置为 97 或其他数量。97 个 stride=1 的五分钟预测原点，从首到末相差 8 小时；这与最后一个窗口的完整未来目标区间不是同一个跨度。

新模块保留原始分数为零不绘边、同一序列共同颜色尺度等约定。旧版每窗口分位数筛图作为 window_quantile 探索模式保留；默认使用与现有评估一致的精确 Top-K。

## 3. 入口参数

所有参数的完整说明由入口帮助提供；以下为主要设置及默认值。

| 参数 | 默认 | 含义 |
|---|---|---|
| suite | 指定的 221703 实验 | 读取哪一次实验，不默认猜测最新目录 |
| models、seeds | 全部已保存模型、种子 | 可缩小到 DynamicGate 或某个种子 |
| splits | test | 可同时选择 train、val、test |
| viz-start-test／start | 0 | 所选分区内部窗口序号，从零开始；名称保留旧版别名 |
| viz-count | 10 | 从起点起连续保存多少个窗口，尾部不足则明确记录截断 |
| panel-count | 10 | 前多少个连续窗口放进组合图；0 关闭组合图 |
| mode | topk | topk、saved_threshold 或 window_quantile |
| fraction | 0.1 | Top-K 的候选边预算 |
| q | 不指定；分位数模式默认 0.9 | 单独提供 q 时选择旧版分位数模式 |
| formats | png、svg、pdf | 图片格式，可只保存 png |
| dpi | 300 | 栅格输出分辨率 |
| norm | log | 序列共同对数尺度或线性尺度 |
| vmin、vmax | 正分数的序列 1% 与 99% 分位数 | 两者必须一起指定；可使不同模型、种子共同使用固定颜色范围 |
| node-labels | index | 显示内部节点索引、原始节点 ID 或不显示 |
| node-size、arrow-size、min-linewidth、max-linewidth | 入口默认值 | 调整点、箭头、线宽 |
| figure-width、figure-height | 8、7 英寸 | 单图画布尺寸 |
| xlim、ylim | 保存坐标的完整范围 | 可放大某个局部；所有帧共用视野 |
| out | 每个分区的 figures/consecutive | 可将补绘产物集中放到另一个根目录 |
| list-only | 关闭 | 只列出可绘制及不适用的运行 |
| allow-gaps | 关闭 | 显式允许时间间断，并在清单中记录；默认拒绝非连续窗口 |

颜色在每个模型、种子的所选序列内共享，默认不自动跨模型共享。要比较不同种子的原始分数尺度，应指定相同 vmin、vmax；颜色一致也不代表分数是校准的贡献概率。

## 4. 三种筛边定义

令候选边数为 E，分数为 s。Top-K 预算为

$$
K=\min(E,\lceil pE\rceil).
$$

每窗按分数降序、候选边顺序稳定打破并列，精确选择 K 条边。选中的零分边仍计入预算，但不绘制为有效激活。文件分别记录 selected_edges 与 positive_drawn_edges，避免空图被伪装成有边。

saved_threshold 使用已经保存的验证期阈值或预声明固定阈值：

$$
E_I^t=\{e\in E_0:s_e^t>0,\ s_e^t\ge\delta_{val}\}.
$$

此模式不会从测试分数重新估计阈值。缺少合法保存阈值时记录失败，不静默改用测试分位数。

window_quantile 按旧版每个窗口计算

$$
\delta_t=Q_q(S_t),\qquad E_I^t=\{e:s_e^t>0,\ s_e^t\ge\delta_t\}.
$$

并列可能令边数超过名义比例；全部正分数相等时会保留全部。此模式仅用于探索图形，不是精确预算，也不提供分数是否整体消退的证据。

FixedGraph 全一门控始终绘制全部候选边，使用 all_candidate 模式并明确标注固定图、无学习排序。StaticGate 连续图片相同是模型定义。Persistence、Ridge-AR 和 Local 无门控文件，清单标记不适用；预期存在却缺失的图文件记为失败。

## 5. 连续性、时间和密度

窗口以保存的 window_id 及 target_times 校验。测试、验证按 eval_stride 连续，训练按 train_stride 连续；默认训练窗口原点相差 15 分钟，测试相差 5 分钟。

历史与目标区间为

$$
[t-W+1,t],\qquad[t+1,t+H].
$$

输入时间由已保存目标起点减去一步，再按 W 向前推导。图中文字和清单保留这些完整区间。时间来自该实验配置的规则网格；原数据真实日历起点尚未独立核验，不能把图上的日期当作已核实交通事件时间。

密度同时报告

$$
D_{cand}=|E_I|/E,\qquad D_{dir}=|E_I|/[N(N-1)].
$$

并分别记录选中密度和实际正分数绘制密度。Top-K 的选中密度由预算决定，平直曲线不是模型异常，也不能证明稀疏学习有效。

## 6. 输出与命名

默认每个有图的模型、种子、分区保存到该分区 figures/consecutive 下新的时间戳目录，再按筛边模式分层；重复补绘不会覆盖原来的图片和数组。

| 文件名主体 | 内容 |
|---|---|
| candidate_directed_graph | 全候选有向图参考 |
| intent_graph_window_0000 等 | 一帧一图，顺序号与 CSV 中 sequence_index 对齐 |
| intent_graphs_10windows 等 | 前若干连续窗口拼图 |
| intent_graph_density | 连续窗口选中／正分数密度曲线 |
| window_index | 分区序号、原点 ID、完整时间、阈值、边数及密度 CSV |
| manifest | 本序列来源、输入数组哈希、设置、颜色范围、输出路径 JSON |
| suite_plot_manifest | 整次批量补绘成功、不适用、失败的运行清单 |

图片主体之后附同一次补绘时间戳与扩展名。未发生模型推断和源数组改写也会写入批量清单。

## 7. 以后训练时保存连续图片的局部修改方案

可选补丁只涉及三个现有文件；本轮没有将该补丁应用到训练主流程。

| 文件 | 局部修改 | 原因 |
|---|---|---|
| run_experiment.py | 增加 sequence-plots 参数，读取绘图 JSON 并传给评价设置 | 通过入口选择是否启用和调节窗口、格式等 |
| experiment/evaluation.py | 增加可为空的 sequence_plots 字段，核验键名 | 与现有严格配置校验兼容；数值评价算法不变 |
| experiment/artifacts.py 的 save_run | 在数组、配置、图评价 JSON 保存后，既有绘图分支内调用 render_saved_split，将返回文件加入 plots | 复用离线渲染器，且清单完成前图文件已生成 |

render_saved_split 读取刚保存的分区产物，内部复用 load_sequence 与 render_sequence。无需在训练器、模型 forward、梯度更新或阈值标定代码中添加绘图。

configs/sequence_plots.json 提供一套可调整的默认设置。应用可选补丁后，入口通过 sequence-plots 指向该文件即可启用；未指定时保持原运行方式，no-plots 仍关闭全部绘图。

当前 save_multisplit_run 只对测试分区启用图片渲染，因此该最小补丁也只自动绘制测试序列。若希望训练、验证也自动出图，需要进一步将分区绘图选择从固定 test 改为显式分区集合；独立补绘入口已经支持三分区。

此方案将绘图设置随 resolved_config 保存；在当前架构中它也会进入 evaluation_signature。启用前后数值算法相同，但严格比较器会把不同绘图配置视为不同评价签名。这是沿用现有 max_plot_windows 的签名方式；后续可将纯展示签名与数值评价签名分开，不能为方便比较而删掉真正改变干预协议的字段。

## 8. 本轮验证记录

连续绘图专项测试覆盖精确预算与并列、零分不绘、训练步幅、时间间断、阈值来源、固定图、缺失运行失败及不加载训练依赖。另对指定实验全部有图模型与种子实际补绘，输入模型数为 3、图运行数为 9，另 5 个无图运行明确不适用。

先完成各 3 帧检查和验证阈值两帧三格式检查；最终批量示例为各 10 个相邻测试窗口、PNG 150 DPI。正式按默认三格式或更多窗口补绘时，可以独立运行入口，无需重新训练。

最终批量清单时间戳为 20261015_012721_351140：9 个图运行共 90 帧，另有候选图、拼图和密度曲线；5 个无图运行不适用，失败为0。可选接入补丁通过 git apply 的只读适用性检查，并在临时副本完成64个原始时刻、一种子、一轮神经训练及自动两帧保存验证。原有 Python 3.10 环境的完整回归测试为57项通过。
