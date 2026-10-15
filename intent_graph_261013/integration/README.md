# 连续绘图接入状态

consecutive_training.patch 保存此前提供的最小接入方案，现已集成到 run_experiment.py、experiment/evaluation.py 和 experiment/artifacts.py，无需再次应用。

构建实验默认启用连续十个测试窗口图片；旧六模型入口可通过 sequence-plots 参数指向 configs/sequence_plots.json 启用。no-plots 关闭渲染。两种入口都复用同一个独立补绘模块。

补丁仅用于追溯局部修改位置，不作为当前版本的安装步骤。
