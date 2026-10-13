"""Conceptual protocol figure; all configurations are illustrative, not results."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = Path(__file__).resolve().parent
font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({
    "font.family": "Microsoft YaHei",
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})
fig = plt.figure(figsize=(15, 8.8), facecolor="#f7f9fc")
fig.text(.055, .947, "面向全路网预测的弹性观测配置", fontsize=24, weight="bold", color="#15283b")
fig.text(.055, .9, "研究协议示意｜固定传感器位置，调整观测集合；示例配置不代表算法结果或最优解",
         fontsize=13, color="#536778")
pos = np.array([[0,1],[1,1.5],[2,1],[3,1.5],[.5,0],[1.5,.2],[2.5,0],[3.5,.5]])
roads = [(0,1),(1,2),(2,3),(0,4),(1,5),(2,5),(2,6),(3,7),(4,5),(5,6),(6,7)]
cases = [
    ("时刻 t：预算 K = 3", {0,2,6}, set(), "观测 1、3、7；预测全部 8 个目标"),
    ("t + 10 min：预算 K = 3", {1,3,5}, set(), "数量相同，允许根据已有信息换点"),
    ("t + 20 min：预算 K = 2", {0,5}, {3}, "扩展示例：预算减少，节点 4 不可用"),
]
for k,(title, selected, failed, subtitle) in enumerate(cases):
    ax=fig.add_axes([.05+k*.315,.48,.285,.325])
    ax.set_facecolor("#ffffff")
    for u,v in roads:
        ax.plot(pos[[u,v],0],pos[[u,v],1],color="#c8d4de",linewidth=2,zorder=1)
    for i,xy in enumerate(pos):
        color = "#079b82" if i in selected else "#d9e4ec"
        if i in failed:
            ax.scatter(*xy, s=530, marker="X", c="#d76255", edgecolors="white", linewidths=2,zorder=3)
        else:
            ax.scatter(*xy,s=550,c=color,edgecolors="white",linewidths=2,zorder=3)
        ax.text(xy[0],xy[1],str(i+1),ha="center",va="center",
                color="white" if i in selected or i in failed else "#294557",fontsize=12,weight="bold",zorder=4)
    ax.set_title(title,fontsize=15,pad=15,color="#20374a",weight="bold")
    ax.set_xlim(-.4,3.95); ax.set_ylim(-.55,2.05)
    ax.set_aspect("equal"); ax.axis("off")
    fig.text(.05+k*.315,.445,subtitle,fontsize=11,color="#536778")
fig.text(.055,.4,"● 绿色：本周期获取观测    ● 浅色：未获取当前观测    × 红色：不可用    — 灰线：固定道路关系",
         fontsize=12,color="#41596b")
ax=fig.add_axes([.04,.15,.92,.20]); ax.axis("off")
labels=[
    ("已有合法历史", "已获取值、掩码、观测年龄"),
    ("选择观测配置", "预算与可用性约束"),
    ("获取选中节点", "其余当前值继续隐藏"),
    ("预测全路网未来", "所有目标节点，多个时距"),
    ("未来到来后评价", "预测误差、费用、可靠性"),
]
for k,(heading,desc) in enumerate(labels):
    x=.008+k*.200
    box=FancyBboxPatch((x,.17),.177,.69,boxstyle="round,pad=0.006,rounding_size=0.025",
        facecolor="#ffffff",edgecolor="#b8cbd8",linewidth=1.3,transform=ax.transAxes)
    ax.add_patch(box)
    ax.text(x+.0885,.61,heading,ha="center",va="center",fontsize=13,weight="bold",
            color="#20374a",transform=ax.transAxes)
    ax.text(x+.0885,.37,desc,ha="center",va="center",fontsize=10,
            color="#536778",transform=ax.transAxes)
    if k<4:
        ax.add_patch(FancyArrowPatch((x+.183,.515),(x+.196,.515),
            arrowstyle="-|>",mutation_scale=13,color="#6a8294",transform=ax.transAxes))
fig.text(.055,.09,"建议初始协议：原生采样 5 min｜集合调整 10 min｜预测更新 5 min｜预测时距 15 / 30 / 60 min",
         fontsize=13,color="#20374a")
fig.text(.055,.045,"关键检验：相同观测成本下，动态配置是否优于强静态选点、周期计划和轮询？",
         fontsize=13,color="#087e6c",weight="bold")
fig.savefig(OUT/"elastic_observation_protocol.png",dpi=160,facecolor=fig.get_facecolor())
fig.savefig(OUT/"elastic_observation_protocol.svg",facecolor=fig.get_facecolor())
plt.close(fig)
print("Saved conceptual protocol: PNG + SVG (not experimental results).")
