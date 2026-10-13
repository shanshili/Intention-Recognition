"""Render a descriptive figure from a saved run summary; no new training."""
from pathlib import Path
from datetime import datetime, timedelta
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "260628/outputs/20260731_205742_all/run_summary_20260731_205742.json"
OUT = Path(__file__).resolve().parent
data = json.loads(SOURCE.read_text(encoding="utf-8"))
density = np.asarray(data["visualization_density"], dtype=float)
hotspots = np.asarray(data["visualization_hotspot_count"], dtype=int)
assert len(density) == len(hotspots) == data["visualization_count"]
n = data["N"]
edge_counts_float = density * n * (n-1)
assert np.allclose(edge_counts_float, np.rint(edge_counts_float))
edge_counts = np.rint(edge_counts_float).astype(int)
start = datetime.fromisoformat(data["visualization_target_start"])
end = datetime.fromisoformat(data["visualization_target_end"])
times = [start + timedelta(hours=i*data["stride"]) for i in range(len(density))]
assert times[-1] == end
font_manager.fontManager.addfont("C:/Windows/Fonts/simhei.ttf")
plt.rcParams.update({
    "font.family": "SimHei", "font.size": 11,
    "axes.unicode_minus": False, "svg.fonttype": "path",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#CED5DF", "axes.labelcolor": "#28344A",
    "xtick.color": "#536178", "ytick.color": "#536178",
    "savefig.facecolor": "white",
})
fig, axes = plt.subplots(2, 1, figsize=(11.8, 6.7), sharex=True)
fig.subplots_adjust(left=.09, right=.96, top=.82, bottom=.21, hspace=.43)
fig.suptitle("现有运行中的“动态”：边数恒定，热点子图数量变化",
             x=.09, y=.965, ha="left", fontsize=19, color="#142239")
fig.text(.09, .905, "真实保存摘要  ·  97 个连续测试窗口  ·  每小时一张图  ·  未重新训练",
         color="#64748B", fontsize=11)

axes[0].plot(times, edge_counts, color="#3477BC", lw=2.6)
axes[0].scatter([times[0],times[-1]], [edge_counts[0],edge_counts[-1]], color="#3477BC",s=25,zorder=3)
axes[0].set_ylim(46,52)
axes[0].set_yticks([47,49,51])
axes[0].set_ylabel("保留边数")
axes[0].set_title("A  边的数量没有变化：所有窗口均保留 49 条",loc="left",fontsize=12,pad=10,color="#26354B")
axes[0].text(.99,.79,"488 条候选边，按 90% 分位数筛选",
             transform=axes[0].transAxes,ha="right",color="#536178",fontsize=10)

axes[1].step(times,hotspots,where="mid",color="#008777",lw=1.7)
axes[1].scatter(times,hotspots,color="#008777",s=9,zorder=3)
axes[1].set_ylim(.3,9.7)
axes[1].set_yticks([1,3,5,7,9])
axes[1].set_ylabel("热点子图数")
axes[1].set_title("B  连通结构发生变化：热点子图数量为 1–9 个",loc="left",fontsize=12,pad=10,color="#26354B")
axes[1].xaxis.set_major_locator(mdates.DayLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
axes[1].xaxis.set_minor_locator(mdates.HourLocator(byhour=[6,12,18]))
axes[1].set_xlabel("图标签时间（原运行的预测目标起点，2025 年）",labelpad=10)
for ax in axes:
    ax.grid(axis="y",color="#E7EBF0",lw=.8)
    ax.set_axisbelow(True)
    ax.margins(x=.01)

fig.text(.09,.115,"热点定义：筛选后意图图中，至少含两个节点的弱连通分量。",
         color="#425268",fontsize=10)
fig.text(.09,.074,"能说明结构重组；还需边权轨迹、边集合变化和对照实验，才能判断变化是否可靠且有用。",
         color="#425268",fontsize=10)
fig.text(.09,.033,"来源：260628 / 20260731_205742_all；观测截止为图标签时间前 1 小时。",
         color="#758297",fontsize=9)
for ext in ["png","svg"]:
    fig.savefig(OUT / ("dynamic_existing_summary."+ext),dpi=160)
plt.close(fig)

audit={
  "source":str(SOURCE),
  "source_sha256":hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
  "scope":"Descriptive analysis of saved summary, no new forecasting experiment",
  "windows":len(hotspots),
  "start_target_time":str(start),
  "end_target_time":str(end),
  "candidate_edges":data["candidate_edges"],
  "retained_edge_counts":np.unique(edge_counts).tolist(),
  "hotspot_min":int(hotspots.min()),
  "hotspot_max":int(hotspots.max()),
  "adjacent_hotspot_count_changes":int(np.count_nonzero(np.diff(hotspots))),
  "adjacent_pairs":len(hotspots)-1,
  "raw_edge_scores_available_in_run_directory":bool(list(SOURCE.parent.glob("edge_intent_scores_*.npz"))),
}
(OUT/"dynamic_summary_audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(audit,ensure_ascii=False,indent=2))

