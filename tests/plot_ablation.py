"""
Visualize ablation_results.json.

Usage:
    conda run -n visualizer python tests/plot_ablation.py
    conda run -n visualizer python tests/plot_ablation.py --results /path/to/ablation_results.json
    conda run -n visualizer python tests/plot_ablation.py --last   # only show most recent run
"""

import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRICS = [
    ("focus_displacement",    "Focus Displacement",    "↓ smaller = more local"),
    ("global_drift",          "Global Drift",          "↓ smaller = more stable"),
    ("neighbor_preservation", "Neighbor Preservation", "↑ larger  = better topology"),
]

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--results", default=os.path.join(ROOT, "ablation_results.json"))
parser.add_argument("--last", action="store_true", help="Show only the most recent run")
parser.add_argument("--no-plot", action="store_true", help="Print table only, skip matplotlib")
args = parser.parse_args()

if not os.path.exists(args.results):
    print(f"[ERROR] Results file not found: {args.results}")
    sys.exit(1)

with open(args.results) as f:
    all_records = json.load(f)

records = [all_records[-1]] if args.last else all_records

# ── Table ─────────────────────────────────────────────────────────────────────
for rec in records:
    print(f"\n{'='*72}")
    print(f"  Timestamp : {rec['timestamp']}   Epoch: {rec['epoch']}   "
          f"N points: {rec['n_points']}")
    print(f"{'='*72}")
    runs = rec["runs"]
    col  = 18

    # Header
    header = f"  {'Metric':<28}" + "".join(f"  {r['config'][:col]:<{col}}" for r in runs)
    print(header)
    print("-" * 72)

    for key, label, hint in METRICS:
        vals = [r[key] for r in runs]
        best = min(vals) if "↓" in hint else max(vals)
        row  = f"  {label:<28}"
        for v in vals:
            marker = " ◀" if v == best else "  "
            row   += f"  {v:<{col-2}.4f}{marker}"
        print(row)
        print(f"  {'  '+hint:<28}" + "  " + "  ".join(f"{'':>{col}}" for _ in runs))

    print(f"\n  {'Time (s)':<28}" +
          "".join(f"  {r['elapsed_s']:<{col}.2f}" for r in runs))
    print(f"{'='*72}")

# ── Plot ──────────────────────────────────────────────────────────────────────
if args.no_plot:
    sys.exit(0)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("\n[INFO] matplotlib not available — table only.")
    sys.exit(0)

# Use last record for the bar chart
rec  = all_records[-1]
runs = rec["runs"]
names = [r["config"] for r in runs]

# Color scheme: A-group=blue, B-group=teal, C-group=coral; best gets accent
GROUP_COLORS = {
    "A": "#5B8DB8",   # steel blue
    "B": "#52A788",   # teal green
    "C": "#E07B54",   # warm coral
}
BEST_EDGE   = "#1a1a2e"
BEST_STAR_C = "#FFD700"  # gold highlight for best bar

def _group_color(name):
    return GROUP_COLORS.get(name[0].upper(), "#888888")

fig, axes = plt.subplots(1, len(METRICS), figsize=(5.2 * len(METRICS), 5))
fig.suptitle(
    f"Ablation Results  |  epoch={rec['epoch']}  n_pts={rec['n_points']}  "
    f"({rec['timestamp']})",
    fontsize=11, fontweight="bold", y=1.02
)

for ax, (key, label, hint) in zip(axes, METRICS):
    vals  = [r[key] for r in runs]
    best  = min(vals) if "↓" in hint else max(vals)
    bar_colors = [_group_color(n) for n in names]

    bars = ax.bar(range(len(names)), vals, color=bar_colors,
                  edgecolor="white", linewidth=0.8, width=0.65, zorder=3)

    # Dynamic y-axis: start slightly below min, end slightly above max
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.35 if hi != lo else hi * 0.05
    ax.set_ylim(lo - pad, hi + pad * 1.6)

    for bar, v, n in zip(bars, vals, names):
        is_best = v == best
        if is_best:
            bar.set_edgecolor(BEST_EDGE)
            bar.set_linewidth(2.2)
            bar.set_zorder(4)
        label_y = v + (hi - lo) * 0.04 + pad * 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                f"{v:.3f}" + (" ★" if is_best else ""),
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold" if is_best else "normal",
                color=BEST_EDGE if is_best else "#333333")

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax.set_title(f"{label}\n{hint}", fontsize=9.5, pad=8)
    ax.set_ylabel(label, fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

# Group legend
from matplotlib.patches import Patch
legend_handles = [Patch(color=c, label=f"{g}-group") for g, c in GROUP_COLORS.items()]
fig.legend(handles=legend_handles, loc="lower center", ncol=3,
           fontsize=8.5, framealpha=0.8, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout()
out = os.path.join(ROOT, "ablation_plot.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\n[Plot saved] → {out}")

# ── Multi-run trend (if more than one record) ─────────────────────────────────
if len(all_records) > 1:
    fig2, axes2 = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 4))
    fig2.suptitle("Ablation Trend across Multiple Runs", fontsize=11)
    cfg_names = [r["config"] for r in all_records[0]["runs"]]

    for ax, (key, label, hint) in zip(axes2, METRICS):
        for ci, cfg in enumerate(cfg_names):
            ys = [rec["runs"][ci][key] for rec in all_records if len(rec["runs"]) > ci]
            ax.plot(range(len(ys)), ys, marker="o", label=cfg,
                    color=colors[ci % len(colors)])
        ax.set_title(f"{label}\n{hint}", fontsize=9)
        ax.set_xlabel("Run index")
        ax.legend(fontsize=7)
        ax.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    out2 = os.path.join(ROOT, "ablation_trend.png")
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"[Trend plot saved] → {out2}")
