"""
Shared style for the bar charts of Sections 5 and 6:
  - vlm_performance.png        (Figure 1, figure6.py)
  - clean_llm_performance.png  (Figure 2, figure5.py)
  - spider_robust_split.pdf    (Figure 7, spider_robust_split.py)

All three are drawn at the same physical size and font size, with the same
background, grid, legend placement, and no per-bar value labels (exact values
are in the appendix tables). Each model keeps one color across all figures.
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Sizing: every figure is 3.25 in wide and placed at 0.35\linewidth, ───────
# so fonts scale identically across figures.
FIG_WIDTH = 3.25
FONT_SIZE = 7

# ── One color per model, shared across figures ───────────────────────────────
COLORS = {
    # Target Ve-LLMs
    "Gemini 3 Flash": "#2ECC96",
    "GPT-4o mini":    "#5B9BD5",
    "Kimi K2.5":      "#F2C14E",
    "Qwen3 VL 235B":  "#8D6E63",
    # Fine-tuned vision models
    "ConvNeXt-B":     "#E05759",
    "Oracle":         "#F28E2B",
    # Surrogate VLMs (blue = CLIP family, purple = SigLIP family)
    "CLIP B/16":      "#90CAF9",
    "CLIP H/14":      "#42A5F5",
    "MetaCLIP H/14":  "#1A237E",
    "SigLIP2 base":   "#CE93D8",
    "SigLIP2 SO400M": "#AB47BC",
}

WHITE_BOX_HATCH = "////"


def apply_rcparams():
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.size":       FONT_SIZE,
        "axes.titlesize":  FONT_SIZE,
        "axes.labelsize":  FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi":      300,
        "hatch.linewidth": 0.6,
    })


def grouped_bars(ax, group_labels, series, ymax, ylabel, percent=True,
                 bar_width=None, group_gap=0.2):
    """Draw grouped bars.

    series: list of (name, values, white_box) tuples. White-box series are drawn
    hatched with the model's color, all others solid.
    """
    n_groups, n_series = len(group_labels), len(series)
    if bar_width is None:
        bar_width = 0.8 / n_series
    x = np.arange(n_groups) * (n_series * bar_width + group_gap)
    for i, (name, values, white_box) in enumerate(series):
        color = COLORS[name]
        pos = x + (i - n_series / 2 + 0.5) * bar_width
        if white_box:
            ax.bar(pos, values, width=bar_width, facecolor="white",
                   edgecolor=color, hatch=WHITE_BOX_HATCH, linewidth=0.6, zorder=3)
        else:
            ax.bar(pos, values, width=bar_width, color=color, zorder=3)

    ax.set_facecolor("#F5F5F5")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, linespacing=0.9)
    ax.set_ylim(0, ymax)
    if percent:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.set_ylabel(ylabel, labelpad=3)
    ax.yaxis.grid(True, color="white", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=2)


def legend_handle(name, white_box=False, label=None):
    color = COLORS[name]
    if white_box:
        return mpatches.Patch(facecolor="white", edgecolor=color,
                              hatch=WHITE_BOX_HATCH, label=label or name)
    return mpatches.Patch(color=color, label=label or name)


def legend_above(ax_or_fig, handles, ncol, anchor_y=1.0):
    """Legend centered above the plotting area, outside the bars."""
    kwargs = dict(handles=handles, loc="lower center", ncol=ncol, frameon=False,
                  handlelength=0.9, handletextpad=0.4, columnspacing=0.8,
                  labelspacing=0.3, borderaxespad=0.2)
    if hasattr(ax_or_fig, "transAxes"):
        return ax_or_fig.legend(bbox_to_anchor=(0.5, anchor_y), **kwargs)
    return ax_or_fig.legend(bbox_to_anchor=(0.5, anchor_y),
                            bbox_transform=ax_or_fig.transFigure, **kwargs)
