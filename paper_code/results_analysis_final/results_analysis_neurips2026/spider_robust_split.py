#!/usr/bin/env python3
"""
Figure 7 (Sec. 6), split into two panels following reviewer W5:
  (a) Clean + common: model-independent perturbations (fair comparison).
  (b) l1 / l2 / linf at the vision-model budgets:
        solid   = transfer attacks from the strongest surrogate per dataset and
                  setting, applied identically to Ve-LLMs and ConvNeXt-B
        hatched = white-box attacks, vision models only (reference)

Same data and aggregation as spider_robust.py (sum of accuracies over the six
datasets, maximum 600). Run from anywhere:
  python results_analysis_neurips2026/spider_robust_split.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE
CONVNEXT_EVAL_DIR = HERE.parents[1] / "results" / "robustgenbench_eval" / "convnext_base_fb_in22k_TRADES_v2"

tbl = pd.read_csv(RESULTS_DIR / "transfer_table.csv")
ft = pd.read_csv(RESULTS_DIR / "fft50.csv", index_col=0)

DATASETS = sorted(tbl["dataset"].unique())
DATASET_NAME_MAP = {
    "Caltech-101": "caltech101",
    "FGVC Aircraft": "fgvc-aircraft-2013b",
    "Flowers-102": "flowers-102",
    "Oxford Pet": "oxford-iiit-pet",
    "Stanford Cars": "stanford_cars",
    "UC Merced": "uc-merced-land-use-dataset",
}
SURROGATES = [
    "zeroshot_clip_vitb16_laion2b", "zeroshot_clip_vith14_laion2b",
    "zeroshot_metaclip_vith14_fullcc2_5b", "zeroshot_siglip2_so400m_patch14_384",
]

# (label, transfer-table column, fft50 column, ConvNeXt eval label fragment)
MODEL_INDEP = [
    ("Clean", "clean", "clean_acc", "clean"),
    ("Common", "sev@3", "common_acc", "common_severity3"),
]
ADVERSARIAL = [
    (r"$\ell_1$@75", "l1@75", "L1_acc", "l1_eps75"),
    (r"$\ell_2$@2", "l2@2", "L2_acc", "l2_eps2"),
    (r"$\ell_\infty$@4/255", "linf@4", "Linf_acc", "linf_eps4"),
]


# ── Helpers (identical aggregation to spider_robust.py) ──────────────────────
def llm_sum(target, col):
    """Sum over datasets; adversarial = worst (strongest) surrogate per dataset."""
    if col == "clean":
        sub = tbl[(tbl["target"] == target) & (tbl["surrogate"] == "CLIP ViT-B/16")]
        vals = sub.set_index("dataset")["clean"]
    else:
        vals = tbl[tbl["target"] == target].groupby("dataset")[col].min()
    return float(sum(vals.get(ds, np.nan) for ds in DATASETS))


_ft_single = ft[(ft["backbone_name"] == "convnext_b,sup,in22k") & (ft["loss_function"] == "TRADES")]


def convnext_wb_sum(col):
    return float(_ft_single[col].sum() * 100)


def oracle_sum(col):
    return float(ft.groupby("dataset")[col].max().sum() * 100)


def _load_convnext_transfer():
    data = {}
    for csv_path in CONVNEXT_EVAL_DIR.glob("*.csv"):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                data[(row["dataset"], row["label"])] = float(row["accuracy"])
    if not data:
        raise FileNotFoundError(f"No ConvNeXt transfer results in {CONVNEXT_EVAL_DIR}")
    return data


_convnext_transfer = _load_convnext_transfer()


def convnext_transfer_sum(fragment):
    total = 0.0
    for ds in DATASETS:
        key = DATASET_NAME_MAP[ds]
        accs = [v for (d, lbl), v in _convnext_transfer.items()
                if d == key and any(s in lbl for s in SURROGATES) and fragment in lbl]
        total += min(accs) if accs else 0.0
    return total * 100


OPEN_SUMMARY = HERE.parents[1] / "vision_enabled_language_models" / "results_tables" / "results_summary.csv"
_open = pd.read_csv(OPEN_SUMMARY)
OPEN_THREAT = {"clean": "clean", "sev@3": "common", "l1@75": "l1_eps75",
               "l2@2": "l2_eps2", "linf@4": "linf_eps4"}


def open_sum(model_key, col):
    """Open-weight Ve-LLMs (Kimi, Qwen): same aggregation as llm_sum, i.e. sum
    over datasets, adversarial = strongest (worst-case) surrogate per dataset."""
    sub = _open[(_open["model"] == model_key) & (_open["threat"] == OPEN_THREAT[col])]
    return float(sub.groupby("dataset")["accuracy"].min().sum() * 100)


def convnext_eval_sum(label):
    """Clean / common accuracy of ConvNeXt-B on the released RobustGenBench
    images, i.e., the same images the Ve-LLMs were evaluated on."""
    return 100 * sum(_convnext_transfer.get((DATASET_NAME_MAP[ds], label), np.nan) for ds in DATASETS)


# ── Values ───────────────────────────────────────────────────────────────────
# Each entry: (model name in paper_style.COLORS, values, white_box)
# Panel (a): clean and common are model-independent, so ConvNeXt-B has one value.
# ConvNeXt-B is taken from the RobustGenBench evaluation (same images as the
# Ve-LLMs). NOTE: the Oracle only exists in the FFT-50 study pipeline (fft50.csv),
# whose common-corruption accuracies differ from the released-image evaluation
# (ConvNeXt-B: 445.3 in fft50.csv vs 414.9 on the released images).
panel_a = [
    ("Gemini 3 Flash", [llm_sum("Gemini Flash (no think)", c) for _, c, _, _ in MODEL_INDEP], False),
    ("GPT-4o mini", [llm_sum("GPT-4o", c) for _, c, _, _ in MODEL_INDEP], False),
    ("Kimi K2.5", [open_sum("kimi", c) for _, c, _, _ in MODEL_INDEP], False),
    ("Qwen3 VL 235B", [open_sum("qwen", c) for _, c, _, _ in MODEL_INDEP], False),
    ("ConvNeXt-B", [convnext_eval_sum("clean"), convnext_eval_sum("common__common_severity3")], False),
    ("Oracle", [oracle_sum(f) for _, _, f, _ in MODEL_INDEP], False),
]
panel_b = [
    ("Gemini 3 Flash", [llm_sum("Gemini Flash (no think)", c) for _, c, _, _ in ADVERSARIAL], False),
    ("GPT-4o mini", [llm_sum("GPT-4o", c) for _, c, _, _ in ADVERSARIAL], False),
    ("Kimi K2.5", [open_sum("kimi", c) for _, c, _, _ in ADVERSARIAL], False),
    ("Qwen3 VL 235B", [open_sum("qwen", c) for _, c, _, _ in ADVERSARIAL], False),
    ("ConvNeXt-B", [convnext_transfer_sum(fr) for _, _, _, fr in ADVERSARIAL], False),
    ("ConvNeXt-B", [convnext_wb_sum(f) for _, _, f, _ in ADVERSARIAL], True),
    ("Oracle", [oracle_sum(f) for _, _, f, _ in ADVERSARIAL], True),
]

print("Panel (a) - clean / common")
for name, v, _ in panel_a:
    print(f"  {name:<16}" + "  ".join(f"{x:6.1f}" for x in v))
print("Panel (b) - l1@75 / l2@2 / linf@4   (WB = white-box, hatched)")
for name, v, wb in panel_b:
    print(f"  {name + (' (WB)' if wb else ''):<16}" + "  ".join(f"{x:6.1f}" for x in v))

# ── Plot (shared style: paper_style.py) ─────────────────────────────────────
import sys
sys.path.insert(0, str(HERE))
from paper_style import FIG_WIDTH, apply_rcparams, grouped_bars, legend_above, legend_handle

apply_rcparams()
fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(FIG_WIDTH, 4.6),
                                 gridspec_kw={"hspace": 0.32})
fig.patch.set_facecolor("white")

grouped_bars(ax_a, [g[0] for g in MODEL_INDEP], panel_a, ymax=600,
             ylabel="Sum score", percent=False, group_gap=0.4)
ax_a.set_title("(a) Model-independent (fair comparison)", loc="left", pad=3)
grouped_bars(ax_b, [g[0] for g in ADVERSARIAL], panel_b, ymax=600,
             ylabel="Sum score", percent=False, group_gap=0.3)
ax_b.set_title("(b) Adversarial perturbations", loc="left", pad=3)
for ax in (ax_a, ax_b):
    ax.set_yticks([0, 200, 400, 600])

handles = [
    legend_handle("Gemini 3 Flash"), legend_handle("GPT-4o mini"),
    legend_handle("Kimi K2.5"), legend_handle("Qwen3 VL 235B"),
    legend_handle("ConvNeXt-B"), legend_handle("Oracle"),
    legend_handle("ConvNeXt-B", white_box=True, label="ConvNeXt-B (white-box)"),
    legend_handle("Oracle", white_box=True, label="Oracle (white-box)"),
]
ax_b.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.12), handlelength=1.2, handleheight=0.9,
           handletextpad=0.4, columnspacing=0.8, labelspacing=0.3)

for ext in ("png", "pdf"):
    out = RESULTS_DIR / f"spider_robust_split.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved -> {out}")
