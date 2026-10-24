"""
Sum-score version of the Figure 4 interaction plots (FFT-50).

Replicates paper_figure3.ipynb (cells 5 and 7), which produced the Borda
figures in the paper, with the Sum score in place of the Borda score:
  - p-value: one-way ANOVA across all (var1, var2) cells (scipy f_oneway)
  - letters: Tukey HSD across all cells, alpha = 0.1, same greedy letter algorithm
  - same plot template (figure size, palette, legend, fonts, reference lines)

Only differences with the Borda template:
  - metric column 'score_sum' and y-label "Mean Sum Score"
  - y-limits use the same relative margins as the Borda plots
    (Borda: -100 / +250 on a range of 1952), since absolute offsets do not
    transfer to the Sum-score scale
  - Tukey letters are matched to bars by bar position rather than by
    container index, so letters stay on the right bar when some
    (var1, var2) combinations do not exist (seaborn-version independent)

Outputs:
  paper_figures/sumscore_interaction_{var1}_{var2}.png
  paper_figures/sumscore_interaction_means.csv   (exact cell means)

Usage (from this folder):
  python make_sum_interactions.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from statsmodels.stats.multicomp import MultiComparison

from load_results import load_result_dataset
from process_database import process_grouped_df, process_rankings

METRIC = 'score_sum'
Y_LABEL = "Mean Sum Score"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', '..', 'results', 'full_fine_tuning50')
OUT_DIR = os.path.join(HERE, 'paper_figures')

# Relative y-margins of the Borda template: ylim = (min - 100, max + 250), range = 2281 - 329
BORDA_RANGE = 2281.0 - 329.0
LOW_MARGIN = 100.0 / BORDA_RANGE
HIGH_MARGIN = 250.0 / BORDA_RANGE

CB_PALETTE = mpl.colormaps['Set2'].colors
sns.set_palette(CB_PALETTE)

# ── Same label maps as paper_figure3.ipynb, cell 7 ────────────────────────────
xtick_label_map = {
    "Base": "Base",
    "Small": "Small",
    "Tiny": "Tiny",
    "hybrid": "Hybrid",
    "fusion": "Fusion",
    "self-supervised": "Self\nSup",
    "self-supervised (multimodal)": "Multi\nmodal",
    "supervised": "Sup.",
    "supervised (robust)": "Robust",
    "supervised (multistep)": "Multi\nstep",
    "fully attention": "Attention",
    "fully convolutional": "Convolution",
    "CLASSIC_AT": "Classic AT",
    "TRADES_v2": "TRADES",
}

variable_title_map = {
    "model_size": "Architecture size",
    "pre_training_strategy": "Pre-training",
    "model_type": "Architecture type",
    "loss_function": "Loss function",
    "ft_strategy": "Fine-tuning strategy",
}

# Same pairs and order as paper_figure3.ipynb, cell 5
interaction_pairs = [
    ("model_size", "model_type"),
    ("loss_function", "model_type"),
    ("loss_function", "model_size"),
    ("loss_function", "pre_training_strategy"),
    ("pre_training_strategy", "model_type"),
    ("pre_training_strategy", "model_size"),
]


def get_tukey_letters(df, group_col, value_col, alpha=0.1):
    """Same greedy letter assignment as paper_figure3.ipynb, cell 7."""
    mc = MultiComparison(df[value_col], df[group_col])
    result = mc.tukeyhsd(alpha=alpha)
    comp = pd.DataFrame(result.summary().data[1:], columns=result.summary().data[0])
    comp['reject'] = comp['reject'].astype(bool)
    groups = list(mc.groupsunique)
    letters = {g: '' for g in groups}
    current_letter = 'a'
    remaining = set(groups)
    while remaining:
        g0 = sorted(remaining)[0]
        group_set = {g0}
        for g in sorted(remaining):
            sig_any = comp.loc[
                ((comp['group1'] == g) & (comp['group2'].isin(group_set))) |
                ((comp['group2'] == g) & (comp['group1'].isin(group_set))), 'reject'
            ].any()
            if not sig_any:
                group_set.add(g)
        for g in group_set:
            letters[g] += current_letter
        remaining -= group_set
        current_letter = chr(ord(current_letter) + 1)
    return letters


def short(lbl):
    return xtick_label_map.get(lbl, lbl)


def pretty_title(var_name):
    return variable_title_map.get(var_name, var_name.replace('_', ' ').capitalize())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    final_data = load_result_dataset(RESULTS_DIR)
    final_data = [{**d, 'ft_strategy': 'FFT (50 epochs)'} for d in final_data]
    grouped_df = process_rankings(process_grouped_df(final_data))
    process = grouped_df['TOTAL'].reset_index()

    palette = sns.color_palette(CB_PALETTE, n_colors=10)
    max_val, min_val = process[METRIC].max(), process[METRIC].min()
    value_range = max_val - min_val
    all_means = []

    for var1, var2 in interaction_pairs:
        df_full = process.copy()
        df_full['interaction'] = df_full[var1].astype(str) + ' | ' + df_full[var2].astype(str)

        # Cell 5: one-way ANOVA across interaction cells
        groups = [g[METRIC].values for _, g in df_full.groupby('interaction')]
        anova_p = stats.f_oneway(*groups).pvalue

        df_means = process.groupby([var1, var2])[METRIC].mean().reset_index()
        df_means['interaction'] = df_means[var1].astype(str) + ' | ' + df_means[var2].astype(str)
        letters = get_tukey_letters(df_full, 'interaction', METRIC, alpha=0.1)
        df_means['letter'] = df_means['interaction'].map(letters)
        all_means.append(df_means.assign(var1=var1, var2=var2, p_value=anova_p)
                         .rename(columns={var1: 'level1', var2: 'level2'}))

        if var1 == "pre_training_strategy":
            x_order = [
                "supervised (multistep)",
                "self-supervised",
                "self-supervised (multimodal)",
                "fusion",
                "supervised",
                "supervised (robust)",
            ]
        else:
            x_order = sorted(df_means[var1].unique())
        hue_order = sorted(df_means[var2].unique())

        print(f"\n{var1} x {var2}: p={anova_p:.3g}")
        print(df_means[[var1, var2, METRIC, 'letter']].round(3).to_string(index=False))

        # Cell 7: plot template
        plt.figure(figsize=(6.4, 4.2))
        ax = sns.barplot(data=df_means,
                         x=var1, y=METRIC, hue=var2,
                         palette=palette,
                         order=x_order, hue_order=hue_order)
        ax.axhline(max_val, color='red', linestyle=':', linewidth=1.5)
        ax.axhline(min_val, color='black', linestyle=':', linewidth=1.5)
        ax.set_ylim(min_val - LOW_MARGIN * value_range, max_val + HIGH_MARGIN * value_range)

        handles, labels = ax.get_legend_handles_labels()
        legend_kwargs = dict(
            handles=handles,
            labels=[short(l) for l in labels],
            title=pretty_title(var2),
            title_fontproperties={'weight': 'bold', 'size': 12},
            frameon=True,
            framealpha=0.4,
            fontsize=13,
        )
        if var2 == "pre_training_strategy":
            legend_kwargs['ncol'] = 2
        ax.legend(**legend_kwargs)

        if set(x_order).issubset(xtick_label_map.keys()):
            ax.set_xticklabels([short(l) for l in x_order])

        # Tukey letters: match each bar to its (x, hue) cell by position.
        # Within an x slot, seaborn places hue bars left to right in hue_order,
        # skipping missing combinations; recover the hue from the bar's offset.
        n_hue = len(hue_order)
        for patch in ax.patches:
            height = patch.get_height()
            width = patch.get_width()
            if width == 0 or pd.isna(height) or height == 0:
                continue
            center = patch.get_x() + width / 2.
            x_idx = int(round(center))
            if not 0 <= x_idx < len(x_order):
                continue
            offset = center - x_idx                       # in (-0.4, 0.4)
            hue_idx = int(round((offset + 0.4) / width - 0.5))
            if not 0 <= hue_idx < n_hue:
                continue
            key_lbl = f"{x_order[x_idx]} | {hue_order[hue_idx]}"
            letter = letters.get(key_lbl, '')
            if letter:
                ax.text(center, height + 0.01,
                        letter, ha='center', va='bottom',
                        fontsize=12, fontweight='bold')

        plt.title(f"p-value={anova_p:.3g}", fontsize=18)
        plt.ylabel(Y_LABEL, fontsize=16)
        plt.xlabel('')
        plt.xticks(rotation=0)
        plt.grid(axis='y', linestyle='--', alpha=.4)
        plt.tick_params(axis='both', labelsize=18)
        plt.tight_layout()
        out = os.path.join(OUT_DIR, f"sumscore_interaction_{var1}_{var2}.png")
        plt.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.05, transparent=False)
        plt.close()
        print(f"Saved {out}")

    pd.concat(all_means).to_csv(os.path.join(OUT_DIR, 'sumscore_interaction_means.csv'), index=False)


if __name__ == '__main__':
    main()
