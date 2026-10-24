"""
Generate interaction figures for both Borda and Sum-Score metrics.

This script reproduces the 6 interaction plots from the paper (Figure 4)
using two different aggregation metrics:
  1. Borda score (pool-dependent, rank-based)
  2. Sum-Score (pool-independent, raw accuracy sum)

The goal is to verify that claims established with Borda also hold with
Sum-Score, addressing the reviewer concern about pool-dependence.

Output:
  - paper_figures/interaction_{var1}_{var2}.png  (Borda)
  - paper_figures/sumscore_interaction_{var1}_{var2}.png  (Sum-Score)
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import MultiComparison

warnings.filterwarnings("ignore")

# Add the current directory to path so we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_results import load_result_dataset
from process_database import process_grouped_df, process_rankings


# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'full_fine_tuning50')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_figures')

# The 6 interaction pairs from the paper
INTERACTION_PAIRS = [
    ('loss_function', 'model_size'),
    ('loss_function', 'model_type'),
    ('loss_function', 'pre_training_strategy'),
    ('model_size', 'model_type'),
    ('pre_training_strategy', 'model_size'),
    ('pre_training_strategy', 'model_type'),
]

# Label mapping for x-axis ticks
XTICK_LABEL_MAP = {
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
    "Classic AT": "Classic AT",
    "TRADES": "TRADES",
}

# NeurIPS-style font sizes
NEURIPS_FONTSIZE = 14
plt.rcParams.update({
    "font.size": NEURIPS_FONTSIZE,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 18,
    "ytick.labelsize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})

CB_PALETTE = mpl.colormaps['Set2'].colors
sns.set_palette(CB_PALETTE)


# ── Helper functions ──────────────────────────────────────────────────────────

def anova_stars(p):
    return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < .05 else '\u2020' if p < .1 else ''


def get_tukey_letters(df, group_col, value_col, alpha=0.1):
    """
    Return dict {group_label: letter(s)} using Tukey HSD results.
    Same letter -> groups NOT significantly different (alpha).
    """
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


def short(lbl: str) -> str:
    return XTICK_LABEL_MAP.get(lbl, lbl)


def compute_nested_anova(df, var1, var2, value_col):
    """Compute nested Welch's ANOVA for interaction effects."""
    formula = f'{value_col} ~ C({var1}) * C({var2})'
    try:
        model = ols(formula, data=df).fit()
        anova_table = anova_lm(model, typ=2)
        return anova_table
    except Exception as e:
        print(f"  ANOVA failed for {var1}/{var2}: {e}")
        return None


def generate_interaction_plot(process_df, var1, var2, value_col, metric_label, output_prefix, palette):
    """Generate a single interaction plot."""
    # Build orders
    if var1 == "pre_training_strategy":
        x_order = [
            "supervised (multistep)",
            "self-supervised",
            "self-supervised (multimodal)",
            "fusion",
            "supervised",
            "supervised (robust)",
        ]
        x_order = [x for x in x_order if x in process_df[var1].unique()]
    else:
        x_order = sorted(process_df[var1].unique())

    hue_order = sorted(process_df[var2].unique())

    # Compute means per group
    df_means = process_df.groupby([var1, var2])[value_col].mean().reset_index()

    # Interaction labels for Tukey
    process_df = process_df.copy()
    process_df['interaction'] = process_df[var1].astype(str) + ' | ' + process_df[var2].astype(str)
    df_means['interaction'] = df_means[var1].astype(str) + ' | ' + df_means[var2].astype(str)

    # Tukey letters
    letters = get_tukey_letters(process_df, 'interaction', value_col, alpha=0.1)
    df_means['letter'] = df_means['interaction'].map(letters)

    # ANOVA p-value
    anova_table = compute_nested_anova(process_df, var1, var2, value_col)
    if anova_table is not None:
        int_label = f'C({var1}):C({var2})'
        if int_label in anova_table.index:
            anova_p = anova_table.loc[int_label, 'PR(>F)']
        else:
            anova_p = anova_table.iloc[-1]['PR(>F)']
    else:
        anova_p = float('nan')

    # Reference lines
    max_val = process_df[value_col].max()
    min_val = process_df[value_col].min()

    # Plot
    plt.figure(figsize=(6.4, 4.2))

    ax = sns.barplot(data=df_means,
                     x=var1, y=value_col, hue=var2,
                     palette=palette,
                     order=x_order, hue_order=hue_order)

    # Reference lines
    ax.axhline(max_val, color='red', linestyle=':', linewidth=1.5, label=f"Max {metric_label}")
    ax.axhline(min_val, color='black', linestyle=':', linewidth=1.5, label=f"Min {metric_label}")
    ax.set_ylim(min_val - (max_val - min_val) * 0.05, max_val + (max_val - min_val) * 0.05)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if var2 == "pre_training_strategy":
        ax.legend(handles, [short(l) for l in labels], title=short(var2),
                  ncol=2, frameon=True, framealpha=0.4, fontsize=11)
    else:
        ax.legend(handles, [short(l) for l in labels], title=short(var2),
                  frameon=True, framealpha=0.4, fontsize=11)

    # X-tick labels
    if set(x_order).issubset(XTICK_LABEL_MAP.keys()):
        ax.set_xticklabels([short(l) for l in x_order])

    # Annotate Tukey letters
    for c_idx, container in enumerate(ax.containers):
        if c_idx >= len(hue_order):
            break
        hue_val = hue_order[c_idx]
        for b_idx, bar in enumerate(container):
            if b_idx >= len(x_order):
                break
            x_val = x_order[b_idx]
            key_lbl = f"{x_val} | {hue_val}"
            letter = letters.get(key_lbl, '')
            if letter:
                ax.text(bar.get_x() + bar.get_width() / 2.,
                        bar.get_height() + (max_val - min_val) * 0.005,
                        letter, ha='center', va='bottom',
                        fontsize=12, fontweight='bold')

    plt.title(f"p-value={anova_p:.3g}", fontsize=18)
    plt.ylabel(f"Mean {metric_label}", fontsize=16)
    plt.xlabel('')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=.4)
    plt.tick_params(axis='both', labelsize=18)
    plt.tight_layout()

    # Save
    key = f"{var1}_{var2}"
    filename = f"{output_prefix}interaction_{key}.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0.05, transparent=False)
    plt.close()

    print(f"  Saved: {filepath} (p={anova_p:.3g})")
    return anova_p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading FFT-50 results...")
    final_data = load_result_dataset(RESULTS_DIR)
    for d in final_data:
        d['ft_strategy'] = 'FFT (50 epochs)'

    print("Processing grouped dataframe...")
    grouped_df = process_grouped_df(final_data)
    grouped_df = process_rankings(grouped_df)

    # Extract the flat process dataframe for ANOVA / plotting
    process = grouped_df['TOTAL'].reset_index()

    print(f"\nData shape: {process.shape}")
    print(f"Borda range: [{process['borda'].min():.1f}, {process['borda'].max():.1f}]")
    print(f"Sum-Score range: [{process['score_sum'].min():.3f}, {process['score_sum'].max():.3f}]")

    palette = sns.color_palette(CB_PALETTE, n_colors=10)

    # Results summary
    results_summary = []

    # ── Generate Borda figures ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING BORDA INTERACTION FIGURES")
    print("=" * 70)

    for var1, var2 in INTERACTION_PAIRS:
        print(f"\n  {var1} x {var2}:")
        p = generate_interaction_plot(
            process, var1, var2, 'borda', 'Borda Score', '', palette
        )
        results_summary.append({
            'metric': 'Borda',
            'var1': var1,
            'var2': var2,
            'anova_p': p,
        })

    # ── Generate Sum-Score figures ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENERATING SUM-SCORE INTERACTION FIGURES")
    print("=" * 70)

    for var1, var2 in INTERACTION_PAIRS:
        print(f"\n  {var1} x {var2}:")
        p = generate_interaction_plot(
            process, var1, var2, 'score_sum', 'Sum-Score', 'sumscore_', palette
        )
        results_summary.append({
            'metric': 'Sum-Score',
            'var1': var1,
            'var2': var2,
            'anova_p': p,
        })

    # ── Summary comparison ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY: Borda vs Sum-Score")
    print("=" * 70)
    print(f"\n{'Interaction':<50} {'Borda p':<12} {'SumScore p':<12} {'Both Sig?':<10}")
    print("-" * 84)

    for var1, var2 in INTERACTION_PAIRS:
        borda_p = next(r['anova_p'] for r in results_summary
                       if r['metric'] == 'Borda' and r['var1'] == var1 and r['var2'] == var2)
        sum_p = next(r['anova_p'] for r in results_summary
                     if r['metric'] == 'Sum-Score' and r['var1'] == var1 and r['var2'] == var2)

        borda_sig = borda_p < 0.1
        sum_sig = sum_p < 0.1
        both = "YES" if (borda_sig and sum_sig) else ("PARTIAL" if (borda_sig or sum_sig) else "NO")

        label = f"{var1} x {var2}"
        print(f"{label:<50} {borda_p:<12.4g} {sum_p:<12.4g} {both:<10}")

    print("\n" + "=" * 70)
    print("ANALYSIS:")
    print("=" * 70)

    # Count agreements
    n_pairs = len(INTERACTION_PAIRS)
    agreements = 0
    for var1, var2 in INTERACTION_PAIRS:
        borda_p = next(r['anova_p'] for r in results_summary
                       if r['metric'] == 'Borda' and r['var1'] == var1 and r['var2'] == var2)
        sum_p = next(r['anova_p'] for r in results_summary
                     if r['metric'] == 'Sum-Score' and r['var1'] == var1 and r['var2'] == var2)
        if (borda_p < 0.1) == (sum_p < 0.1):
            agreements += 1

    print(f"\nOut of {n_pairs} interaction tests:")
    print(f"  - {agreements}/{n_pairs} show consistent significance (both sig or both non-sig at alpha=0.1)")
    print(f"  - Agreement rate: {agreements/n_pairs*100:.0f}%")
    print()

    if agreements == n_pairs:
        print("CONCLUSION: All claims established with Borda score ALSO hold with Sum-Score.")
        print("The findings are NOT artifacts of the pool-dependent Borda aggregation.")
    elif agreements >= n_pairs - 1:
        print("CONCLUSION: Nearly all claims established with Borda score also hold with Sum-Score.")
        print("The findings are largely robust to the choice of aggregation method.")
    else:
        print("CONCLUSION: Some claims differ between Borda and Sum-Score.")
        print("Further investigation may be needed for discrepant interactions.")

    print("\nDone.")


if __name__ == '__main__':
    main()
