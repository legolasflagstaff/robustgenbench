"""
Concrete figure-by-figure comparison: Borda vs Sum-Score.

For each of the 6 interaction figures, this prints the mean value of each
bar (each combination of var1 level x var2 level), ranked from highest to
lowest. Then it checks whether the ordering is preserved across metrics.

This directly answers: "Do the same categories land in the same ordering?"
"""

import sys
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_results import load_result_dataset
from process_database import process_grouped_df, process_rankings


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'full_fine_tuning50')

INTERACTION_PAIRS = [
    ('loss_function', 'model_size'),
    ('loss_function', 'model_type'),
    ('loss_function', 'pre_training_strategy'),
    ('model_size', 'model_type'),
    ('pre_training_strategy', 'model_size'),
    ('pre_training_strategy', 'model_type'),
]

FIGURE_NAMES = {
    ('loss_function', 'model_size'): "Fig (a): Loss Function x Arch. Size",
    ('loss_function', 'model_type'): "Fig (b): Loss Function x Arch. Type",
    ('loss_function', 'pre_training_strategy'): "Fig (c): Loss Function x Pre-training",
    ('model_size', 'model_type'): "Fig (d): Arch. Type x Size",
    ('pre_training_strategy', 'model_size'): "Fig (e): Pre-training x Arch. Size",
    ('pre_training_strategy', 'model_type'): "Fig (f): Pre-training x Arch. Type",
}


def main():
    print("Loading FFT-50 results...")
    final_data = load_result_dataset(RESULTS_DIR)
    for d in final_data:
        d['ft_strategy'] = 'FFT (50 epochs)'

    grouped_df = process_grouped_df(final_data)
    grouped_df = process_rankings(grouped_df)
    process = grouped_df['TOTAL'].reset_index()

    print(f"N = {len(process)} model configurations\n")

    all_agree = True

    for var1, var2 in INTERACTION_PAIRS:
        fig_name = FIGURE_NAMES[(var1, var2)]
        print("=" * 80)
        print(f"  {fig_name}")
        print(f"  x-axis: {var1}  |  hue: {var2}")
        print("=" * 80)

        # Group means for each bar in the figure
        means = process.groupby([var1, var2])[['borda', 'score_sum']].mean().reset_index()
        means['label'] = means[var1].astype(str) + " | " + means[var2].astype(str)

        # Rank by Borda (descending = best first)
        means['rank_borda'] = means['borda'].rank(ascending=False, method='min').astype(int)
        means['rank_sum'] = means['score_sum'].rank(ascending=False, method='min').astype(int)

        # Sort by Borda rank
        means_sorted = means.sort_values('rank_borda').reset_index(drop=True)

        # Check if ordering is identical
        borda_order = list(means_sorted['label'])
        sum_order = list(means.sort_values('rank_sum')['label'])
        ordering_same = (borda_order == sum_order)

        # Print the table
        print(f"\n  {'Rank(B)':<8} {'Rank(S)':<8} {'Borda':<10} {'SumScore':<10} {'Group'}")
        print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*50}")

        disagreements = []
        for _, row in means_sorted.iterrows():
            rb = int(row['rank_borda'])
            rs = int(row['rank_sum'])
            flag = "" if rb == rs else " <-- RANK DIFFERS"
            if rb != rs:
                disagreements.append((row['label'], rb, rs))
            print(f"  {rb:<8} {rs:<8} {row['borda']:<10.1f} {row['score_sum']:<10.3f} {row['label']}{flag}")

        print()
        if ordering_same:
            print(f"  VERDICT: Ordering is IDENTICAL between Borda and Sum-Score.")
        else:
            all_agree = False
            print(f"  VERDICT: Ordering DIFFERS. Swaps:")
            for label, rb, rs in disagreements:
                print(f"    - '{label}': Borda rank {rb}, Sum-Score rank {rs}")

        # Also show within each x-axis group, which hue is higher
        print(f"\n  Within-group ordering (for each {var1} level, which {var2} is best?):")
        for level in sorted(means[var1].unique()):
            sub = means[means[var1] == level].copy()
            sub_borda = sub.sort_values('borda', ascending=False)[var2].tolist()
            sub_sum = sub.sort_values('score_sum', ascending=False)[var2].tolist()
            match = "SAME" if sub_borda == sub_sum else "DIFFERENT"
            print(f"    {level:<35}: Borda={sub_borda}")
            print(f"    {'':35}  Sum  ={sub_sum}  [{match}]")

        print()

    # Final summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    if all_agree:
        print("\nAll 6 figures show IDENTICAL bar orderings between Borda and Sum-Score.")
        print("Every claim about relative performance holds regardless of metric choice.")
    else:
        print("\nSome figures show ordering differences. See details above.")
        print("Check whether the swaps are between adjacent, close-valued bars")
        print("(minor, likely non-significant) or between distant bars (concerning).")


if __name__ == '__main__':
    main()
