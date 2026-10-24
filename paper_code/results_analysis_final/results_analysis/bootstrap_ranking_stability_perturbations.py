"""
Bootstrap Ranking-Stability Analysis – Perturbation Resampling.

Addresses reviewer concern: "no subsample-stability checks for the benchmark's own rankings."

This variant resamples PERTURBATION TYPES (metrics) with replacement instead of datasets.
It answers: "How stable are the rankings if we change which threat models are included?"

The 5 metrics used in Borda scoring are:
  - clean_acc    (no perturbation / clean accuracy)
  - Linf_acc     (L∞ adversarial robustness)
  - L2_acc       (L2 adversarial robustness)
  - L1_acc       (L1 adversarial robustness)
  - common_acc   (common corruptions robustness)

For each bootstrap iteration we resample these 5 metrics WITH REPLACEMENT,
recompute Borda scores across all 6 datasets using only the resampled metrics,
rank configurations, and record the result.

Output
------
- bootstrap_perturbation_rank_stability.csv: per-configuration rank statistics.
- bootstrap_perturbation_summary.csv: overall stability metrics.
- kendall_tau_perturbation_distribution.npy: full tau distribution.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from pathlib import Path


def compute_borda_scores(df_wide, datasets, metrics):
    """
    Compute Borda score for each configuration using the given datasets and metrics.
    Borda = sum of per-(dataset, metric) ascending ranks.
    Higher Borda = better.
    """
    borda = np.zeros(len(df_wide))
    for ds in datasets:
        for metric in metrics:
            col = (ds, metric)
            if col in df_wide.columns:
                ranks = df_wide[col].rank(ascending=True, method='min')
                borda += ranks.values
    return borda


def run_bootstrap_perturbations(csv_path, B=10000, top_k_frac=0.33, seed=42):
    """
    Run bootstrap ranking stability analysis by resampling perturbation types.

    Parameters
    ----------
    csv_path : str or Path
        Path to fft50.csv (flat results file).
    B : int
        Number of bootstrap iterations.
    top_k_frac : float
        Fraction of configurations considered "Tier-1".
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # ── Load data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, index_col=0)

    datasets = sorted(df['dataset'].unique())
    all_metrics = ['clean_acc', 'Linf_acc', 'L2_acc', 'L1_acc', 'common_acc']

    # Build wide-format: rows = configurations, columns = (dataset, metric)
    index_cols = ['backbone_name', 'loss_function', 'model_size']

    df_wide = df.pivot_table(
        index=index_cols,
        columns='dataset',
        values=all_metrics,
        aggfunc='mean'
    )
    # Swap levels so columns are (dataset, metric)
    df_wide.columns = df_wide.columns.swaplevel(0, 1)
    df_wide = df_wide.sort_index(axis=1)

    n_configs = len(df_wide)
    top_k = int(np.ceil(top_k_frac * n_configs))

    # ── Full-data ranking (all datasets, all metrics) ──────────────────────────
    full_borda = compute_borda_scores(df_wide, datasets, all_metrics)
    full_ranks = pd.Series(full_borda, index=df_wide.index).rank(ascending=False, method='min')

    # ── Bootstrap over perturbations ───────────────────────────────────────────
    bootstrap_ranks = np.zeros((B, n_configs))
    kendall_taus = np.zeros(B)

    for b in range(B):
        # Resample metrics (perturbation types) with replacement
        resampled_metrics = rng.choice(all_metrics, size=len(all_metrics), replace=True).tolist()

        # Compute Borda on all datasets but only resampled metrics
        borda_b = compute_borda_scores(df_wide, datasets, resampled_metrics)
        ranks_b = pd.Series(borda_b).rank(ascending=False, method='min').values
        bootstrap_ranks[b] = ranks_b

        # Kendall's tau vs full ranking
        tau, _ = kendalltau(full_ranks.values, ranks_b)
        kendall_taus[b] = tau

    # ── Per-configuration statistics ───────────────────────────────────────────
    mean_ranks = bootstrap_ranks.mean(axis=0)
    ci_lower = np.percentile(bootstrap_ranks, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_ranks, 97.5, axis=0)

    # Probability of being in top-k
    in_tier1 = (bootstrap_ranks <= top_k).mean(axis=0)

    results_df = pd.DataFrame({
        'backbone_name': df_wide.index.get_level_values('backbone_name'),
        'loss_function': df_wide.index.get_level_values('loss_function'),
        'model_size': df_wide.index.get_level_values('model_size'),
        'original_rank': full_ranks.values,
        'mean_bootstrap_rank': mean_ranks,
        'ci_lower_2.5': ci_lower,
        'ci_upper_97.5': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'prob_tier1': in_tier1,
    }).sort_values('original_rank')

    # ── Summary statistics ─────────────────────────────────────────────────────
    summary = {
        'resampling_dimension': 'perturbation_types',
        'n_configurations': n_configs,
        'n_datasets': len(datasets),
        'n_metrics': len(all_metrics),
        'metrics': str(all_metrics),
        'n_bootstrap_iterations': B,
        'top_k': top_k,
        'top_k_fraction': top_k_frac,
        'kendall_tau_mean': kendall_taus.mean(),
        'kendall_tau_std': kendall_taus.std(),
        'kendall_tau_median': np.median(kendall_taus),
        'kendall_tau_ci_lower': np.percentile(kendall_taus, 2.5),
        'kendall_tau_ci_upper': np.percentile(kendall_taus, 97.5),
        'mean_ci_width': (ci_upper - ci_lower).mean(),
        'median_ci_width': np.median(ci_upper - ci_lower),
    }
    summary_df = pd.DataFrame([summary])

    return results_df, summary_df, kendall_taus


if __name__ == '__main__':
    csv_path = Path(__file__).parent / 'fft50.csv'

    print("Running bootstrap ranking stability analysis (perturbation resampling)...")
    results_df, summary_df, kendall_taus = run_bootstrap_perturbations(csv_path, B=10000, seed=42)

    out_dir = Path(__file__).parent / 'bootstrap_results'
    out_dir.mkdir(exist_ok=True)

    results_df.to_csv(out_dir / 'bootstrap_perturbation_rank_stability.csv', index=False)
    summary_df.to_csv(out_dir / 'bootstrap_perturbation_summary.csv', index=False)
    np.save(out_dir / 'kendall_tau_perturbation_distribution.npy', kendall_taus)

    print("\n=== Bootstrap Ranking Stability (Perturbation Resampling) ===")
    print(f"Configurations: {summary_df['n_configurations'].iloc[0]}")
    print(f"Datasets: {summary_df['n_datasets'].iloc[0]}")
    print(f"Metrics (perturbations): {summary_df['n_metrics'].iloc[0]}")
    print(f"Bootstrap iterations: {summary_df['n_bootstrap_iterations'].iloc[0]}")
    print(f"Kendall's tau (mean ± std): {summary_df['kendall_tau_mean'].iloc[0]:.4f} ± {summary_df['kendall_tau_std'].iloc[0]:.4f}")
    print(f"Kendall's tau 95% CI: [{summary_df['kendall_tau_ci_lower'].iloc[0]:.4f}, {summary_df['kendall_tau_ci_upper'].iloc[0]:.4f}]")
    print(f"Mean rank CI width: {summary_df['mean_ci_width'].iloc[0]:.2f}")
    print(f"Median rank CI width: {summary_df['median_ci_width'].iloc[0]:.2f}")

    print("\n=== Top-10 Configurations (by original rank) ===")
    print(results_df.head(10).to_string(index=False))

    print(f"\nResults saved to: {out_dir}")
