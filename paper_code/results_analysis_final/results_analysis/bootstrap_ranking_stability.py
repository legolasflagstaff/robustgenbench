"""
Bootstrap Ranking-Stability Analysis for the FFT-50 Benchmark.

Addresses reviewer concern: "no subsample-stability checks for the benchmark's own rankings."

Approach
--------
1. Load the flat per-configuration × per-dataset results (fft50.csv).
2. For B bootstrap iterations, resample the 6 datasets WITH REPLACEMENT,
   recompute per-configuration Borda scores on the resampled set, and record ranks.
3. Report:
   - Mean rank and 95% CI for each configuration.
   - Kendall's tau between each bootstrap ranking and the full-data ranking.
   - Probability that each configuration remains in Tier-1 (top-k) across resamples.

Output
------
- bootstrap_results.csv: per-configuration rank statistics.
- bootstrap_summary.csv: overall stability metrics.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from pathlib import Path


def compute_borda_scores(df_wide, datasets, metrics):
    """
    Given a wide-format DataFrame (rows=configurations, columns=dataset×metric),
    compute the Borda score for each configuration using only `datasets`.
    
    Borda score = sum of rank positions across all (dataset, metric) pairs.
    Higher rank = better → higher Borda = better.
    """
    borda = np.zeros(len(df_wide))
    for ds in datasets:
        for metric in metrics:
            col = (ds, metric)
            if col in df_wide.columns:
                # rank ascending: highest accuracy gets highest rank number → best
                ranks = df_wide[col].rank(ascending=True, method='min')
                borda += ranks.values
    return borda


def run_bootstrap(csv_path, B=10000, top_k_frac=0.33, seed=42):
    """
    Run bootstrap ranking stability analysis.
    
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
    metrics = ['clean_acc', 'Linf_acc', 'L2_acc', 'L1_acc', 'common_acc']
    
    # Build wide-format: rows = configurations, columns = (dataset, metric)
    index_cols = ['backbone_name', 'loss_function', 'model_size']
    value_cols = metrics
    
    df_wide = df.pivot_table(
        index=index_cols,
        columns='dataset',
        values=value_cols,
        aggfunc='mean'
    )
    # Swap levels so columns are (dataset, metric)
    df_wide.columns = df_wide.columns.swaplevel(0, 1)
    df_wide = df_wide.sort_index(axis=1)
    
    n_configs = len(df_wide)
    top_k = int(np.ceil(top_k_frac * n_configs))
    
    # ── Full-data ranking ──────────────────────────────────────────────────────
    full_borda = compute_borda_scores(df_wide, datasets, metrics)
    full_ranks = pd.Series(full_borda, index=df_wide.index).rank(ascending=False, method='min')
    
    # ── Bootstrap ──────────────────────────────────────────────────────────────
    bootstrap_ranks = np.zeros((B, n_configs))
    kendall_taus = np.zeros(B)
    
    for b in range(B):
        # Resample datasets with replacement
        resampled_datasets = rng.choice(datasets, size=len(datasets), replace=True).tolist()
        
        # Compute Borda on resampled datasets
        borda_b = compute_borda_scores(df_wide, resampled_datasets, metrics)
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
        'n_configurations': n_configs,
        'n_datasets': len(datasets),
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
    
    print("Running bootstrap ranking stability analysis...")
    results_df, summary_df, kendall_taus = run_bootstrap(csv_path, B=10000, seed=42)
    
    out_dir = Path(__file__).parent / 'bootstrap_results'
    out_dir.mkdir(exist_ok=True)
    
    results_df.to_csv(out_dir / 'bootstrap_rank_stability.csv', index=False)
    summary_df.to_csv(out_dir / 'bootstrap_summary.csv', index=False)
    np.save(out_dir / 'kendall_tau_distribution.npy', kendall_taus)
    
    print("\n=== Bootstrap Ranking Stability Summary ===")
    print(f"Configurations: {summary_df['n_configurations'].iloc[0]}")
    print(f"Datasets: {summary_df['n_datasets'].iloc[0]}")
    print(f"Bootstrap iterations: {summary_df['n_bootstrap_iterations'].iloc[0]}")
    print(f"Kendall's tau (mean ± std): {summary_df['kendall_tau_mean'].iloc[0]:.4f} ± {summary_df['kendall_tau_std'].iloc[0]:.4f}")
    print(f"Kendall's tau 95% CI: [{summary_df['kendall_tau_ci_lower'].iloc[0]:.4f}, {summary_df['kendall_tau_ci_upper'].iloc[0]:.4f}]")
    print(f"Mean rank CI width: {summary_df['mean_ci_width'].iloc[0]:.2f}")
    print(f"Median rank CI width: {summary_df['median_ci_width'].iloc[0]:.2f}")
    
    print("\n=== Top-10 Configurations (by original rank) ===")
    print(results_df.head(10).to_string(index=False))
    
    print(f"\nResults saved to: {out_dir}")
