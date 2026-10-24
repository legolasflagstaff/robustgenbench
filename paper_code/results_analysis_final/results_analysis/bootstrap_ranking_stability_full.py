"""
Unified Bootstrap Ranking-Stability Analysis for the FFT-50 Benchmark.

Tests ranking stability along three axes, reporting BOTH Borda and SumScore:

(i)   Dataset composition       – resample the 6 datasets with replacement (LODO).
(ii)  Threat-model composition  – resample the 5 perturbation settings with replacement (LOPO).
(iii) Test-sample noise:
      A) Optimistic (Binomial)  – each cell a[i,d,z] as i.i.d. proportion.
      B) Conservative (Beta-Binomial) – overdispersion rho inflates per-cell variance.

Parameters: B=1000 iterations, random seed=42.
Each iteration recomputes both Borda and SumScore rankings and reports
Kendall's tau against the original ranking.

Output
------
- bootstrap_stability_full_summary.csv: table with tau (mean +/- std) and 95% CI
  for each (analysis, scoring method) combination.
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
DATASET_TEST_SIZES = {
    'caltech101': 1000,
    'flowers-102': 1000,
    'oxford-iiit-pet': 1000,
    'stanford_cars': 1000,
    'fgvc-aircraft-2013b': 1000,
    'uc-merced-land-use-dataset': 420,
}

METRICS = ['clean_acc', 'Linf_acc', 'L2_acc', 'L1_acc', 'common_acc']

B = 1000
SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# Scoring functions
# ─────────────────────────────────────────────────────────────────────────────
def compute_borda_from_wide(df_wide, datasets, metrics):
    """
    Borda score = sum of per-(dataset, metric) ascending ranks.
    Higher accuracy -> higher rank number -> better.
    """
    borda = np.zeros(len(df_wide))
    for ds in datasets:
        for metric in metrics:
            col = (ds, metric)
            if col in df_wide.columns:
                ranks = df_wide[col].rank(ascending=True, method='min')
                borda += ranks.values
    return borda


def compute_sumscore_from_wide(df_wide, datasets, metrics):
    """
    SumScore = direct sum of accuracy values across all (dataset, metric) cells.
    """
    total = np.zeros(len(df_wide))
    for ds in datasets:
        for metric in metrics:
            col = (ds, metric)
            if col in df_wide.columns:
                total += df_wide[col].fillna(0).values
    return total


def compute_borda_from_tensor(A, n_configs, n_datasets, n_metrics):
    """Borda score from 3-D tensor A[config, dataset, metric]."""
    borda = np.zeros(n_configs)
    for d in range(n_datasets):
        for z in range(n_metrics):
            borda += rankdata(A[:, d, z], method='min')
    return borda


def compute_sumscore_from_tensor(A, n_configs, n_datasets, n_metrics):
    """SumScore from 3-D tensor A[config, dataset, metric]."""
    return A.sum(axis=(1, 2))


def rank_descending(scores):
    """Rank: highest score = rank 1. Ties: method='min'."""
    return rankdata(-scores, method='min')


# ─────────────────────────────────────────────────────────────────────────────
# Analysis (i): Dataset resampling (LODO)
# ─────────────────────────────────────────────────────────────────────────────
def analysis_dataset_resampling(df_wide, datasets, metrics, rng):
    """Resample the 6 datasets with replacement."""
    n_configs = len(df_wide)

    # Original rankings
    borda_orig = compute_borda_from_wide(df_wide, datasets, metrics)
    sum_orig = compute_sumscore_from_wide(df_wide, datasets, metrics)
    rank_borda_orig = rank_descending(borda_orig)
    rank_sum_orig = rank_descending(sum_orig)

    taus_borda = np.zeros(B)
    taus_sum = np.zeros(B)

    for b in range(B):
        resampled_ds = rng.choice(datasets, size=len(datasets), replace=True).tolist()
        borda_b = compute_borda_from_wide(df_wide, resampled_ds, metrics)
        sum_b = compute_sumscore_from_wide(df_wide, resampled_ds, metrics)

        rank_borda_b = rank_descending(borda_b)
        rank_sum_b = rank_descending(sum_b)

        taus_borda[b], _ = kendalltau(rank_borda_orig, rank_borda_b)
        taus_sum[b], _ = kendalltau(rank_sum_orig, rank_sum_b)

    return taus_borda, taus_sum


# ─────────────────────────────────────────────────────────────────────────────
# Analysis (ii): Perturbation resampling (LOPO)
# ─────────────────────────────────────────────────────────────────────────────
def analysis_perturbation_resampling(df_wide, datasets, metrics, rng):
    """Resample the 5 perturbation settings with replacement."""
    n_configs = len(df_wide)

    # Original rankings
    borda_orig = compute_borda_from_wide(df_wide, datasets, metrics)
    sum_orig = compute_sumscore_from_wide(df_wide, datasets, metrics)
    rank_borda_orig = rank_descending(borda_orig)
    rank_sum_orig = rank_descending(sum_orig)

    taus_borda = np.zeros(B)
    taus_sum = np.zeros(B)

    for b in range(B):
        resampled_metrics = rng.choice(metrics, size=len(metrics), replace=True).tolist()
        borda_b = compute_borda_from_wide(df_wide, datasets, resampled_metrics)
        sum_b = compute_sumscore_from_wide(df_wide, datasets, resampled_metrics)

        rank_borda_b = rank_descending(borda_b)
        rank_sum_b = rank_descending(sum_b)

        taus_borda[b], _ = kendalltau(rank_borda_orig, rank_borda_b)
        taus_sum[b], _ = kendalltau(rank_sum_orig, rank_sum_b)

    return taus_borda, taus_sum


# ─────────────────────────────────────────────────────────────────────────────
# Analysis (iii-A): Binomial noise (Optimistic)
# ─────────────────────────────────────────────────────────────────────────────
def analysis_binomial(A, dataset_sizes, n_configs, n_datasets, n_metrics, rng):
    """
    Each cell: k* ~ Binomial(n_d, a[i,d,z]), a* = k*/n_d.
    Assumes images are independent (minimum per-cell noise).
    """
    # Original rankings
    borda_orig = compute_borda_from_tensor(A, n_configs, n_datasets, n_metrics)
    sum_orig = compute_sumscore_from_tensor(A, n_configs, n_datasets, n_metrics)
    rank_borda_orig = rank_descending(borda_orig)
    rank_sum_orig = rank_descending(sum_orig)

    taus_borda = np.zeros(B)
    taus_sum = np.zeros(B)

    for b in range(B):
        A_star = np.empty_like(A)
        for d in range(n_datasets):
            n_d = dataset_sizes[d]
            k_star = rng.binomial(n_d, A[:, d, :])
            A_star[:, d, :] = k_star / n_d

        borda_b = compute_borda_from_tensor(A_star, n_configs, n_datasets, n_metrics)
        sum_b = compute_sumscore_from_tensor(A_star, n_configs, n_datasets, n_metrics)

        taus_borda[b], _ = kendalltau(rank_borda_orig, rank_descending(borda_b))
        taus_sum[b], _ = kendalltau(rank_sum_orig, rank_descending(sum_b))

    return taus_borda, taus_sum


# ─────────────────────────────────────────────────────────────────────────────
# Analysis (iii-B): Beta-Binomial noise (Conservative)
# ─────────────────────────────────────────────────────────────────────────────
def analysis_beta_binomial(A, dataset_sizes, n_configs, n_datasets, n_metrics, rng, rho):
    """
    Beta-Binomial noise with overdispersion rho.
    Variance inflation factor = 1 + (n-1)*rho relative to Binomial.

    For each cell with accuracy p and test-set size n:
      alpha = p * ((1/rho) - 1)
      beta  = (1-p) * ((1/rho) - 1)
      theta ~ Beta(alpha, beta)
      k*    ~ Binomial(n, theta)
      a*    = k*/n

    At rho=0.05: variance ~51x i.i.d. baseline.
    At rho=0.10: variance ~101x i.i.d. baseline.
    """
    # Original rankings
    borda_orig = compute_borda_from_tensor(A, n_configs, n_datasets, n_metrics)
    sum_orig = compute_sumscore_from_tensor(A, n_configs, n_datasets, n_metrics)
    rank_borda_orig = rank_descending(borda_orig)
    rank_sum_orig = rank_descending(sum_orig)

    taus_borda = np.zeros(B)
    taus_sum = np.zeros(B)

    # Precompute alpha/beta parameters for all cells
    # For p=0 or p=1, Beta is degenerate; clamp to avoid issues
    eps = 1e-6
    A_clamped = np.clip(A, eps, 1.0 - eps)
    concentration = (1.0 / rho) - 1.0  # total concentration parameter
    alpha_params = A_clamped * concentration  # shape (n_configs, n_datasets, n_metrics)
    beta_params = (1.0 - A_clamped) * concentration

    for b in range(B):
        A_star = np.empty_like(A)
        for d in range(n_datasets):
            n_d = dataset_sizes[d]
            # Draw theta ~ Beta(alpha, beta) for all (config, metric) at dataset d
            theta = rng.beta(alpha_params[:, d, :], beta_params[:, d, :])
            # Draw k* ~ Binomial(n_d, theta)
            k_star = rng.binomial(n_d, theta)
            A_star[:, d, :] = k_star / n_d

        borda_b = compute_borda_from_tensor(A_star, n_configs, n_datasets, n_metrics)
        sum_b = compute_sumscore_from_tensor(A_star, n_configs, n_datasets, n_metrics)

        taus_borda[b], _ = kendalltau(rank_borda_orig, rank_descending(borda_b))
        taus_sum[b], _ = kendalltau(rank_sum_orig, rank_descending(sum_b))

    return taus_borda, taus_sum


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    csv_path = Path(__file__).parent / 'fft50.csv'
    rng = np.random.default_rng(SEED)

    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)

    datasets = sorted(df['dataset'].unique())
    n_datasets = len(datasets)
    n_metrics = len(METRICS)

    # Build wide-format for dataset/perturbation resampling
    index_cols = ['backbone_name', 'loss_function', 'model_size']
    df_wide = df.pivot_table(
        index=index_cols,
        columns='dataset',
        values=METRICS,
        aggfunc='mean'
    )
    df_wide.columns = df_wide.columns.swaplevel(0, 1)
    df_wide = df_wide.sort_index(axis=1)
    n_configs = len(df_wide)

    print(f"Configurations: {n_configs}")
    print(f"Datasets: {n_datasets}")
    print(f"Metrics (perturbations): {n_metrics}")
    print(f"Bootstrap iterations: {B}")
    print(f"Seed: {SEED}")
    print()

    # Build 3-D tensor for binomial/beta-binomial analyses
    # Use the same data as df_wide (which averages duplicates via pivot_table)
    # to ensure tensor and wide-format produce identical rankings.
    dataset_sizes = np.zeros(n_datasets, dtype=int)
    A = np.full((n_configs, n_datasets, n_metrics), np.nan)

    for d_idx, ds in enumerate(datasets):
        dataset_sizes[d_idx] = DATASET_TEST_SIZES[ds]
        for z_idx, metric in enumerate(METRICS):
            col = (ds, metric)
            if col in df_wide.columns:
                A[:, d_idx, z_idx] = df_wide[col].values

    nan_count = np.isnan(A).sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN cells in tensor. Filling with 0.")
        A = np.nan_to_num(A, nan=0.0)

    # Verify tensor Borda matches wide-format Borda
    borda_tensor = compute_borda_from_tensor(A, n_configs, n_datasets, n_metrics)
    borda_wide = compute_borda_from_wide(df_wide, datasets, METRICS)
    # Rankings should match (not necessarily scores due to ordering differences)
    rank_t = rank_descending(borda_tensor)
    rank_w = rank_descending(borda_wide)
    tau_check, _ = kendalltau(rank_t, rank_w)
    print(f"[Sanity] Tensor vs Wide Borda ranking tau: {tau_check:.4f}")
    print()

    # ── Run all analyses ──────────────────────────────────────────────────────
    results = []

    # (i) Dataset resampling
    print("Running (i) Dataset resampling (LODO)...")
    taus_borda, taus_sum = analysis_dataset_resampling(df_wide, datasets, METRICS, rng)
    results.append({
        'analysis': 'Dataset resampling',
        'what_varies': 'Dataset composition',
        'tau_borda_mean': taus_borda.mean(),
        'tau_borda_std': taus_borda.std(),
        'tau_borda_ci_lo': np.percentile(taus_borda, 2.5),
        'tau_borda_ci_hi': np.percentile(taus_borda, 97.5),
        'tau_sumscore_mean': taus_sum.mean(),
        'tau_sumscore_std': taus_sum.std(),
        'tau_sumscore_ci_lo': np.percentile(taus_sum, 2.5),
        'tau_sumscore_ci_hi': np.percentile(taus_sum, 97.5),
    })
    print(f"  Borda:    tau = {taus_borda.mean():.3f} +/- {taus_borda.std():.3f}  "
          f"95% CI [{np.percentile(taus_borda, 2.5):.3f}, {np.percentile(taus_borda, 97.5):.3f}]")
    print(f"  SumScore: tau = {taus_sum.mean():.3f} +/- {taus_sum.std():.3f}  "
          f"95% CI [{np.percentile(taus_sum, 2.5):.3f}, {np.percentile(taus_sum, 97.5):.3f}]")
    print()

    # (ii) Perturbation resampling
    print("Running (ii) Perturbation resampling (LOPO)...")
    taus_borda, taus_sum = analysis_perturbation_resampling(df_wide, datasets, METRICS, rng)
    results.append({
        'analysis': 'Perturbation resampling',
        'what_varies': 'Threat-model composition',
        'tau_borda_mean': taus_borda.mean(),
        'tau_borda_std': taus_borda.std(),
        'tau_borda_ci_lo': np.percentile(taus_borda, 2.5),
        'tau_borda_ci_hi': np.percentile(taus_borda, 97.5),
        'tau_sumscore_mean': taus_sum.mean(),
        'tau_sumscore_std': taus_sum.std(),
        'tau_sumscore_ci_lo': np.percentile(taus_sum, 2.5),
        'tau_sumscore_ci_hi': np.percentile(taus_sum, 97.5),
    })
    print(f"  Borda:    tau = {taus_borda.mean():.3f} +/- {taus_borda.std():.3f}  "
          f"95% CI [{np.percentile(taus_borda, 2.5):.3f}, {np.percentile(taus_borda, 97.5):.3f}]")
    print(f"  SumScore: tau = {taus_sum.mean():.3f} +/- {taus_sum.std():.3f}  "
          f"95% CI [{np.percentile(taus_sum, 2.5):.3f}, {np.percentile(taus_sum, 97.5):.3f}]")
    print()

    # (iii-A) Binomial (Optimistic)
    print("Running (iii-A) Binomial noise (Optimistic, i.i.d.)...")
    taus_borda, taus_sum = analysis_binomial(A, dataset_sizes, n_configs, n_datasets, n_metrics, rng)
    results.append({
        'analysis': 'Binomial (i.i.d.)',
        'what_varies': 'Sampling noise',
        'tau_borda_mean': taus_borda.mean(),
        'tau_borda_std': taus_borda.std(),
        'tau_borda_ci_lo': np.percentile(taus_borda, 2.5),
        'tau_borda_ci_hi': np.percentile(taus_borda, 97.5),
        'tau_sumscore_mean': taus_sum.mean(),
        'tau_sumscore_std': taus_sum.std(),
        'tau_sumscore_ci_lo': np.percentile(taus_sum, 2.5),
        'tau_sumscore_ci_hi': np.percentile(taus_sum, 97.5),
    })
    print(f"  Borda:    tau = {taus_borda.mean():.3f} +/- {taus_borda.std():.3f}  "
          f"95% CI [{np.percentile(taus_borda, 2.5):.3f}, {np.percentile(taus_borda, 97.5):.3f}]")
    print(f"  SumScore: tau = {taus_sum.mean():.3f} +/- {taus_sum.std():.3f}  "
          f"95% CI [{np.percentile(taus_sum, 2.5):.3f}, {np.percentile(taus_sum, 97.5):.3f}]")
    print()

    # (iii-B) Beta-Binomial (Conservative, rho=0.05)
    print("Running (iii-B) Beta-Binomial noise (rho=0.05, 51x variance)...")
    taus_borda, taus_sum = analysis_beta_binomial(
        A, dataset_sizes, n_configs, n_datasets, n_metrics, rng, rho=0.05)
    results.append({
        'analysis': 'Beta-Binomial (rho=0.05)',
        'what_varies': 'Correlated noise (51x)',
        'tau_borda_mean': taus_borda.mean(),
        'tau_borda_std': taus_borda.std(),
        'tau_borda_ci_lo': np.percentile(taus_borda, 2.5),
        'tau_borda_ci_hi': np.percentile(taus_borda, 97.5),
        'tau_sumscore_mean': taus_sum.mean(),
        'tau_sumscore_std': taus_sum.std(),
        'tau_sumscore_ci_lo': np.percentile(taus_sum, 2.5),
        'tau_sumscore_ci_hi': np.percentile(taus_sum, 97.5),
    })
    print(f"  Borda:    tau = {taus_borda.mean():.3f} +/- {taus_borda.std():.3f}  "
          f"95% CI [{np.percentile(taus_borda, 2.5):.3f}, {np.percentile(taus_borda, 97.5):.3f}]")
    print(f"  SumScore: tau = {taus_sum.mean():.3f} +/- {taus_sum.std():.3f}  "
          f"95% CI [{np.percentile(taus_sum, 2.5):.3f}, {np.percentile(taus_sum, 97.5):.3f}]")
    print()

    # (iii-B) Beta-Binomial (Conservative, rho=0.10)
    print("Running (iii-B) Beta-Binomial noise (rho=0.10, 101x variance)...")
    taus_borda, taus_sum = analysis_beta_binomial(
        A, dataset_sizes, n_configs, n_datasets, n_metrics, rng, rho=0.10)
    results.append({
        'analysis': 'Beta-Binomial (rho=0.10)',
        'what_varies': 'Correlated noise (101x)',
        'tau_borda_mean': taus_borda.mean(),
        'tau_borda_std': taus_borda.std(),
        'tau_borda_ci_lo': np.percentile(taus_borda, 2.5),
        'tau_borda_ci_hi': np.percentile(taus_borda, 97.5),
        'tau_sumscore_mean': taus_sum.mean(),
        'tau_sumscore_std': taus_sum.std(),
        'tau_sumscore_ci_lo': np.percentile(taus_sum, 2.5),
        'tau_sumscore_ci_hi': np.percentile(taus_sum, 97.5),
    })
    print(f"  Borda:    tau = {taus_borda.mean():.3f} +/- {taus_borda.std():.3f}  "
          f"95% CI [{np.percentile(taus_borda, 2.5):.3f}, {np.percentile(taus_borda, 97.5):.3f}]")
    print(f"  SumScore: tau = {taus_sum.mean():.3f} +/- {taus_sum.std():.3f}  "
          f"95% CI [{np.percentile(taus_sum, 2.5):.3f}, {np.percentile(taus_sum, 97.5):.3f}]")
    print()

    # ── Build summary table ───────────────────────────────────────────────────
    results_df = pd.DataFrame(results)

    out_dir = Path(__file__).parent / 'bootstrap_results'
    out_dir.mkdir(exist_ok=True)
    results_df.to_csv(out_dir / 'bootstrap_stability_full_summary.csv', index=False)

    # ── Print formatted table ─────────────────────────────────────────────────
    print("=" * 100)
    print("  RANKING STABILITY SUMMARY TABLE (B={}, seed={})".format(B, SEED))
    print("=" * 100)
    print()
    header = f"{'Analysis':<30} {'What varies':<25} {'tau Borda (mean +/- std)':<25} {'95% CI Borda':<20} {'tau SumScore (mean +/- std)':<28} {'95% CI SumScore':<20}"
    print(header)
    print("-" * len(header))
    for _, row in results_df.iterrows():
        borda_str = f"{row['tau_borda_mean']:.3f} +/- {row['tau_borda_std']:.3f}"
        borda_ci = f"[{row['tau_borda_ci_lo']:.3f}, {row['tau_borda_ci_hi']:.3f}]"
        sum_str = f"{row['tau_sumscore_mean']:.3f} +/- {row['tau_sumscore_std']:.3f}"
        sum_ci = f"[{row['tau_sumscore_ci_lo']:.3f}, {row['tau_sumscore_ci_hi']:.3f}]"
        print(f"{row['analysis']:<30} {row['what_varies']:<25} {borda_str:<25} {borda_ci:<20} {sum_str:<28} {sum_ci:<20}")
    print()

    # ── LaTeX table ───────────────────────────────────────────────────────────
    print("\n% LaTeX table for paper appendix")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Ranking stability of the FFT-50 leaderboard (B=1{,}000, seed=42). "
          r"Kendall's $\tau$ between original and bootstrap rankings.}")
    print(r"\label{tab:ranking_stability}")
    print(r"\small")
    print(r"\begin{tabular}{llcccc}")
    print(r"\toprule")
    print(r"& & \multicolumn{2}{c}{\textbf{Borda}} & \multicolumn{2}{c}{\textbf{SumScore}} \\")
    print(r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}")
    print(r"\textbf{Analysis} & \textbf{What varies} & $\tau$ (mean $\pm$ std) & 95\% CI & $\tau$ (mean $\pm$ std) & 95\% CI \\")
    print(r"\midrule")
    for _, row in results_df.iterrows():
        borda_str = f"${row['tau_borda_mean']:.3f} \\pm {row['tau_borda_std']:.3f}$"
        borda_ci = f"$[{row['tau_borda_ci_lo']:.3f},\\, {row['tau_borda_ci_hi']:.3f}]$"
        sum_str = f"${row['tau_sumscore_mean']:.3f} \\pm {row['tau_sumscore_std']:.3f}$"
        sum_ci = f"$[{row['tau_sumscore_ci_lo']:.3f},\\, {row['tau_sumscore_ci_hi']:.3f}]$"
        analysis_name = row['analysis'].replace('rho=', r'$\rho$=')
        what_varies = row['what_varies'].replace('x', r'$\times$')
        print(f"{analysis_name} & {what_varies} & {borda_str} & {borda_ci} & {sum_str} & {sum_ci} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print(f"\nCSV saved to: {out_dir / 'bootstrap_stability_full_summary.csv'}")


if __name__ == '__main__':
    main()
