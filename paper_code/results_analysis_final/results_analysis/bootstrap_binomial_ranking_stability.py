"""
Parametric Binomial Bootstrap Ranking-Stability Analysis for FFT-50.

Addresses reviewer concern: "single deterministic subsample — no stability checks."

Approach
--------
Each cell accuracy a[i, d, z] was computed on n_d test images. Since we only
have aggregate accuracies (not per-image predictions), we model each accuracy
as a binomial proportion: k = round(a * n_d) ~ Binomial(n_d, p).

For B bootstrap iterations we:
  1. Draw k* ~ Binomial(n_d, a[i,d,z]) for every cell.
  2. Set a*[i,d,z] = k* / n_d.
  3. Recompute the Borda leaderboard from a*.
  4. Record ranks.

This directly tests: "Would the rankings change if we had drawn a different
1,000-image test set from the same distribution?"

Outputs
-------
- bootstrap_binomial_rank_stability.csv  (per-config rank statistics)
- bootstrap_binomial_summary.csv         (overall stability metrics)
- bootstrap_binomial_results.npz         (full arrays for appendix reproduction)
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset test sizes (from the benchmark protocol)
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


def compute_borda(A, n_configs, n_datasets, n_metrics):
    """
    Compute Borda score from a 3-D accuracy tensor A[i, d, z].
    Borda = sum of per-(d, z) ascending ranks (higher accuracy → higher rank → better).
    Uses method='min' for ties, matching the paper's convention.
    """
    borda = np.zeros(n_configs)
    for d in range(n_datasets):
        for z in range(n_metrics):
            col = A[:, d, z]
            # rankdata with method='min': ties get lowest rank in group
            # We rank ascending (higher value = higher rank number = better)
            ranks = rankdata(col, method='min')
            borda += ranks
    return borda


def rank_from_borda(scores):
    """Rank configs: highest Borda = rank 1. Ties: method='min'."""
    return rankdata(-scores, method='min')


def run_binomial_bootstrap(csv_path, B=1000, top_k_frac=0.33, seed=42):
    """
    Parametric binomial bootstrap over cell-level accuracies.
    """
    rng = np.random.default_rng(seed)

    # ── Load and reshape data into 3-D tensor ─────────────────────────────────
    df = pd.read_csv(csv_path, index_col=0)

    datasets = sorted(df['dataset'].unique())
    n_datasets = len(datasets)
    n_metrics = len(METRICS)

    # Create config index (backbone_name × loss_function)
    config_cols = ['backbone_name', 'loss_function', 'model_size']
    configs = df.drop_duplicates(subset=config_cols)[config_cols].reset_index(drop=True)
    # Sort configs for consistent ordering
    configs = configs.sort_values(config_cols).reset_index(drop=True)
    n_configs = len(configs)

    # Build tensor A[i, d, z]
    A = np.full((n_configs, n_datasets, n_metrics), np.nan)
    dataset_sizes = np.zeros(n_datasets, dtype=int)

    for d_idx, ds in enumerate(datasets):
        dataset_sizes[d_idx] = DATASET_TEST_SIZES[ds]
        for i_idx, (_, cfg) in enumerate(configs.iterrows()):
            mask = (
                (df['backbone_name'] == cfg['backbone_name']) &
                (df['loss_function'] == cfg['loss_function']) &
                (df['dataset'] == ds)
            )
            row = df[mask]
            if len(row) == 1:
                for z_idx, metric in enumerate(METRICS):
                    A[i_idx, d_idx, z_idx] = row[metric].values[0]

    # Check for NaN (should be zero based on earlier check)
    nan_count = np.isnan(A).sum()
    if nan_count > 0:
        print(f"WARNING: {nan_count} NaN cells in accuracy tensor. Filling with 0.")
        A = np.nan_to_num(A, nan=0.0)

    # ── Sanity checks ─────────────────────────────────────────────────────────
    # Check degenerate cases (a=1.0 → k*=n_d always; a=0.0 → k*=0 always)
    assert np.all(A >= 0) and np.all(A <= 1), "Accuracies must be in [0, 1]"

    # ── Original ranking ──────────────────────────────────────────────────────
    borda_original = compute_borda(A, n_configs, n_datasets, n_metrics)
    rank_original = rank_from_borda(borda_original)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    top_k = int(np.ceil(top_k_frac * n_configs))
    borda_boot = np.zeros((B, n_configs))
    rank_boot = np.zeros((B, n_configs), dtype=int)
    kendall_taus = np.zeros(B)

    for b in range(B):
        A_star = np.empty_like(A)
        for d in range(n_datasets):
            n_d = dataset_sizes[d]
            # Draw k* ~ Binomial(n_d, a) for all (config, perturbation) at dataset d
            k_star = rng.binomial(n_d, A[:, d, :])  # shape (n_configs, n_metrics)
            A_star[:, d, :] = k_star / n_d
        borda_boot[b] = compute_borda(A_star, n_configs, n_datasets, n_metrics)
        rank_boot[b] = rank_from_borda(borda_boot[b])
        tau, _ = kendalltau(rank_original, rank_boot[b])
        kendall_taus[b] = tau

    # ── Sanity 4: bootstrap is unbiased ───────────────────────────────────────
    # A_star mean across bootstrap iterations should ≈ A
    # (We can't easily compute this without storing all A_star, but we verify
    # borda_boot mean ≈ borda_original as a proxy)
    borda_bias = (borda_boot.mean(axis=0) - borda_original).mean()
    print(f"[Sanity] Mean Borda bias (should be ~0): {borda_bias:.4f}")

    # ── Per-configuration statistics ──────────────────────────────────────────
    mean_ranks = rank_boot.mean(axis=0)
    ci_lower = np.percentile(rank_boot, 2.5, axis=0)
    ci_upper = np.percentile(rank_boot, 97.5, axis=0)
    ci_width = ci_upper - ci_lower

    borda_ci_lower = np.percentile(borda_boot, 2.5, axis=0)
    borda_ci_upper = np.percentile(borda_boot, 97.5, axis=0)

    in_tier1 = (rank_boot <= top_k).astype(float).mean(axis=0)

    # ── Top-k overlap ─────────────────────────────────────────────────────────
    top_k_overlap = {}
    for k in [3, 5, 10]:
        original_top_k = set(np.where(rank_original <= k)[0])
        overlaps = []
        for b in range(B):
            boot_top_k = set(np.where(rank_boot[b] <= k)[0])
            overlaps.append(len(original_top_k & boot_top_k) / k)
        top_k_overlap[k] = np.mean(overlaps)

    # ── Top-3 stability per size ──────────────────────────────────────────────
    size_stability = {}
    for size in configs['model_size'].unique():
        size_mask = (configs['model_size'] == size).values
        # Original top-3 in this size
        size_ranks_orig = rank_from_borda(borda_original[size_mask])
        orig_top3_indices = np.where(size_ranks_orig <= 3)[0]

        # For each bootstrap: compute within-size ranking, check if orig top-3 stay
        stay_count = np.zeros(len(orig_top3_indices))
        for b in range(B):
            size_ranks_b = rank_from_borda(borda_boot[b][size_mask])
            for j, idx in enumerate(orig_top3_indices):
                if size_ranks_b[idx] <= 3:
                    stay_count[j] += 1
        probs = stay_count / B
        # Get config names for these
        size_configs = configs[size_mask].iloc[orig_top3_indices]
        for (_, cfg), prob in zip(size_configs.iterrows(), probs):
            key = f"{cfg['backbone_name']}_{cfg['loss_function']}_{size}"
            size_stability[key] = prob

    # ── Build results DataFrame ───────────────────────────────────────────────
    results_df = pd.DataFrame({
        'backbone_name': configs['backbone_name'].values,
        'loss_function': configs['loss_function'].values,
        'model_size': configs['model_size'].values,
        'original_borda': borda_original,
        'original_rank': rank_original,
        'mean_bootstrap_rank': mean_ranks,
        'rank_ci_lower_2.5': ci_lower,
        'rank_ci_upper_97.5': ci_upper,
        'rank_ci_width': ci_width,
        'borda_ci_lower_2.5': borda_ci_lower,
        'borda_ci_upper_97.5': borda_ci_upper,
        'prob_tier1': in_tier1,
    }).sort_values('original_rank')

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        'method': 'parametric_binomial_bootstrap',
        'n_configurations': n_configs,
        'n_datasets': n_datasets,
        'n_metrics': n_metrics,
        'n_bootstrap_iterations': B,
        'top_k': top_k,
        'top_k_fraction': top_k_frac,
        'kendall_tau_mean': kendall_taus.mean(),
        'kendall_tau_std': kendall_taus.std(),
        'kendall_tau_median': np.median(kendall_taus),
        'kendall_tau_ci_lower': np.percentile(kendall_taus, 2.5),
        'kendall_tau_ci_upper': np.percentile(kendall_taus, 97.5),
        'mean_rank_ci_width': ci_width.mean(),
        'median_rank_ci_width': np.median(ci_width),
        'top3_overlap_mean': top_k_overlap[3],
        'top5_overlap_mean': top_k_overlap[5],
        'top10_overlap_mean': top_k_overlap[10],
        'borda_bias_mean': borda_bias,
    }
    # Add per-size top-3 stability
    for key, prob in size_stability.items():
        summary[f'top3_stay_prob_{key}'] = prob

    summary_df = pd.DataFrame([summary])

    return results_df, summary_df, kendall_taus, borda_boot, rank_boot


if __name__ == '__main__':
    csv_path = Path(__file__).parent / 'fft50.csv'

    print("Running parametric binomial bootstrap ranking stability analysis...")
    print(f"Dataset test sizes: {DATASET_TEST_SIZES}")
    print()

    results_df, summary_df, kendall_taus, borda_boot, rank_boot = \
        run_binomial_bootstrap(csv_path, B=1000, seed=42)

    out_dir = Path(__file__).parent / 'bootstrap_results'
    out_dir.mkdir(exist_ok=True)

    results_df.to_csv(out_dir / 'bootstrap_binomial_rank_stability.csv', index=False)
    summary_df.to_csv(out_dir / 'bootstrap_binomial_summary.csv', index=False)
    np.savez_compressed(
        out_dir / 'bootstrap_binomial_results.npz',
        kendall_taus=kendall_taus,
        borda_boot=borda_boot,
        rank_boot=rank_boot,
    )

    print("\n" + "=" * 70)
    print("  PARAMETRIC BINOMIAL BOOTSTRAP RANKING STABILITY")
    print("=" * 70)
    print(f"Configurations: {summary_df['n_configurations'].iloc[0]}")
    print(f"Datasets: {summary_df['n_datasets'].iloc[0]}")
    print(f"Metrics (perturbation settings): {summary_df['n_metrics'].iloc[0]}")
    print(f"Bootstrap iterations: {summary_df['n_bootstrap_iterations'].iloc[0]}")
    print()
    print(f"Kendall's τ (mean ± std): {summary_df['kendall_tau_mean'].iloc[0]:.4f} ± {summary_df['kendall_tau_std'].iloc[0]:.4f}")
    print(f"Kendall's τ 95% CI: [{summary_df['kendall_tau_ci_lower'].iloc[0]:.4f}, {summary_df['kendall_tau_ci_upper'].iloc[0]:.4f}]")
    print(f"Mean rank CI width: {summary_df['mean_rank_ci_width'].iloc[0]:.2f}")
    print(f"Median rank CI width: {summary_df['median_rank_ci_width'].iloc[0]:.2f}")
    print()
    print(f"Top-3 overlap (mean): {summary_df['top3_overlap_mean'].iloc[0]:.4f}")
    print(f"Top-5 overlap (mean): {summary_df['top5_overlap_mean'].iloc[0]:.4f}")
    print(f"Top-10 overlap (mean): {summary_df['top10_overlap_mean'].iloc[0]:.4f}")
    print()
    print("Top-3 per-size stability (prob of staying in top-3):")
    for col in summary_df.columns:
        if col.startswith('top3_stay_prob_'):
            name = col.replace('top3_stay_prob_', '')
            print(f"  {name}: {summary_df[col].iloc[0]:.4f}")

    print("\n=== Top-10 Configurations (by original rank) ===")
    print(results_df.head(10).to_string(index=False))

    print(f"\nResults saved to: {out_dir}")
