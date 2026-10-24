"""
Unified bootstrap + LOO stability table for MAC (Table 2 extension).

Produces:
  - Point estimates (matching Table 2)
  - 95% bootstrap CIs (Method A: config resampling)
  - Leave-one-dataset-out (LOO) range for stability

Also computes:
  - Method E paired-difference CIs for the two headline claims
  - Per-dataset MAC breakdown

Usage:
    python bootstrap_mac_full.py --input fft_50_full.csv --B 10000 --seed 42
"""

import argparse
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core MAC metric (matches generate_mac_table.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
ACC_COLS = ['clean_acc', 'Linf_acc', 'L2_acc', 'L1_acc', 'common_acc']


def mean_absolute_correlation(sub: pd.DataFrame, min_rows: int = 3) -> float:
    if len(sub) < min_rows:
        return np.nan
    corr = sub[ACC_COLS].corr(method='spearman')
    mask = ~np.eye(len(ACC_COLS), dtype=bool)
    return np.abs(corr.values[mask]).mean()


# ─────────────────────────────────────────────────────────────────────────────
# MAC computation
# ─────────────────────────────────────────────────────────────────────────────
def compute_mac_avg(df_by_dataset: dict, indices, datasets: list) -> float:
    macs = []
    for ds in datasets:
        sub = df_by_dataset[ds].iloc[indices]
        m = mean_absolute_correlation(sub)
        if not np.isnan(m):
            macs.append(m)
    return np.mean(macs) if macs else np.nan


def compute_mac_avg_stratified(
    df_by_dataset: dict, indices, stratify_col: str,
    datasets: list, min_rows: int = 4
) -> float:
    dataset_macs = []
    for ds in datasets:
        ds_df = df_by_dataset[ds].iloc[indices]
        stratum_macs = []
        for _, sdf in ds_df.groupby(stratify_col):
            if len(sdf) >= min_rows:
                m = mean_absolute_correlation(sdf)
                if not np.isnan(m):
                    stratum_macs.append(m)
        if stratum_macs:
            dataset_macs.append(np.mean(stratum_macs))
    return np.mean(dataset_macs) if dataset_macs else np.nan


def compute_mac_loo(
    df_by_dataset: dict, indices, datasets: list,
    stratify_col: str = None, min_rows: int = 4
) -> tuple:
    """Leave-one-dataset-out: return (min, max) of MAC(avg) over 6 LOO folds."""
    loo_vals = []
    for skip_ds in datasets:
        remaining = [ds for ds in datasets if ds != skip_ds]
        if stratify_col is None:
            val = compute_mac_avg(df_by_dataset, indices, remaining)
        else:
            val = compute_mac_avg_stratified(
                df_by_dataset, indices, stratify_col, remaining, min_rows
            )
        if not np.isnan(val):
            loo_vals.append(val)
    if loo_vals:
        return min(loo_vals), max(loo_vals)
    return np.nan, np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_mac(
    df_by_dataset: dict, config_indices: np.ndarray, datasets: list,
    B: int, rng: np.random.Generator,
    stratify_col: str = None, min_rows: int = 4
) -> dict:
    n_J = len(config_indices)
    mac_boot = np.zeros(B)
    for b in range(B):
        resampled = rng.choice(config_indices, size=n_J, replace=True)
        if stratify_col is None:
            mac_boot[b] = compute_mac_avg(df_by_dataset, resampled, datasets)
        else:
            mac_boot[b] = compute_mac_avg_stratified(
                df_by_dataset, resampled, stratify_col, datasets, min_rows
            )
    ci_lo, ci_hi = np.nanpercentile(mac_boot, [2.5, 97.5])
    return {'ci_lo': ci_lo, 'ci_hi': ci_hi}


def bootstrap_paired_difference(
    df_by_dataset: dict, idx_a: np.ndarray, idx_b: np.ndarray,
    datasets: list, B: int, rng: np.random.Generator
) -> dict:
    """Bootstrap the mean per-dataset difference MAC(A,d) - MAC(B,d)."""
    # Compute observed per-dataset differences
    obs_diffs = []
    for ds in datasets:
        ma = mean_absolute_correlation(df_by_dataset[ds].iloc[idx_a])
        mb = mean_absolute_correlation(df_by_dataset[ds].iloc[idx_b])
        obs_diffs.append(ma - mb)
    obs_diffs = np.array(obs_diffs)

    # Bootstrap the mean difference (resample dataset-level diffs)
    diff_boot = np.zeros(B)
    for b in range(B):
        sample = rng.choice(obs_diffs, size=len(obs_diffs), replace=True)
        diff_boot[b] = sample.mean()

    ci_lo, ci_hi = np.percentile(diff_boot, [2.5, 97.5])
    return {
        'mean_diff': obs_diffs.mean(),
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'per_dataset': obs_diffs,
        'positive_count': int(np.sum(obs_diffs > 0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 categories
# ─────────────────────────────────────────────────────────────────────────────
CATEGORIES = [
    ('Loss', 'loss_function', 'model_size'),
    ('Size', 'model_size', 'model_type'),
    ('Type', 'model_type', 'model_size'),
    ('Pre-training', 'pre_training_strategy', 'model_size'),
]

SIZE_DISPLAY = {'large': 'Base', 'medium': 'Small', 'small': 'Tiny'}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="./fft_50_full.csv")
    parser.add_argument("--B", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-rows", type=int, default=4)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.input)
    datasets = sorted(df['dataset'].unique())

    # Build aligned per-dataset dataframes
    sort_cols = ['backbone_name', 'loss_function', 'pre_training_strategy', 'model_size']
    df_by_dataset = {}
    for ds in datasets:
        df_by_dataset[ds] = df[df['dataset'] == ds].sort_values(sort_cols).reset_index(drop=True)

    n_configs = len(df_by_dataset[datasets[0]])
    meta = df_by_dataset[datasets[0]][
        ['backbone_name', 'model_size', 'model_type', 'loss_function', 'pre_training_strategy']
    ].reset_index(drop=True)

    print(f"Bootstrap MAC + LOO stability — B={args.B}, seed={args.seed}")
    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Configs per dataset: {n_configs}")

    # ─── Compute full table ───
    results = []

    # Global
    global_idx = np.arange(n_configs)
    mac_global = compute_mac_avg(df_by_dataset, global_idx, datasets)
    ci_global = bootstrap_mac(df_by_dataset, global_idx, datasets, args.B, rng)
    loo_global = compute_mac_loo(df_by_dataset, global_idx, datasets)

    results.append({
        'category': 'All', 'subset': f'All {n_configs} configs', 'n': n_configs,
        'mac': mac_global, 'ci_lo': ci_global['ci_lo'], 'ci_hi': ci_global['ci_hi'],
        'loo_lo': loo_global[0], 'loo_hi': loo_global[1],
        'macs': np.nan, 'macs_ci_lo': np.nan, 'macs_ci_hi': np.nan,
        'macs_loo_lo': np.nan, 'macs_loo_hi': np.nan,
    })

    # Per category
    for cat_name, group_col, stratify_col in CATEGORIES:
        for val in sorted(meta[group_col].unique()):
            idx = meta[meta[group_col] == val].index.values
            n = len(idx)
            display = SIZE_DISPLAY.get(val, val)

            # Naive
            mac = compute_mac_avg(df_by_dataset, idx, datasets)
            ci = bootstrap_mac(df_by_dataset, idx, datasets, args.B, rng)
            loo = compute_mac_loo(df_by_dataset, idx, datasets)

            # Stratified
            macs = compute_mac_avg_stratified(
                df_by_dataset, idx, stratify_col, datasets, args.min_rows
            )
            ci_s = bootstrap_mac(
                df_by_dataset, idx, datasets, args.B, rng,
                stratify_col=stratify_col, min_rows=args.min_rows
            )
            loo_s = compute_mac_loo(
                df_by_dataset, idx, datasets,
                stratify_col=stratify_col, min_rows=args.min_rows
            )

            results.append({
                'category': cat_name, 'subset': display, 'n': n,
                'mac': mac, 'ci_lo': ci['ci_lo'], 'ci_hi': ci['ci_hi'],
                'loo_lo': loo[0], 'loo_hi': loo[1],
                'macs': macs, 'macs_ci_lo': ci_s['ci_lo'], 'macs_ci_hi': ci_s['ci_hi'],
                'macs_loo_lo': loo_s[0], 'macs_loo_hi': loo_s[1],
            })

    # ─── Print unified table ───
    print("\n" + "=" * 140)
    hdr = (f"{'Category':<12} {'Subset':<28} {'n':>3}  "
           f"{'MACavg':>6} {'95% CI':>15} {'LOO range':>15}  "
           f"{'MACavg_s':>7} {'95% CI':>15} {'LOO range':>15}")
    print(hdr)
    print("-" * 140)

    for r in results:
        ci_n = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        loo_n = f"[{r['loo_lo']:.3f}, {r['loo_hi']:.3f}]" if not np.isnan(r['loo_lo']) else "---"

        if np.isnan(r['macs']):
            macs_str, ci_s, loo_s = "---", "---", "---"
        else:
            macs_str = f"{r['macs']:.3f}"
            ci_s = f"[{r['macs_ci_lo']:.3f}, {r['macs_ci_hi']:.3f}]"
            loo_s = f"[{r['macs_loo_lo']:.3f}, {r['macs_loo_hi']:.3f}]" if not np.isnan(r['macs_loo_lo']) else "---"

        print(f"{r['category']:<12} {r['subset']:<28} {r['n']:>3}  "
              f"{r['mac']:.3f}  {ci_n:>15} {loo_n:>15}  "
              f"{macs_str:>7}  {ci_s:>15} {loo_s:>15}")

    # ─── LOO spread summary ───
    print("\n" + "=" * 140)
    print("LOO SPREAD SUMMARY (smaller = more stable across datasets)")
    print("-" * 80)
    print(f"{'Subset':<28} {'MACavg LOO spread':>18} {'MACavg_s LOO spread':>20}")
    print("-" * 80)
    for r in results[1:]:  # skip global
        spread_n = f"{r['loo_hi'] - r['loo_lo']:.3f}" if not np.isnan(r['loo_lo']) else "---"
        spread_s = f"{r['macs_loo_hi'] - r['macs_loo_lo']:.3f}" if not np.isnan(r['macs_loo_lo']) else "---"
        print(f"{r['subset']:<28} {spread_n:>18} {spread_s:>20}")

    # ─── Method E: Paired-difference CIs for headline claims ───
    print("\n" + "=" * 140)
    print("METHOD E: PAIRED-DIFFERENCE BOOTSTRAP (headline claims)")
    print("=" * 140)

    hybrid_idx = meta[meta['model_type'] == 'hybrid'].index.values
    attn_idx = meta[meta['model_type'] == 'fully attention'].index.values
    conv_idx = meta[meta['model_type'] == 'fully convolutional'].index.values
    base_idx = meta[meta['model_size'] == 'large'].index.values
    small_idx = meta[meta['model_size'] == 'medium'].index.values

    claims = [
        ("Hybrid − Attention", hybrid_idx, attn_idx),
        ("Hybrid − Conv", hybrid_idx, conv_idx),
        ("Base − Small", base_idx, small_idx),
    ]

    for name, idx_a, idx_b in claims:
        res = bootstrap_paired_difference(
            df_by_dataset, idx_a, idx_b, datasets, args.B, rng
        )
        excludes = "YES ✓" if res['ci_lo'] > 0 else "NO"
        print(f"\n  {name}:")
        print(f"    Mean diff: {res['mean_diff']:.3f}")
        print(f"    95% CI: [{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]")
        print(f"    CI excludes 0: {excludes}")
        print(f"    Positive in {res['positive_count']}/{len(datasets)} datasets")
        print(f"    Per-dataset: {[f'{d:+.3f}' for d in res['per_dataset']]}")


if __name__ == "__main__":
    main()
