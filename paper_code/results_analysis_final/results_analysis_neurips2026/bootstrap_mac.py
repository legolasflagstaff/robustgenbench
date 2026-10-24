"""
Bootstrap 95% confidence intervals for MAC(avg, J) — naive and stratified.

Addresses reviewer concerns (YDZp W4, sJ4U W4) about correlation stability
in small subsets. Reuses the MAC definition from generate_mac_table.py.

Usage:
    python bootstrap_mac.py --input fft_50_full.csv --B 1000 --seed 42
"""

import argparse
import pandas as pd
import numpy as np
from itertools import combinations

# ─────────────────────────────────────────────────────────────────────────────
# Core MAC metric (matches generate_mac_table.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
ACC_COLS = ['clean_acc', 'Linf_acc', 'L2_acc', 'L1_acc', 'common_acc']


def mean_absolute_correlation(sub: pd.DataFrame, min_rows: int = 3) -> float:
    """
    MAC for a subset: mean absolute off-diagonal Spearman correlation
    across perturbation types.
    """
    if len(sub) < min_rows:
        return np.nan
    corr = sub[ACC_COLS].corr(method='spearman')
    mask = ~np.eye(len(ACC_COLS), dtype=bool)
    return np.abs(corr.values[mask]).mean()


# ─────────────────────────────────────────────────────────────────────────────
# MAC computation: naive and stratified
# ─────────────────────────────────────────────────────────────────────────────
def compute_mac_avg(df_subset: pd.DataFrame, datasets: list) -> float:
    """Naive MAC(avg, J): per-dataset MAC averaged across datasets."""
    macs = []
    for ds in datasets:
        sub = df_subset[df_subset['dataset'] == ds]
        m = mean_absolute_correlation(sub)
        if not np.isnan(m):
            macs.append(m)
    return np.mean(macs) if macs else np.nan


def compute_mac_avg_stratified(
    df_subset: pd.DataFrame,
    stratify_col: str,
    datasets: list,
    min_rows: int = 4
) -> float:
    """
    Stratified MAC(avg_s, J): for each dataset, compute MAC per stratum,
    average across strata (equal weight), then average across datasets.
    """
    dataset_macs = []
    for ds in datasets:
        ds_df = df_subset[df_subset['dataset'] == ds]
        stratum_macs = []
        for _, sdf in ds_df.groupby(stratify_col):
            if len(sdf) >= min_rows:
                m = mean_absolute_correlation(sdf)
                if not np.isnan(m):
                    stratum_macs.append(m)
        if stratum_macs:
            dataset_macs.append(np.mean(stratum_macs))
    return np.mean(dataset_macs) if dataset_macs else np.nan


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_mac(
    df_by_dataset: dict,
    config_indices: np.ndarray,
    datasets: list,
    B: int,
    rng: np.random.Generator,
    stratify_col: str = None,
    min_rows: int = 4
) -> dict:
    """
    Bootstrap MAC(avg, J) by resampling configurations with replacement.

    df_by_dataset: dict mapping dataset -> DataFrame with positional index
                   matching config_indices.
    config_indices: integer indices (0..79) identifying configurations in subset J.
    
    Resampling indices are shared across datasets to preserve the cross-dataset
    structure of MAC (same configs selected for every dataset).

    Returns dict with: point, mean, ci_lo, ci_hi, std
    """
    n_J = len(config_indices)

    # Build point-estimate subset
    subset_frames = []
    for ds in datasets:
        subset_frames.append(df_by_dataset[ds].iloc[config_indices])
    df_subset = pd.concat(subset_frames, ignore_index=True)

    if stratify_col is None:
        point = compute_mac_avg(df_subset, datasets)
    else:
        point = compute_mac_avg_stratified(df_subset, stratify_col, datasets, min_rows)

    # Bootstrap
    mac_boot = np.zeros(B)
    for b in range(B):
        resampled_idx = rng.choice(config_indices, size=n_J, replace=True)
        # Build resampled dataframe (same resampled configs across all datasets)
        boot_frames = []
        for ds in datasets:
            boot_frames.append(df_by_dataset[ds].iloc[resampled_idx])
        df_boot = pd.concat(boot_frames, ignore_index=True)

        if stratify_col is None:
            mac_boot[b] = compute_mac_avg(df_boot, datasets)
        else:
            mac_boot[b] = compute_mac_avg_stratified(
                df_boot, stratify_col, datasets, min_rows
            )

    ci_lo, ci_hi = np.nanpercentile(mac_boot, [2.5, 97.5])

    return {
        'point': point,
        'mean': np.nanmean(mac_boot),
        'ci_lo': ci_lo,
        'ci_hi': ci_hi,
        'std': np.nanstd(mac_boot),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Table 2 subset definitions
# ─────────────────────────────────────────────────────────────────────────────
# Stratification rules from the paper:
#   - For all categories except 'Size': stratify by model_size
#   - For 'Size' category: stratify by model_type
CATEGORIES = [
    # (category_name, group_col, stratify_col)
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
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs for MAC table (Table 2)."
    )
    parser.add_argument(
        "--input", type=str,
        default="./fft_50_full.csv",
        help="Path to input CSV"
    )
    parser.add_argument("--B", type=int, default=1000, help="Bootstrap iterations")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument(
        "--min-rows", type=int, default=4,
        help="Minimum rows per stratum (default: 4)"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.input)
    datasets = sorted(df['dataset'].unique())

    # ─── Build aligned per-dataset dataframes ───
    # Sort each dataset's rows identically so positional index is consistent.
    # The sort key must produce a unique ordering for the 80 rows per dataset.
    sort_cols = ['backbone_name', 'loss_function', 'pre_training_strategy', 'model_size']
    df_by_dataset = {}
    for ds in datasets:
        ds_df = df[df['dataset'] == ds].sort_values(sort_cols).reset_index(drop=True)
        df_by_dataset[ds] = ds_df

    n_configs = len(df_by_dataset[datasets[0]])  # 80

    # Build a metadata frame for config indices (from first dataset, order-aligned)
    meta = df_by_dataset[datasets[0]][
        ['backbone_name', 'model_size', 'model_type',
         'loss_function', 'pre_training_strategy']
    ].reset_index(drop=True)

    print(f"Bootstrap MAC table — B={args.B}, seed={args.seed}, "
          f"min_rows={args.min_rows}")
    print(f"Datasets ({len(datasets)}): {datasets}")
    print(f"Total configs per dataset: {n_configs}")
    print("=" * 100)

    # ─── Results collection ───
    results = []

    # Global row
    global_indices = np.arange(n_configs)
    res_naive = bootstrap_mac(df_by_dataset, global_indices, datasets, args.B, rng)
    results.append({
        'category': 'All',
        'subset': f'All {n_configs} configs',
        'n': n_configs,
        'mac_point': res_naive['point'],
        'mac_ci_lo': res_naive['ci_lo'],
        'mac_ci_hi': res_naive['ci_hi'],
        'macs_point': np.nan,
        'macs_ci_lo': np.nan,
        'macs_ci_hi': np.nan,
    })

    # Per-category rows
    for cat_name, group_col, stratify_col in CATEGORIES:
        for val in sorted(df[group_col].unique()):
            config_indices = meta[meta[group_col] == val].index.values
            n = len(config_indices)

            display_val = SIZE_DISPLAY.get(val, val)

            # Naive bootstrap
            res_naive = bootstrap_mac(
                df_by_dataset, config_indices, datasets, args.B, rng
            )

            # Stratified bootstrap
            res_strat = bootstrap_mac(
                df_by_dataset, config_indices, datasets, args.B, rng,
                stratify_col=stratify_col, min_rows=args.min_rows
            )

            results.append({
                'category': cat_name,
                'subset': display_val,
                'n': n,
                'mac_point': res_naive['point'],
                'mac_ci_lo': res_naive['ci_lo'],
                'mac_ci_hi': res_naive['ci_hi'],
                'macs_point': res_strat['point'],
                'macs_ci_lo': res_strat['ci_lo'],
                'macs_ci_hi': res_strat['ci_hi'],
            })

    # ─── Print table ───
    print(f"\n{'Category':<15} {'Subset':<30} {'n':>3}  "
          f"{'MACavg':>7} {'95% CI':>15}  "
          f"{'MACavg_s':>8} {'95% CI':>15}")
    print("-" * 100)

    for r in results:
        ci_naive = f"[{r['mac_ci_lo']:.3f}, {r['mac_ci_hi']:.3f}]"
        if np.isnan(r['macs_point']):
            ci_strat = "---"
            strat_pt = "---"
        else:
            ci_strat = f"[{r['macs_ci_lo']:.3f}, {r['macs_ci_hi']:.3f}]"
            strat_pt = f"{r['macs_point']:.3f}"

        print(f"{r['category']:<15} {r['subset']:<30} {r['n']:>3}  "
              f"{r['mac_point']:.3f}   {ci_naive:<15}  "
              f"{strat_pt:>8}   {ci_strat}")

    # ─── Priority claim checks ───
    print("\n" + "=" * 100)
    print("PRIORITY CLAIM CHECKS")
    print("=" * 100)

    # Build lookup
    lookup = {(r['category'], r['subset']): r for r in results}

    # Claim 1: Hybrid > Fully convolutional and Fully attention
    print("\n--- Claim 1: Hybrid architecture headline ---")
    hybrid = lookup[('Type', 'hybrid')]
    conv = lookup[('Type', 'fully convolutional')]
    attn = lookup[('Type', 'fully attention')]

    print(f"  Hybrid MAC:       {hybrid['mac_point']:.3f}  CI [{hybrid['mac_ci_lo']:.3f}, {hybrid['mac_ci_hi']:.3f}]")
    print(f"  Conv MAC:         {conv['mac_point']:.3f}  CI [{conv['mac_ci_lo']:.3f}, {conv['mac_ci_hi']:.3f}]")
    print(f"  Attention MAC:    {attn['mac_point']:.3f}  CI [{attn['mac_ci_lo']:.3f}, {attn['mac_ci_hi']:.3f}]")

    check1a = hybrid['mac_ci_lo'] > conv['mac_point']
    check1b = hybrid['mac_ci_lo'] > attn['mac_point']
    print(f"\n  Hybrid CI_lo ({hybrid['mac_ci_lo']:.3f}) > Conv point ({conv['mac_point']:.3f})? "
          f"{'YES ✓' if check1a else 'NO ✗'}")
    print(f"  Hybrid CI_lo ({hybrid['mac_ci_lo']:.3f}) > Attention point ({attn['mac_point']:.3f})? "
          f"{'YES ✓' if check1b else 'NO ✗'}")

    # Stratified version
    print(f"\n  [Stratified]")
    print(f"  Hybrid MACavg_s:    {hybrid['macs_point']:.3f}  CI [{hybrid['macs_ci_lo']:.3f}, {hybrid['macs_ci_hi']:.3f}]")
    print(f"  Conv MACavg_s:      {conv['macs_point']:.3f}  CI [{conv['macs_ci_lo']:.3f}, {conv['macs_ci_hi']:.3f}]")
    print(f"  Attention MACavg_s: {attn['macs_point']:.3f}  CI [{attn['macs_ci_lo']:.3f}, {attn['macs_ci_hi']:.3f}]")

    check1c = hybrid['macs_ci_lo'] > conv['macs_point']
    check1d = hybrid['macs_ci_lo'] > attn['macs_point']
    print(f"\n  Hybrid CI_lo ({hybrid['macs_ci_lo']:.3f}) > Conv point ({conv['macs_point']:.3f})? "
          f"{'YES ✓' if check1c else 'NO ✗'}")
    print(f"  Hybrid CI_lo ({hybrid['macs_ci_lo']:.3f}) > Attention point ({attn['macs_point']:.3f})? "
          f"{'YES ✓' if check1d else 'NO ✗'}")

    # Claim 2: Base > Small (size monotonicity)
    print("\n--- Claim 2: Size monotonicity (Base > Small) ---")
    base = lookup[('Size', 'Base')]
    small = lookup[('Size', 'Small')]
    tiny = lookup[('Size', 'Tiny')]

    print(f"  Base MAC:   {base['mac_point']:.3f}  CI [{base['mac_ci_lo']:.3f}, {base['mac_ci_hi']:.3f}]")
    print(f"  Small MAC:  {small['mac_point']:.3f}  CI [{small['mac_ci_lo']:.3f}, {small['mac_ci_hi']:.3f}]")
    print(f"  Tiny MAC:   {tiny['mac_point']:.3f}  CI [{tiny['mac_ci_lo']:.3f}, {tiny['mac_ci_hi']:.3f}]")

    check2a = base['mac_ci_lo'] > small['mac_point']
    check2b = base['mac_ci_lo'] > tiny['mac_point']
    print(f"\n  Base CI_lo ({base['mac_ci_lo']:.3f}) > Small point ({small['mac_point']:.3f})? "
          f"{'YES ✓' if check2a else 'NO ✗'}")
    print(f"  Base CI_lo ({base['mac_ci_lo']:.3f}) > Tiny point ({tiny['mac_point']:.3f})? "
          f"{'YES ✓' if check2b else 'NO ✗'}")

    # Stratified
    print(f"\n  [Stratified]")
    print(f"  Base MACavg_s:  {base['macs_point']:.3f}  CI [{base['macs_ci_lo']:.3f}, {base['macs_ci_hi']:.3f}]")
    print(f"  Small MACavg_s: {small['macs_point']:.3f}  CI [{small['macs_ci_lo']:.3f}, {small['macs_ci_hi']:.3f}]")
    print(f"  Tiny MACavg_s:  {tiny['macs_point']:.3f}  CI [{tiny['macs_ci_lo']:.3f}, {tiny['macs_ci_hi']:.3f}]")

    check2c = base['macs_ci_lo'] > small['macs_point']
    print(f"\n  Base CI_lo ({base['macs_ci_lo']:.3f}) > Small point ({small['macs_point']:.3f})? "
          f"{'YES ✓' if check2c else 'NO ✗'}")

    # ─── CI overlap check ───
    print("\n--- CI overlap summary ---")
    print("  (Non-overlapping CIs = stronger evidence than point-estimate comparison)")

    # Hybrid vs Conv CIs
    overlap_hc = hybrid['mac_ci_lo'] <= conv['mac_ci_hi'] and conv['mac_ci_lo'] <= hybrid['mac_ci_hi']
    print(f"\n  Hybrid vs Conv CIs overlap (naive)?      {'YES' if overlap_hc else 'NO (non-overlapping)'}")

    overlap_ha = hybrid['mac_ci_lo'] <= attn['mac_ci_hi'] and attn['mac_ci_lo'] <= hybrid['mac_ci_hi']
    print(f"  Hybrid vs Attention CIs overlap (naive)?  {'YES' if overlap_ha else 'NO (non-overlapping)'}")

    overlap_bs = base['mac_ci_lo'] <= small['mac_ci_hi'] and small['mac_ci_lo'] <= base['mac_ci_hi']
    print(f"  Base vs Small CIs overlap (naive)?        {'YES' if overlap_bs else 'NO (non-overlapping)'}")

    # Stratified
    overlap_hc_s = hybrid['macs_ci_lo'] <= conv['macs_ci_hi'] and conv['macs_ci_lo'] <= hybrid['macs_ci_hi']
    print(f"\n  Hybrid vs Conv CIs overlap (stratified)?      {'YES' if overlap_hc_s else 'NO (non-overlapping)'}")

    overlap_ha_s = hybrid['macs_ci_lo'] <= attn['macs_ci_hi'] and attn['macs_ci_lo'] <= hybrid['macs_ci_hi']
    print(f"  Hybrid vs Attention CIs overlap (stratified)?  {'YES' if overlap_ha_s else 'NO (non-overlapping)'}")

    overlap_bs_s = base['macs_ci_lo'] <= small['macs_ci_hi'] and small['macs_ci_lo'] <= base['macs_ci_hi']
    print(f"  Base vs Small CIs overlap (stratified)?        {'YES' if overlap_bs_s else 'NO (non-overlapping)'}")


if __name__ == "__main__":
    main()
