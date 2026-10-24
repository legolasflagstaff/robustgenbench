"""Aggregate metrics of the paper (Section 3).

All functions take a long-format table of accuracies with columns
``model``, ``dataset``, ``setting``, ``accuracy`` (in [0, 1]).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def _pivot(results: pd.DataFrame) -> pd.DataFrame:
    """models x (dataset, setting) accuracy matrix."""
    missing = {"model", "dataset", "setting", "accuracy"} - set(results.columns)
    if missing:
        raise ValueError(f"results is missing columns {missing}")
    return results.pivot_table(index="model", columns=["dataset", "setting"],
                               values="accuracy", aggfunc="first")


def sum_score(results: pd.DataFrame) -> pd.Series:
    """Sum of accuracies over all (dataset, setting) pairs, per model.

    Pool-independent: use it to compare a new model with published results.
    With the 5 white-box settings on 6 datasets the maximum is 30.
    """
    return _pivot(results).sum(axis=1).sort_values(ascending=False).rename("sum_score")


def borda_score(results: pd.DataFrame) -> pd.Series:
    """Borda score per model: within each (dataset, setting) pair, models are ranked
    by accuracy (ties share the lowest rank, worst model = 1) and ranks are summed.

    Pool-dependent: only comparable within a fixed set of models. This is the
    convention used to produce the leaderboards of the paper.
    """
    acc = _pivot(results)
    if acc.isna().any().any():
        raise ValueError("borda_score requires every model to be evaluated on every pair")
    ranks = acc.rank(axis=0, ascending=True, method="min")
    return ranks.sum(axis=1).sort_values(ascending=False).rename("borda_score")


def mean_absolute_correlation(results: pd.DataFrame) -> float:
    """MAC of a group of models (paper Section 3): for each dataset, the mean absolute
    Spearman correlation between perturbation settings, computed over the models of
    the group; then averaged over datasets.

    Close to 1: robustness transfers uniformly across settings within the group.
    """
    macs = []
    for _, sub in results.groupby("dataset"):
        acc = sub.pivot_table(index="model", columns="setting", values="accuracy")
        if len(acc) < 3 or acc.shape[1] < 2:
            continue
        corr = acc.rank().corr(method="pearson").to_numpy()  # Spearman on ranks
        z = acc.shape[1]
        macs.append(np.nansum(np.abs(corr[~np.eye(z, dtype=bool)])) / (z * (z - 1)))
    if not macs:
        raise ValueError("MAC needs at least 3 models and 2 settings on one dataset")
    return float(np.mean(macs))


def kendall_tau(a: pd.Series, b: pd.Series) -> float:
    """Kendall's tau-b between two score series indexed by model (e.g. Borda vs Sum)."""
    common = a.index.intersection(b.index)
    x, y = a[common].to_numpy(), b[common].to_numpy()
    con = dis = tx = ty = 0
    for i, j in combinations(range(len(x)), 2):
        dx, dy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        if dx == 0 and dy == 0:
            continue
        if dx == 0:
            tx += 1
        elif dy == 0:
            ty += 1
        elif dx == dy:
            con += 1
        else:
            dis += 1
    return (con - dis) / np.sqrt((con + dis + tx) * (con + dis + ty))
