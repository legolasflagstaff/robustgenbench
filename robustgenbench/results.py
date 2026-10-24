"""Result records, persistence, and comparison with the paper's leaderboards."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable

import pandas as pd

from .metrics import borda_score, sum_score


@dataclass
class Result:
    """Accuracy of one model on one (dataset, perturbation setting) pair."""

    model: str
    dataset: str
    setting: str
    accuracy: float
    num_examples: int
    mode: str  # "whitebox" or "transfer"
    source: str | None = None  # attack source: "whitebox_autoattack" or the surrogate used


def to_frame(results: Iterable[Result]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def save(results: Iterable[Result], path: str | Path) -> None:
    """Append results to a JSON-lines file (one record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def load(path: str | Path) -> pd.DataFrame:
    with Path(path).open() as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def reference(name: str = "vision_whitebox", protocol: str | None = "fft50") -> pd.DataFrame:
    """Published per-(dataset, setting) accuracies from the paper.

    Args:
        name: ``"vision_whitebox"`` (80 configurations, Section 4) or
            ``"vellm_transfer"`` (4 Ve-LLMs, per surrogate, Section 5).
        protocol: for vision results, ``"fft50"``, ``"fft5"``, ``"lp50"``, or None (all).

    For ``vellm_transfer``, adversarial rows are per surrogate; use
    ``worst_case_over_surrogates`` to obtain the paper's aggregation.
    """
    with resources.files("robustgenbench.reference_data").joinpath(f"{name}.csv").open() as f:
        df = pd.read_csv(f)
    if name == "vision_whitebox" and protocol is not None:
        df = df[df.protocol == protocol]
    return df


def worst_case_over_surrogates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep, for each (model, dataset, setting), the lowest accuracy over surrogates."""
    return (df.sort_values("accuracy").groupby(["model", "dataset", "setting"], as_index=False)
              .first())


def leaderboard(results: pd.DataFrame | Iterable[Result], compare_with: str | None = "fft50",
                datasets: Iterable[str] | None = None) -> pd.DataFrame:
    """Rank new results, optionally together with a published leaderboard.

    ``compare_with``: ``"fft50"``, ``"fft5"``, ``"lp50"`` (vision, white-box),
    ``"vellm"`` (Ve-LLMs, transfer), or None. Only (dataset, setting) pairs
    evaluated by the new models are kept, so partial evaluations rank fairly.
    Sum scores are pool-independent; Borda scores depend on the pool shown.
    """
    new = results if isinstance(results, pd.DataFrame) else to_frame(results)
    new = new[["model", "dataset", "setting", "accuracy"]]
    if compare_with == "vellm":
        ref = worst_case_over_surrogates(reference("vellm_transfer"))
    elif compare_with is not None:
        ref = reference("vision_whitebox", protocol=compare_with)
    else:
        ref = new.iloc[:0]
    pairs = set(map(tuple, new[["dataset", "setting"]].drop_duplicates().to_numpy()))
    if datasets is not None:
        pairs = {p for p in pairs if p[0] in set(datasets)}
    ref = ref[[(d, s) in pairs for d, s in zip(ref.dataset, ref.setting)]]
    ref = ref[["model", "dataset", "setting", "accuracy"]]
    both = pd.concat([new, ref], ignore_index=True)
    table = pd.concat([sum_score(both), borda_score(both)], axis=1)
    table["new"] = table.index.isin(set(new.model))
    table["rank_sum"] = table.sum_score.rank(ascending=False, method="min").astype(int)
    return table.sort_values("sum_score", ascending=False)
