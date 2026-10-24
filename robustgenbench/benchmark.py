"""Benchmark definition: datasets, perturbation settings, and data access.

Data lives on the Hugging Face Hub (``legolasflagstaff/RobustGenBench``):

* config ``<dataset>``: splits ``train``, ``validation``, ``test``, ``common``,
  ``random_linf_30``
* config ``<dataset>-<surrogate>``: one split per transfer budget, e.g.
  ``linf_30``

All perturbed splits are aligned with ``test`` by ``image_id`` and share its labels.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

HF_REPO = "legolasflagstaff/RobustGenBench"

DATASETS = (
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
)
COARSE_DATASETS = ("caltech101", "flowers-102", "uc-merced-land-use-dataset")

SURROGATES = ("clip_vitb16", "clip_vith14", "metaclip_vith14", "siglip2_so400m", "siglip2_base")


@dataclass(frozen=True)
class Threat:
    """An adversarial threat model: an l_p ball of radius ``eps`` on [0, 1] pixels."""

    norm: str  # "L1", "L2", or "Linf" (AutoAttack naming)
    eps: float
    name: str  # split suffix on the Hub, e.g. "linf_30"


THREATS = {
    "l1_75": Threat("L1", 75.0, "l1_75"),
    "l1_300": Threat("L1", 300.0, "l1_300"),
    "l2_2": Threat("L2", 2.0, "l2_2"),
    "l2_8": Threat("L2", 8.0, "l2_8"),
    "linf_4": Threat("Linf", 4 / 255, "linf_4"),
    "linf_8": Threat("Linf", 8 / 255, "linf_8"),
    "linf_30": Threat("Linf", 30 / 255, "linf_30"),
}

# Protocol for models with gradient access (Section 4 of the paper): 5 settings,
# adversarial perturbations crafted white-box on the evaluated model.
WHITEBOX_SETTINGS = ("clean", "common", "l1_75", "l2_2", "linf_4")

# Protocol for black-box models, e.g. API-only Ve-LLMs (Section 5): 9 settings,
# adversarial perturbations transferred from the released surrogate attacks.
BLACKBOX_SETTINGS = ("clean", "common", "l1_75", "l1_300", "l2_2", "l2_8",
                     "linf_4", "linf_8", "linf_30")

# Perturbations not provided on the Hub (attack signal too weak; see dataset card).
_MISSING = {("siglip2_so400m", d) for d in DATASETS if d not in COARSE_DATASETS}
_SIGLIP2_BASE_ONLY = {"linf_30"}


def available_surrogates(dataset: str, threat: str) -> list[str]:
    """Surrogates for which the transfer split ``threat`` exists on ``dataset``."""
    out = []
    for s in SURROGATES:
        if (s, dataset) in _MISSING:
            continue
        if s == "siglip2_base" and threat not in _SIGLIP2_BASE_ONLY:
            continue
        out.append(s)
    return out


def _check_dataset(dataset: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; choose from {DATASETS}")


def load_split(dataset: str, setting: str, surrogate: str | None = None,
               revision: str | None = None, **kwargs):
    """Load one RobustGenBench split as a ``datasets.Dataset``.

    Args:
        dataset: one of ``DATASETS``.
        setting: ``train``, ``validation``, ``test`` (alias ``clean``), ``common``,
            ``random_linf_30``, or a threat name in ``THREATS`` (requires ``surrogate``).
        surrogate: surrogate used to craft transfer attacks (threat settings only).
        revision: Hub revision to pin, e.g. a commit hash for exact reproducibility.
        **kwargs: forwarded to ``datasets.load_dataset`` (e.g. ``streaming=True``).
    """
    from datasets import load_dataset

    _check_dataset(dataset)
    if setting == "clean":
        setting = "test"
    if setting in THREATS:
        if surrogate is None:
            raise ValueError(f"Setting {setting!r} is a transfer attack: pass surrogate=")
        if surrogate not in available_surrogates(dataset, setting):
            raise ValueError(f"No {surrogate} attack for {dataset}/{setting}; "
                             f"available: {available_surrogates(dataset, setting)}")
        config, split = f"{dataset}-{surrogate}", setting
    else:
        config, split = dataset, setting
    return load_dataset(HF_REPO, config, split=split, revision=revision, **kwargs)


@lru_cache(maxsize=None)
def class_names(dataset: str) -> tuple[str, ...]:
    """Class names of ``dataset``, indexed by integer label."""
    _check_dataset(dataset)
    ds = load_split(dataset, "test", streaming=True)
    return tuple(ds.features["label"].names)
