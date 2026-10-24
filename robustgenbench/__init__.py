"""RobustGenBench: robust generalization of image classifiers across common and
adversarial (l1, l2, linf) perturbations.

Data: https://huggingface.co/datasets/legolasflagstaff/RobustGenBench

Two entry points:

* ``evaluate_model``: PyTorch classifiers (white-box AutoAttack, or transfer attacks).
* ``evaluate_classifier``: any black-box model, e.g. a vision-enabled LLM, behind a
  ``BlackBoxClassifier.predict(image, class_names)`` method.
"""
from .benchmark import (BLACKBOX_SETTINGS, COARSE_DATASETS, DATASETS, HF_REPO, SURROGATES,
                        THREATS, WHITEBOX_SETTINGS, available_surrogates, class_names,
                        load_split)
from .metrics import borda_score, kendall_tau, mean_absolute_correlation, sum_score
from .results import (Result, leaderboard, load, reference, save, to_frame,
                      worst_case_over_surrogates)
from .vlm import (BlackBoxClassifier, OpenAICompatibleClassifier, build_prompt,
                  evaluate_classifier, parse_prediction)

__version__ = "2.0.0"


def __getattr__(name):
    # Torch-dependent API is imported lazily so that black-box evaluation does not
    # require PyTorch.
    if name in ("evaluate_model", "with_normalization", "TorchSplit", "num_classes"):
        from . import vision
        return getattr(vision, name)
    raise AttributeError(name)


__all__ = [
    "BLACKBOX_SETTINGS", "COARSE_DATASETS", "DATASETS", "HF_REPO", "SURROGATES", "THREATS",
    "WHITEBOX_SETTINGS", "available_surrogates", "class_names", "load_split",
    "borda_score", "kendall_tau", "mean_absolute_correlation", "sum_score",
    "Result", "leaderboard", "load", "reference", "save", "to_frame",
    "worst_case_over_surrogates", "BlackBoxClassifier", "OpenAICompatibleClassifier",
    "build_prompt", "evaluate_classifier", "parse_prediction",
    "evaluate_model", "with_normalization", "TorchSplit", "num_classes",
]
