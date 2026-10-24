"""Evaluate a PyTorch image classifier on RobustGenBench.

The model contract is the one used by AutoAttack: ``model(x) -> logits`` where ``x``
is a float tensor of shape (B, 3, 224, 224) with pixel values in [0, 1]. Put any
normalization inside the model (see ``with_normalization``) so that perturbation
budgets are measured on raw pixels.

Clean and common settings use the released images. Adversarial settings are crafted
white-box on your model with AutoAttack (standard version) from the clean test
images, at the paper's budgets. Black-box (transfer) evaluation of a torch model is
also available with ``mode="transfer"``.
"""
from __future__ import annotations

import time
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn

from .benchmark import (BLACKBOX_SETTINGS, DATASETS, THREATS, WHITEBOX_SETTINGS,
                        available_surrogates, class_names, load_split)
from .results import Result


class _Normalize(nn.Module):
    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def with_normalization(model: nn.Module, mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225)) -> nn.Module:
    """Wrap ``model`` so it accepts [0, 1] images (defaults: ImageNet statistics)."""
    return _Normalize(model, mean, std)


def _to_tensor(batch):
    x = np.stack([np.asarray(img.convert("RGB"), dtype=np.float32) for img in batch["image"]])
    return torch.from_numpy(x).permute(0, 3, 1, 2) / 255.0, torch.tensor(batch["label"])


def _batches(ds, batch_size: int, max_examples: int | None) -> Iterable:
    n = len(ds) if max_examples is None else min(max_examples, len(ds))
    for start in range(0, n, batch_size):
        yield _to_tensor(ds[start:min(start + batch_size, n)])


@torch.no_grad()
def _accuracy(model, ds, device, batch_size, max_examples) -> tuple[float, int]:
    correct = total = 0
    for x, y in _batches(ds, batch_size, max_examples):
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += len(y)
    return correct / total, total


def _robust_accuracy(model, ds, threat, device, batch_size, max_examples,
                     attack_version, seed) -> tuple[float, int]:
    try:
        from autoattack import AutoAttack
    except ImportError as e:  # pragma: no cover
        raise ImportError("White-box evaluation requires AutoAttack: "
                          "pip install git+https://github.com/fra31/auto-attack") from e
    adversary = AutoAttack(model, norm=threat.norm, eps=threat.eps,
                           version=attack_version, verbose=False, device=device, seed=seed)
    correct = total = 0
    for x, y in _batches(ds, batch_size, max_examples):
        x, y = x.to(device), y.to(device)
        x_adv = adversary.run_standard_evaluation(x, y, bs=len(x))
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += len(y)
    return correct / total, total


def evaluate_model(
    model: nn.Module,
    model_name: str,
    datasets: Iterable[str] = DATASETS,
    mode: str = "whitebox",
    settings: Iterable[str] | None = None,
    model_for_dataset: Callable[[str, nn.Module], nn.Module] | None = None,
    device: str | None = None,
    batch_size: int = 64,
    max_examples: int | None = None,
    attack_version: str = "standard",
    seed: int = 0,
    log: Callable[[str], None] = print,
) -> list[Result]:
    """Evaluate a classifier and return one ``Result`` per (dataset, setting).

    Args:
        model: classifier on [0, 1] images. RobustGenBench datasets have different
            label spaces, so you usually need one fine-tuned model per dataset:
            pass ``model_for_dataset(dataset, model) -> nn.Module`` to swap heads or
            load checkpoints per dataset.
        mode: ``"whitebox"`` (5 paper settings, AutoAttack on ``model``) or
            ``"transfer"`` (9 settings, released surrogate attacks, worst case over
            surrogates as in the paper).
        settings: subset of the protocol's settings (default: all of them).
        max_examples: evaluate on the first N test images only (quick checks; results
            are then not comparable with published numbers).
    """
    if mode not in ("whitebox", "transfer"):
        raise ValueError("mode must be 'whitebox' or 'transfer'")
    protocol = WHITEBOX_SETTINGS if mode == "whitebox" else BLACKBOX_SETTINGS
    settings = tuple(settings or protocol)
    unknown = set(settings) - set(protocol)
    if unknown:
        raise ValueError(f"Settings {unknown} are not part of the {mode} protocol {protocol}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for dataset in datasets:
        net = model_for_dataset(dataset, model) if model_for_dataset else model
        net = net.to(device).eval()
        test = load_split(dataset, "test")
        for setting in settings:
            t0 = time.time()
            if setting == "clean":
                acc, n = _accuracy(net, test, device, batch_size, max_examples)
                source = None
            elif setting == "common":
                acc, n = _accuracy(net, load_split(dataset, "common"), device, batch_size,
                                   max_examples)
                source = None
            elif mode == "whitebox":
                acc, n = _robust_accuracy(net, test, THREATS[setting], device, batch_size,
                                          max_examples, attack_version, seed)
                source = "whitebox_autoattack"
            else:
                per_surrogate = {
                    s: _accuracy(net, load_split(dataset, setting, surrogate=s), device,
                                 batch_size, max_examples)[0]
                    for s in available_surrogates(dataset, setting)}
                source, acc = min(per_surrogate.items(), key=lambda kv: kv[1])
                n = min(max_examples or len(test), len(test))
            results.append(Result(model=model_name, dataset=dataset, setting=setting,
                                  accuracy=acc, num_examples=n, mode=mode, source=source))
            log(f"{model_name:30s} {dataset:28s} {setting:8s} acc={acc:.4f} "
                f"(n={n}, {time.time() - t0:.0f}s)")
    return results


class TorchSplit(torch.utils.data.Dataset):
    """A RobustGenBench split as a torch ``Dataset`` of ([0, 1] tensor, label) pairs.

    Example (fine-tuning)::

        train = TorchSplit("flowers-102", "train", transform=my_augmentation)
        loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
    """

    def __init__(self, dataset: str, split: str = "train", transform=None, **load_kwargs):
        self.ds = load_split(dataset, split, **load_kwargs)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        row = self.ds[i]
        img = row["image"].convert("RGB")
        if self.transform is not None:
            return self.transform(img), row["label"]
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1) / 255.0
        return x, row["label"]


def num_classes(dataset: str) -> int:
    """Number of classes of ``dataset`` (to build classification heads)."""
    return len(class_names(dataset))
