"""Evaluate your own PyTorch classifiers on RobustGenBench (white-box protocol).

RobustGenBench datasets have different label spaces, so you provide one model per
dataset through ``model_for_dataset``. The model must take [0, 1] images of shape
(B, 3, 224, 224); wrap it with ``with_normalization`` if it expects normalized input.

    python examples/evaluate_torch_model.py --checkpoints "ckpt/{dataset}.pt" --max-examples 64
"""
import argparse

import timm
import torch

import robustgenbench as rgb
from robustgenbench.vision import evaluate_model, num_classes, with_normalization

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="convnext_base.fb_in22k")
ap.add_argument("--checkpoints", required=True, help="path template with {dataset}")
ap.add_argument("--datasets", nargs="+", default=list(rgb.DATASETS))
ap.add_argument("--mode", choices=["whitebox", "transfer"], default="whitebox")
ap.add_argument("--max-examples", type=int, default=None)
a = ap.parse_args()


def model_for_dataset(dataset, _):
    # Replace with any nn.Module: this is the only model-specific code.
    net = timm.create_model(a.arch, num_classes=num_classes(dataset))
    net.load_state_dict(torch.load(a.checkpoints.format(dataset=dataset), map_location="cpu"))
    cfg = timm.data.resolve_data_config({}, model=net)
    return with_normalization(net, cfg["mean"], cfg["std"])


results = evaluate_model(None, a.arch, datasets=a.datasets, mode=a.mode,
                         model_for_dataset=model_for_dataset, max_examples=a.max_examples)
rgb.save(results, "results.jsonl")
# Rank against the 80 FFT-50 configurations of the paper (Sum score, pool-independent).
print(rgb.leaderboard(results, compare_with="fft50").head(15))
