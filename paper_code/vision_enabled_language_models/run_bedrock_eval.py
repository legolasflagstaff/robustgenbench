#!/usr/bin/env python3
"""
Run RobustGenBench evaluation for Bedrock models (Kimi K2.5 and Qwen3 VL).

Produces results for:
  - Table 15: Adversarial perturbations (Linf eps=30, CLIP ViT-B/16 surrogate)
  - Table 16: Common corruptions (severity 3)
  - Clean baseline

Usage:
    python run_bedrock_eval.py --model moonshotai.kimi-k2.5 --condition all
    python run_bedrock_eval.py --model qwen.qwen3-vl-235b-a22b --condition adversarial
    python run_bedrock_eval.py --model all --condition all
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

MODELS = {
    "kimi": "moonshotai.kimi-k2.5",
    "qwen": "qwen.qwen3-vl-235b-a22b",
}

# Data roots for each condition
DATA_ROOTS = {
    "clean": os.path.expanduser("~/data_processed"),
    "adversarial": os.path.expanduser("~/data_processed/adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard"),
    "common": os.path.expanduser("~/data_processed/adversarial/common/common_severity3"),
}

CLASS_NAMES_DIR = os.path.expanduser("~/data_processed/class_names")
OUTPUT_DIR = str(Path(__file__).parent / "llm_classification_results")


def get_dataset_name(base_dataset: str, condition: str) -> str:
    """Get the archive name for a dataset under a given condition."""
    if condition == "clean":
        return base_dataset
    elif condition == "adversarial":
        return f"{base_dataset}__zeroshot_clip_vitb16_laion2b__linf_eps30__autoattack_standard"
    elif condition == "common":
        return f"{base_dataset}__common_severity3"
    else:
        raise ValueError(f"Unknown condition: {condition}")


def get_run_name(base_dataset: str, model_key: str, condition: str) -> str:
    """Generate a run name."""
    return f"{base_dataset}__{model_key}__{condition}"


def run_single(model_id: str, model_key: str, dataset: str, condition: str, concurrency: int):
    """Run evaluation for a single dataset/condition/model."""
    data_root = DATA_ROOTS[condition]
    dataset_name = get_dataset_name(dataset, condition)
    run_name = get_run_name(dataset, model_key, condition)

    # Use test split for clean, test for adversarial (they are already perturbed test sets)
    split = "test"

    cmd = [
        sys.executable, "llm_classify.py",
        "--provider", "bedrock",
        "--model", model_id,
        "--dataset", dataset_name,
        "--split", split,
        "--data_root", data_root,
        "--class_names_dir", CLASS_NAMES_DIR,
        "--output_dir", OUTPUT_DIR,
        "--run_name", run_name,
        "--batch_concurrency", str(concurrency),
    ]

    print(f"\n{'='*70}")
    print(f"  Model:     {model_id} ({model_key})")
    print(f"  Dataset:   {dataset}")
    print(f"  Condition: {condition}")
    print(f"  Run name:  {run_name}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="RobustGenBench Bedrock Evaluation")
    parser.add_argument("--model", default="all", choices=["all", "kimi", "qwen",
                        "moonshotai.kimi-k2.5", "qwen.qwen3-vl-235b-a22b"])
    parser.add_argument("--condition", default="all", choices=["all", "clean", "adversarial", "common"])
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to run (default: all 6)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Parallel requests per dataset evaluation")
    args = parser.parse_args()

    # Resolve models
    if args.model == "all":
        models = list(MODELS.items())
    elif args.model in MODELS:
        models = [(args.model, MODELS[args.model])]
    else:
        # Direct model ID
        key = "kimi" if "kimi" in args.model else "qwen"
        models = [(key, args.model)]

    # Resolve conditions
    if args.condition == "all":
        conditions = ["clean", "adversarial", "common"]
    else:
        conditions = [args.condition]

    # Resolve datasets
    datasets = args.datasets if args.datasets else DATASETS

    total = len(models) * len(conditions) * len(datasets)
    print(f"\nRobustGenBench Bedrock Evaluation")
    print(f"  Models:     {[m[0] for m in models]}")
    print(f"  Conditions: {conditions}")
    print(f"  Datasets:   {datasets}")
    print(f"  Total runs: {total}")
    print(f"  Concurrency: {args.concurrency}")

    successes = 0
    failures = 0

    for model_key, model_id in models:
        for condition in conditions:
            for dataset in datasets:
                ok = run_single(model_id, model_key, dataset, condition, args.concurrency)
                if ok:
                    successes += 1
                else:
                    failures += 1

    print(f"\n{'='*70}")
    print(f"  COMPLETE: {successes} succeeded, {failures} failed out of {total}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
