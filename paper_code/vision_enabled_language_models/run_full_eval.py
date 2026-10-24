#!/usr/bin/env python3
"""
Run FULL RobustGenBench evaluation for Bedrock models (Kimi K2.5 and Qwen3 VL).

Replicates the exact structure of Tables 15 & 16 from the paper:
  - Table 15: Adversarial transfer across 4 surrogates x 7 threat models + common + clean
  - Table 16: Common corruptions (severity 3) + clean

Surrogates:
  1. CLIP ViT-B/16 (all 6 datasets)
  2. CLIP ViT-H/14 (all 6 datasets)
  3. MetaCLIP ViT-H/14 (all 6 datasets)
  4. SigLIP2 SO400M (3 datasets: caltech101, flowers-102, uc-merced)

Threat models per surrogate:
  l1_eps75, l1_eps300, l2_eps2, l2_eps8, linf_eps4, linf_eps8, linf_eps30

Usage:
    python run_full_eval.py --model kimi --surrogate clip_vitb16
    python run_full_eval.py --model qwen --surrogate all
    python run_full_eval.py --model kimi --condition clean
    python run_full_eval.py --model kimi --condition common
    python run_full_eval.py --model all --surrogate all --concurrency 5
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# All 6 datasets
ALL_DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

# SigLIP2 SO400M only has these 3 datasets
SIGLIP2_DATASETS = [
    "caltech101",
    "flowers-102",
    "uc-merced-land-use-dataset",
]

MODELS = {
    "kimi": "moonshotai.kimi-k2.5",
    "qwen": "qwen.qwen3-vl-235b-a22b",
}

# Surrogate model mapping: short key -> directory name on disk
SURROGATES = {
    "clip_vitb16": "zeroshot_clip_vitb16_laion2b",
    "clip_vith14": "zeroshot_clip_vith14_laion2b",
    "metaclip_vith14": "zeroshot_metaclip_vith14_fullcc2_5b",
    "siglip2_so400m": "zeroshot_siglip2_so400m_patch14_384",
}

# All threat models (norm/epsilon combinations)
THREAT_MODELS = [
    "l1_eps75_autoattack_standard",
    "l1_eps300_autoattack_standard",
    "l2_eps2_autoattack_standard",
    "l2_eps8_autoattack_standard",
    "linf_eps4_autoattack_standard",
    "linf_eps8_autoattack_standard",
    "linf_eps30_autoattack_standard",
]

DATA_BASE = os.path.expanduser("~/data_processed")
CLASS_NAMES_DIR = os.path.expanduser("~/data_processed/class_names")
OUTPUT_DIR = str(Path(__file__).parent / "llm_classification_results")


def get_datasets_for_surrogate(surrogate_key: str) -> list:
    """Return the dataset list for a given surrogate."""
    if surrogate_key == "siglip2_so400m":
        return SIGLIP2_DATASETS
    return ALL_DATASETS


def get_archive_name(dataset: str, surrogate_dir: str, threat_model: str) -> str:
    """Get the archive/dataset name for adversarial data."""
    # Extract norm info for the dataset name
    # e.g., linf_eps30_autoattack_standard -> linf_eps30__autoattack_standard
    parts = threat_model.split("_autoattack_")
    norm_eps = parts[0]  # e.g., linf_eps30
    return f"{dataset}__{surrogate_dir}__{norm_eps}_autoattack_standard"


def run_single(model_id: str, model_key: str, dataset: str,
               surrogate_key: str, surrogate_dir: str, threat_model: str,
               concurrency: int) -> bool:
    """Run evaluation for a single adversarial condition."""
    data_root = os.path.join(DATA_BASE, "adversarial", surrogate_dir, threat_model)
    archive_name = get_archive_name(dataset, surrogate_dir, threat_model)

    # Short threat model name for run_name (e.g., linf_eps30)
    threat_short = threat_model.replace("_autoattack_standard", "")
    run_name = f"{dataset}__{model_key}__{surrogate_key}__{threat_short}"

    cmd = [
        sys.executable, "llm_classify.py",
        "--provider", "bedrock",
        "--model", model_id,
        "--dataset", archive_name,
        "--split", "test",
        "--data_root", data_root,
        "--class_names_dir", CLASS_NAMES_DIR,
        "--output_dir", OUTPUT_DIR,
        "--run_name", run_name,
        "--batch_concurrency", str(concurrency),
    ]

    print(f"\n{'='*70}")
    print(f"  Model:     {model_id} ({model_key})")
    print(f"  Dataset:   {dataset}")
    print(f"  Surrogate: {surrogate_key} ({surrogate_dir})")
    print(f"  Threat:    {threat_short}")
    print(f"  Run name:  {run_name}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    return result.returncode == 0


def run_clean(model_id: str, model_key: str, dataset: str, concurrency: int) -> bool:
    """Run clean baseline evaluation."""
    data_root = os.path.join(DATA_BASE)
    run_name = f"{dataset}__{model_key}__clean"

    cmd = [
        sys.executable, "llm_classify.py",
        "--provider", "bedrock",
        "--model", model_id,
        "--dataset", dataset,
        "--split", "test",
        "--data_root", data_root,
        "--class_names_dir", CLASS_NAMES_DIR,
        "--output_dir", OUTPUT_DIR,
        "--run_name", run_name,
        "--batch_concurrency", str(concurrency),
    ]

    print(f"\n{'='*70}")
    print(f"  Model:     {model_id} ({model_key})")
    print(f"  Dataset:   {dataset}")
    print(f"  Condition: clean")
    print(f"  Run name:  {run_name}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    return result.returncode == 0


def run_common(model_id: str, model_key: str, dataset: str, concurrency: int) -> bool:
    """Run common corruption evaluation."""
    data_root = os.path.join(DATA_BASE, "adversarial", "common", "common_severity3")
    archive_name = f"{dataset}__common_severity3"
    run_name = f"{dataset}__{model_key}__common"

    cmd = [
        sys.executable, "llm_classify.py",
        "--provider", "bedrock",
        "--model", model_id,
        "--dataset", archive_name,
        "--split", "test",
        "--data_root", data_root,
        "--class_names_dir", CLASS_NAMES_DIR,
        "--output_dir", OUTPUT_DIR,
        "--run_name", run_name,
        "--batch_concurrency", str(concurrency),
    ]

    print(f"\n{'='*70}")
    print(f"  Model:     {model_id} ({model_key})")
    print(f"  Dataset:   {dataset}")
    print(f"  Condition: common (severity 3)")
    print(f"  Run name:  {run_name}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Full RobustGenBench Bedrock Evaluation")
    parser.add_argument("--model", default="all",
                        choices=["all", "kimi", "qwen"])
    parser.add_argument("--surrogate", default=None,
                        choices=["all", "clip_vitb16", "clip_vith14",
                                 "metaclip_vith14", "siglip2_so400m"],
                        help="Run adversarial for specified surrogate(s)")
    parser.add_argument("--threat", default="all",
                        choices=["all"] + [t.replace("_autoattack_standard", "") for t in THREAT_MODELS],
                        help="Specific threat model (default: all)")
    parser.add_argument("--condition", default=None,
                        choices=["clean", "common"],
                        help="Run clean or common condition instead of adversarial")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets to run (default: all available)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Parallel requests per dataset evaluation")
    args = parser.parse_args()

    # Resolve models
    if args.model == "all":
        models = list(MODELS.items())
    else:
        models = [(args.model, MODELS[args.model])]

    successes = 0
    failures = 0
    total = 0

    for model_key, model_id in models:
        # If condition is specified, run that (clean or common)
        if args.condition == "clean":
            datasets = args.datasets or ALL_DATASETS
            total += len(datasets)
            for dataset in datasets:
                ok = run_clean(model_id, model_key, dataset, args.concurrency)
                if ok:
                    successes += 1
                else:
                    failures += 1

        elif args.condition == "common":
            datasets = args.datasets or ALL_DATASETS
            total += len(datasets)
            for dataset in datasets:
                ok = run_common(model_id, model_key, dataset, args.concurrency)
                if ok:
                    successes += 1
                else:
                    failures += 1

        else:
            # Adversarial: run for specified surrogate(s) and threat model(s)
            if args.surrogate is None:
                parser.error("Must specify --surrogate or --condition")

            if args.surrogate == "all":
                surrogates = list(SURROGATES.items())
            else:
                surrogates = [(args.surrogate, SURROGATES[args.surrogate])]

            # Resolve threat models
            if args.threat == "all":
                threats = THREAT_MODELS
            else:
                threats = [f"{args.threat}_autoattack_standard"]

            for surr_key, surr_dir in surrogates:
                datasets = args.datasets or get_datasets_for_surrogate(surr_key)
                total += len(datasets) * len(threats)
                for threat in threats:
                    for dataset in datasets:
                        ok = run_single(model_id, model_key, dataset,
                                       surr_key, surr_dir, threat,
                                       args.concurrency)
                        if ok:
                            successes += 1
                        else:
                            failures += 1

    print(f"\n{'='*70}")
    print(f"  COMPLETE: {successes} succeeded, {failures} failed out of {total}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
