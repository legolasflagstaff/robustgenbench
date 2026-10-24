#!/usr/bin/env python3
"""
Prompt sensitivity ablation — remaining 4 datasets.
(caltech101 and flowers-102 already done)
"""

import subprocess
import sys
from pathlib import Path

MODEL_ID = "moonshotai.kimi-k2.5"
MODEL_KEY = "kimi"
CONCURRENCY = 5

DATASETS = ["fgvc-aircraft-2013b", "oxford-iiit-pet", "stanford_cars", "uc-merced-land-use-dataset"]

DATA_ROOTS = {
    "clean": str(Path.home() / "data_processed"),
    "adversarial": str(Path.home() / "data_processed/adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard"),
}

CLASS_NAMES_DIR = str(Path.home() / "data_processed/class_names")
OUTPUT_DIR = str(Path(__file__).parent / "llm_classification_results")

PROMPTS = {
    "default": (
        "Classify the image into exactly one of these classes: {classes}. "
        "Output ONLY the exact class name. No explanation, no punctuation, no extra words."
        "Your response must exactly match one of the class names above."
    ),
    "minimal": (
        "What is this? Choose from: {classes}. Reply with just the class name."
    ),
    "detailed": (
        "You are an expert image classifier. Carefully examine this image and determine "
        "which of the following categories it belongs to: {classes}. "
        "Respond with the exact class name only, nothing else."
    ),
    "cot": (
        "Look at this image and classify it into one of these categories: {classes}. "
        "Think about what you see, then on the final line output ONLY the exact class name."
    ),
}


def get_dataset_name(dataset, condition):
    if condition == "clean":
        return dataset
    else:
        return f"{dataset}__zeroshot_clip_vitb16_laion2b__linf_eps30__autoattack_standard"


def run_single(dataset, condition, prompt_key, prompt_template):
    data_root = DATA_ROOTS[condition]
    dataset_name = get_dataset_name(dataset, condition)
    run_name = f"{dataset}__{MODEL_KEY}__prompt_{prompt_key}__{condition}"

    cmd = [
        sys.executable, "llm_classify.py",
        "--provider", "bedrock",
        "--model", MODEL_ID,
        "--dataset", dataset_name,
        "--split", "test",
        "--data_root", data_root,
        "--class_names_dir", CLASS_NAMES_DIR,
        "--output_dir", OUTPUT_DIR,
        "--run_name", run_name,
        "--batch_concurrency", str(CONCURRENCY),
        "--prompt_template", prompt_template,
    ]

    print(f"\n{'='*70}")
    print(f"  Dataset:   {dataset}")
    print(f"  Condition: {condition}")
    print(f"  Prompt:    {prompt_key}")
    print(f"  Run name:  {run_name}")
    print(f"{'='*70}")

    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    return result.returncode == 0


def main():
    total = len(DATASETS) * len(PROMPTS) * len(DATA_ROOTS)
    print(f"\nPrompt Sensitivity Ablation (remaining datasets)")
    print(f"  Model:      {MODEL_ID}")
    print(f"  Datasets:   {DATASETS}")
    print(f"  Prompts:    {list(PROMPTS.keys())}")
    print(f"  Conditions: {list(DATA_ROOTS.keys())}")
    print(f"  Total runs: {total}")

    successes = 0
    failures = 0

    for dataset in DATASETS:
        for condition in DATA_ROOTS:
            for prompt_key, prompt_template in PROMPTS.items():
                ok = run_single(dataset, condition, prompt_key, prompt_template)
                if ok:
                    successes += 1
                else:
                    failures += 1

    print(f"\n{'='*70}")
    print(f"  COMPLETE: {successes} succeeded, {failures} failed out of {total}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
