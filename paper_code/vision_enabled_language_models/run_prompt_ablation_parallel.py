#!/usr/bin/env python3
"""
Run all missing prompt ablation evaluations in PARALLEL.
Each dataset-condition-prompt combo gets its own subprocess.
"""

import subprocess
import sys
import json
import concurrent.futures
from pathlib import Path

MODEL_ID = "moonshotai.kimi-k2.5"
MODEL_KEY = "kimi"
CONCURRENCY = 3  # per-process concurrency (lower to avoid throttling with many parallel)

DATASETS = ["fgvc-aircraft-2013b", "oxford-iiit-pet", "stanford_cars", "uc-merced-land-use-dataset"]
DATA_ROOTS = {
    "clean": str(Path.home() / "data_processed"),
    "adversarial": str(Path.home() / "data_processed/adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard"),
}
CLASS_NAMES_DIR = str(Path.home() / "data_processed/class_names")
OUTPUT_DIR = str(Path(__file__).parent / "llm_classification_results")
RESULTS_DIR = Path(__file__).parent / "llm_classification_results"

PROMPTS = {
    "default": "Classify the image into exactly one of these classes: {classes}. Output ONLY the exact class name. No explanation, no punctuation, no extra words.Your response must exactly match one of the class names above.",
    "minimal": "What is this? Choose from: {classes}. Reply with just the class name.",
    "detailed": "You are an expert image classifier. Carefully examine this image and determine which of the following categories it belongs to: {classes}. Respond with the exact class name only, nothing else.",
    "cot": "Look at this image and classify it into one of these categories: {classes}. Think about what you see, then on the final line output ONLY the exact class name.",
}

MAX_PARALLEL = 8  # max parallel processes


def get_dataset_name(dataset, condition):
    return dataset if condition == "clean" else f"{dataset}__zeroshot_clip_vitb16_laion2b__linf_eps30__autoattack_standard"


def is_done(dataset, condition, prompt_key):
    run_name = f"{dataset}__{MODEL_KEY}__prompt_{prompt_key}__{condition}"
    return (RESULTS_DIR / run_name / "results.json").exists()


def run_single(args):
    dataset, condition, prompt_key = args
    run_name = f"{dataset}__{MODEL_KEY}__prompt_{prompt_key}__{condition}"
    prompt_template = PROMPTS[prompt_key]

    cmd = [
        sys.executable, "llm_classify.py",
        "--provider", "bedrock",
        "--model", MODEL_ID,
        "--dataset", get_dataset_name(dataset, condition),
        "--split", "test",
        "--data_root", DATA_ROOTS[condition],
        "--class_names_dir", CLASS_NAMES_DIR,
        "--output_dir", OUTPUT_DIR,
        "--run_name", run_name,
        "--batch_concurrency", str(CONCURRENCY),
        "--prompt_template", prompt_template,
    ]

    print(f"[START] {run_name}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent),
                            capture_output=True, text=True)
    if result.returncode == 0:
        # Extract accuracy from results
        rfile = RESULTS_DIR / run_name / "results.json"
        if rfile.exists():
            acc = json.load(open(rfile))["accuracy"]
            print(f"[DONE]  {run_name}  acc={acc*100:.1f}%")
        else:
            print(f"[DONE]  {run_name}  (no results.json?)")
        return True
    else:
        print(f"[FAIL]  {run_name}")
        print(result.stderr[-500:] if result.stderr else "")
        return False


def main():
    # Collect missing runs
    tasks = []
    for dataset in DATASETS:
        for condition in DATA_ROOTS:
            for prompt_key in PROMPTS:
                if not is_done(dataset, condition, prompt_key):
                    tasks.append((dataset, condition, prompt_key))

    print(f"\nPrompt Ablation — Parallel Execution")
    print(f"  Missing runs: {len(tasks)}")
    print(f"  Max parallel: {MAX_PARALLEL}")
    print(f"  Per-process concurrency: {CONCURRENCY}")
    print()

    if not tasks:
        print("Nothing to do!")
        return

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        results = list(executor.map(run_single, tasks))

    successes = sum(results)
    failures = len(results) - successes
    print(f"\n{'='*70}")
    print(f"  COMPLETE: {successes} succeeded, {failures} failed out of {len(tasks)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
