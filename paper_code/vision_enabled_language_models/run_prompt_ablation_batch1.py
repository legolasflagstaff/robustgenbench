#!/usr/bin/env python3
"""Prompt ablation batch 1: fgvc-aircraft-2013b and oxford-iiit-pet"""
import subprocess, sys
from pathlib import Path

MODEL_ID = "moonshotai.kimi-k2.5"
MODEL_KEY = "kimi"
CONCURRENCY = 5
DATASETS = ["fgvc-aircraft-2013b", "oxford-iiit-pet"]
DATA_ROOTS = {
    "clean": str(Path.home() / "data_processed"),
    "adversarial": str(Path.home() / "data_processed/adversarial/zeroshot_clip_vitb16_laion2b/linf_eps30_autoattack_standard"),
}
CLASS_NAMES_DIR = str(Path.home() / "data_processed/class_names")
OUTPUT_DIR = str(Path(__file__).parent / "llm_classification_results")
PROMPTS = {
    "default": "Classify the image into exactly one of these classes: {classes}. Output ONLY the exact class name. No explanation, no punctuation, no extra words.Your response must exactly match one of the class names above.",
    "minimal": "What is this? Choose from: {classes}. Reply with just the class name.",
    "detailed": "You are an expert image classifier. Carefully examine this image and determine which of the following categories it belongs to: {classes}. Respond with the exact class name only, nothing else.",
    "cot": "Look at this image and classify it into one of these categories: {classes}. Think about what you see, then on the final line output ONLY the exact class name.",
}

def get_dataset_name(dataset, condition):
    return dataset if condition == "clean" else f"{dataset}__zeroshot_clip_vitb16_laion2b__linf_eps30__autoattack_standard"

successes, failures, total = 0, 0, len(DATASETS)*len(PROMPTS)*len(DATA_ROOTS)
for dataset in DATASETS:
    for condition in DATA_ROOTS:
        for prompt_key, prompt_template in PROMPTS.items():
            run_name = f"{dataset}__{MODEL_KEY}__prompt_{prompt_key}__{condition}"
            cmd = [sys.executable, "llm_classify.py", "--provider", "bedrock", "--model", MODEL_ID,
                   "--dataset", get_dataset_name(dataset, condition), "--split", "test",
                   "--data_root", DATA_ROOTS[condition], "--class_names_dir", CLASS_NAMES_DIR,
                   "--output_dir", OUTPUT_DIR, "--run_name", run_name,
                   "--batch_concurrency", str(CONCURRENCY), "--prompt_template", prompt_template]
            print(f"\n{'='*70}\n  {run_name}\n{'='*70}")
            r = subprocess.run(cmd, cwd=str(Path(__file__).parent))
            if r.returncode == 0: successes += 1
            else: failures += 1
print(f"\n  COMPLETE: {successes} succeeded, {failures} failed out of {total}")
