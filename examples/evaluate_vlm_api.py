"""Evaluate a vision-enabled LLM served behind an OpenAI-compatible API.

Works with OpenAI and with any OpenAI-compatible server. Examples:

    # OpenAI (reads OPENAI_API_KEY)
    python examples/evaluate_vlm_api.py --model gpt-4o-mini

    # Open-weight model served locally with vLLM
    vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 127.0.0.1 --port 8000
    python examples/evaluate_vlm_api.py --model Qwen/Qwen2.5-VL-7B-Instruct \
        --base-url http://127.0.0.1:8000/v1

The full protocol (6 datasets x 9 settings x all surrogates) sends about 150k
requests. Start with --max-examples and a few datasets to estimate the cost; raw
answers are cached in rgb_cache/, so interrupted runs resume where they stopped.
"""
import argparse

import robustgenbench as rgb

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--base-url", default=None)
ap.add_argument("--datasets", nargs="+", default=list(rgb.DATASETS))
ap.add_argument("--max-examples", type=int, default=None)
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

clf = rgb.OpenAICompatibleClassifier(a.model, base_url=a.base_url, max_workers=a.workers)
results = rgb.evaluate_classifier(clf, a.model, datasets=a.datasets, max_examples=a.max_examples)
rgb.save(results, "results.jsonl")

# Rank against GPT-4o mini, Gemini 3 Flash, Kimi K2.5, and Qwen3 VL from the paper.
print(rgb.leaderboard(results, compare_with="vellm"))
