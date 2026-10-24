"""Evaluate a black-box classifier (e.g. a vision-enabled LLM) on RobustGenBench.

Any model can be plugged in by implementing one method::

    class MyModel(robustgenbench.BlackBoxClassifier):
        def predict(self, image: PIL.Image.Image, class_names: list[str]) -> str:
            ...  # return the predicted class name (free text is parsed, see below)

Adversarial settings use the transfer attacks released on the Hub: the paper reports,
for each (dataset, budget), the accuracy under the strongest surrogate.

``OpenAICompatibleClassifier`` covers OpenAI and any OpenAI-compatible endpoint
(vLLM, SGLang, Ollama, LiteLLM, Together, OpenRouter, ...).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable

from .benchmark import (BLACKBOX_SETTINGS, DATASETS, THREATS, available_surrogates,
                        load_split)
from .results import Result

# Prompt used for all Ve-LLM experiments in the paper (Appendix, Figure "prompt").
SYSTEM_PROMPT = (
    "Classify the image into exactly one of these classes: {classes}. "
    "Output ONLY the exact class name. No explanation, no punctuation, no extra words. "
    "Your response must exactly match one of the class names above."
)
USER_PROMPT = "Classify this image."


def build_prompt(class_names: list[str]) -> tuple[str, str]:
    """(system, user) prompts of the paper for a given class list."""
    return SYSTEM_PROMPT.format(classes=", ".join(class_names)), USER_PROMPT


def encode_image(image, quality: int = 95) -> str:
    """Base64 JPEG (quality 95), as sent to the APIs in the paper."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _normalize(s: str) -> str:
    s = re.sub(r"[^a-z0-9\s]", " ", s.lower().strip())
    return " ".join(s.split())


def parse_prediction(raw: str, class_names: list[str]) -> int | None:
    """Map a free-text answer to a class index (None if no class matches).

    Same rules as in the paper: exact match after normalization, then longest class
    name contained in the answer, then answer contained in a class name, then
    closest string (similarity >= 0.6).
    """
    if raw is None:
        return None
    pred = _normalize(raw)
    norm = [_normalize(c) for c in class_names]
    if pred in norm:
        return norm.index(pred)
    contained = [(len(c), i) for i, c in enumerate(norm) if c and c in pred]
    if contained and max(contained)[0] > 3:
        return max(contained)[1]
    for i, c in enumerate(norm):
        if pred and pred in c and len(pred) > 3:
            return i
    score, best = max((SequenceMatcher(None, pred, c).ratio(), i) for i, c in enumerate(norm))
    return best if score >= 0.6 else None


class BlackBoxClassifier:
    """Interface for black-box models. Implement ``predict``."""

    #: number of concurrent requests during evaluation (raise for API models)
    max_workers: int = 1

    def predict(self, image, class_names: list[str]) -> str | int:
        """Return the predicted class name (str, parsed) or class index (int)."""
        raise NotImplementedError


class OpenAICompatibleClassifier(BlackBoxClassifier):
    """Prompt-based zero-shot classifier through an OpenAI-compatible chat API.

    Args:
        model: model name on the endpoint, e.g. ``"gpt-4o-mini"``.
        base_url: endpoint URL (default: OpenAI). For a local vLLM server:
            ``"http://127.0.0.1:8000/v1"``.
        api_key: defaults to the ``OPENAI_API_KEY`` environment variable.
        system_role: send the instruction as a system message (False: prepend it to
            the user message, as done for Gemini in the paper).
    """

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 system_role: bool = True, max_workers: int = 8, max_retries: int = 6,
                 temperature: float = 0.0, **request_kwargs):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url,
                             api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.system_role = system_role
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.request_kwargs = {"temperature": temperature, **request_kwargs}

    def predict(self, image, class_names):
        system, user = build_prompt(class_names)
        image_part = {"type": "image_url",
                      "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image)}"}}
        if self.system_role:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": [image_part, {"type": "text", "text": user}]}]
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": system}, image_part, {"type": "text", "text": user}]}]
        for attempt in range(self.max_retries):
            try:
                out = self.client.chat.completions.create(
                    model=self.model, messages=messages, **self.request_kwargs)
                return out.choices[0].message.content or ""
            except Exception:  # rate limits, transient errors
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)


class _Cache:
    """Append-only JSONL cache of raw predictions, so interrupted runs resume for free."""

    def __init__(self, path: Path | None):
        self.path, self.lock, self.data = path, threading.Lock(), {}
        if path and path.exists():
            for line in path.read_text().splitlines():
                rec = json.loads(line)
                self.data[rec["key"]] = rec["raw"]

    def get(self, key):
        return self.data.get(key)

    def put(self, key, raw, **meta):
        self.data[key] = raw
        if self.path:
            with self.lock, self.path.open("a") as f:
                f.write(json.dumps({"key": key, "raw": raw, **meta}) + "\n")


def _accuracy(clf, ds, class_names, cache, tag, max_examples, log) -> tuple[float, int]:
    n = len(ds) if max_examples is None else min(max_examples, len(ds))
    rows = ds.select(range(n))

    def one(i):
        row = rows[i]
        key = f"{tag}/{row['image_id']}"
        raw = cache.get(key)
        if raw is None:
            raw = clf.predict(row["image"], list(class_names))
            cache.put(key, raw, label=row["label"])
        pred = raw if isinstance(raw, int) else parse_prediction(str(raw), list(class_names))
        return pred == row["label"]

    with ThreadPoolExecutor(max_workers=max(1, clf.max_workers)) as pool:
        correct = sum(pool.map(one, range(n)))
    return correct / n, n


def evaluate_classifier(
    classifier: BlackBoxClassifier,
    model_name: str,
    datasets: Iterable[str] = DATASETS,
    settings: Iterable[str] = BLACKBOX_SETTINGS,
    surrogates: Iterable[str] | None = None,
    cache_dir: str | Path | None = "rgb_cache",
    max_examples: int | None = None,
    log: Callable[[str], None] = print,
) -> list[Result]:
    """Evaluate a black-box classifier; returns one ``Result`` per (dataset, setting).

    Adversarial settings are evaluated on every available surrogate and the lowest
    accuracy is reported (``source`` = that surrogate). Restrict ``surrogates`` to
    reduce cost (results then are not comparable with the paper's aggregation).

    Raw predictions are cached in ``cache_dir/<model_name>.jsonl``: re-running the
    same call resumes without re-querying the model. Set ``cache_dir=None`` to disable.
    """
    settings = tuple(settings)
    unknown = set(settings) - set(BLACKBOX_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown settings {unknown}; choose from {BLACKBOX_SETTINGS}")
    cache = None
    if cache_dir is not None:
        safe = hashlib.sha1(model_name.encode()).hexdigest()[:8]
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name) + f"-{safe}.jsonl"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache = _Cache(Path(cache_dir) / name)
    else:
        cache = _Cache(None)

    results = []
    for dataset in datasets:
        test = load_split(dataset, "test")
        names = test.features["label"].names
        for setting in settings:
            t0 = time.time()
            if setting in ("clean", "common"):
                ds = test if setting == "clean" else load_split(dataset, "common")
                acc, n = _accuracy(classifier, ds, names, cache, f"{dataset}/{setting}",
                                   max_examples, log)
                source = None
            else:
                pool = [s for s in available_surrogates(dataset, setting)
                        if surrogates is None or s in set(surrogates)]
                if not pool:
                    log(f"skip {dataset}/{setting}: no requested surrogate available")
                    continue
                per = {}
                for s in pool:
                    per[s], n = _accuracy(classifier, load_split(dataset, setting, surrogate=s),
                                          names, cache, f"{dataset}/{setting}/{s}",
                                          max_examples, log)
                source, acc = min(per.items(), key=lambda kv: kv[1])
            results.append(Result(model=model_name, dataset=dataset, setting=setting,
                                  accuracy=acc, num_examples=n, mode="transfer", source=source))
            log(f"{model_name:30s} {dataset:28s} {setting:8s} acc={acc:.4f} "
                f"(n={n}, {source or '-'}, {time.time() - t0:.0f}s)")
    return results
