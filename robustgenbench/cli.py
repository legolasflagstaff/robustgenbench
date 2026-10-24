"""Command-line interface: ``robustgenbench {eval-vlm, eval-timm, leaderboard, info}``."""
from __future__ import annotations

import argparse
import sys

from .benchmark import BLACKBOX_SETTINGS, DATASETS, SURROGATES, WHITEBOX_SETTINGS


def _common(p):
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--max-examples", type=int, default=None,
                   help="first N test images per split (quick check; not comparable)")
    p.add_argument("--out", default="results.jsonl", help="results file (JSON lines, appended)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="robustgenbench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("eval-vlm", help="evaluate a model behind an OpenAI-compatible API")
    v.add_argument("--model", required=True, help="model name on the endpoint")
    v.add_argument("--name", help="name in the results (default: --model)")
    v.add_argument("--base-url", default=None, help="endpoint, e.g. http://127.0.0.1:8000/v1")
    v.add_argument("--no-system-role", action="store_true",
                   help="put the instruction in the user turn (models without system role)")
    v.add_argument("--workers", type=int, default=8, help="concurrent requests")
    v.add_argument("--settings", nargs="+", default=list(BLACKBOX_SETTINGS),
                   choices=BLACKBOX_SETTINGS)
    v.add_argument("--surrogates", nargs="+", default=None, choices=SURROGATES)
    v.add_argument("--cache-dir", default="rgb_cache")
    _common(v)

    t = sub.add_parser("eval-timm", help="evaluate fine-tuned timm models (one per dataset)")
    t.add_argument("--arch", required=True, help="timm architecture, e.g. convnext_base")
    t.add_argument("--checkpoint", required=True,
                   help="state-dict path template with {dataset}, e.g. ckpt/{dataset}.pt")
    t.add_argument("--name", help="name in the results (default: --arch)")
    t.add_argument("--mode", choices=["whitebox", "transfer"], default="whitebox")
    t.add_argument("--settings", nargs="+", default=None,
                   help=f"whitebox: {WHITEBOX_SETTINGS}; transfer: {BLACKBOX_SETTINGS}")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--device", default=None)
    _common(t)

    lb = sub.add_parser("leaderboard", help="rank results against the paper's leaderboard")
    lb.add_argument("results", help="results file written by eval-*")
    lb.add_argument("--compare", default="fft50", choices=["fft50", "fft5", "lp50", "vellm", "none"])
    lb.add_argument("--top", type=int, default=15)

    sub.add_parser("info", help="list datasets, settings, and surrogates")

    a = ap.parse_args(argv)
    from . import results as R

    if a.cmd == "info":
        from .benchmark import available_surrogates
        print("datasets:          ", ", ".join(DATASETS))
        print("white-box settings:", ", ".join(WHITEBOX_SETTINGS))
        print("black-box settings:", ", ".join(BLACKBOX_SETTINGS))
        for d in DATASETS:
            print(f"  {d:28s} linf_30 surrogates: {', '.join(available_surrogates(d, 'linf_30'))}")
        return 0

    if a.cmd == "eval-vlm":
        from .vlm import OpenAICompatibleClassifier, evaluate_classifier
        clf = OpenAICompatibleClassifier(a.model, base_url=a.base_url,
                                         system_role=not a.no_system_role, max_workers=a.workers)
        res = evaluate_classifier(clf, a.name or a.model, datasets=a.datasets,
                                  settings=a.settings, surrogates=a.surrogates,
                                  cache_dir=a.cache_dir, max_examples=a.max_examples)
        R.save(res, a.out)
        print(f"\nwrote {len(res)} results to {a.out}")
        return 0

    if a.cmd == "eval-timm":
        import timm
        import torch
        from .vision import evaluate_model, num_classes, with_normalization

        def build(dataset, _):
            net = timm.create_model(a.arch, pretrained=False, num_classes=num_classes(dataset))
            net.load_state_dict(torch.load(a.checkpoint.format(dataset=dataset),
                                           map_location="cpu"))
            cfg = timm.data.resolve_data_config({}, model=net)
            return with_normalization(net, cfg["mean"], cfg["std"])

        res = evaluate_model(None, a.name or a.arch, datasets=a.datasets, mode=a.mode,
                             settings=a.settings, model_for_dataset=build, device=a.device,
                             batch_size=a.batch_size, max_examples=a.max_examples)
        R.save(res, a.out)
        print(f"\nwrote {len(res)} results to {a.out}")
        return 0

    if a.cmd == "leaderboard":
        import pandas as pd
        table = R.leaderboard(R.load(a.results),
                              compare_with=None if a.compare == "none" else a.compare)
        with pd.option_context("display.width", 140, "display.max_colwidth", 60):
            print(table.head(a.top).to_string())
            new = table[table.new]
            if len(new) and not set(new.index) <= set(table.head(a.top).index):
                print("...\n" + new.to_string(header=False))
        return 0


if __name__ == "__main__":
    sys.exit(main())
