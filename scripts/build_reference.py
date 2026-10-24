"""Build robustgenbench/reference_data/*.csv from the paper's raw results in paper_code/.

Outputs long-format tables (model, dataset, setting, accuracy, ...) that
``robustgenbench.results.reference`` exposes for leaderboard comparisons.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_code"
OUT = ROOT / "robustgenbench" / "reference_data"
OUT.mkdir(exist_ok=True)

# Canonical configuration names (the shortcut column of the raw CSV has collisions).
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "load_results", PAPER / "results_analysis_final" / "results_analysis" / "load_results.py")
_lr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_lr)
NAMES = {**_lr.backbone_name,
         "CLIP-convnext_base_w-laion2B-s13B-b82K": "convnext_b,clip,laion2b",
         "CLIP-convnext_base_w-laion_aesthetic-s13B-b82K": "convnext_b,clip,laiona",
         "resnet50.a1_in1k": "resnet50,sup,in1k", "robust_resnet50": "resnet50,rob-sup,in1k",
         "deit_small_patch16_224.fb_in1k": "deit_s,sup,in1k",
         "convnext_base.fb_in1k": "convnext_b,sup,in1k",
         "vit_small_patch16_224.augreg_in21k": "vit_s,sup,in22k"}

# ── Vision: 80 configurations x 3 fine-tuning protocols, white-box (Section 4) ──
vis = pd.read_csv(PAPER / "results" / "results_dataset.csv", index_col=0)
protocol = {"FFT (50 epochs)": "fft50", "FFT (5 epochs)": "fft5", "LP (50 epochs)": "lp50"}
cols = {"clean_acc": "clean", "common_acc": "common", "L1_acc": "l1_75",
        "L2_acc": "l2_2", "Linf_acc": "linf_4"}
rows = []
for _, r in vis.iterrows():
    for c, setting in cols.items():
        rows.append(dict(
            protocol=protocol[r.ft_strategy], model=f"{NAMES[r.backbone]} | {r.loss_function}",
            hf_backbone=r.backbone, loss=r.loss_function, architecture=r.model_type,
            size={"large": "base", "medium": "small", "small": "tiny"}.get(r.model_size, r.model_size),
            pretraining=r.pre_training_strategy, dataset=r.dataset, setting=setting,
            accuracy=r[c], mode="whitebox"))
vis_long = pd.DataFrame(rows)
assert vis_long.groupby(["protocol", "model", "dataset", "setting"]).size().max() == 1
vis_long.to_csv(OUT / "vision_whitebox.csv", index=False)

# ── Ve-LLMs: 4 targets, per surrogate, transfer (Section 5) ──────────────────
DS = {"Caltech-101": "caltech101", "FGVC Aircraft": "fgvc-aircraft-2013b",
      "Flowers-102": "flowers-102", "Oxford Pet": "oxford-iiit-pet",
      "Stanford Cars": "stanford_cars", "UC Merced": "uc-merced-land-use-dataset"}
SUR = {"CLIP ViT-B/16": "clip_vitb16", "CLIP ViT-H/14": "clip_vith14",
       "MetaCLIP ViT-H/14": "metaclip_vith14", "SigLIP2 SO400M": "siglip2_so400m"}
TH = {"l1@75": "l1_75", "l1@300": "l1_300", "l2@2": "l2_2", "l2@8": "l2_8",
      "linf@4": "linf_4", "linf@8": "linf_8", "linf@30": "linf_30"}
NAME = {"Gemini Flash (no think)": "gemini-3-flash (no think)", "GPT-4o": "gpt-4o-mini"}
t = pd.read_csv(PAPER / "results_analysis_final" / "results_analysis_neurips2026" / "transfer_table.csv")
rows = []
for _, r in t.iterrows():
    m, d = NAME[r.target], DS[r.dataset]
    if r.surrogate == "CLIP ViT-B/16":  # clean and common are surrogate-independent
        rows += [dict(model=m, dataset=d, setting="clean", surrogate=None, accuracy=r.clean / 100),
                 dict(model=m, dataset=d, setting="common", surrogate=None, accuracy=r["sev@3"] / 100)]
    for c, s in TH.items():
        if pd.notna(r[c]):
            rows.append(dict(model=m, dataset=d, setting=s, surrogate=SUR[r.surrogate],
                             accuracy=r[c] / 100))
o = pd.read_csv(PAPER / "vision_enabled_language_models" / "results_tables" / "results_summary.csv")
OPEN = {"kimi": "kimi-k2.5", "qwen": "qwen3-vl-235b-a22b"}
OT = {"l1_eps75": "l1_75", "l1_eps300": "l1_300", "l2_eps2": "l2_2", "l2_eps8": "l2_8",
      "linf_eps4": "linf_4", "linf_eps8": "linf_8", "linf_eps30": "linf_30"}
for _, r in o.iterrows():
    setting = {"clean": "clean", "common": "common"}.get(r.threat, OT.get(r.threat))
    rows.append(dict(model=OPEN[r.model], dataset=r.dataset, setting=setting,
                     surrogate=None if r.surrogate == "none" else r.surrogate,
                     accuracy=r.accuracy))
vl = pd.DataFrame(rows)
vl["mode"] = "transfer"
vl.to_csv(OUT / "vellm_transfer.csv", index=False)
print(len(vis_long), "vision rows;", len(vl), "Ve-LLM rows")
