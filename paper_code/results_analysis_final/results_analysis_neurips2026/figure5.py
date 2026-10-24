"""Figure 2 (Sec. 5): prompt-based zero-shot accuracy of the target Ve-LLMs (clean)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_style import FIG_WIDTH, apply_rcparams, grouped_bars, legend_above, legend_handle

# Closed-source targets: clean accuracy from transfer_table.csv (as in
# tab:transfer_full). Open-weight targets: compiled API evaluation
# (vision_enabled_language_models/compile_results.py).
TRANSFER_TABLE = HERE / "transfer_table.csv"
OPEN_SUMMARY = HERE.parents[1] / "vision_enabled_language_models" / "results_tables" / "results_summary.csv"

DATASET_KEYS = ["caltech101", "fgvc-aircraft-2013b", "flowers-102",
                "oxford-iiit-pet", "stanford_cars", "uc-merced-land-use-dataset"]
DATASET_DISPLAY = ["Caltech-101", "FGVC Aircraft", "Flowers-102",
                   "Oxford Pet", "Stanford Cars", "UC Merced"]
datasets = ["Caltech-101", "FGVC\nAircraft", "Flowers-102",
            "Oxford\nPet", "Stanford\nCars", "UC Merced"]

_tbl = pd.read_csv(TRANSFER_TABLE)
_tbl = _tbl[_tbl["surrogate"] == "CLIP ViT-B/16"].set_index(["target", "dataset"])["clean"]
_open = pd.read_csv(OPEN_SUMMARY)
_open = _open[_open["threat"] == "clean"].set_index(["model", "dataset"])["accuracy"] * 100

models = {
    "Gemini 3 Flash": [float(_tbl[("Gemini Flash (no think)", d)]) for d in DATASET_DISPLAY],
    "GPT-4o mini":    [float(_tbl[("GPT-4o", d)]) for d in DATASET_DISPLAY],
    "Kimi K2.5":      [float(_open[("kimi", d)]) for d in DATASET_KEYS],
    "Qwen3 VL 235B":  [float(_open[("qwen", d)]) for d in DATASET_KEYS],
}
for name, vals in models.items():
    print(f"{name:<15}" + "  ".join(f"{v:5.1f}" for v in vals) + f"   avg {np.mean(vals):.1f}")

apply_rcparams()
fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))
fig.patch.set_facecolor("white")
grouped_bars(ax, datasets, [(m, v, False) for m, v in models.items()],
             ymax=105, ylabel="Prompt-based zero-shot accuracy (%)")
legend_above(ax, [legend_handle(m) for m in models], ncol=2)

plt.tight_layout(pad=0.4)
out = HERE / "clean_llm_performance.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Figure saved to {out}")
