"""Figure 1 (Sec. 5): embedding-based zero-shot accuracy of the surrogate VLMs."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from paper_style import FIG_WIDTH, apply_rcparams, grouped_bars, legend_above, legend_handle

datasets = ["Caltech-101", "FGVC\nAircraft", "Flowers-102",
            "Oxford\nPet", "Stanford\nCars", "UC Merced"]

models = {
    "CLIP B/16":      [88.9, 29.2, 71.5, 91.7, 88.7, 68.1],
    "CLIP H/14":      [90.1, 47.6, 81.5, 94.4, 93.7, 74.5],
    "MetaCLIP H/14":  [89.0, 53.5, 84.5, 95.4, 87.9, 79.3],
    "SigLIP2 base":   [92.1, 13.7, 84.7, 15.1,  4.2, 73.8],
    "SigLIP2 SO400M": [92.1, 23.8, 91.7, 14.0,  5.3, 87.6],
}

apply_rcparams()
fig, ax = plt.subplots(figsize=(FIG_WIDTH, 2.6))
fig.patch.set_facecolor("white")
grouped_bars(ax, datasets, [(m, v, False) for m, v in models.items()],
             ymax=105, ylabel="Embedding-based zero-shot accuracy (%)")
legend_above(ax, [legend_handle(m) for m in models], ncol=3)

plt.tight_layout(pad=0.4)
out = HERE / "vlm_performance.png"
plt.savefig(out, dpi=300, bbox_inches="tight")
print(f"Figure saved to {out}")
