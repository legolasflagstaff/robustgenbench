#!/usr/bin/env python3
"""
Compile RobustGenBench results into Markdown tables matching
the paper's Table 15 & 16 structure — both Kimi K2.5 and Qwen3 VL 235B.

Produces one table per surrogate per model (8 tables total for Table 15),
grouped under a model heading so both are shown cleanly.
"""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "llm_classification_results"
OUTPUT_FILE = Path(__file__).parent / "results_tables" / "results.md"

DATASETS = [
    "caltech101",
    "fgvc-aircraft-2013b",
    "flowers-102",
    "oxford-iiit-pet",
    "stanford_cars",
    "uc-merced-land-use-dataset",
]

DATASET_DISPLAY = {
    "caltech101": "Caltech-101",
    "fgvc-aircraft-2013b": "FGVC Aircraft",
    "flowers-102": "Flowers-102",
    "oxford-iiit-pet": "Oxford Pet",
    "stanford_cars": "Stanford Cars",
    "uc-merced-land-use-dataset": "UC Merced",
}

SURROGATES = {
    "clip_vitb16": "CLIP ViT-B/16",
    "clip_vith14": "CLIP ViT-H/14",
    "metaclip_vith14": "MetaCLIP ViT-H/14",
    "siglip2_so400m": "SigLIP2 SO400M",
}

SIGLIP2_DATASETS = ["caltech101", "flowers-102", "uc-merced-land-use-dataset"]

THREAT_MODELS = ["l1_eps75", "l1_eps300", "l2_eps2", "l2_eps8", "linf_eps4", "linf_eps8", "linf_eps30"]

THREAT_DISPLAY = {
    "l1_eps75": "ℓ1₇₅",
    "l1_eps300": "ℓ1₃₀₀",
    "l2_eps2": "ℓ2₂",
    "l2_eps8": "ℓ2₈",
    "linf_eps4": "ℓ∞₄",
    "linf_eps8": "ℓ∞₈",
    "linf_eps30": "ℓ∞₃₀",
}

MODELS = {
    "kimi": "Kimi K2.5",
    "qwen": "Qwen3 VL 235B",
}


def load_acc(run_name):
    """Accuracy with one prediction per test image (see compile_results.py)."""
    from compile_results import deduplicated_accuracy
    p = RESULTS_DIR / run_name / "results.json"
    if not p.exists():
        return None
    with open(p) as f:
        return deduplicated_accuracy(json.load(f))


def fmt(v):
    if v is None:
        return "—"
    return f"{v * 100:.1f}"


def collect_results():
    results = {}
    for mk in MODELS:
        results[mk] = {}
        results[mk]["clean"] = {d: load_acc(f"{d}__{mk}__clean") for d in DATASETS}
        results[mk]["common"] = {d: load_acc(f"{d}__{mk}__common") for d in DATASETS}
        for sk in SURROGATES:
            results[mk][sk] = {}
            ds_for_surr = SIGLIP2_DATASETS if sk == "siglip2_so400m" else DATASETS
            for t in THREAT_MODELS:
                results[mk][sk][t] = {}
                for d in DATASETS:
                    if d not in ds_for_surr:
                        results[mk][sk][t][d] = None
                        continue
                    acc = load_acc(f"{d}__{mk}__{sk}__{t}")
                    if acc is None and sk == "clip_vitb16" and t == "linf_eps30":
                        acc = load_acc(f"{d}__{mk}__adversarial")
                    results[mk][sk][t][d] = acc
    return results


def generate_model_table15(results, mk, mname):
    """Generate Table 15 for a single model."""
    lines = []
    lines.append(f"### {mname}")
    lines.append("")

    for sk, sname in SURROGATES.items():
        ds_for_surr = SIGLIP2_DATASETS if sk == "siglip2_so400m" else DATASETS

        lines.append(f"#### Surrogate: {sname}")
        lines.append("")

        threat_cols = " | ".join(THREAT_DISPLAY[t] for t in THREAT_MODELS)
        header = f"| Dataset | Clean | {threat_cols} | sev@3 | **Avg** |"
        sep = "|---|---:" + "|---:" * 7 + "|---:|---:|"
        lines.append(header)
        lines.append(sep)

        surr_avgs = []

        for d in DATASETS:
            clean = results[mk]["clean"][d]
            row = [DATASET_DISPLAY[d], fmt(clean)]

            threat_vals = []
            for t in THREAT_MODELS:
                v = results[mk][sk][t][d]
                row.append(fmt(v))
                if v is not None:
                    threat_vals.append(v * 100)

            common = results[mk]["common"][d]
            row.append(fmt(common))

            all_vals = list(threat_vals)
            if common is not None:
                all_vals.append(common * 100)
            if all_vals:
                avg = sum(all_vals) / len(all_vals)
                row.append(f"{avg:.1f}")
                surr_avgs.append(avg)
            else:
                row.append("—")

            lines.append("| " + " | ".join(row) + " |")

        # Average row
        clean_vals = [results[mk]["clean"][d] for d in ds_for_surr if results[mk]["clean"].get(d) is not None]
        avg_row = ["**Avg.**", f"**{sum(clean_vals)/len(clean_vals)*100:.1f}**" if clean_vals else "—"]

        for t in THREAT_MODELS:
            tv = [results[mk][sk][t][d] for d in ds_for_surr if results[mk][sk][t].get(d) is not None]
            avg_row.append(f"**{sum(tv)/len(tv)*100:.1f}**" if tv else "—")

        common_vals = [results[mk]["common"][d] for d in ds_for_surr if results[mk]["common"].get(d) is not None]
        avg_row.append(f"**{sum(common_vals)/len(common_vals)*100:.1f}**" if common_vals else "—")

        if surr_avgs:
            avg_row.append(f"**{sum(surr_avgs)/len(surr_avgs):.1f}**")
        else:
            avg_row.append("—")

        lines.append("| " + " | ".join(avg_row) + " |")
        lines.append("")

    # Global average
    lines.append("#### Global Average")
    lines.append("")
    threat_cols = " | ".join(THREAT_DISPLAY[t] for t in THREAT_MODELS)
    header = f"| | Clean | {threat_cols} | sev@3 | **Avg** |"
    sep = "|---|---:" + "|---:" * 7 + "|---:|---:|"
    lines.append(header)
    lines.append(sep)

    global_row = ["**All surrogates**"]
    clean_vals = [results[mk]["clean"][d] for d in DATASETS if results[mk]["clean"].get(d) is not None]
    global_row.append(f"**{sum(clean_vals)/len(clean_vals)*100:.1f}**" if clean_vals else "—")

    for t in THREAT_MODELS:
        all_t_vals = []
        for sk2 in SURROGATES:
            ds2 = SIGLIP2_DATASETS if sk2 == "siglip2_so400m" else DATASETS
            for d in ds2:
                v = results[mk][sk2][t].get(d)
                if v is not None:
                    all_t_vals.append(v)
        global_row.append(f"**{sum(all_t_vals)/len(all_t_vals)*100:.1f}**" if all_t_vals else "—")

    common_vals = [results[mk]["common"][d] for d in DATASETS if results[mk]["common"].get(d) is not None]
    global_row.append(f"**{sum(common_vals)/len(common_vals)*100:.1f}**" if common_vals else "—")

    all_global = []
    for sk2 in SURROGATES:
        ds2 = SIGLIP2_DATASETS if sk2 == "siglip2_so400m" else DATASETS
        for d in ds2:
            for t in THREAT_MODELS:
                v = results[mk][sk2][t].get(d)
                if v is not None:
                    all_global.append(v * 100)
            c = results[mk]["common"].get(d)
            if c is not None:
                all_global.append(c * 100)
    global_row.append(f"**{sum(all_global)/len(all_global):.1f}**" if all_global else "—")

    lines.append("| " + " | ".join(global_row) + " |")
    lines.append("")

    return lines


def generate_table16(results):
    """Generate Table 16: Common corruptions for both models."""
    lines = []
    lines.append("## Table 16: Common Corruptions (Severity 3)")
    lines.append("")
    lines.append("| Dataset | Kimi K2.5 Clean | Kimi K2.5 sev@3 | Qwen3 VL Clean | Qwen3 VL sev@3 |")
    lines.append("|---|---:|---:|---:|---:|")

    for d in DATASETS:
        lines.append(
            f"| {DATASET_DISPLAY[d]} "
            f"| {fmt(results['kimi']['clean'][d])} "
            f"| {fmt(results['kimi']['common'][d])} "
            f"| {fmt(results['qwen']['clean'][d])} "
            f"| {fmt(results['qwen']['common'][d])} |"
        )

    kc = [v for v in results["kimi"]["clean"].values() if v is not None]
    km = [v for v in results["kimi"]["common"].values() if v is not None]
    qc = [v for v in results["qwen"]["clean"].values() if v is not None]
    qm = [v for v in results["qwen"]["common"].values() if v is not None]
    lines.append(
        f"| **Average** "
        f"| **{sum(kc)/len(kc)*100:.1f}** "
        f"| **{sum(km)/len(km)*100:.1f}** "
        f"| **{sum(qc)/len(qc)*100:.1f}** "
        f"| **{sum(qm)/len(qm)*100:.1f}** |"
    )
    lines.append("")
    return lines


def main():
    results = collect_results()

    total, missing = 0, 0
    for mk in MODELS:
        for d in DATASETS:
            for cond in ["clean", "common"]:
                if results[mk][cond].get(d) is not None:
                    total += 1
                else:
                    missing += 1
        for sk in SURROGATES:
            ds_for_surr = SIGLIP2_DATASETS if sk == "siglip2_so400m" else DATASETS
            for t in THREAT_MODELS:
                for d in ds_for_surr:
                    if results[mk][sk][t].get(d) is not None:
                        total += 1
                    else:
                        missing += 1

    print(f"Results: {total} available, {missing} missing")

    all_lines = []
    all_lines.append("# RobustGenBench Evaluation Results")
    all_lines.append("")
    all_lines.append("Models evaluated via Amazon Bedrock:")
    all_lines.append("- **Kimi K2.5** (`moonshotai.kimi-k2.5`)")
    all_lines.append("- **Qwen3 VL 235B** (`qwen.qwen3-vl-235b-a22b`)")
    all_lines.append("")
    all_lines.append("---")
    all_lines.append("")

    # Table 16
    all_lines.extend(generate_table16(results))
    all_lines.append("---")
    all_lines.append("")

    # Table 15
    all_lines.append("## Table 15: Adversarial Transfer Accuracy (%)")
    all_lines.append("")
    all_lines.append("Columns: Clean | 7 adversarial threat models | sev@3 (common corruptions) | Avg")
    all_lines.append("")

    for mk, mname in MODELS.items():
        all_lines.extend(generate_model_table15(results, mk, mname))
        all_lines.append("---")
        all_lines.append("")

    md = "\n".join(all_lines)

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(md)

    print(f"Written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
