#!/usr/bin/env python3
"""
Compile RobustGenBench results into Tables 15 & 16 format.

Table 15: Adversarial transfer accuracy (%) under each surrogate and threat model.
Table 16: Common corruptions (severity 3) with clean baselines.

Outputs both LaTeX tables and a summary CSV.
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "llm_classification_results"

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

# SigLIP2 only has these datasets
SIGLIP2_DATASETS = ["caltech101", "flowers-102", "uc-merced-land-use-dataset"]

THREAT_MODELS = ["l1_eps75", "l1_eps300", "l2_eps2", "l2_eps8", "linf_eps4", "linf_eps8", "linf_eps30"]

THREAT_DISPLAY = {
    "l1_eps75": r"$\ell_1^{75}$",
    "l1_eps300": r"$\ell_1^{300}$",
    "l2_eps2": r"$\ell_2^{2}$",
    "l2_eps8": r"$\ell_2^{8}$",
    "linf_eps4": r"$\ell_\infty^{4}$",
    "linf_eps8": r"$\ell_\infty^{8}$",
    "linf_eps30": r"$\ell_\infty^{30}$",
}

MODELS = {
    "kimi": "Kimi K2.5",
    "qwen": "Qwen3 VL 235B",
}


def deduplicated_accuracy(data: dict):
    """Accuracy with one prediction per test image.

    Resumed runs can append a second pass to ``per_sample``, so some files hold
    more rows than the test set (e.g. 2000 rows for 1000 images). Keep the first
    prediction for each image ``index`` so every image is counted exactly once.
    Falls back to the stored accuracy when per-sample records are unavailable.
    """
    per_sample = data.get("per_sample")
    if not per_sample or "index" not in per_sample[0]:
        return data.get("accuracy")
    first = {}
    for sample in per_sample:
        first.setdefault(sample["index"], sample)
    if len(first) != len(per_sample):
        print(f"  [dedup] {data.get('run_name', '?')}: {len(per_sample)} rows -> "
              f"{len(first)} unique images", file=sys.stderr)
    return sum(bool(s.get("correct")) for s in first.values()) / len(first)


def load_accuracy(run_name: str):
    """Load accuracy from a results.json file (deduplicated per test image)."""
    results_path = RESULTS_DIR / run_name / "results.json"
    if not results_path.exists():
        return None
    with open(results_path) as f:
        data = json.load(f)
    return deduplicated_accuracy(data)


def get_run_name_adversarial(dataset: str, model_key: str, surrogate: str, threat: str) -> str:
    """Get run name for adversarial results."""
    return f"{dataset}__{model_key}__{surrogate}__{threat}"


def get_run_name_clean(dataset: str, model_key: str) -> str:
    """Get run name for clean results."""
    return f"{dataset}__{model_key}__clean"


def get_run_name_common(dataset: str, model_key: str) -> str:
    """Get run name for common corruption results."""
    return f"{dataset}__{model_key}__common"


def get_run_name_adversarial_legacy(dataset: str, model_key: str) -> str:
    """Get run name for the original adversarial run (clip_vitb16 linf_eps30)."""
    return f"{dataset}__{model_key}__adversarial"


def collect_all_results():
    """Collect all results into a nested dictionary."""
    results = {}
    
    for model_key in MODELS:
        results[model_key] = {}
        
        # Clean
        results[model_key]["clean"] = {}
        for dataset in DATASETS:
            acc = load_accuracy(get_run_name_clean(dataset, model_key))
            results[model_key]["clean"][dataset] = acc
        
        # Common
        results[model_key]["common"] = {}
        for dataset in DATASETS:
            acc = load_accuracy(get_run_name_common(dataset, model_key))
            results[model_key]["common"][dataset] = acc
        
        # Adversarial per surrogate
        for surrogate in SURROGATES:
            results[model_key][surrogate] = {}
            datasets_for_surr = SIGLIP2_DATASETS if surrogate == "siglip2_so400m" else DATASETS
            for threat in THREAT_MODELS:
                results[model_key][surrogate][threat] = {}
                for dataset in DATASETS:
                    if dataset not in datasets_for_surr:
                        results[model_key][surrogate][threat][dataset] = None  # not evaluated
                        continue
                    # Check new naming first, then legacy naming for linf_eps30
                    acc = load_accuracy(get_run_name_adversarial(dataset, model_key, surrogate, threat))
                    if acc is None and surrogate == "clip_vitb16" and threat == "linf_eps30":
                        acc = load_accuracy(get_run_name_adversarial_legacy(dataset, model_key))
                    results[model_key][surrogate][threat][dataset] = acc
    
    return results


def format_acc(acc, multiply=True):
    """Format accuracy as percentage string."""
    if acc is None:
        return r"{\textemdash}"
    val = acc * 100 if multiply else acc
    return f"{val:.1f}"


def generate_table15_latex(results):
    """Generate LaTeX for Table 15 (adversarial transfer table)."""
    lines = []
    lines.append(r"% Requires: booktabs, multirow, array")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\small")
    lines.append(r"\caption{%")
    lines.append(r"  Adversarial transfer accuracy (\%) under each surrogate and threat model,")
    lines.append(r"  evaluated on two open-weight Ve-LLMs (Kimi K2.5, Qwen3 VL 235B) accessed via API")
    lines.append(r"  with the same prompt as the closed-source targets (Table~\ref{tab:transfer_full}).")
    lines.append(r"  \emph{Clean}: no attack.")
    lines.append(r"  Bold values = column/block averages.")
    lines.append(r"  {\textemdash}: not evaluated.")
    lines.append(r"  See Table~\ref{tab:common} for common corruption results.")
    lines.append(r"}")
    lines.append(r"\label{tab:transfer_open}")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    
    # Header: model blocks with Clean + 7 threats + sev@3 + Avg
    ncols_per_model = 10  # Clean + 7 threats + sev@3 + Avg
    lines.append(r"\begin{tabular}{l || r | rrrrrrr | r || r | rrrrrrr | r}")
    lines.append(r"\toprule")
    
    # Model super-headers
    lines.append(r"\multirow{3}{*}{\textbf{Dataset}}")
    lines.append(r"  & \multicolumn{9}{c||}{\textbf{" + MODELS["kimi"] + r"}}")
    lines.append(r"  & \multicolumn{9}{c}{\textbf{" + MODELS["qwen"] + r"}} \\")
    
    # Sub-headers
    lines.append(r"  & \multirow{2}{*}{Clean} & \multicolumn{7}{c|}{Adversarial} & \multirow{2}{*}{\textbf{Avg.}}")
    lines.append(r"  & \multirow{2}{*}{Clean} & \multicolumn{7}{c|}{Adversarial} & \multirow{2}{*}{\textbf{Avg.}} \\")
    
    # Threat model labels
    threat_labels = " & ".join([THREAT_DISPLAY[t] for t in THREAT_MODELS])
    lines.append(f"  & & {threat_labels} &")
    lines.append(f"  & & {threat_labels} & \\\\")
    lines.append(r"\midrule")
    
    global_avgs = {m: [] for m in MODELS}
    
    for surrogate, surr_display in SURROGATES.items():
        lines.append(r"\multicolumn{19}{l}{\textit{Surrogate: \textbf{" + surr_display + r"}}}  \\")
        lines.append(r"\addlinespace[1pt]")
        
        surr_avgs = {m: [] for m in MODELS}
        datasets_for_surr = SIGLIP2_DATASETS if surrogate == "siglip2_so400m" else DATASETS
        
        for dataset in DATASETS:
            row_parts = [r"\hspace{4pt}" + DATASET_DISPLAY[dataset]]
            
            for model_key in MODELS:
                clean_acc = results[model_key]["clean"].get(dataset)
                row_parts.append(format_acc(clean_acc))
                
                threat_accs = []
                for threat in THREAT_MODELS:
                    acc = results[model_key][surrogate][threat].get(dataset)
                    row_parts.append(format_acc(acc))
                    if acc is not None:
                        threat_accs.append(acc * 100)
                
                # Average (over all adversarial threats for this dataset)
                if threat_accs:
                    avg = sum(threat_accs) / len(threat_accs)
                    row_parts.append(f"{avg:.1f}")
                    surr_avgs[model_key].append(avg)
                    global_avgs[model_key].append(avg)
                else:
                    row_parts.append(r"{\textemdash}")
            
            lines.append(" & ".join(row_parts) + r" \\")
        
        # Surrogate average row
        lines.append(r"\addlinespace[1pt]")
        avg_row = [r"\hspace{4pt}\textit{Avg.}"]
        for model_key in MODELS:
            # Clean average
            clean_vals = [results[model_key]["clean"][d] for d in datasets_for_surr if results[model_key]["clean"].get(d) is not None]
            if clean_vals:
                avg_row.append(r"\textbf{" + f"{sum(clean_vals)/len(clean_vals)*100:.1f}" + "}")
            else:
                avg_row.append(r"{\textemdash}")
            
            # Per-threat averages
            for threat in THREAT_MODELS:
                vals = [results[model_key][surrogate][threat][d] for d in datasets_for_surr 
                        if results[model_key][surrogate][threat].get(d) is not None]
                if vals:
                    avg_row.append(r"\textbf{" + f"{sum(vals)/len(vals)*100:.1f}" + "}")
                else:
                    avg_row.append(r"{\textemdash}")
            
            # Overall surrogate average
            if surr_avgs[model_key]:
                avg_row.append(r"\textbf{" + f"{sum(surr_avgs[model_key])/len(surr_avgs[model_key]):.1f}" + "}")
            else:
                avg_row.append(r"{\textemdash}")
        
        lines.append(" & ".join(avg_row) + r" \\")
        lines.append(r"\midrule")
    
    # Global average
    global_row = [r"\textbf{Global avg.}"]
    for model_key in MODELS:
        clean_vals = [results[model_key]["clean"][d] for d in DATASETS if results[model_key]["clean"].get(d) is not None]
        if clean_vals:
            global_row.append(r"\textbf{" + f"{sum(clean_vals)/len(clean_vals)*100:.1f}" + "}")
        else:
            global_row.append(r"{\textemdash}")
        
        for threat in THREAT_MODELS:
            all_vals = []
            for surrogate in SURROGATES:
                datasets_for_surr = SIGLIP2_DATASETS if surrogate == "siglip2_so400m" else DATASETS
                for d in datasets_for_surr:
                    v = results[model_key][surrogate][threat].get(d)
                    if v is not None:
                        all_vals.append(v)
            if all_vals:
                global_row.append(r"\textbf{" + f"{sum(all_vals)/len(all_vals)*100:.1f}" + "}")
            else:
                global_row.append(r"{\textemdash}")
        
        if global_avgs[model_key]:
            global_row.append(r"\textbf{" + f"{sum(global_avgs[model_key])/len(global_avgs[model_key]):.1f}" + "}")
        else:
            global_row.append(r"{\textemdash}")
    
    lines.append(" & ".join(global_row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table*}")
    
    return "\n".join(lines)


def generate_table16_latex(results):
    """Generate LaTeX for Table 16 (common corruptions)."""
    lines = []
    lines.append(r"% Requires: booktabs, multirow")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.08}")
    lines.append(r"\small")
    lines.append(r"\caption{%")
    lines.append(r"  Robustness to common corruptions (severity 3).")
    lines.append(r"  \emph{Clean} accuracy reported for reference.")
    lines.append(r"}")
    lines.append(r"\label{tab:common_bedrock}")
    lines.append(r"\begin{tabular}{l || r r || r r}")
    lines.append(r"\toprule")
    lines.append(r"\multirow{2}{*}{\textbf{Dataset}} & \multicolumn{2}{c||}{\textbf{" + MODELS["kimi"] + r"}} & \multicolumn{2}{c}{\textbf{" + MODELS["qwen"] + r"}} \\")
    lines.append(r" & Clean & Sev.$^3$ & Clean & Sev.$^3$ \\")
    lines.append(r"\midrule")
    
    for dataset in DATASETS:
        row = [DATASET_DISPLAY[dataset]]
        for model_key in MODELS:
            clean = results[model_key]["clean"].get(dataset)
            common = results[model_key]["common"].get(dataset)
            row.append(format_acc(clean))
            row.append(format_acc(common))
        lines.append(" & ".join(row) + r" \\")
    
    # Average row
    lines.append(r"\midrule")
    avg_row = [r"\textbf{Average}"]
    for model_key in MODELS:
        clean_vals = [results[model_key]["clean"][d] for d in DATASETS if results[model_key]["clean"].get(d) is not None]
        common_vals = [results[model_key]["common"][d] for d in DATASETS if results[model_key]["common"].get(d) is not None]
        avg_row.append(r"\textbf{" + (f"{sum(clean_vals)/len(clean_vals)*100:.1f}" if clean_vals else "—") + "}")
        avg_row.append(r"\textbf{" + (f"{sum(common_vals)/len(common_vals)*100:.1f}" if common_vals else "—") + "}")
    lines.append(" & ".join(avg_row) + r" \\")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    return "\n".join(lines)


def generate_csv_summary(results):
    """Generate a CSV summary of all results."""
    lines = ["model,surrogate,threat,dataset,accuracy"]
    
    for model_key in MODELS:
        for dataset in DATASETS:
            acc = results[model_key]["clean"].get(dataset)
            if acc is not None:
                lines.append(f"{model_key},none,clean,{dataset},{acc:.4f}")
            
            acc = results[model_key]["common"].get(dataset)
            if acc is not None:
                lines.append(f"{model_key},none,common,{dataset},{acc:.4f}")
        
        for surrogate in SURROGATES:
            for threat in THREAT_MODELS:
                for dataset in DATASETS:
                    acc = results[model_key][surrogate][threat].get(dataset)
                    if acc is not None:
                        lines.append(f"{model_key},{surrogate},{threat},{dataset},{acc:.4f}")
    
    return "\n".join(lines)


def print_summary(results):
    """Print a human-readable summary."""
    print("\n" + "=" * 80)
    print("  RobustGenBench Results Summary")
    print("=" * 80)
    
    for model_key, model_name in MODELS.items():
        print(f"\n{'─' * 80}")
        print(f"  {model_name}")
        print(f"{'─' * 80}")
        
        # Clean + Common
        print(f"\n  {'Dataset':<20} {'Clean':>8} {'Common':>8} {'Drop':>8}")
        print(f"  {'─'*20} {'─'*8} {'─'*8} {'─'*8}")
        clean_vals, common_vals = [], []
        for dataset in DATASETS:
            clean = results[model_key]["clean"].get(dataset)
            common = results[model_key]["common"].get(dataset)
            clean_str = f"{clean*100:.1f}" if clean else "—"
            common_str = f"{common*100:.1f}" if common else "—"
            drop_str = f"{(clean-common)*100:.1f}" if (clean and common) else "—"
            print(f"  {DATASET_DISPLAY[dataset]:<20} {clean_str:>8} {common_str:>8} {drop_str:>8}")
            if clean: clean_vals.append(clean)
            if common: common_vals.append(common)
        
        if clean_vals and common_vals:
            avg_clean = sum(clean_vals)/len(clean_vals)*100
            avg_common = sum(common_vals)/len(common_vals)*100
            print(f"  {'Average':<20} {avg_clean:>8.1f} {avg_common:>8.1f} {avg_clean-avg_common:>8.1f}")
        
        # Adversarial per surrogate (linf_eps30 only for compact display)
        print(f"\n  Adversarial (linf_eps30 only):")
        print(f"  {'Dataset':<20}", end="")
        for surr_key, surr_name in SURROGATES.items():
            print(f" {surr_name:>18}", end="")
        print()
        
        for dataset in DATASETS:
            print(f"  {DATASET_DISPLAY[dataset]:<20}", end="")
            for surr_key in SURROGATES:
                acc = results[model_key][surr_key]["linf_eps30"].get(dataset)
                val = f"{acc*100:.1f}" if acc else "—"
                print(f" {val:>18}", end="")
            print()


def main():
    print("Loading results...")
    results = collect_all_results()
    
    # Count available results
    total = 0
    missing = 0
    for model_key in MODELS:
        for dataset in DATASETS:
            if results[model_key]["clean"].get(dataset) is not None:
                total += 1
            else:
                missing += 1
            if results[model_key]["common"].get(dataset) is not None:
                total += 1
            else:
                missing += 1
        for surrogate in SURROGATES:
            datasets_for_surr = SIGLIP2_DATASETS if surrogate == "siglip2_so400m" else DATASETS
            for threat in THREAT_MODELS:
                for dataset in datasets_for_surr:
                    if results[model_key][surrogate][threat].get(dataset) is not None:
                        total += 1
                    else:
                        missing += 1
    
    print(f"  Available: {total} results")
    print(f"  Missing:   {missing} results")
    
    # Print summary
    print_summary(results)
    
    # Generate LaTeX tables
    output_dir = Path(__file__).parent / "results_tables"
    output_dir.mkdir(exist_ok=True)
    
    table15 = generate_table15_latex(results)
    table16 = generate_table16_latex(results)
    csv_data = generate_csv_summary(results)
    
    with open(output_dir / "table15_adversarial.tex", "w") as f:
        f.write(table15)
    with open(output_dir / "table16_common.tex", "w") as f:
        f.write(table16)
    with open(output_dir / "results_summary.csv", "w") as f:
        f.write(csv_data)
    
    print(f"\n\nOutput saved to: {output_dir}")
    print(f"  - table15_adversarial.tex")
    print(f"  - table16_common.tex")
    print(f"  - results_summary.csv")
    
    # Also print the LaTeX tables
    print("\n\n" + "=" * 80)
    print("TABLE 16: Common Corruptions")
    print("=" * 80)
    print(table16)
    
    print("\n\n" + "=" * 80)
    print("TABLE 15: Adversarial Transfer (excerpt)")
    print("=" * 80)
    print(table15[:3000] + "\n... [truncated]")


if __name__ == "__main__":
    main()
