# RobustGenBench

**Evaluate how image classifiers and vision-enabled LLMs hold up across common corruptions and $\ell_1$, $\ell_2$, $\ell_\infty$ adversarial perturbations, on six downstream datasets.**

[🌐 Project page](https://legolasflagstaff.github.io/robustgenbench/) · [📦 Dataset on Hugging Face](https://huggingface.co/datasets/legolasflagstaff/RobustGenBench) · [📄 Paper](https://openreview.net/forum?id=KDsuVST8by) · [📜 Citation](#citation)

This repository provides the `robustgenbench` Python package, which connects any model to the benchmark in a few lines:

- **PyTorch classifiers** (fine-tuned ViTs, ConvNeXts, your own architecture): white-box AutoAttack at the paper's budgets, or transfer attacks.
- **Black-box models** (GPT-4o, Gemini, open-weight VLMs behind vLLM, CLIP, anything with a `predict` function): evaluated by inference on the 147k pre-computed transfer attacks, no GPU needed.
- **Leaderboards**: rank your results against the 80 fine-tuned configurations and 4 Ve-LLMs reported in the paper.

The code used for the paper itself is in [`paper_code/`](paper_code/).

## Installation

```bash
pip install -e .                    # core: data access, metrics, black-box evaluation
pip install -e ".[vlm]"             # + OpenAI-compatible API client
pip install -e ".[torch]"           # + PyTorch / timm models
pip install git+https://github.com/fra31/auto-attack   # white-box attacks (not on PyPI)
```

## Evaluate a vision-enabled LLM

Anything served behind an OpenAI-compatible API works, including open-weight models served locally:

```bash
# OpenAI (reads OPENAI_API_KEY)
robustgenbench eval-vlm --model gpt-4o-mini --max-examples 50 --datasets flowers-102

# Open-weight model with vLLM
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 127.0.0.1 --port 8000
robustgenbench eval-vlm --model Qwen/Qwen2.5-VL-7B-Instruct --base-url http://127.0.0.1:8000/v1

robustgenbench leaderboard results.jsonl --compare vellm
```

The model receives the paper's prompt (`robustgenbench.build_prompt`) with each 224×224 image as JPEG (quality 95). Free-text answers are mapped to classes with the paper's parser (`robustgenbench.parse_prediction`). Raw answers are cached in `rgb_cache/`, so an interrupted run resumes without new requests.

**Cost.** The full protocol is about 150k requests (6 datasets × 9 settings × every surrogate). Start with `--max-examples` and a few `--datasets`, or restrict `--surrogates` (results are then not comparable with the paper's worst-case aggregation).

### Any other model

Implement one method:

```python
import robustgenbench as rgb

class MyModel(rgb.BlackBoxClassifier):
    max_workers = 4                      # concurrent calls

    def predict(self, image, class_names):
        ...                              # PIL image (224x224), list of class names
        return "golden retriever"        # a class name (free text is parsed) or an index

results = rgb.evaluate_classifier(MyModel(), "my-model")
rgb.save(results, "results.jsonl")
print(rgb.leaderboard(results, compare_with="vellm"))
```

[`examples/clip_zero_shot.py`](examples/clip_zero_shot.py) wraps zero-shot CLIP this way.

## Evaluate a PyTorch classifier

The model takes images in $[0, 1]$ of shape (B, 3, 224, 224) and returns logits. Put normalization inside the model with `with_normalization`, so budgets are measured on raw pixels. The six datasets have different label spaces, so you provide one model per dataset:

```python
import timm, torch
import robustgenbench as rgb
from robustgenbench.vision import evaluate_model, num_classes, with_normalization

def model_for_dataset(dataset, _):
    net = timm.create_model("convnext_base", num_classes=num_classes(dataset))
    net.load_state_dict(torch.load(f"ckpt/{dataset}.pt"))
    return with_normalization(net)       # ImageNet mean/std by default

results = evaluate_model(None, "my-convnext", model_for_dataset=model_for_dataset)
print(rgb.leaderboard(results, compare_with="fft50"))
```

Or from the command line: `robustgenbench eval-timm --arch convnext_base --checkpoint "ckpt/{dataset}.pt"`.

- `mode="whitebox"` (default): the paper's 5 settings for vision models. Clean and common use the released images; $\ell_1 = 75$, $\ell_2 = 2$, $\ell_\infty = 4/255$ are crafted on your model with AutoAttack (standard version).
- `mode="transfer"`: the 9 black-box settings with the released surrogate attacks, as for Ve-LLMs.

Need a model to start from? [`examples/robust_finetune.py`](examples/robust_finetune.py) robustly fine-tunes a timm backbone with TRADES or Classic AT on any RobustGenBench dataset and evaluates it. `TorchSplit(dataset, "train")` gives a standard torch `Dataset` for your own training loops.

## Protocol

| | Fine-tuned vision models (white-box) | Black-box models, e.g. Ve-LLMs (transfer) |
|---|---|---|
| Settings | clean, common, $\ell_1$@75, $\ell_2$@2, $\ell_\infty$@4/255 | clean, common, $\ell_1$@{75, 300}, $\ell_2$@{2, 8}, $\ell_\infty$@{4, 8, 30}/255 |
| Adversarial images | AutoAttack on the evaluated model | released attacks, worst case over surrogates |
| Max. Sum score (6 datasets) | 30 | 54 |

Metrics (`robustgenbench.metrics`):

- **Sum score**: sum of accuracies over (dataset, setting) pairs. Pool-independent: **use it to position new models** against published results.
- **Borda score**: rank consistency across pairs (worst model in a pair gets 1). Depends on the pool of models, so it is only meaningful within a fixed pool.
- **MAC**: mean absolute Spearman correlation between settings within a group of models (e.g. one architecture type), averaged over datasets.

`rgb.leaderboard` keeps only the (dataset, setting) pairs you evaluated, so partial evaluations are ranked fairly. Results obtained with `max_examples` are for quick checks only.

## Reference results

`rgb.reference("vision_whitebox", protocol="fft50" | "fft5" | "lp50")` and `rgb.reference("vellm_transfer")` return the paper's per-(dataset, setting) accuracies. The test suite checks that they reproduce the paper's leaderboard (top Sum score 19.79, Borda 2281, Kendall's τ = 0.854 between Borda and Sum, MAC 0.795).

## Data access

```python
rgb.load_split("flowers-102", "test")                              # clean
rgb.load_split("flowers-102", "common")                            # 11 corruptions, severity 3
rgb.load_split("flowers-102", "linf_30", surrogate="clip_vith14")  # transfer attack
rgb.available_surrogates("stanford_cars", "linf_30")
```

These return `datasets.Dataset` objects aligned by `image_id` with `test`. See the [dataset card](https://huggingface.co/datasets/legolasflagstaff/RobustGenBench) for coverage, construction, and licensing.

## Tests

```bash
pytest                  # offline: metrics, parser, reference leaderboards
pytest -m network       # end-to-end against the Hub (a few minutes)
```

## Repository layout

```
robustgenbench/        the package (benchmark, vision, vlm, metrics, results, cli)
examples/              CLIP zero-shot, VLM via API, PyTorch evaluation, robust fine-tuning
scripts/               build_reference.py (reference results from paper_code/)
tests/
paper_code/            code, configs, raw results, and analyses of the paper
```

## License

Code: [CC BY-NC 4.0](LICENSE), the same license as the dataset, for non-commercial research use. Files adapted from other projects keep their licenses ([third-party notices](THIRD_PARTY_NOTICES.md)). The images additionally follow the terms of their six source datasets (see the [dataset card](https://huggingface.co/datasets/legolasflagstaff/RobustGenBench#license)).

## Citation

```bibtex
@inproceedings{heuillet2026robustgenbench,
  title     = {RobustGenBench: A Benchmark for Robust Generalization to Adversarial and Common Perturbations, with Applications to Vision and Vision-Enabled Large Language Models},
  author    = {Heuillet, Maxime and Ngnaw{\'e}, Jonas and Pequignot, Yann and Bhagwatkar, Rishika and Larouche, Alexandre and Gagn{\'e}, Christian and Rish, Irina and Ahmad, Ola and Durand, Audrey},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  url       = {https://openreview.net/forum?id=KDsuVST8by},
  year      = {2026}
}
```
