"""Build the final Croissant metadata for RobustGenBench v2.1.

Structure (distribution + recordSet) is taken verbatim from the Croissant
that Hugging Face generates for the live Parquet dataset, so it always
matches the Hub layout. This script adds the dataset-level metadata HF does
not provide: authors, version, citation, and the Responsible-AI (RAI) and
provenance fields.

Usage:
    curl -sL https://huggingface.co/api/datasets/legolasflagstaff/RobustGenBench/croissant -o hf_croissant.json
    python make_croissant.py hf_croissant.json croissant.json
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src))

ctx = d["@context"]
ctx.setdefault("rai", "http://mlcommons.org/croissant/RAI/")
ctx.setdefault("prov", "http://www.w3.org/ns/prov#")

AUTHORS = [
    "Maxime Heuillet", "Jonas Ngnawé", "Yann Pequignot", "Rishika Bhagwatkar",
    "Alexandre Larouche", "Christian Gagné", "Irina Rish", "Ola Ahmad", "Audrey Durand",
]
TITLE = ("RobustGenBench: Benchmarking Robust Generalization to Unforeseen Adversarial "
         "and Common Perturbations in Vision Models and Vision-Enabled LLMs")
BIBTEX = """@inproceedings{heuillet2026robustgenbench,
  title     = {RobustGenBench: Benchmarking Robust Generalization to Unforeseen Adversarial and Common Perturbations in Vision Models and Vision-Enabled LLMs},
  author    = {Heuillet, Maxime and Ngnaw{\\'e}, Jonas and Pequignot, Yann and Bhagwatkar, Rishika and Larouche, Alexandre and Gagn{\\'e}, Christian and Rish, Irina and Ahmad, Ola and Durand, Audrey},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems Evaluations and Datasets Track},
  url       = {https://openreview.net/forum?id=KDsuVST8by},
  year      = {2026}
}"""

meta = {
    "name": "RobustGenBench",
    "description": (
        "RobustGenBench is a standardized benchmark for robust generalization of image "
        "classifiers: how accuracy holds up across perturbation types a model was not "
        "trained against. It packages six classification datasets (Caltech-101, "
        "FGVC-Aircraft, Flowers-102, Oxford-IIIT Pet, Stanford Cars, UC Merced Land Use) "
        "with fixed train/validation/test splits, images resized to 224x224, and test sets "
        "capped at 1,000 images (420 for UC Merced; 5,420 test images in total). Each "
        "dataset config holds the clean splits, a common-corruption split (11 corruption "
        "types at severity 3, assigned deterministically, one per image) and a random "
        "uniform-noise control (l-infinity, 30/255). Each <dataset>-<surrogate> config holds "
        "adversarial test images crafted with AutoAttack on an open-weight vision-language "
        "surrogate (CLIP ViT-B/16, CLIP ViT-H/14, MetaCLIP ViT-H/14, SigLIP2 SO400M, SigLIP2 "
        "base) at l1 in {75, 300}, l2 in {2, 8} and l-infinity in {4, 8, 30}/255. Because these "
        "perturbations transfer, any model, including API-only vision-enabled LLMs, can be "
        "evaluated by inference alone. Companion code: "
        "https://github.com/legolasflagstaff/robustgenbench"),
    "alternateName": ["legolasflagstaff/RobustGenBench", "RobustGenBench"],
    "version": "2.1.0",
    "datePublished": "2026-04-28",
    "dateModified": "2026-10-24",
    "license": "https://creativecommons.org/licenses/by-nc/4.0/",
    "url": "https://huggingface.co/datasets/legolasflagstaff/RobustGenBench",
    "sameAs": ["https://github.com/legolasflagstaff/robustgenbench",
               "https://legolasflagstaff.github.io/robustgenbench/"],
    "creator": [{"@type": "sc:Person", "name": n} for n in AUTHORS],
    "publisher": {"@type": "sc:Organization", "name": "Hugging Face",
                  "url": "https://huggingface.co"},
    "citeAs": BIBTEX,
    "citation": TITLE,
    "isLiveDataset": False,
    "keywords": ["robust generalization", "unforeseen robustness", "adversarial robustness",
                 "common corruptions", "transfer attacks", "AutoAttack", "image classification",
                 "vision-enabled large language models", "benchmark"],

    "rai:dataCollection": (
        "No new images were collected. The clean images come from six public datasets, "
        "organized into fixed train/validation/test splits, resized to 224x224 and stored as "
        "lossless PNG. Test sets were reduced to at most 1,000 images by uniform sampling that "
        "preserves all classes."),
    "rai:dataLimitations": [
        "Coverage is bounded by the six source datasets (objects, aircraft, flowers, pets, "
        "cars, aerial land use). Robustness measured here should not be read as robustness "
        "on open-world imagery or other domains.",
        "Adversarial images are crafted white-box on five fixed open-weight surrogates, all "
        "attention-based, and act as transfer (black-box) attacks for any other model. For a "
        "model with gradient access, they are not a substitute for white-box or adaptive "
        "evaluation, which the companion code runs on the clean test splits (as done for the "
        "fine-tuned vision models in the paper). Transfer is also weaker to architectures "
        "unlike the surrogates (e.g., convolutional networks).",
        "Surrogates only attack images they classify correctly; images the surrogate "
        "misclassifies are stored unmodified (is_modified = false). SigLIP2 SO400M adversarial "
        "splits exist only for the coarse-grained datasets (Caltech-101, Flowers-102, UC "
        "Merced), and SigLIP2 base only at l-infinity 30/255.",
        "Common corruptions are provided at a single severity (3); adversarial perturbations "
        "are provided at several budgets per norm (l1 75/300, l2 2/8, l-infinity "
        "4/8/30 /255). Budgets are not matched across norms, so cross-norm comparisons depend "
        "on the chosen intensities.",
        "Images are 224x224; results may not transfer to models evaluated at other "
        "resolutions without resizing."],
    "rai:dataBiases": [
        "The source datasets reflect the content and collection choices of their original "
        "authors (e.g., car models from a narrow period, pet breeds common in the UK, "
        "web-scraped Caltech-101 imagery with known class imbalance). These selection biases "
        "carry over to every robustness score.",
        "Transfer-attack strength varies by dataset and class with the surrogate's own "
        "zero-shot accuracy (e.g., weak SigLIP2 accuracy on fine-grained datasets), so "
        "per-class and per-dataset robustness signals are uneven."],
    "rai:personalSensitiveInformation": (
        "The dataset contains no annotated personal or sensitive attributes. Some source "
        "images may incidentally show people, faces or license plates (e.g., Caltech-101, "
        "Stanford Cars, Oxford-IIIT Pet). No anonymization beyond the source datasets was "
        "applied."),
    "rai:dataUseCases": [
        "Intended: held-out evaluation of how image classifiers (fine-tuned vision models, "
        "zero-shot models, API-only vision-enabled LLMs) generalize across common and "
        "l1/l2/l-infinity perturbations, using the protocol and metrics (Sum, Borda, MAC) of "
        "the associated paper.",
        "Intended: crafting new perturbations on the clean test splits for white-box "
        "evaluation, and robust fine-tuning on the clean train/validation splits.",
        "Not validated for: training general-purpose recognition models, fairness or "
        "demographic auditing, or robustness claims outside the six domains and the "
        "included budgets."],
    "rai:dataSocialImpact": (
        "The benchmark aims to make robustness comparisons reproducible and cheap: the "
        "pre-computed attacks remove the need for high-memory GPUs, so models can be tested "
        "before deployment. The released adversarial images could also be used to probe "
        "deployed systems; they are standard l-p transfer attacks on public datasets, the "
        "release is limited to non-commercial research use (CC BY-NC 4.0, plus the source "
        "datasets' terms), and the domains covered are narrow."),
    "rai:hasSyntheticData": True,
    "rai:hasSyntheticDataDescription": (
        "Common-corruption, random-noise and adversarial splits are procedurally perturbed "
        "versions of the natural source images (imgaug corruptions, uniform l-infinity noise, "
        "AutoAttack). No images were produced by generative models."),
    "prov:wasDerivedFrom": [
        {"@id": u, "name": n} for n, u in [
            ("Caltech-101", "https://data.caltech.edu/records/mzrjq-6wc02"),
            ("FGVC-Aircraft", "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"),
            ("Oxford 102 Flowers", "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"),
            ("Oxford-IIIT Pet", "https://www.robots.ox.ac.uk/~vgg/data/pets/"),
            ("Stanford Cars", "https://ai.stanford.edu/~jkrause/cars/car_dataset.html"),
            ("UC Merced Land Use", "http://weegee.vision.ucmerced.edu/datasets/landuse.html"),
        ]],
    "prov:wasGeneratedBy": (
        "(1) Source datasets were downloaded from their official pages and organized into "
        "fixed train/validation/test splits; test sets were capped at 1,000 images with "
        "class-preserving uniform sampling. (2) Original images were resized from their "
        "native resolutions to a common 224x224 and stored as lossless PNG. (3) Common corruptions: 11 ImageNet-C-style corruption types (imgaug) "
        "at severity 3, assigned to test images in contiguous blocks (type floor(i/s) mod 11, "
        "s = floor(N/11)), one corruption per image. (4) Random "
        "control: uniform noise in the l-infinity ball of radius 30/255, clipped to [0, 1]. "
        "(5) Adversarial: white-box AutoAttack (standard version) against zero-shot "
        "classifiers built from each surrogate ('a photo of a {class}' text prompts), released "
        "for use as transfer attacks on other models, at l1 75/300, l2 2/8, "
        "l-infinity 4/8/30 (/255), in the 224x224 input space (SigLIP2 SO400M upsamples to 384 "
        "inside its forward pass). (6) All splits are stored as Parquet. Version 2.1 restores "
        "the common split of the original processed archives (tag v1.0-archives). Code: "
        "https://github.com/legolasflagstaff/robustgenbench"),
}

# Put metadata first, then the HF-generated structure unchanged.
out = {"@context": ctx, "@type": d.get("@type", "sc:Dataset"),
       "conformsTo": d.get("conformsTo", "http://mlcommons.org/croissant/1.1")}
out.update(meta)
out["distribution"] = d["distribution"]
out["recordSet"] = d["recordSet"]
json.dump(out, open(dst, "w"), indent=2, ensure_ascii=False)
print(f"wrote {dst}: {len(out['distribution'])} distribution entries, "
      f"{len(out['recordSet'])} record sets")
