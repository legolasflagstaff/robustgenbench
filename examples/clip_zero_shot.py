"""Plug a custom black-box model into RobustGenBench: zero-shot CLIP.

The only requirement is a ``predict(image, class_names)`` method. Here it returns a
class index; returning free text (e.g. an LLM answer) works too and is parsed.

    pip install "robustgenbench[torch]" open_clip_torch
    python examples/clip_zero_shot.py --datasets uc-merced-land-use-dataset --max-examples 100
"""
import argparse

import open_clip
import torch

import robustgenbench as rgb


class CLIPZeroShot(rgb.BlackBoxClassifier):
    def __init__(self, arch="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(arch)
        self._text = {}

    @torch.no_grad()
    def _text_features(self, class_names):
        key = tuple(class_names)
        if key not in self._text:
            tokens = self.tokenizer([f"a photo of a {c}" for c in class_names]).to(self.device)
            self._text[key] = torch.nn.functional.normalize(
                self.model.encode_text(tokens), dim=-1)
        return self._text[key]

    @torch.no_grad()
    def predict(self, image, class_names):
        x = self.preprocess(image).unsqueeze(0).to(self.device)
        img = torch.nn.functional.normalize(self.model.encode_image(x), dim=-1)
        return int((img @ self._text_features(class_names).T).argmax())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(rgb.DATASETS))
    ap.add_argument("--max-examples", type=int, default=None)
    a = ap.parse_args()

    results = rgb.evaluate_classifier(CLIPZeroShot(), "clip-vit-b32-zeroshot",
                                      datasets=a.datasets, max_examples=a.max_examples)
    rgb.save(results, "results.jsonl")
    print(rgb.leaderboard(results, compare_with="vellm"))
