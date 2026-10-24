"""Robustly fine-tune a timm backbone on a RobustGenBench dataset, then evaluate it.

A compact version of the paper's FFT protocol: full fine-tuning with AdamW, separate
learning rates for backbone and head, and Classic AT or TRADES (beta = 1) against
linf 4/255 perturbations. The paper crafts training perturbations with APGD-10 and
tunes hyper-parameters per configuration (paper_code/configs/HPO_results/); this
example uses PGD-10 and fixed defaults for simplicity.

    pip install "robustgenbench[torch]" && pip install git+https://github.com/fra31/auto-attack
    python examples/robust_finetune.py --arch convnext_tiny.fb_in22k --dataset flowers-102 \
        --loss trades --epochs 50
    robustgenbench leaderboard results.jsonl --compare fft50
"""
import argparse

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import robustgenbench as rgb
from robustgenbench.vision import TorchSplit, evaluate_model, num_classes, with_normalization


def pgd_linf(model, x, y, eps=4 / 255, steps=10, alpha=None, target_logits=None):
    """PGD in the linf ball; maximizes CE (Classic AT) or KL to clean logits (TRADES)."""
    alpha = alpha or 2.5 * eps / steps
    x_adv = (x + 0.001 * torch.randn_like(x)).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = (F.cross_entropy(logits, y) if target_logits is None else
                F.kl_div(F.log_softmax(logits, 1), F.softmax(target_logits, 1),
                         reduction="batchmean"))
        grad, = torch.autograd.grad(loss, x_adv)
        x_adv = torch.min(torch.max(x_adv.detach() + alpha * grad.sign(), x - eps), x + eps)
        x_adv = x_adv.clamp(0, 1)
    return x_adv.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="convnext_tiny.fb_in22k")
    ap.add_argument("--dataset", default="flowers-102", choices=rgb.DATASETS)
    ap.add_argument("--loss", choices=["trades", "classic_at"], default="trades")
    ap.add_argument("--beta", type=float, default=1.0, help="TRADES weight (paper: 1)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--eval-max-examples", type=int, default=None)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = timm.create_model(a.arch, pretrained=True, num_classes=num_classes(a.dataset))
    cfg = timm.data.resolve_data_config({}, model=net)
    model = with_normalization(net, cfg["mean"], cfg["std"]).to(device)

    # Same augmentations as the paper; images are already 224x224.
    aug = transforms.Compose([transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(0.25, 0.25, 0.25),
                              transforms.RandomRotation(2), transforms.ToTensor()])
    loader = DataLoader(TorchSplit(a.dataset, "train", transform=aug),
                        batch_size=a.batch_size, shuffle=True, num_workers=4, drop_last=True)

    head = set(id(p) for p in net.get_classifier().parameters())
    opt = torch.optim.AdamW([
        {"params": [p for p in net.parameters() if id(p) not in head], "lr": a.lr_backbone},
        {"params": list(net.get_classifier().parameters()), "lr": a.lr_head}],
        weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(loader))

    for epoch in range(a.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if a.loss == "classic_at":
                loss = F.cross_entropy(model(pgd_linf(model, x, y)), y)
            else:
                with torch.no_grad():
                    clean_logits = model(x)
                x_adv = pgd_linf(model, x, y, target_logits=clean_logits)
                logits = model(x)
                loss = F.cross_entropy(logits, y) + a.beta * F.kl_div(
                    F.log_softmax(model(x_adv), 1), F.softmax(logits, 1), reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        print(f"epoch {epoch + 1}/{a.epochs} loss={loss.item():.3f}")

    name = f"{a.arch} | {a.loss}"
    torch.save(net.state_dict(), f"{a.arch}_{a.dataset}_{a.loss}.pt")
    results = evaluate_model(model, name, datasets=[a.dataset],
                             max_examples=a.eval_max_examples)
    rgb.save(results, "results.jsonl")
    print(rgb.leaderboard(results, compare_with="fft50").head(15))


if __name__ == "__main__":
    main()
