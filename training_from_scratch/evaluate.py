"""
Evaluate a trained ResNet18 checkpoint on ImageNet-1k validation set.

Usage
-----
    python evaluate.py --checkpoint ./checkpoints/best_model.pth
    python evaluate.py --checkpoint ./checkpoints/checkpoint_epoch_90.pth --batch-size 128
"""

import sys, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ── make src/ importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import ResNet18

# ── constants ───────────────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:, :k].reshape(-1).float().sum(0) * 100.0 / batch_size)
    return res


@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    n_batches  = 0

    for i, (images, targets) in enumerate(val_loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, targets)

        acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
        total_loss += loss.item()
        total_top1 += acc1.item()
        total_top5 += acc5.item()
        n_batches  += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(val_loader)}]  "
                  f"top-1 {total_top1/n_batches:.2f}%  "
                  f"top-5 {total_top5/n_batches:.2f}%")

    avg_loss = total_loss / n_batches
    avg_top1 = total_top1 / n_batches
    avg_top5 = total_top5 / n_batches

    print("=" * 50)
    print(f"Loss:          {avg_loss:.4f}")
    print(f"Top-1 Accuracy: {avg_top1:.2f}%")
    print(f"Top-5 Accuracy: {avg_top5:.2f}%")
    print("=" * 50)

    return avg_top1, avg_top5, avg_loss


def main():
    p = argparse.ArgumentParser(description="Evaluate trained ResNet18 on ImageNet val")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--data-dir",    type=str, default="/home/pf4636/imagenet")
    p.add_argument("--batch-size",  type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device",      type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)

    # model
    model = ResNet18(num_classes=1000, pretrained=False).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded {args.checkpoint}  (epoch {ckpt['epoch']+1}, best {ckpt.get('best_acc', '?')}%)")

    # data
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])
    dataset = torchvision.datasets.ImageNet(root=args.data_dir, split="val", transform=transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    evaluate(model, loader, device)


if __name__ == "__main__":
    main()
