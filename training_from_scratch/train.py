"""
Train the custom ResNet18 (from src/model.py) on ImageNet-1k from scratch.

Saves checkpoints to  ./checkpoints/
  • checkpoint_epoch_<N>.pth   – every epoch
  • best_model.pth             – best val top-1

Usage
-----
    cd training_from_scratch
    python train.py                          # defaults: 90 epochs, bs 256, SGD 0.1
    python train.py --epochs 100 --lr 0.05   # override
    python train.py --resume ./checkpoints/checkpoint_epoch_50.pth
"""

import os, sys, time, argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# ── make src/ importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import ResNet18                          # your custom arch

# ── ImageNet constants ──────────────────────────────────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

# ── helpers ─────────────────────────────────────────────────────────────

def _build_train_transform():
    """Standard ImageNet training augmentation."""
    return T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])

def _build_val_transform():
    """Standard ImageNet validation transform."""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=_MEAN, std=_STD),
    ])

def _make_loader(data_root: str, split: str, batch_size: int,
                 num_workers: int, is_train: bool) -> DataLoader:
    transform = _build_train_transform() if is_train else _build_val_transform()
    dataset = torchvision.datasets.ImageNet(
        root=data_root, split=split, transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )

@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor,
             topk=(1,)):
    """Compute top-k accuracy (returns list of %-values)."""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:, :k].reshape(-1).float().sum(0) * 100.0 / batch_size)
    return res

# ── Trainer ─────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device = device
        self.args   = args

        self.criterion = nn.CrossEntropyLoss()

        # optimiser
        if args.optimizer == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(), lr=args.lr,
                momentum=args.momentum, weight_decay=args.weight_decay,
            )
        elif args.optimizer == "adam":
            self.optimizer = optim.Adam(
                model.parameters(), lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")

        # scheduler
        if args.scheduler == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=args.epochs, eta_min=args.min_lr,
            )
        elif args.scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer, step_size=args.step_size, gamma=args.gamma,
            )
        else:
            self.scheduler = None

        self.checkpoint_dir = Path(args.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_acc   = 0.0
        self.best_epoch = 0

    # ── one training epoch ──────────────────────────────────────────────
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_top1 = 0.0
        n_batches  = 0

        t0 = time.time()
        for i, (images, targets) in enumerate(self.train_loader):
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_top1 += accuracy(logits, targets, topk=(1,))[0].item()
            n_batches  += 1

            if (i + 1) % self.args.log_interval == 0:
                print(f"  [{i+1}/{len(self.train_loader)}]  "
                      f"loss {total_loss/n_batches:.4f}  "
                      f"top-1 {total_top1/n_batches:.2f}%")

        elapsed = time.time() - t0
        avg_loss = total_loss / n_batches
        avg_acc  = total_top1 / n_batches
        print(f"Epoch {epoch+1} train  loss={avg_loss:.4f}  "
              f"top-1={avg_acc:.2f}%  ({elapsed:.0f}s)")
        return avg_loss, avg_acc

    # ── validation ──────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        n_batches  = 0

        for images, targets in self.val_loader:
            images  = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss   = self.criterion(logits, targets)

            acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
            total_loss += loss.item()
            total_top1 += acc1.item()
            total_top5 += acc5.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches
        avg_top1 = total_top1 / n_batches
        avg_top5 = total_top5 / n_batches
        print(f"         val    loss={avg_loss:.4f}  "
              f"top-1={avg_top1:.2f}%  top-5={avg_top5:.2f}%")
        return avg_loss, avg_top1, avg_top5

    # ── checkpoint ──────────────────────────────────────────────────────
    def save_checkpoint(self, epoch, is_best=False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        torch.save(state, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(state, best_path)
            print(f"  ★ new best saved → {best_path}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_acc = ckpt.get("best_acc", 0.0)
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {path}  (epoch {start_epoch}, best {self.best_acc:.2f}%)")
        return start_epoch

    # ── main loop ───────────────────────────────────────────────────────
    def train(self, start_epoch=0):
        print(f"Training on {self.device}  |  epochs {start_epoch+1}→{self.args.epochs}  "
              f"bs={self.args.batch_size}  lr={self.args.lr}  opt={self.args.optimizer}")
        print("=" * 60)

        wall_t0 = time.time()
        for epoch in range(start_epoch, self.args.epochs):
            self.train_epoch(epoch)
            _, val_top1, _ = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            is_best = val_top1 > self.best_acc
            if is_best:
                self.best_acc   = val_top1
                self.best_epoch = epoch
            self.save_checkpoint(epoch, is_best=is_best)

            lr_now = self.optimizer.param_groups[0]["lr"]
            print(f"  lr={lr_now:.6f}  best so far={self.best_acc:.2f}% (ep {self.best_epoch+1})\n")

        hours = (time.time() - wall_t0) / 3600
        print("=" * 60)
        print(f"Done — best top-1 {self.best_acc:.2f}% @ epoch {self.best_epoch+1}  "
              f"({hours:.1f} h)")


# ── CLI ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet18 from scratch on ImageNet-1k")

    # training
    p.add_argument("--epochs",       type=int,   default=90)
    p.add_argument("--batch-size",   type=int,   default=256)
    p.add_argument("--lr",           type=float, default=0.1)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer",    type=str,   default="sgd", choices=["sgd", "adam"])

    # scheduler
    p.add_argument("--scheduler",    type=str,   default="step", choices=["cosine", "step", "none"])
    p.add_argument("--step-size",    type=int,   default=30,  help="for StepLR")
    p.add_argument("--gamma",        type=float, default=0.1, help="for StepLR")
    p.add_argument("--min-lr",       type=float, default=0.0, help="for CosineAnnealing")

    # data
    p.add_argument("--data-dir",     type=str,   default="/home/pf4636/imagenet")
    p.add_argument("--num-workers",  type=int,   default=8)

    # misc
    p.add_argument("--device",         type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--log-interval",   type=int, default=100)
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    p.add_argument("--resume",         type=str, default=None, help="path to checkpoint to resume")
    p.add_argument("--seed",           type=int, default=42)

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # model — pretrained=False → uses Kaiming init from _init_weights()
    print("Creating ResNet18 (from scratch) …")
    model = ResNet18(num_classes=1000, pretrained=False).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    # data
    print("Building data loaders …")
    train_loader = _make_loader(args.data_dir, "train", args.batch_size, args.num_workers, is_train=True)
    val_loader   = _make_loader(args.data_dir, "val",   args.batch_size, args.num_workers, is_train=False)
    print(f"  train: {len(train_loader)} batches  |  val: {len(val_loader)} batches")

    # trainer
    trainer = Trainer(model, train_loader, val_loader, device, args)

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)

    trainer.train(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
