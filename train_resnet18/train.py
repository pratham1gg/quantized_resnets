"""
train.py
--------
Train ResNet-18 from scratch on ImageNet-1K.

Usage
-----
    # Single GPU, defaults
    python train.py

    # Override anything
    python train.py --epochs 90 --batch-size 256 --lr 0.1 --workers 8

    # Resume from checkpoint
    python train.py --resume checkpoints/epoch_10.pth
"""

import argparse
import os
import random
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from model import ResNet18


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ResNet-18 on ImageNet-1K")
    p.add_argument("--data",           default="/home/pf4636/imagenet")
    p.add_argument("--epochs",         default=90,   type=int)
    p.add_argument("--batch-size",     default=256,  type=int)
    p.add_argument("--lr",             default=0.1,  type=float)
    p.add_argument("--momentum",       default=0.9,  type=float)
    p.add_argument("--weight-decay",   default=1e-4, type=float)
    p.add_argument("--workers",        default=8,    type=int)
    p.add_argument("--num-classes",    default=100,  type=int)
    p.add_argument("--dropout",        default=0.0,  type=float)
    p.add_argument("--resume",         default=None, type=str)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--seed",           default=42,   type=int)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def subset_dataset(dataset: datasets.ImageFolder, num_classes: int) -> Subset:
    """Keep all samples from the first `num_classes` classes."""
    kept_classes = set(range(num_classes))
    selected = [idx for idx, (_, label) in enumerate(dataset.samples) if label in kept_classes]
    return Subset(dataset, selected)


def get_dataloaders(args: argparse.Namespace):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.ImageFolder(os.path.join(args.data, "train"), train_transform)
    val_dataset   = datasets.ImageFolder(os.path.join(args.data, "val"),   val_transform)

    if args.num_classes < 1000:
        train_dataset = subset_dataset(train_dataset, args.num_classes)
        val_dataset   = subset_dataset(val_dataset,   args.num_classes)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True,
    )

    print(f"[Data] Train: {len(train_dataset):,} samples  (classes={args.num_classes})")
    print(f"[Data] Val  : {len(val_dataset):,} samples")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
) -> tuple[float, float]:

    model.train()
    print("Training")

    running_loss    = 0.0
    running_correct = 0
    counter         = 0  # number of batches — mirrors your training_utils

    for images, labels in tqdm(loader, total=len(loader)):
        counter += 1
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):                        # mixed-precision forward
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()             # sum of per-batch mean losses
        _, preds      = torch.max(outputs.data, 1)
        running_correct += (preds == labels).sum().item()

    # Loss averaged over batches; accuracy over all samples — same as your utils
    epoch_loss = running_loss / counter
    epoch_acc  = 100. * running_correct / len(loader.dataset)
    return epoch_loss, epoch_acc


# ---------------------------------------------------------------------------
# Validate one epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:

    model.eval()
    print("Validation")

    running_loss    = 0.0
    running_correct = 0
    counter         = 0

    for images, labels in tqdm(loader, total=len(loader)):
        counter += 1
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        running_loss += loss.item()
        _, preds      = torch.max(outputs.data, 1)
        running_correct += (preds == labels).sum().item()

    epoch_loss = running_loss / counter
    epoch_acc  = 100. * running_correct / len(loader.dataset)
    return epoch_loss, epoch_acc


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, directory: str, epoch: int, is_best: bool) -> None:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"epoch_{epoch:03d}.pth")
    torch.save(state, path)
    if is_best:
        best = os.path.join(directory, "best.pth")
        torch.save(state, best)
        print(f"  [Checkpoint] Best model saved → {best}")


def load_checkpoint(path, model, optimizer, scaler, scheduler):
    print(f"[Resume] Loading {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"] + 1, ckpt.get("best_acc", 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args   = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    train_loader, val_loader = get_dataloaders(args)

    model  = ResNet18(num_classes=args.num_classes, pretrained=False).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Total params    : {total:,}")
    print(f"[Model] Trainable params: {trainable:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Cosine annealing: lr decays smoothly from args.lr → ~0 over all epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler()

    start_epoch = 1
    best_acc    = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            args.resume, model, optimizer, scaler, scheduler
        )

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist,  val_acc_hist  = [], []

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n[INFO]: Epoch {epoch} of {args.epochs}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        print(f"Training   loss: {train_loss:.3f}, acc: {train_acc:.3f}%")
        print(f"Validation loss: {val_loss:.3f}, acc: {val_acc:.3f}%")
        print("-" * 50)

        is_best  = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint(
            state={
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler":    scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc":  best_acc,
            },
            directory=args.checkpoint_dir,
            epoch=epoch,
            is_best=is_best,
        )

    print(f"\nTRAINING COMPLETE — Best val acc: {best_acc:.3f}%")
    print(f"Best weights saved at: {args.checkpoint_dir}/best.pth")