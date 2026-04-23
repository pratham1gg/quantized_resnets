"""
train_qat_modelopt.py
---------------------
ModelOpt QAT fine-tuning for ResNet-18 on ImageNet-100.

Supports INT8 and INT4 via NVIDIA ModelOpt (mtq.quantize):
  int8 → INT8_DEFAULT_CFG   (per-channel W8 + per-tensor A8)
  int4 → INT4_BLOCKWISE_WEIGHT_ONLY_CFG  (blockwise W4, activations in fp)

Usage
-----
  # INT8, default settings
  python training/train_qat_modelopt.py --precision int8

  # INT4
  python training/train_qat_modelopt.py --precision int4

  # Custom run
  python training/train_qat_modelopt.py --precision int8 --epochs 20 --lr 5e-5

  # Resume
  python training/train_qat_modelopt.py --precision int8 \\
      --resume checkpoints/qat_modelopt/resnet18_modelopt_int8_in8b_cuda_bs256/qat_modelopt_epoch_005.pth \\
      --resume-mostate checkpoints/qat_modelopt/resnet18_modelopt_int8_in8b_cuda_bs256/qat_modelopt_epoch_005_mostate.pt
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "qat_modelopt"))

from config import ExperimentConfig
from data import build_imagenet_dataset
from qat_modelopt.quantize import get_model, get_quant_cfg, quantize_model
from qat_modelopt.train_utils import (
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ModelOpt QAT — INT8 / INT4")
    p.add_argument("--precision",       default="int8", choices=["int8", "int4"],
                   help="Quantization precision: int8 or int4")
    p.add_argument("--data",            default="/home/pf4636/imagenet",
                   help="ImageNet root containing train/ and val/")
    p.add_argument("--checkpoint",      default=str(ROOT / "checkpoints" / "best.pth"),
                   help="FP32 pretrained checkpoint to start from")
    p.add_argument("--checkpoint-dir",  default=str(ROOT / "checkpoints" / "qat_modelopt"),
                   help="Root directory for QAT checkpoints")
    p.add_argument("--epochs",          default=15,   type=int)
    p.add_argument("--batch-size",      default=256,  type=int)
    p.add_argument("--lr",              default=1e-4, type=float)
    p.add_argument("--momentum",        default=0.9,  type=float)
    p.add_argument("--weight-decay",    default=1e-4, type=float)
    p.add_argument("--workers",         default=8,    type=int)
    p.add_argument("--num-classes",     default=100,  type=int)
    p.add_argument("--calib-batches",   default=32,   type=int,
                   help="Calibration batches fed to mtq.quantize()")
    p.add_argument("--input-quant-bits", default=8,  type=int,
                   help="Input quantization bits for data transforms (1,2,4,8)")
    p.add_argument("--resume",          default=None, type=str,
                   help="Path to a .pth training checkpoint to resume from")
    p.add_argument("--resume-mostate",  default=None, type=str,
                   help="Path to the matching _mostate.pt file (required with --resume)")
    p.add_argument("--seed",            default=42,   type=int)
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
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _subset(dataset, num_classes: int) -> Subset:
    kept    = set(range(num_classes))
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label in kept]
    return Subset(dataset, indices)


def get_dataloaders(args: argparse.Namespace):
    cfg = ExperimentConfig(
        imagenet_path=args.data,
        batch_size=args.batch_size,
        num_workers=args.workers,
        input_quant_bits=args.input_quant_bits,
    )

    train_ds = build_imagenet_dataset(cfg, "train")
    val_ds   = build_imagenet_dataset(cfg, "val")

    if args.num_classes < 1000:
        train_ds = _subset(train_ds, args.num_classes)
        val_ds   = _subset(val_ds,   args.num_classes)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    print(f"[Data] Train: {len(train_ds):,}  Val: {len(val_ds):,}  "
          f"(num_classes={args.num_classes})")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    if args.resume and not args.resume_mostate:
        raise ValueError("--resume requires --resume-mostate (path to _mostate.pt)")

    device_str = device.type
    run_name   = (f"resnet18_modelopt_{args.precision}"
                  f"_in{args.input_quant_bits}b_{device_str}_bs{args.batch_size}")
    run_dir    = Path(args.checkpoint_dir) / run_name
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Checkpoints] {run_dir}")

    # 1. Load plain FP32 model
    model = get_model(args.checkpoint, num_classes=args.num_classes)
    print(f"[Model] FP32 weights loaded from {args.checkpoint}")

    train_loader, val_loader = get_dataloaders(args)

    # 2. Quantize (calibrate + insert fake-quant) or restore from checkpoint
    quant_cfg = get_quant_cfg(args.precision)
    if args.resume is None:
        model = quantize_model(model, quant_cfg, train_loader, args.calib_batches, device)
    else:
        # Restore quantizer structure first, then load weights in load_checkpoint
        model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = GradScaler()

    start_epoch = 1
    best_acc    = 0.0
    if args.resume:
        start_epoch, best_acc = load_checkpoint(
            ckpt_path=args.resume,
            mo_path=args.resume_mostate,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
        )
        print(f"[Resume] Epoch {start_epoch}, best_acc={best_acc:.3f}%")

    # 3. QAT fine-tuning loop
    for epoch in range(start_epoch, args.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch}/{args.epochs}]  lr={lr:.2e}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_loss, val_top1, val_top5 = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val    loss={val_loss:.4f}  top1={val_top1:.2f}%  top5={val_top5:.2f}%")
        print("-" * 60)

        is_best  = val_top1 > best_acc
        best_acc = max(val_top1, best_acc)

        save_checkpoint(
            model=model,
            state={
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler":    scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc":  best_acc,
            },
            directory=str(run_dir),
            epoch=epoch,
            is_best=is_best,
        )

    print(f"\nQAT COMPLETE — Best val top-1: {best_acc:.3f}%")
    print(f"Best weights saved to: {run_dir}/qat_modelopt_best.pth")
