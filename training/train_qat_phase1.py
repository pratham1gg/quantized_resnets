"""
train_qat_phase1.py
-------------------
Phase 1 QAT: pytorch-quantization INT8 fine-tuning on ImageNet-100.

Quantization scheme
-------------------
  Weights     : per-channel INT8, max calibration
  Activations : per-tensor INT8, max calibration

Usage
-----
  # Default run (15 epochs, lr=1e-4, 32 calib batches)
  python training/train_qat_phase1.py

  # Override anything
  python training/train_qat_phase1.py --epochs 20 --lr 5e-5 --calib-batches 64

  # Resume from a mid-run checkpoint
  python training/train_qat_phase1.py --resume checkpoints/qat/qat_phase1_epoch_005.pth
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

# ---------------------------------------------------------------------------
# Path setup — make src/ and src/qat/ importable
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "qat"))

# pytorch-quantization must be imported before the quantize module triggers
# quant_modules.initialize(), which happens in the call below.
from config import ExperimentConfig  # noqa: E402
from data import build_imagenet_dataset  # noqa: E402
from qat.quantize import (  # noqa: E402
    calibrate,
    get_quantized_model,
    initialize_quant_modules,
    setup_quantization_descriptors,
)
from qat.train_utils import (  # noqa: E402
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 QAT — pytorch-quantization INT8")
    p.add_argument("--data",           default=str("/home/pf4636/imagenet"),
                   help="ImageNet root containing train/ and val/")
    p.add_argument("--checkpoint",     default=str(ROOT / "checkpoints" / "best.pth"),
                   help="FP32 pretrained checkpoint to start from")
    p.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints" / "qat"),
                   help="Directory to save QAT checkpoints")
    p.add_argument("--epochs",         default=15,   type=int)
    p.add_argument("--batch-size",     default=256,  type=int)
    p.add_argument("--lr",             default=1e-4, type=float)
    p.add_argument("--momentum",       default=0.9,  type=float)
    p.add_argument("--weight-decay",   default=1e-4, type=float)
    p.add_argument("--workers",        default=8,    type=int)
    p.add_argument("--num-classes",    default=100,  type=int)
    p.add_argument("--calib-batches",  default=32,   type=int,
                   help="Number of train batches used for max calibration")
    p.add_argument("--input-quant-bits", default=8, type=int,
                   help="Input quantization bits for data loading (1,2,4,8)")
    p.add_argument("--resume",         default=None, type=str,
                   help="Path to a QAT checkpoint to resume from")
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

    device_str = device.type
    run_name   = f"resnet18_qat_int8_in{args.input_quant_bits}b_{device_str}_bs{args.batch_size}"
    args.checkpoint_dir = str(Path(args.checkpoint_dir) / run_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print(f"[Checkpoints] {args.checkpoint_dir}")

    # 1. Configure quantization descriptors and patch nn.Conv2d / nn.Linear
    #    — must happen before ResNet18 is instantiated inside get_quantized_model
    setup_quantization_descriptors()
    initialize_quant_modules()

    # 2. Build quantized model and load FP32 weights
    model = get_quantized_model(args.checkpoint, num_classes=args.num_classes).to(device)
    print(f"[Model] FP32 weights loaded from {args.checkpoint}")

    train_loader, val_loader = get_dataloaders(args)

    # 3. Calibration (skipped on resume — amax is stored in the checkpoint)
    if args.resume is None:
        calibrate(model, train_loader, args.calib_batches, device)

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
            args.resume, model, optimizer, scaler, scheduler
        )
        print(f"[Resume] Epoch {start_epoch}, best_acc={best_acc:.3f}%")

    # 4. QAT fine-tuning loop
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

    print(f"\nQAT COMPLETE — Best val top-1: {best_acc:.3f}%")
    print(f"Best weights saved to: {args.checkpoint_dir}/qat_phase1_best.pth")