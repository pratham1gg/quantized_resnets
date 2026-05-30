"""
QAT training with INT4 weights + INT4 activations.

This is an experimental config — INT4 activations (only 16 discrete levels)
are expected to cause significant accuracy degradation.

Usage:
    python misc/qat_training_int4act.py --input-quant-bits 8 --seed 42
    python misc/qat_training_int4act.py --input-quant-bits 4 --seed 1
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

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pyfiles"))
sys.path.insert(0, str(ROOT / "pyfiles" / "qat_modelopt"))

from src.config import ExperimentConfig
from src.data import build_imagenet_dataset
from train_utils import (
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate,
)
from quantize import get_model

DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints" / "qat"

INT4_ACT_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {"num_bits": 4, "axis": 0},
        "*input_quantizer": {"num_bits": 4, "axis": None},
    },
    "algorithm": "max",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QAT — INT4 weights + INT4 activations")
    p.add_argument("--data",            default="/home/pf4636/imagenet")
    p.add_argument("--checkpoint",      default=None,
                   help="FP32 pretrained checkpoint (default: checkpoints/fp32_{bits}bit/seed_{seed}/best.pth)")
    p.add_argument("--checkpoint-dir",  default=str(DEFAULT_CHECKPOINT_DIR))
    p.add_argument("--epochs",          default=15,   type=int)
    p.add_argument("--batch-size",      default=256,  type=int)
    p.add_argument("--lr",              default=1e-4, type=float)
    p.add_argument("--momentum",        default=0.9,  type=float)
    p.add_argument("--weight-decay",    default=1e-4, type=float)
    p.add_argument("--workers",         default=8,    type=int)
    p.add_argument("--num-classes",     default=100,  type=int)
    p.add_argument("--calib-batches",   default=32,   type=int)
    p.add_argument("--input-quant-bits", default=8,   type=int)
    p.add_argument("--resume",          default=None, type=str)
    p.add_argument("--resume-mostate",  default=None, type=str)
    p.add_argument("--seed",            default=42,   type=int)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


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


def quantize_model(model, calib_loader, num_calib_batches, device):
    model = model.to(device)

    def forward_loop(m: nn.Module) -> None:
        m.eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(calib_loader):
                if i >= num_calib_batches:
                    break
                m(images.to(device))

    print(f"[Calibration] Running {num_calib_batches} batches for INT4w+INT4a quantizer init ...")
    model = mtq.quantize(model, INT4_ACT_CFG, forward_loop)
    print("[Calibration] Done — fake-quant active (INT4 weights + INT4 activations).")
    return model


if __name__ == "__main__":
    args = parse_args()
    if args.checkpoint is None:
        args.checkpoint = str(
            ROOT / "checkpoints" / f"fp32_{args.input_quant_bits}bit" / f"seed_{args.seed}" / "best.pth"
        )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    if args.resume and not args.resume_mostate:
        raise ValueError("--resume requires --resume-mostate (path to _mostate.pt)")

    run_name = f"int4act_in{args.input_quant_bits}b"
    run_dir  = Path(args.checkpoint_dir) / run_name / f"seed_{args.seed}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Checkpoints] {run_dir}")

    model = get_model(args.checkpoint, num_classes=args.num_classes)
    print(f"[Model] FP32 weights loaded from {args.checkpoint}")

    train_loader, val_loader = get_dataloaders(args)

    if args.resume is None:
        model = quantize_model(model, train_loader, args.calib_batches, device)
    else:
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
