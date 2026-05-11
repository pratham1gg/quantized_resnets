import argparse
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "qat_modelopt"))

import numpy as np
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from quantize import get_quant_cfg, get_model, quantize_model, restore_modelopt_state
from train_utils import train_one_epoch, validate, save_checkpoint, load_training_state
from data import build_train_holdout_split

BEST_CHECKPOINT_PATH = ROOT / "checkpoints" / "best.pth"
DEFAULT_OUTPUT_DIR   = ROOT / "checkpoints" / "qat"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="INT8/INT4 QAT for ResNet-18 using ModelOpt")
    p.add_argument("--data",              default="/home/pf4636/imagenet")
    p.add_argument("--checkpoint",        default=str(BEST_CHECKPOINT_PATH),
                   help="FP32 checkpoint to start QAT from")
    p.add_argument("--output-dir",        default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--precision",         choices=["int8", "int4"], default="int8")
    p.add_argument("--epochs",            type=int,   default=15)
    p.add_argument("--batch-size",        type=int,   default=64)
    p.add_argument("--num-workers",       type=int,   default=8)
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--momentum",          type=float, default=0.9)
    p.add_argument("--weight-decay",      type=float, default=1e-4)
    p.add_argument("--num-calib-batches", type=int,   default=8,
                   help="Calibration batches (~512 images at default batch-size=64)")
    p.add_argument("--num-classes",       type=int,   default=100)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--gpu",               type=int,   default=None)
    p.add_argument("--resume",            type=str,   default=None,
                   help="Resume from a QAT .pth checkpoint")
    p.add_argument("--resume-mostate",    type=str,   default=None,
                   help="Paired _mostate.pt for --resume (inferred from --resume path if omitted)")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _subset(dataset, num_classes: int) -> Subset:
    indices = [i for i, (_, label) in enumerate(dataset.samples) if label < num_classes]
    return Subset(dataset, indices)


def get_dataloaders(args):
    normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
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

    train_subset, holdout_subset = build_train_holdout_split(
        data_root=args.data,
        num_classes=args.num_classes,
        seed=args.seed,
        train_transform=train_transform,
        eval_transform=val_transform,
    )

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        holdout_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    # Calibration draws from train_subset to avoid leaking holdout-val statistics.
    # num_workers=0 avoids multiprocessing issues during mtq.quantize calibration.
    calib_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader, calib_loader


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu is not None
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[Device] {device}")

    train_loader, val_loader, calib_loader = get_dataloaders(args)

    model = get_model(args.checkpoint, num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    quant_cfg = get_quant_cfg(args.precision)

    if args.resume:
        mo_path = args.resume_mostate or args.resume.replace(".pth", "_mostate.pt")
        print(f"[Resume] Restoring modelopt quantization state from {mo_path}")
        restore_modelopt_state(model, mo_path)
        model = model.to(device)
        ptq_top1 = 0.0
    else:
        print("[FP32] Evaluating baseline ...")
        _, fp32_top1, fp32_top5 = validate(model, val_loader, criterion, device)
        print(f"[FP32] Top-1: {fp32_top1:.2f}%  Top-5: {fp32_top5:.2f}%")

        print(f"[PTQ] Calibrating for {args.precision.upper()} ...")
        model = quantize_model(model, quant_cfg, calib_loader, args.num_calib_batches, device)

        _, ptq_top1, ptq_top5 = validate(model, val_loader, criterion, device)
        print(f"[PTQ] Top-1: {ptq_top1:.2f}%  Top-5: {ptq_top5:.2f}%")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    scaler = GradScaler()

    start_epoch = 1
    best_acc = ptq_top1
    if args.resume:
        start_epoch, best_acc = load_training_state(
            args.resume, model, optimizer, scaler, scheduler
        )

    print(f"[QAT] Fine-tuning for {args.epochs} epochs ...")
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n[INFO] Epoch {epoch}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.6f}")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        scheduler.step()
        val_loss, top1, top5 = validate(model, val_loader, criterion, device)
        print(f"  Train loss: {train_loss:.4f}  acc: {train_acc:.2f}%")
        print(f"  Val   loss: {val_loss:.4f}  top-1: {top1:.2f}%  top-5: {top5:.2f}%")

        is_best  = top1 > best_acc
        best_acc = max(top1, best_acc)

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
            directory=args.output_dir,
            epoch=epoch,
            is_best=is_best,
        )

    print(f"\nQAT COMPLETE — Best val top-1: {best_acc:.2f}%")
    print(f"Best weights: {os.path.join(args.output_dir, 'qat_modelopt_best.pth')}")
