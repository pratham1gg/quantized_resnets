"""PT2E INT8 QAT for ResNet18 (TRT-compatible Q/DQ scheme)."""

import argparse
import copy
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_FP32_CKPT = ROOT / "checkpoints" / "best.pth"
DEFAULT_QAT_DIR   = ROOT / "checkpoints" / "qat_pt2e"

import numpy as np
import PIL.ImageFile
PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.fx as fx
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
from torchao.quantization.pt2e import (
    move_exported_model_to_train,
    move_exported_model_to_eval,
    disable_observer,
)

from torch.ao.quantization.quantizer import (
    Quantizer,
    QuantizationAnnotation,
    QuantizationSpec,
)
from torch.ao.quantization import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)

from data import build_train_holdout_split
from model import ResNet18


# ----------------------------- Quant specs -----------------------------

ACT_SPEC = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MovingAverageMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
        quant_min=-128,
        quant_max=127,
    ),
)

WEIGHT_SPEC = QuantizationSpec(
    dtype=torch.int8,
    quant_min=-127,
    quant_max=127,
    qscheme=torch.per_channel_symmetric,
    ch_axis=0,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=-127,
        quant_max=127,
        ch_axis=0,
    ),
)


# ----------------------------- Quantizer -------------------------------

CONV_TARGETS = {
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
}
LINEAR_TARGETS = {
    torch.ops.aten.linear.default,
}
ADDMM_TARGETS = {
    torch.ops.aten.addmm.default,
}
RESIDUAL_ADD_TARGETS = {
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add_.Tensor,
}


class TRTInt8Quantizer(Quantizer):
    """Annotates the aten graph to match modelopt's INT8 QDQ scheme.

    Conv/Linear: input act + weight (per-channel symmetric) + output act.
    Residual Add: output act only.
    """

    def annotate(self, model):
        for node in model.graph.nodes:
            if node.op != "call_function":
                continue
            existing = node.meta.get("quantization_annotation", None)
            if existing is not None and getattr(existing, "_annotated", False):
                continue

            target = node.target

            if target in CONV_TARGETS or target in LINEAR_TARGETS:
                input_node  = node.args[0]
                weight_node = node.args[1]
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={
                        input_node:  ACT_SPEC,
                        weight_node: WEIGHT_SPEC,
                    },
                    output_qspec=ACT_SPEC,
                    _annotated=True,
                )

            elif target in ADDMM_TARGETS:
                # addmm signature: (bias, input, weight)
                input_node  = node.args[1]
                weight_node = node.args[2]
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={
                        input_node:  ACT_SPEC,
                        weight_node: WEIGHT_SPEC,
                    },
                    output_qspec=ACT_SPEC,
                    _annotated=True,
                )

            elif target in RESIDUAL_ADD_TARGETS:
                # Only annotate residual adds — both operands must be fx.Node tensors.
                if (
                    len(node.args) >= 2
                    and isinstance(node.args[0], fx.Node)
                    and isinstance(node.args[1], fx.Node)
                ):
                    node.meta["quantization_annotation"] = QuantizationAnnotation(
                        output_qspec=ACT_SPEC,
                        _annotated=True,
                    )

        return model

    def validate(self, model):
        return None


# ----------------------------- Utilities -------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, val_tf


def get_dataloaders(args):
    train_tf, val_tf = build_transforms()
    train_ds, val_ds = build_train_holdout_split(
        data_root=args.data,
        num_classes=args.num_classes,
        seed=args.seed,
        train_transform=train_tf,
        eval_transform=val_tf,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False,
    )
    return train_loader, val_loader


@torch.no_grad()
def evaluate(model, loader, device, desc="eval"):
    correct1 = 0
    correct5 = 0
    total = 0
    for images, labels in tqdm(loader, total=len(loader), desc=desc):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        _, top5 = logits.topk(5, dim=1)
        correct1 += (top5[:, 0] == labels).sum().item()
        correct5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total    += labels.size(0)
    return 100.0 * correct1 / total, 100.0 * correct5 / total


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    move_exported_model_to_train(model)
    running_loss = 0.0
    running_correct = 0
    total = 0
    pbar = tqdm(loader, total=len(loader), desc=f"train e{epoch}")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss   += loss.item() * labels.size(0)
        running_correct += (logits.argmax(1) == labels).sum().item()
        total          += labels.size(0)
        pbar.set_postfix(loss=running_loss / max(total, 1), acc=100. * running_correct / max(total, 1))
    return running_loss / total, 100.0 * running_correct / total


def freeze_bn_stats(prepared_model) -> int:
    """Set BN training flag to False in the graph. Returns number of nodes patched."""
    bn_targets = {
        torch.ops.aten._native_batch_norm_legit.default,
        torch.ops.aten.cudnn_batch_norm.default,
    }
    patched = 0
    for n in prepared_model.graph.nodes:
        if n.target in bn_targets:
            new_args = list(n.args)
            if len(new_args) > 5:
                new_args[5] = False
                n.args = tuple(new_args)
                patched += 1
    if patched:
        prepared_model.recompile()
    return patched


def count_fake_quants(prepared_model) -> int:
    n = 0
    for mod in prepared_model.modules():
        if mod.__class__.__name__.endswith("FakeQuantize") or mod.__class__.__name__.endswith("FakeQuantizeBase"):
            n += 1
    return n


# ----------------------------- Main ------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="PT2E INT8 QAT for ResNet18")
    p.add_argument("--data",          default="/home/pf4636/imagenet")
    p.add_argument("--fp32-ckpt",     default=str(DEFAULT_FP32_CKPT))
    p.add_argument("--out-dir",       default=str(DEFAULT_QAT_DIR))
    p.add_argument("--epochs",        default=10,    type=int)
    p.add_argument("--batch-size",    default=32,    type=int)
    p.add_argument("--lr",            default=1e-5,  type=float)
    p.add_argument("--weight-decay",  default=1e-5,  type=float)
    p.add_argument("--workers",       default=8,     type=int)
    p.add_argument("--num-classes",   default=100,   type=int)
    p.add_argument("--seed",          default=42,    type=int)
    p.add_argument("--observer-freeze-after", default=4, type=int,
                   help="Disable observers after this epoch (1-indexed).")
    p.add_argument("--bn-freeze-after",       default=3, type=int,
                   help="Freeze BN stats after this epoch (1-indexed).")
    p.add_argument("--eval-every",            default=2, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type != "cuda":
        raise RuntimeError("PT2E QAT requires CUDA for cudnn_batch_norm export.")

    train_loader, val_loader = get_dataloaders(args)

    # ---- Load FP32 model ----
    model = ResNet18(num_classes=args.num_classes, pretrained=False)
    ckpt  = torch.load(args.fp32_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model = model.to(device)

    # ---- FP32 baseline ----
    model.eval()
    fp32_top1, fp32_top5 = evaluate(model, val_loader, device, desc="fp32")
    print(f"[FP32 baseline] top1={fp32_top1:.3f}%  top5={fp32_top5:.3f}%")

    # ---- Export to aten IR ----
    model.train()
    example_inputs = (torch.randn(2, 3, 224, 224, device=device),)
    exported_model = torch.export.export_for_training(model, example_inputs).module()

    bn_op_name = None
    for n in exported_model.graph.nodes:
        if n.op == "call_function" and "batch_norm" in str(n.target):
            bn_op_name = str(n.target)
            break
    print(f"[Export] First BN op in graph: {bn_op_name}")

    # Swap _native_batch_norm_legit → cudnn_batch_norm if needed for better QAT numerics.
    swapped = 0
    for n in exported_model.graph.nodes:
        if n.target == torch.ops.aten._native_batch_norm_legit.default:
            n.target = torch.ops.aten.cudnn_batch_norm.default
            swapped += 1
    if swapped:
        exported_model.recompile()
        print(f"[Export] Swapped {swapped} _native_batch_norm_legit → cudnn_batch_norm")

    # ---- Prepare for QAT ----
    quantizer = TRTInt8Quantizer()
    prepared_model = prepare_qat_pt2e(exported_model, quantizer)
    n_fq = count_fake_quants(prepared_model)
    print(f"[Prepare] FakeQuantize modules inserted: {n_fq} (modelopt ref: 41 Q+DQ pairs)")

    # ---- Optimizer / scheduler / criterion ----
    optimizer = optim.AdamW(
        prepared_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---- Training loop ----
    best_q_top1 = -1.0
    best_prepared_sd = None
    best_epoch = -1
    bn_frozen = False
    obs_frozen = False

    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}] lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_acc = train_one_epoch(
            prepared_model, train_loader, criterion, optimizer, device, epoch,
        )
        print(f"  train loss={train_loss:.4f}  acc={train_acc:.3f}%")

        if epoch > args.observer_freeze_after and not obs_frozen:
            prepared_model.apply(disable_observer)
            obs_frozen = True
            print(f"  [Freeze] Observers disabled (epoch {epoch})")

        if epoch > args.bn_freeze_after and not bn_frozen:
            patched = freeze_bn_stats(prepared_model)
            bn_frozen = True
            print(f"  [Freeze] BN stats frozen — patched {patched} BN nodes (epoch {epoch})")

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model_copy = copy.deepcopy(prepared_model)
            move_exported_model_to_eval(model_copy)
            quantized = convert_pt2e(model_copy)
            q_top1, q_top5 = evaluate(quantized, val_loader, device, desc=f"q-eval e{epoch}")
            print(f"  [INT8 eval] top1={q_top1:.3f}%  top5={q_top5:.3f}%")

            if q_top1 > best_q_top1:
                best_q_top1 = q_top1
                best_epoch = epoch
                best_prepared_sd = copy.deepcopy(prepared_model.state_dict())
                print(f"  [Best] new best INT8 top1={q_top1:.3f}% @ epoch {epoch}")

            del model_copy, quantized

        scheduler.step()

    # ---- Restore best, convert, save ----
    if best_prepared_sd is not None:
        prepared_model.load_state_dict(best_prepared_sd)
        print(f"\n[Restore] loaded best prepared state from epoch {best_epoch}")
    else:
        best_prepared_sd = copy.deepcopy(prepared_model.state_dict())

    move_exported_model_to_eval(prepared_model)
    quantized_model = convert_pt2e(prepared_model)

    final_top1, final_top5 = evaluate(quantized_model, val_loader, device, desc="final")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    converted_path = out_dir / "best_converted.pt"
    prepared_path  = out_dir / "best_prepared.pt"
    torch.save(quantized_model.state_dict(), converted_path)
    torch.save(best_prepared_sd,             prepared_path)

    # ---- Summary ----
    delta = final_top1 - fp32_top1
    print("\n" + "=" * 60)
    print(f"FP32 baseline:       top1={fp32_top1:6.3f}%  top5={fp32_top5:6.3f}%")
    print(f"INT8 QAT (best):     top1={best_q_top1:6.3f}%  (epoch {best_epoch})")
    print(f"INT8 QAT (final):    top1={final_top1:6.3f}%  top5={final_top5:6.3f}%")
    print(f"Accuracy delta:      {delta:+.3f}%")
    print(f"Checkpoints saved:   {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
