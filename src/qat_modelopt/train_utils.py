
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_one_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    epoch:     int,
) -> tuple[float, float]:
    model.train()
    running_loss    = 0.0
    running_correct = 0
    n_seen          = 0
    n_batches       = 0

    for images, labels in tqdm(loader, desc=f"Epoch {epoch} [train]"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss    += loss.item()
        _, preds         = torch.max(outputs.detach(), 1)
        running_correct += (preds == labels).sum().item()
        n_seen          += labels.size(0)
        n_batches       += 1

    epoch_loss = running_loss / n_batches
    epoch_acc  = 100.0 * running_correct / n_seen
    return epoch_loss, epoch_acc

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float, float]:
    model.eval()
    running_loss    = 0.0
    running_correct = 0
    top5_correct    = 0
    n_batches       = 0

    for images, labels in tqdm(loader, desc="[val]"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda"):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        running_loss    += loss.item()
        _, preds         = torch.max(outputs, 1)
        running_correct += (preds == labels).sum().item()
        _, top5          = outputs.topk(5, dim=1)
        top5_correct    += (top5 == labels.view(-1, 1)).sum().item()
        n_batches       += 1

    n        = len(loader.dataset)
    val_loss = running_loss / n_batches
    top1     = 100.0 * running_correct / n
    top5     = 100.0 * top5_correct    / n
    return val_loss, top1, top5

def save_checkpoint(
    model:     nn.Module,
    state:     dict,
    directory: str,
    epoch:     int,
    is_best:   bool,
) -> None:
    import modelopt.torch.opt as mto

    os.makedirs(directory, exist_ok=True)
    stem      = f"qat_modelopt_epoch_{epoch:03d}"
    ckpt_path = os.path.join(directory, f"{stem}.pth")
    mo_path   = os.path.join(directory, f"{stem}_mostate.pt")

    torch.save(state, ckpt_path)
    torch.save(mto.modelopt_state(model), mo_path)

    if is_best:
        best_ckpt = os.path.join(directory, "qat_modelopt_best.pth")
        best_mo   = os.path.join(directory, "qat_modelopt_best_mostate.pt")
        torch.save(state, best_ckpt)
        torch.save(mto.modelopt_state(model), best_mo)
        print(f"  [Checkpoint] Best model → {best_ckpt}")

def load_training_state(
    ckpt_path: str,
    model:     nn.Module,
    optimizer: optim.Optimizer,
    scaler:    GradScaler,
    scheduler,
) -> tuple[int, float]:
    print(f"[Resume] Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"] + 1, ckpt.get("best_acc", 0.0)
