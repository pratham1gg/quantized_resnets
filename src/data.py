import os
import random
from typing import Optional, Tuple, List

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from config import ExperimentConfig

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
_HOLDOUT_PER_CLASS = 50

class Quantize01:
    def __init__(self, num_bits: int | None):
        if num_bits is None or int(num_bits) == 0:
            self.num_bits = 8
            return

        num_bits = int(num_bits)
        if not (1 <= num_bits <= 8):
            raise ValueError(f"num_bits must be in [1,8] (or None/0 to disable), got {num_bits}")
        self.num_bits = num_bits

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError(f"Quantize01 expected torch.Tensor, got {type(x)}")

        x = x.to(torch.float32).clamp(0.0, 1.0)
        levels = (1 << self.num_bits) - 1
        xq = torch.round(x * levels) / levels
        return xq

def build_imagenet_transform(cfg: ExperimentConfig) -> T.Compose:
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            Quantize01(cfg.input_quant_bits),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )

def build_imagenet_dataset(cfg: ExperimentConfig, split: str):
    dataset = torchvision.datasets.ImageNet(
        root=cfg.imagenet_path,
        split=split,
        transform=build_imagenet_transform(cfg),
    )
    if cfg.num_classes < 1000:
        keep = [(path, cls) for path, cls in dataset.samples if cls < cfg.num_classes]
        if not keep:
            raise ValueError(
                f"No samples found for num_classes={cfg.num_classes}. "
                "Check that the ImageNet root contains the expected classes."
            )
        dataset.samples = keep
        dataset.imgs    = keep
        dataset.targets = [cls for _, cls in keep]
        print(f"[data] Filtered to {len(keep)} samples across {cfg.num_classes} classes.")
    return dataset

def _stratified_split_indices(
    samples: List[Tuple[str, int]],
    num_classes: int,
    val_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {c: [] for c in range(num_classes)}
    for idx, (_, label) in enumerate(samples):
        if label < num_classes:
            by_class[label].append(idx)

    train_idx: list[int] = []
    holdout_idx: list[int] = []
    for c in range(num_classes):
        idxs = by_class[c]
        if len(idxs) < val_per_class:
            raise ValueError(
                f"Class {c} has only {len(idxs)} samples; cannot hold out {val_per_class}."
            )
        chosen = rng.sample(idxs, val_per_class)
        chosen_set = set(chosen)
        holdout_idx.extend(chosen)
        train_idx.extend(i for i in idxs if i not in chosen_set)

    return train_idx, holdout_idx

def build_train_holdout_split(
    data_root: str,
    num_classes: int = 100,
    val_per_class: int = _HOLDOUT_PER_CLASS,
    seed: int = 42,
    train_transform=None,
    eval_transform=None,
) -> Tuple[Subset, Subset]:
    if eval_transform is None:
        eval_transform = train_transform

    train_dir = os.path.join(data_root, "train")
    eval_ds   = datasets.ImageFolder(train_dir, transform=eval_transform)

    train_idx, holdout_idx = _stratified_split_indices(
        eval_ds.samples, num_classes, val_per_class, seed,
    )

    if train_transform is None or train_transform is eval_transform:
        train_subset = Subset(eval_ds, train_idx)
    else:
        train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
        train_subset = Subset(train_ds, train_idx)

    holdout_subset = Subset(eval_ds, holdout_idx)

    assert not (set(train_idx) & set(holdout_idx)), "train/holdout indices overlap"
    print(f"[data] Train: {len(train_subset):,}  Holdout-Val: {len(holdout_subset):,}  "
          f"(num_classes={num_classes}, val_per_class={val_per_class}, seed={seed})")
    return train_subset, holdout_subset

def build_runner_loaders(cfg: ExperimentConfig) -> Tuple[DataLoader, DataLoader]:
    transform = build_imagenet_transform(cfg)
    train_subset, holdout_subset = build_train_holdout_split(
        data_root=cfg.imagenet_path,
        num_classes=cfg.num_classes,
        val_per_class=_HOLDOUT_PER_CLASS,
        seed=cfg.seed,
        train_transform=transform,
        eval_transform=transform,
    )
    pin_memory = str(cfg.device).startswith("cuda")
    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        holdout_subset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_memory, drop_last=True,
    )
    return train_loader, val_loader

def get_dataloader(cfg: ExperimentConfig, split: str = "val") -> DataLoader:
    dataset = build_imagenet_dataset(cfg, split)

    pin_memory = str(cfg.device).startswith("cuda")

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
