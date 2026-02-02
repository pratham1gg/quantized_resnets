# config.py
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional
import torch


@dataclass(frozen=True)
class ExperimentConfig:
    # Stable settings (usually constant across experiments)
    imagenet_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    num_workers: int = 4

    # Experiment knobs (baseline defaults; overridden in Jupyter)
    input_quant_bits: int = 8
    model_precision: str = "fp32"   # fp32 / fp16 (int8 later via OpenVINO)

    # Optional convenience
    num_eval_batches: Optional[int] = None  # None = full validation set


def with_overrides(cfg: ExperimentConfig, **kwargs) -> ExperimentConfig:
    """
    Create a new config with updated fields (nice for Jupyter).
    """
    return replace(cfg, **kwargs)
