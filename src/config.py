
from dataclasses import dataclass, replace
from typing import Optional
import torch
import random
import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    imagenet_path: str = "/home/pf4636/imagenet2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    num_workers: int = 4

    # Experiment knobs
    input_quant_bits: int = 8
    model_precision: str = "fp32"   # fp32 / fp16 / int 8

    num_eval_batches: Optional[int] = None  # None = full validation set
    
    seed: int = 42
    
def set_seed(cfg: ExperimentConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def with_overrides(cfg: ExperimentConfig, **kwargs) -> ExperimentConfig:
    """
    Create a new config with updated fields.
    """
    return replace(cfg, **kwargs)
