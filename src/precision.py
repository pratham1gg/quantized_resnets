import torch
import torch.nn as nn

from config import ExperimentConfig


def keep_batchnorm_fp32(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.float()


def apply_precision(model: nn.Module, cfg: ExperimentConfig) -> nn.Module:
    prec = str(cfg.model_precision).lower().strip()
    model = model.to(cfg.device).eval()

    if prec in ("fp32", "float32"):
        return model.float()

    if prec in ("fp16", "float16", "half"):
        if not str(cfg.device).startswith("cuda"):
            raise ValueError("fp16 requires CUDA.")
        model = model.half()
        keep_batchnorm_fp32(model)
        return model

    raise ValueError(f"apply_precision only supports fp32/fp16, got {cfg.model_precision}")


@torch.no_grad()
def ensure_input_dtype(x: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    prec = str(cfg.model_precision).lower().strip()
    if prec in ("fp16", "float16", "half"):
        return x.to(dtype=torch.float16, device=cfg.device, non_blocking=True)
    # INT8 CPU PTQ still takes fp32 inputs; cfg.device should be "cpu" for that backend.
    return x.to(dtype=torch.float32, device=cfg.device, non_blocking=True)