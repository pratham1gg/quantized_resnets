from __future__ import annotations

import torch
import torch.nn as nn

from config import ExperimentConfig


def _keep_batchnorm_fp32(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.float()


def apply_model_precision(model: nn.Module, cfg: ExperimentConfig) -> nn.Module:
    prec = str(cfg.model_precision).lower()

    model = model.to(cfg.device)
    model.eval()

    if prec in ("fp32", "float32"):
        return model.float()

    if prec in ("fp16", "float16", "half"):
        if not str(cfg.device).startswith("cuda"):
            raise ValueError("model_precision='fp16' is intended for CUDA.")
        model = model.half()
        _keep_batchnorm_fp32(model)  # <- important
        return model

    if prec in ("int8", "int4", "int2"):
        raise NotImplementedError(f"model_precision='{prec}' not implemented in PyTorch path yet.")

    raise ValueError(f"Unknown model_precision: {cfg.model_precision}")


@torch.no_grad()
def ensure_input_dtype_for_model(x: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    prec = str(cfg.model_precision).lower()

    if prec in ("fp16", "float16", "half"):
        return x.to(dtype=torch.float16, device=cfg.device, non_blocking=True)

    return x.to(dtype=torch.float32, device=cfg.device, non_blocking=True)
