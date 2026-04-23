"""
quantize.py
-----------
ModelOpt QAT setup for INT8 and INT4.

Quantization schemes
--------------------
  int8 : INT8_DEFAULT_CFG  — per-channel INT8 weights + per-tensor INT8 activations
  int4 : INT4_BLOCKWISE_WEIGHT_ONLY_CFG — INT4 blockwise weight-only (activations in fp)

Call order
----------
  1. get_quant_cfg(precision)              # pick the right mtq config
  2. quantize_model(model, cfg, loader)    # calibrates + inserts fake-quant in-place
  3. (train normally with fake-quant live)
  4. modelopt_state / restore helpers for checkpointing
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Map CLI precision names → mtq built-in configs
_QUANT_CFGS = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int4": mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
}


def get_quant_cfg(precision: str):
    """Return the mtq config dict for ``precision`` ('int8' or 'int4')."""
    precision = precision.lower()
    if precision not in _QUANT_CFGS:
        raise ValueError(f"Unsupported precision '{precision}'. Choose from {list(_QUANT_CFGS)}")
    return _QUANT_CFGS[precision]


def get_model(checkpoint_path: str, num_classes: int = 100) -> nn.Module:
    """Load the FP32 ResNet-18 from a checkpoint (plain weights, no quantization yet)."""
    from model import ResNet18

    model = ResNet18(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


def quantize_model(
    model: nn.Module,
    quant_cfg,
    calib_loader,
    num_calib_batches: int,
    device: torch.device,
) -> nn.Module:
    """
    Insert fake-quant nodes and calibrate in one call via mtq.quantize().

    mtq.quantize() runs the forward_loop to collect calibration statistics,
    then switches all quantizers to fake-quant mode ready for QAT fine-tuning.
    The model is modified in-place and returned.
    """
    model = model.to(device)

    def forward_loop(m: nn.Module) -> None:
        m.eval()
        with torch.no_grad():
            for i, (images, _) in enumerate(calib_loader):
                if i >= num_calib_batches:
                    break
                m(images.to(device))

    print(f"[Calibration] Running {num_calib_batches} batches for quantizer init ...")
    model = mtq.quantize(model, quant_cfg, forward_loop)
    print("[Calibration] Done — fake-quant active.")
    return model


def save_modelopt_state(model: nn.Module, path: str) -> None:
    """Save ModelOpt quantizer states (scales, zero-points) to a separate file."""
    torch.save(mto.modelopt_state(model), path)


def restore_modelopt_state(model: nn.Module, path: str) -> None:
    """
    Restore quantizer states into a freshly-instantiated (unquantized) model.

    Must be called BEFORE loading the model weight state_dict so that
    the quantized layer parameters already exist in the model.
    """
    state = torch.load(path, map_location="cpu")
    mto.restore_from_modelopt_state(model, state)
