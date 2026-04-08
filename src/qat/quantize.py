"""
quantize.py
-----------
pytorch-quantization INT8 QAT setup for Phase 1.

Quantization scheme
-------------------
  Weights     : per-channel INT8, max calibration  (axis=0 = output channels)
  Activations : per-tensor INT8, max calibration

Call order
----------
  1. setup_quantization_descriptors()   # configure before any model init
  2. initialize_quant_modules()         # monkey-patch nn.Conv2d / nn.Linear
  3. get_quantized_model(...)           # instantiate + load FP32 weights
  4. calibrate(...)                     # collect amax, switch to fake-quant
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor

# Make src/ importable when this module is used directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def setup_quantization_descriptors() -> None:
    """
    Configure per-tensor activation + per-channel weight descriptors with max
    calibration. Must be called before initialize_quant_modules() and before
    any model instantiation.
    """
    input_desc  = QuantDescriptor(num_bits=8, calib_method="max")
    weight_desc = QuantDescriptor(num_bits=8, axis=(0,), calib_method="max")

    quant_nn.QuantConv2d.set_default_quant_desc_input(input_desc)
    quant_nn.QuantConv2d.set_default_quant_desc_weight(weight_desc)
    quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
    quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)


def initialize_quant_modules() -> None:
    """Monkey-patch torch.nn.Conv2d / Linear with quantized equivalents."""
    quant_modules.initialize()


def get_quantized_model(checkpoint_path: str, num_classes: int = 100) -> nn.Module:
    """
    Instantiate ResNet-18 with quantized layers and load FP32 weights.

    initialize_quant_modules() must have been called before this so that
    nn.Conv2d / nn.Linear inside ResNet18.__init__ are already patched.
    """
    from model import ResNet18

    model = ResNet18(num_classes=num_classes, pretrained=False)
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    return model


@torch.no_grad()
def calibrate(
    model:       nn.Module,
    loader,
    num_batches: int,
    device:      torch.device,
) -> None:
    """
    Collect per-tensor / per-channel amax values via max calibration, then
    switch all TensorQuantizers from calibration mode to fake-quant mode.
    """
    model.eval()

    # --- enable calibrators, disable fake-quant ---
    for module in model.modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    print(f"[Calibration] Collecting stats over {num_batches} batches ...")
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        model(images.to(device))

    # --- load amax, switch to fake-quant mode ---
    for module in model.modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(percentile=99.99)
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

    print("[Calibration] Done. amax values loaded, fake-quant active.")