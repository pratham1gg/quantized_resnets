
import sys
from pathlib import Path

import torch
import torch.nn as nn

from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def setup_quantization_descriptors() -> None:
    input_desc  = QuantDescriptor(num_bits=8, calib_method="max")
    weight_desc = QuantDescriptor(num_bits=8, axis=(0,), calib_method="max")

    quant_nn.QuantConv2d.set_default_quant_desc_input(input_desc)
    quant_nn.QuantConv2d.set_default_quant_desc_weight(weight_desc)
    quant_nn.QuantLinear.set_default_quant_desc_input(input_desc)
    quant_nn.QuantLinear.set_default_quant_desc_weight(weight_desc)

def initialize_quant_modules() -> None:
    quant_modules.initialize()

def get_quantized_model(checkpoint_path: str, num_classes: int = 100) -> nn.Module:
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
    model.eval()

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