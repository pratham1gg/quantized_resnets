import torch
import torch.nn as nn

# TorchAO Imports
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config
import torchao.quantization.pt2e as pt2e_utils

from torch.export import export

from config import ExperimentConfig

class INT8Strategy:

    def apply_to_model(self, model: nn.Module, _device: str, dataloader=None) -> nn.Module:
        if dataloader is None:
            raise ValueError("INT8 Static Quantization requires a dataloader for calibration!")
        
        # FORCE CPU for X86 Quantization
        quant_device = "cpu"
        model = model.to(quant_device)
        
        print(f"Starting PT2E Static Quantization on {quant_device}...")
        
        # 1. Exporting Model
        print("  Step 1: Exporting model graph...")
        # Reduce dummy batch size to 1 to save memory; the graph structure is the same.
        images0, _ = next(iter(dataloader))  # images0: [B,3,224,224]
        example_x = images0.to(quant_device, non_blocking=True)

        print("  PT2E example_x shape:", tuple(example_x.shape))
        exported_model = export(model, (example_x,))
        
        # 2. Selecting Quantizer
        print("  Step 2: Preparing observers (X86Inductor Config)...")
        ##### quantizer is optimized for CPU
        quantizer = X86InductorQuantizer()
        quantization_config = get_default_x86_inductor_quantization_config()
        quantizer.set_global(quantization_config)
        
        prepared_model = prepare_pt2e(exported_model.module(), quantizer)
        
        # 3. Calibration
        print("  Step 3: Calibrating with representative data...")
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >=10: break 
                images = images.to(quant_device)
                prepared_model(images) # Runs data through observers
        
        # 4. Convert
        print("  Step 4: Converting to quantized model...")
        quantized_model = convert_pt2e(prepared_model)
        pt2e_utils.move_exported_model_to_eval(quantized_model)
        print("quantized device:", next(quantized_model.parameters()).device)


        
        print("INT8 Quantization complete.")
        return quantized_model

    def prepare_input(self, x: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
        # PT2E graph handles quantization internally; input remains fp32
        return x.to(dtype=torch.float32, device=cfg.device, non_blocking=True)


def _keep_batchnorm_fp32(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.float()


def apply_model_precision(model: nn.Module, cfg: ExperimentConfig, dataloader=None) -> nn.Module:
    """
    Applies precision strategies.
    WARNING: INT8 requires 'dataloader' to be passed for calibration.
    """
    prec = str(cfg.model_precision).lower()
    model = model.to(cfg.device)
    model.eval()
    
    # 1. Handle INT8 (Static PTQ)
    if prec == "int8": # Fixed tuple syntax error from 'in ("int8")'
        strategy = INT8Strategy()
        return strategy.apply_to_model(model, cfg.device, dataloader=dataloader)

    # 2. Handle FP32 (Baseline)
    if prec in ("fp32", "float32"):
        return model.float()

    # 3. Handle FP16
    if prec in ("fp16", "float16", "half"):
        if not str(cfg.device).startswith("cuda"):
            raise ValueError("model_precision='fp16' is intended for CUDA.")
        model = model.half()
        _keep_batchnorm_fp32(model)
        return model
    
    raise ValueError(f"Unknown model_precision: {cfg.model_precision}")


@torch.no_grad()
def ensure_input_dtype_for_model(x: torch.Tensor, cfg: ExperimentConfig) -> torch.Tensor:
    prec = str(cfg.model_precision).lower()

    if prec in ("fp16", "float16", "half"):
        return x.to(dtype=torch.float16, device=cfg.device, non_blocking=True)

    if prec == "int8":
        return x.to(dtype=torch.float32, device="cpu", non_blocking=True)

    return x.to(dtype=torch.float32, device=cfg.device, non_blocking=True)
