
import sys
from pathlib import Path
import torch
import torch.nn as nn

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_QUANT_CFGS = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int4": mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
}

def get_quant_cfg(precision: str):
    precision = precision.lower()
    if precision not in _QUANT_CFGS:
        raise ValueError(f"Unsupported precision '{precision}'. Choose from {list(_QUANT_CFGS)}")
    return _QUANT_CFGS[precision]

def get_model(checkpoint_path: str, num_classes: int = 100) -> nn.Module:
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
    torch.save(mto.modelopt_state(model), path)

def restore_modelopt_state(model: nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu")
    mto.restore_from_modelopt_state(model, state)
