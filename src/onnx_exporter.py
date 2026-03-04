from pathlib import Path
import torch
from torch.onnx import TrainingMode, export

from model import get_model
from config import ExperimentConfig

cfg = ExperimentConfig()
model = get_model(cfg, pretrained=True)

# Export
model.eval()
model.to(cfg.device, dtype=torch.float32)
dummy = torch.randn(1, 3, 224, 224, device=cfg.device, dtype=torch.float32)

with torch.no_grad():
    export(
        model, (dummy,), "resnet18.onnx",
        training=TrainingMode.EVAL,
        input_names=["images"],
        output_names=["logits"],
        export_params=True,
        do_constant_folding=True,
        opset_version=17,
    )

print("Model exported to resnet18.onnx")