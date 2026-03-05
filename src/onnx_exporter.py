from pathlib import Path
import torch
import torch.nn as nn


class ONNXExporter:
    def __init__(self, model: nn.Module, device: str, onnx_path: str | Path):
        self.model     = model
        self.device    = torch.device(device)
        self.onnx_path = Path(onnx_path)

    def export_model(
        self,
        opset_version: int = 17,
        dynamic_batch: bool = True,
        dummy_input_shape: tuple = (1, 3, 224, 224),
    ) -> Path:
        self.model.eval().to(self.device)
        self.onnx_path.parent.mkdir(parents=True, exist_ok=True)

        dummy = torch.randn(*dummy_input_shape, device=self.device, dtype=torch.float32)

        # Only mark batch dim as dynamic if requested
        dynamic_axes = {"images": {0: "batch"}, "logits": {0: "batch"}} if dynamic_batch else None

        print(f"[onnx_exporter] Exporting to {self.onnx_path} ...")
        with torch.no_grad():
            torch.onnx.export(
                self.model, (dummy,), str(self.onnx_path),
                input_names=["images"],
                output_names=["logits"],
                opset_version=opset_version,
                dynamic_axes=dynamic_axes,
                export_params=True,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL,
            )

        print(f"[onnx_exporter] Saved ({self.onnx_path.stat().st_size / 1e6:.1f} MB)")
        return self.onnx_path