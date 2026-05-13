from pathlib import Path
import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_DIR = _PROJECT_ROOT / "training" / "checkpoints" / "fp32"
_ONNX_DIR = _PROJECT_ROOT / "onnx"


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


def export_all_seeds(device: str = "cpu") -> list[Path]:
    import sys
    sys.path.insert(0, str(_PROJECT_ROOT / "pyfiles"))
    from model import ResNet18

    seed_dirs = sorted(_CHECKPOINT_DIR.iterdir())
    exported = []

    for seed_dir in seed_dirs:
        ckpt_path = seed_dir / "best.pth"
        if not ckpt_path.exists():
            print(f"[onnx_exporter] Skipping {seed_dir.name}: no best.pth found")
            continue

        seed_name = seed_dir.name  # e.g. "seed_42"
        seed_num = seed_name.split("_", 1)[1]  # e.g. "42"
        onnx_path = _ONNX_DIR / f"resnet_{seed_num}.onnx"

        print(f"\n[onnx_exporter] Loading checkpoint {ckpt_path}")
        model = ResNet18(num_classes=100, pretrained=False)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state_dict)

        exporter = ONNXExporter(model, device, onnx_path)
        exporter.export_model()
        exported.append(onnx_path)

    print(f"\n[onnx_exporter] Exported {len(exported)} models")
    return exported


if __name__ == "__main__":
    export_all_seeds()
