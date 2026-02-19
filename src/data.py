import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from config import ExperimentConfig

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

class Quantize01:
    # Quantization of Input
    def __init__(self, num_bits: int | None):
        if num_bits is None or int(num_bits) == 0:
            self.num_bits = None
            return

        num_bits = int(num_bits)
        if not (1 <= num_bits <= 8):
            raise ValueError(f"num_bits must be in [1,8] (or None/0 to disable), got {num_bits}")
        self.num_bits = num_bits

    def __call__(self, x: torch.Tensor) -> torch.Tensor: 
        # Ensure floating-point tensor and restrict values to valid [0,1] range
        if self.num_bits is None:
            return x

        if not torch.is_tensor(x):
            raise TypeError(f"Quantize01 expected torch.Tensor, got {type(x)}")

        # Convert to float32 and clamp to expected quantization range
        x = x.to(torch.float32).clamp(0.0, 1.0)

        levels = (1 << self.num_bits) - 1  # 8 bits -> 255 levels, 4 bits -> 15 levels, 2 bits -> 3 levels
        xq = torch.round(x * levels) / levels
        return xq

def build_imagenet_transform(cfg: ExperimentConfig) -> T.Compose:
    return T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),  # -> float in [0,1], CHW
            Quantize01(cfg.input_quant_bits), # Quantize01 expects num_bits
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )

def build_imagenet_dataset(cfg: ExperimentConfig, split: str):
    return torchvision.datasets.ImageNet(
        root=cfg.imagenet_path,
        split=split,
        transform=build_imagenet_transform(cfg),
    )
    
def get_dataloader(cfg: ExperimentConfig, split: str = "val") -> DataLoader:
    dataset = build_imagenet_dataset(cfg, split)

    pin_memory = str(cfg.device).startswith("cuda")

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

