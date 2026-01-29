import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from quantization import InputQuantizer
from config import ExperimentConfig

def get_dataloader(config, split: str = "val"):
    """
    Create ImageNet dataloader with specified quantization
    
    Args:
        config: ExperimentConfig object
        split: Dataset split ('train' or 'val')
    
    Returns:
        DataLoader object
    """
    # Standard ImageNet normalization
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Input quantizer
    input_quantizer = InputQuantizer(num_bits=config.input_quant_bits)
    
    # Compose transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        input_quantizer,  # Apply quantization before normalization, if done after values would be centered around zero making quant much more difficult
        normalize,
    ])
    
    # Create dataset
    dataset = torchvision.datasets.ImageNet(
        root=config.imagenet_path,
        split=split,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True if config.device == "cuda" else False
    )
    
    return dataloader