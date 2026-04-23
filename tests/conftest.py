"""
Shared fixtures and sys.path setup for all tests.
"""
import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture
def tiny_model():
    """ResNet-18 with 3 classes and random weights — no checkpoint needed."""
    from model import ResNet18
    return ResNet18(num_classes=3, pretrained=False).eval()


@pytest.fixture
def cpu_device():
    return torch.device("cpu")


@pytest.fixture
def tiny_batch():
    """2-image batch, float32, CPU."""
    return torch.randn(2, 3, 224, 224)
