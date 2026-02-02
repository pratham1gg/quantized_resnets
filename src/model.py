# model.py
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from config import ExperimentConfig
from quantization import apply_model_precision

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.downsample = None
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.in_c = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(out_c=64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(out_c=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(out_c=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(out_c=512, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, out_c: int, blocks: int, stride: int) -> nn.Sequential:
        layers = []
        layers.append(BasicBlock(self.in_c, out_c, stride=stride))
        self.in_c = out_c * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def resnet18(num_classes: int = 1000) -> ResNet18:
    return ResNet18(num_classes=num_classes)


def get_model(cfg: ExperimentConfig) -> nn.Module:
    """
    Build ResNet-18 and apply model precision (fp32/fp16).
    """
    model = resnet18(num_classes=1000)

    # Apply precision + move to device + eval() (handled inside apply_model_precision)
    model = apply_model_precision(model, cfg)
    return model
