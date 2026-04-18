from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from config import ExperimentConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINT = _REPO_ROOT / "checkpoints" / "best.pth"


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

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))


        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 1000, pretrained: bool = True):
        super().__init__()
        self.in_c = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(out_c=64,  blocks=2, stride=1)
        self.layer2 = self._make_layer(out_c=128, blocks=2, stride=2)
        self.layer3 = self._make_layer(out_c=256, blocks=2, stride=2)
        self.layer4 = self._make_layer(out_c=512, blocks=2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        if pretrained:
            if num_classes != 1000:
                raise ValueError("Pretrained ImageNet weights require num_classes=1000.")
            sd = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).state_dict()
            self.load_state_dict(sd)
        else:
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def get_model(
    cfg,
    pretrained: bool = False,
    checkpoint_path: str | None = None,
) -> nn.Module:
    if pretrained:
        return ResNet18(num_classes=1000, pretrained=True)
    path = checkpoint_path or str(_DEFAULT_CHECKPOINT)
    model = ResNet18(num_classes=100, pretrained=False)
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    return model