

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):

    expansion: int = 1

    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)

        self.shortcut: nn.Module | None = None
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x) if self.shortcut is not None else x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        return F.relu(out + identity, inplace=True)


# ---------------------------------------------------------------------------
# ResNet-18
# ---------------------------------------------------------------------------

class ResNet18(nn.Module):
    """
    ResNet-18 built for scratch training on a custom dataset.

    Args:
        num_classes:  Number of output classes.
        dropout:      Dropout probability applied before the FC layer (0 = off).
    """

    def __init__(self, num_classes: int = 1000, dropout: float = 0.0) -> None:
        super().__init__()

        self._in_c = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_stage(out_c=64,  blocks=2, stride=1)
        self.layer2 = self._make_stage(out_c=128, blocks=2, stride=2)
        self.layer3 = self._make_stage(out_c=256, blocks=2, stride=2)
        self.layer4 = self._make_stage(out_c=512, blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_stage(self, out_c: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self._in_c, out_c, stride=stride)]
        self._in_c = out_c * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self._in_c, out_c, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Kaiming initialisation -- well suited for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def count_params(self) -> dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_model(num_classes: int = 1000, dropout: float = 0.0) -> ResNet18:
    return ResNet18(num_classes=num_classes, dropout=dropout)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_model(num_classes=10).to(device)
    dummy = torch.randn(4, 3, 224, 224, device=device)
    logits = model(dummy)

    counts = model.count_params()
    print(f"Output shape : {logits.shape}")
    print(f"Total params : {counts['total']:,}")
    print(f"Trainable    : {counts['trainable']:,}")