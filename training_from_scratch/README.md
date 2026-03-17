# Training ResNet18 from Scratch on ImageNet-1k

Scripts to train your custom `ResNet18` architecture (defined in `src/model.py`) from scratch, without any pre-trained weights.

## Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script (SGD/Adam, StepLR/Cosine, checkpoint & resume) |
| `evaluate.py` | Evaluate a checkpoint on the ImageNet validation set |
| `load_weights.py` | Helper to load a `.pth` checkpoint back into `ResNet18` |

## Quick Start

```bash
cd training_from_scratch

# Train with defaults (90 epochs, bs 256, SGD lr=0.1, StepLR γ=0.1 every 30 ep)
python train.py

# Custom run
python train.py --epochs 100 --batch-size 128 --lr 0.05 --scheduler cosine

# Resume from a checkpoint
python train.py --resume ./checkpoints/checkpoint_epoch_50.pth
```

Checkpoints are saved to `./checkpoints/`:
- `checkpoint_epoch_<N>.pth` – every epoch
- `best_model.pth` – best validation top-1

## Evaluate

```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pth
```

## Load Weights in Your Code

```python
from load_weights import load_best_model

model = load_best_model("./checkpoints")   # returns ResNet18 in eval mode
```

Or manually:

```python
from src.model import ResNet18
import torch

model = ResNet18(num_classes=1000, pretrained=False)
ckpt  = torch.load("./checkpoints/best_model.pth")
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
```

## Training Arguments

| Arg | Default | Notes |
|-----|---------|-------|
| `--epochs` | 90 | |
| `--batch-size` | 256 | reduce to 128/64 if OOM |
| `--lr` | 0.1 | |
| `--optimizer` | sgd | sgd / adam |
| `--scheduler` | step | step / cosine / none |
| `--step-size` | 30 | for StepLR |
| `--gamma` | 0.1 | for StepLR |
| `--data-dir` | /home/pf4636/imagenet2 | ImageNet root |
| `--resume` | — | path to checkpoint |

## Expected Results

Standard ResNet18 from scratch on ImageNet-1k:
- **Top-1 ≈ 69.8 %**
- **Top-5 ≈ 89.1 %**
