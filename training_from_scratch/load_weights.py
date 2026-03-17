"""
Load trained ResNet18 weights from a checkpoint .pth file.

Usage
-----
    # as a library
    from load_weights import load_best_model
    model = load_best_model("./checkpoints")

    # from the command line (quick sanity check)
    python load_weights.py --checkpoint-dir ./checkpoints --best
"""

import sys, argparse
from pathlib import Path

import torch

# ── make src/ importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from model import ResNet18


def load_trained_model(checkpoint_path, device=None):
    """Load a ResNet18 model from a training checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNet18(num_classes=1000, pretrained=False).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    epoch    = ckpt.get("epoch", "?")
    best_acc = ckpt.get("best_acc", "?")
    print(f"Loaded {checkpoint_path}  (epoch {epoch}, best_acc {best_acc}%)")
    return model


def load_best_model(checkpoint_dir="./checkpoints", device=None):
    """Convenience: load best_model.pth from the checkpoint directory."""
    path = Path(checkpoint_dir) / "best_model.pth"
    if not path.exists():
        raise FileNotFoundError(f"No best_model.pth in {checkpoint_dir}")
    return load_trained_model(str(path), device)


def load_epoch_model(epoch, checkpoint_dir="./checkpoints", device=None):
    """Load checkpoint_epoch_<N>.pth from the checkpoint directory."""
    path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return load_trained_model(str(path), device)


# ── CLI ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Load a trained ResNet18 checkpoint")
    p.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    p.add_argument("--best",  action="store_true", help="load best_model.pth")
    p.add_argument("--epoch", type=int, default=None, help="load specific epoch")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    if args.epoch is not None:
        model = load_epoch_model(args.epoch, args.checkpoint_dir, args.device)
    else:
        model = load_best_model(args.checkpoint_dir, args.device)

    model.eval()
    print(f"Model ready  (eval mode, {sum(p.numel() for p in model.parameters()):,} params)")
