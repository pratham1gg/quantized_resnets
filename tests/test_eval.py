"""Tests for src/eval.py — evaluate() loop."""
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock

from eval import evaluate
from config import ExperimentConfig
from metrics import MetricsTracker, WARMUP_BATCHES


def _make_loader(num_batches: int, batch_size: int = 4, num_classes: int = 10):
    """Synthetic DataLoader with random images and labels."""
    images  = torch.randn(num_batches * batch_size, 3, 224, 224)
    targets = torch.randint(0, num_classes, (num_batches * batch_size,))
    ds = TensorDataset(images, targets)
    return DataLoader(ds, batch_size=batch_size)


def _tiny_model(num_classes: int = 10):
    """Tiny linear model — fast, no real ResNet needed."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, num_classes),
    ).eval()


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_evaluate_returns_metrics_tracker():
    cfg = ExperimentConfig(backend="pytorch", model_precision="fp32",
                           device="cpu", num_eval_batches=5)
    loader = _make_loader(num_batches=WARMUP_BATCHES + 5)
    tracker = evaluate(_tiny_model(), loader, cfg)
    assert isinstance(tracker, MetricsTracker)


def test_evaluate_skips_warmup_batches():
    total = WARMUP_BATCHES + 10
    cfg = ExperimentConfig(backend="pytorch", model_precision="fp32",
                           device="cpu", num_eval_batches=total)
    loader = _make_loader(num_batches=total, batch_size=4)
    tracker = evaluate(_tiny_model(), loader, cfg)
    # Only the 10 post-warmup batches contribute to metrics
    assert tracker.total == 10 * 4


def test_evaluate_respects_num_eval_batches():
    cfg = ExperimentConfig(backend="pytorch", model_precision="fp32",
                           device="cpu", num_eval_batches=WARMUP_BATCHES + 2)
    loader = _make_loader(num_batches=WARMUP_BATCHES + 50, batch_size=4)
    tracker = evaluate(_tiny_model(), loader, cfg)
    # Only 2 post-warmup batches
    assert tracker.total == 2 * 4


def test_evaluate_none_num_eval_batches_uses_full_loader():
    cfg = ExperimentConfig(backend="pytorch", model_precision="fp32",
                           device="cpu", num_eval_batches=None)
    n_post = 5
    loader = _make_loader(num_batches=WARMUP_BATCHES + n_post, batch_size=4)
    tracker = evaluate(_tiny_model(), loader, cfg)
    assert tracker.total == n_post * 4


def test_evaluate_with_criterion_records_loss():
    cfg = ExperimentConfig(backend="pytorch", model_precision="fp32",
                           device="cpu", num_eval_batches=WARMUP_BATCHES + 2)
    loader  = _make_loader(num_batches=WARMUP_BATCHES + 2)
    tracker = evaluate(_tiny_model(), loader, cfg,
                       criterion=nn.CrossEntropyLoss())
    assert len(tracker.losses) > 0


def test_evaluate_top1_in_valid_range():
    cfg = ExperimentConfig(backend="pytorch", model_precision="fp32",
                           device="cpu", num_eval_batches=WARMUP_BATCHES + 5)
    loader  = _make_loader(num_batches=WARMUP_BATCHES + 5)
    tracker = evaluate(_tiny_model(), loader, cfg)
    s = tracker.summary()
    assert 0.0 <= s["top1_acc"] <= 100.0
    assert 0.0 <= s["top5_acc"] <= 100.0
