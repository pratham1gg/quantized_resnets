"""Tests for src/metrics.py — MetricsTracker"""
import pytest
import torch
import numpy as np

from metrics import MetricsTracker, WARMUP_BATCHES


def _make_batch(n=4, num_classes=10, correct_top1=True):
    """Return (outputs, targets) where top-1 is correct if correct_top1=True."""
    targets = torch.arange(n) % num_classes
    outputs = torch.zeros(n, num_classes)
    if correct_top1:
        outputs.scatter_(1, targets.unsqueeze(1), 10.0)  # push correct class high
    else:
        # Make class 0 always highest, targets are 1..n
        outputs[:, 0] = 10.0
    return outputs, targets


class TestMetricsTrackerEmpty:
    def test_summary_returns_zeros_when_empty(self):
        t = MetricsTracker()
        s = t.summary()
        assert s["top1_acc"] == 0.0
        assert s["top5_acc"] == 0.0
        assert s["total_samples"] == 0
        assert s["infer_ms_avg"] is None

    def test_reset_clears_state(self):
        t = MetricsTracker()
        outputs, targets = _make_batch()
        t.update(outputs, targets, loss_value=0.5,
                 batch_time_s=0.01, infer_time_s=0.009, batch_size=4)
        t.reset()
        assert t.total == 0
        assert t.correct_top1 == 0


class TestMetricsTrackerUpdate:
    def test_perfect_top1(self):
        t = MetricsTracker()
        outputs, targets = _make_batch(n=8, correct_top1=True)
        t.update(outputs, targets, loss_value=None,
                 batch_time_s=0.01, infer_time_s=0.008, batch_size=8)
        s = t.summary()
        assert s["top1_acc"] == pytest.approx(100.0)

    def test_zero_top1(self):
        t = MetricsTracker()
        # targets are 1..7, model always predicts class 0
        outputs, targets = _make_batch(n=8, correct_top1=False)
        t.update(outputs, targets, loss_value=None,
                 batch_time_s=0.01, infer_time_s=0.008, batch_size=8)
        s = t.summary()
        # targets[0] = 0 % 10 = 0, which matches class-0 prediction, so top1 = 1/8
        assert s["top1_acc"] == pytest.approx(100.0 * 1 / 8)

    def test_top5_all_correct_when_classes_le5(self):
        # 5 classes, correct answer is always in top-5
        t = MetricsTracker()
        outputs, targets = _make_batch(n=4, num_classes=5, correct_top1=True)
        t.update(outputs, targets, loss_value=None,
                 batch_time_s=0.01, infer_time_s=0.008, batch_size=4)
        s = t.summary()
        assert s["top5_acc"] == pytest.approx(100.0)

    def test_accumulates_across_updates(self):
        t = MetricsTracker()
        outputs, targets = _make_batch(n=4, correct_top1=True)
        for _ in range(3):
            t.update(outputs, targets, loss_value=0.1,
                     batch_time_s=0.01, infer_time_s=0.008, batch_size=4)
        assert t.total == 12

    def test_loss_avg_computed(self):
        t = MetricsTracker()
        outputs, targets = _make_batch(n=4)
        t.update(outputs, targets, loss_value=1.0,
                 batch_time_s=0.01, infer_time_s=0.008, batch_size=4)
        t.update(outputs, targets, loss_value=3.0,
                 batch_time_s=0.01, infer_time_s=0.008, batch_size=4)
        assert t.summary()["loss_avg"] == pytest.approx(2.0)

    def test_none_loss_not_stored(self):
        t = MetricsTracker()
        outputs, targets = _make_batch(n=4)
        t.update(outputs, targets, loss_value=None,
                 batch_time_s=0.01, infer_time_s=0.008, batch_size=4)
        assert t.summary()["loss_avg"] is None

    def test_timing_stats_in_ms(self):
        t = MetricsTracker()
        outputs, targets = _make_batch(n=4)
        t.update(outputs, targets, loss_value=None,
                 batch_time_s=0.01, infer_time_s=0.005, batch_size=4)
        s = t.summary()
        assert s["infer_ms_avg"] == pytest.approx(5.0)
        assert s["batch_ms_avg"] == pytest.approx(10.0)

    def test_throughput_positive(self):
        t = MetricsTracker()
        outputs, targets = _make_batch(n=4)
        t.update(outputs, targets, loss_value=None,
                 batch_time_s=0.1, infer_time_s=0.05, batch_size=4)
        s = t.summary()
        assert s["throughput_infer_sps"] == pytest.approx(4 / 0.05)
        assert s["throughput_sps"] == pytest.approx(4 / 0.1)

    def test_running_accuracy_length_matches_updates(self):
        t = MetricsTracker()
        outputs, targets = _make_batch(n=4)
        for _ in range(5):
            t.update(outputs, targets, loss_value=None,
                     batch_time_s=0.01, infer_time_s=0.008, batch_size=4)
        assert len(t.top1_running) == 5
        assert len(t.top5_running) == 5

    def test_summary_total_batches(self):
        t = MetricsTracker()
        outputs, targets = _make_batch(n=4)
        for _ in range(3):
            t.update(outputs, targets, loss_value=None,
                     batch_time_s=0.01, infer_time_s=0.008, batch_size=4)
        assert t.summary()["total_batches"] == 3


def test_warmup_batches_constant():
    assert WARMUP_BATCHES == 30
