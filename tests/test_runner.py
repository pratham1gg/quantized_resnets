"""Tests for src/runner.py — run_experiment routing (all heavy deps mocked)."""
import sys
import types
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch, call

from config import ExperimentConfig
from metrics import MetricsTracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_tracker():
    t = MetricsTracker()
    outputs = torch.zeros(4, 10)
    outputs[:, 0] = 10.0
    targets = torch.zeros(4, dtype=torch.long)
    t.update(outputs, targets, loss_value=0.1,
             batch_time_s=0.01, infer_time_s=0.008, batch_size=4)
    return t


def _tiny_loader():
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.randn(8, 3, 224, 224), torch.zeros(8, dtype=torch.long))
    return DataLoader(ds, batch_size=4)


def _tiny_model():
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(3, 10),
    ).eval()


# ---------------------------------------------------------------------------
# pytorch backend
# ---------------------------------------------------------------------------

class TestRunnerPytorch:
    def test_pytorch_fp32_returns_payload_and_tracker(self, tmp_path):
        cfg = ExperimentConfig(
            backend="pytorch", model_precision="fp32", device="cpu",
            batch_size=4, num_eval_batches=35, output_root=str(tmp_path / "runs"),
        )
        with (
            patch("runner.get_model", return_value=_tiny_model()),
            patch("runner.get_dataloader", return_value=_tiny_loader()),
            patch("runner.apply_precision", side_effect=lambda m, c: m),
        ):
            from runner import run_experiment
            payload, tracker = run_experiment(cfg, save_results_flag=False)

        assert payload["status"] == "ok"
        assert isinstance(tracker, MetricsTracker)

    def test_pytorch_saves_result_json(self, tmp_path):
        cfg = ExperimentConfig(
            backend="pytorch", model_precision="fp32", device="cpu",
            batch_size=4, num_eval_batches=35, output_root=str(tmp_path / "runs"),
        )
        with (
            patch("runner.get_model", return_value=_tiny_model()),
            patch("runner.get_dataloader", return_value=_tiny_loader()),
            patch("runner.apply_precision", side_effect=lambda m, c: m),
        ):
            from runner import run_experiment
            run_experiment(cfg, save_results_flag=True)

        assert cfg.result_json_path().exists()

    def test_pytorch_payload_contains_run_id(self, tmp_path):
        cfg = ExperimentConfig(
            backend="pytorch", model_precision="fp32", device="cpu",
            batch_size=4, num_eval_batches=35, output_root=str(tmp_path / "runs"),
        )
        with (
            patch("runner.get_model", return_value=_tiny_model()),
            patch("runner.get_dataloader", return_value=_tiny_loader()),
            patch("runner.apply_precision", side_effect=lambda m, c: m),
        ):
            from runner import run_experiment
            payload, _ = run_experiment(cfg, save_results_flag=False)

        assert payload["run_id"] == cfg.run_id()


# ---------------------------------------------------------------------------
# torchao_cpu_ptq backend
# ---------------------------------------------------------------------------

class TestRunnerTorchaoCPU:
    def test_torchao_routes_to_quantize(self, tmp_path):
        cfg = ExperimentConfig(
            backend="torchao_cpu_ptq", model_precision="int8", device="cpu",
            batch_size=4, num_eval_batches=35, output_root=str(tmp_path / "runs"),
        )
        mock_quantized = _tiny_model()
        with (
            patch("runner.get_model", return_value=_tiny_model()),
            patch("runner.get_dataloader", return_value=_tiny_loader()),
            patch("runner.quantize_int8_x86_pt2e", return_value=mock_quantized) as mock_q,
        ):
            from runner import run_experiment
            payload, _ = run_experiment(cfg, save_results_flag=False)

        mock_q.assert_called_once()
        assert payload["status"] == "ok"


# ---------------------------------------------------------------------------
# Unknown backend
# ---------------------------------------------------------------------------

def test_unknown_backend_raises(tmp_path):
    cfg = ExperimentConfig(
        backend="pytorch", model_precision="fp32", device="cpu",
        output_root=str(tmp_path / "runs"),
    )
    # Manually bypass validate() to inject bad backend
    from dataclasses import replace
    bad_cfg = replace(cfg, backend="nonexistent")
    with pytest.raises(ValueError, match="nonexistent"):
        from runner import run_experiment
        run_experiment.__wrapped__ if hasattr(run_experiment, "__wrapped__") else None
        # Call runner._run with bad backend directly
        import runner
        runner.run_experiment(bad_cfg, save_results_flag=False)


# ---------------------------------------------------------------------------
# TRT path helpers
# ---------------------------------------------------------------------------

def test_get_trt_paths_plain_onnx_for_fp32(tmp_path):
    cfg = ExperimentConfig(
        backend="tensorrt", model_precision="fp32", device="cuda",
        output_root=str(tmp_path / "runs"),
    )
    import runner
    onnx_path, engine_path, _ = runner._get_trt_paths(cfg)
    assert onnx_path.name == "resnet18.onnx"


def test_get_trt_paths_qdq_onnx_for_int8(tmp_path):
    cfg = ExperimentConfig(
        backend="tensorrt", model_precision="int8", device="cuda",
        output_root=str(tmp_path / "runs"),
    )
    import runner
    onnx_path, _, _ = runner._get_trt_paths(cfg)
    assert onnx_path.name == "resnet18_int8_qdq.onnx"


def test_get_trt_paths_qdq_onnx_for_fp8(tmp_path):
    cfg = ExperimentConfig(
        backend="tensorrt", model_precision="fp8", device="cuda",
        output_root=str(tmp_path / "runs"),
    )
    import runner
    onnx_path, _, _ = runner._get_trt_paths(cfg)
    assert onnx_path.name == "resnet18_fp8_qdq.onnx"
