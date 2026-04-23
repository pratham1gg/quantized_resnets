"""Tests for src/config.py"""
import pytest
from dataclasses import replace

from config import ExperimentConfig, set_seed, with_overrides


# ---------------------------------------------------------------------------
# run_id
# ---------------------------------------------------------------------------

def test_run_id_format():
    cfg = ExperimentConfig(backend="pytorch", model_precision="fp32",
                           input_quant_bits=8, device="cpu", batch_size=4)
    rid = cfg.run_id()
    assert rid == "resnet18_pytorch_fp32_in8b_cpu_bs4"


def test_run_id_tensorrt_with_tag():
    cfg = ExperimentConfig(backend="tensorrt", model_precision="int8",
                           input_quant_bits=4, device="cuda", batch_size=1,
                           trt_engine_tag="v2")
    rid = cfg.run_id()
    assert rid == "resnet18_tensorrt_int8_in4b_cuda_bs1_v2"


def test_run_id_tensorrt_no_tag():
    cfg = ExperimentConfig(backend="tensorrt", model_precision="fp16",
                           device="cuda", batch_size=8, trt_engine_tag="")
    rid = cfg.run_id()
    assert "v" not in rid.split("_")[-1]  # no tag appended


# ---------------------------------------------------------------------------
# normalized
# ---------------------------------------------------------------------------

def test_normalized_strips_whitespace_and_lowercases():
    cfg = ExperimentConfig(model_precision="FP32 ", device=" CUDA", backend="PyTorch")
    n = cfg.normalized()
    assert n.model_precision == "fp32"
    assert n.device == "cuda"
    assert n.backend == "pytorch"


# ---------------------------------------------------------------------------
# validate — happy paths
# ---------------------------------------------------------------------------

def test_validate_pytorch_fp32_cpu():
    ExperimentConfig(backend="pytorch", model_precision="fp32", device="cpu").validate()


def test_validate_pytorch_fp16_cuda():
    ExperimentConfig(backend="pytorch", model_precision="fp16", device="cuda").validate()


def test_validate_torchao_cpu_int8():
    ExperimentConfig(backend="torchao_cpu_ptq", model_precision="int8", device="cpu").validate()


def test_validate_tensorrt_precisions():
    for prec in ("fp32", "fp16", "int8", "fp8", "int4"):
        ExperimentConfig(backend="tensorrt", model_precision=prec, device="cuda").validate()


# ---------------------------------------------------------------------------
# validate — error paths
# ---------------------------------------------------------------------------

def test_validate_pytorch_rejects_int8():
    with pytest.raises(ValueError, match="fp32/fp16"):
        ExperimentConfig(backend="pytorch", model_precision="int8", device="cpu").validate()


def test_validate_torchao_rejects_gpu():
    with pytest.raises(ValueError, match="CPU"):
        ExperimentConfig(backend="torchao_cpu_ptq", model_precision="int8", device="cuda").validate()


def test_validate_torchao_rejects_fp32():
    with pytest.raises(ValueError, match="int8"):
        ExperimentConfig(backend="torchao_cpu_ptq", model_precision="fp32", device="cpu").validate()


def test_validate_tensorrt_rejects_cpu():
    with pytest.raises(ValueError, match="cuda"):
        ExperimentConfig(backend="tensorrt", model_precision="fp32", device="cpu").validate()


def test_validate_bad_input_quant_bits():
    with pytest.raises(ValueError, match="input_quant_bits"):
        ExperimentConfig(input_quant_bits=3, backend="pytorch",
                         model_precision="fp32", device="cpu").validate()


def test_validate_bad_batch_size():
    with pytest.raises(ValueError, match="batch_size"):
        ExperimentConfig(batch_size=0, backend="pytorch",
                         model_precision="fp32", device="cpu").validate()


def test_validate_fp16_requires_cuda():
    with pytest.raises(ValueError, match="fp16"):
        ExperimentConfig(backend="pytorch", model_precision="fp16", device="cpu").validate()


# ---------------------------------------------------------------------------
# with_overrides
# ---------------------------------------------------------------------------

def test_with_overrides_returns_new_config():
    base = ExperimentConfig(backend="pytorch", model_precision="fp32",
                            device="cpu", batch_size=1)
    new  = with_overrides(base, batch_size=8)
    assert new.batch_size == 8
    assert base.batch_size == 1  # frozen, unchanged


def test_with_overrides_validates():
    base = ExperimentConfig(backend="pytorch", model_precision="fp32", device="cpu")
    with pytest.raises(ValueError):
        with_overrides(base, model_precision="int8")  # pytorch doesn't support int8


# ---------------------------------------------------------------------------
# result_json_path
# ---------------------------------------------------------------------------

def test_result_json_path_contains_run_id():
    cfg = ExperimentConfig(backend="pytorch", model_precision="fp32",
                           device="cpu", batch_size=1, output_root="/tmp/runs")
    p = cfg.result_json_path()
    assert cfg.run_id() in str(p)
    assert p.name == "result.json"


# ---------------------------------------------------------------------------
# set_seed — smoke test (no crash, determinism)
# ---------------------------------------------------------------------------

def test_set_seed_determinism():
    import torch
    cfg = ExperimentConfig()
    set_seed(cfg)
    a = torch.randn(10)
    set_seed(cfg)
    b = torch.randn(10)
    assert torch.allclose(a, b)
