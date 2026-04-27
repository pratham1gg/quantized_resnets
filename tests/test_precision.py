import pytest
import torch
import torch.nn as nn

from precision import apply_precision, keep_batchnorm_fp32
from config import ExperimentConfig

def test_keep_batchnorm_fp32_restores_bn_dtype():
    model = nn.Sequential(
        nn.Conv2d(3, 3, 1),
        nn.BatchNorm2d(3),
    ).half()
    keep_batchnorm_fp32(model)
    bn = model[1]
    assert bn.weight.dtype == torch.float32
    assert model[0].weight.dtype == torch.float16  

def test_apply_precision_fp32(tiny_model):
    cfg = ExperimentConfig(model_precision="fp32", device="cpu",
                           backend="pytorch")
    m = apply_precision(tiny_model, cfg)
    assert next(m.parameters()).dtype == torch.float32

def test_apply_precision_fp16_requires_cuda(tiny_model):
    cfg = ExperimentConfig(model_precision="fp16", device="cpu",
                           backend="pytorch")
    with pytest.raises(ValueError, match="CUDA"):
        apply_precision(tiny_model, cfg)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_apply_precision_fp16_cuda(tiny_model):
    cfg = ExperimentConfig(model_precision="fp16", device="cuda",
                           backend="pytorch")
    m = apply_precision(tiny_model, cfg)
    convs = [p for p in m.parameters() if p.dtype == torch.float16]
    assert len(convs) > 0

def test_apply_precision_unsupported_raises(tiny_model):
    cfg = ExperimentConfig(model_precision="int8", device="cpu",
                           backend="torchao_cpu_ptq")
    with pytest.raises(ValueError, match="fp32/fp16"):
        apply_precision(tiny_model, cfg)

def test_ensure_input_dtype_fp32_stays_fp32():
    from precision import ensure_input_dtype
    cfg = ExperimentConfig(model_precision="fp32", device="cpu", backend="pytorch")
    x = torch.randn(2, 3, 224, 224)
    out = ensure_input_dtype(x, cfg)
    assert out.dtype == torch.float32

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_ensure_input_dtype_fp16_casts():
    from precision import ensure_input_dtype
    cfg = ExperimentConfig(model_precision="fp16", device="cuda", backend="pytorch")
    x = torch.randn(2, 3, 224, 224)
    out = ensure_input_dtype(x, cfg)
    assert out.dtype == torch.float16
