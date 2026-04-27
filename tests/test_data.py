import pytest
import torch
from data import Quantize01, build_imagenet_transform
from config import ExperimentConfig

class TestQuantize01:
    def test_8bit_identity_on_uniform_grid(self):
        q = Quantize01(8)
        levels = 255.0
        x = torch.tensor([0.0, 0.5, 1.0])
        out = q(x)
        expected = torch.round(x * levels) / levels
        assert torch.allclose(out, expected)

    def test_1bit_maps_to_0_or_1(self):
        q = Quantize01(1)
        x = torch.linspace(0.0, 1.0, 20)
        out = q(x)
        assert set(out.unique().tolist()).issubset({0.0, 1.0})

    def test_clamps_out_of_range(self):
        q = Quantize01(8)
        x = torch.tensor([-0.5, 1.5])
        out = q(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_output_in_0_1(self):
        for bits in (1, 2, 4, 8):
            q = Quantize01(bits)
            x = torch.rand(100)
            out = q(x)
            assert out.min() >= 0.0
            assert out.max() <= 1.0

    def test_none_defaults_to_8bit(self):
        q = Quantize01(None)
        assert q.num_bits == 8

    def test_zero_defaults_to_8bit(self):
        q = Quantize01(0)
        assert q.num_bits == 8

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError, match="num_bits"):
            Quantize01(9)
        with pytest.raises(ValueError):
            Quantize01(-1)

    def test_non_tensor_raises(self):
        q = Quantize01(8)
        with pytest.raises(TypeError):
            q([0.5, 0.5])

    def test_preserves_shape(self):
        q = Quantize01(4)
        x = torch.rand(3, 4, 5)
        assert q(x).shape == x.shape

def test_transform_pipeline_length():
    cfg = ExperimentConfig(input_quant_bits=8)
    t = build_imagenet_transform(cfg)
    assert len(t.transforms) == 5

def test_transform_produces_correct_shape():
    from PIL import Image
    import numpy as np
    cfg = ExperimentConfig(input_quant_bits=8)
    t = build_imagenet_transform(cfg)
    img = Image.fromarray(
        (np.random.rand(300, 400, 3) * 255).astype("uint8")
    )
    out = t(img)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32
