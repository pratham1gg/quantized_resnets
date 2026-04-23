"""Tests for src/model.py — ResNet18 architecture and get_model."""
import sys
import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock

from model import ResNet18, BasicBlock, get_model


# ---------------------------------------------------------------------------
# BasicBlock
# ---------------------------------------------------------------------------

class TestBasicBlock:
    def test_no_downsample_same_shape(self):
        blk = BasicBlock(64, 64, stride=1)
        x = torch.randn(1, 64, 56, 56)
        assert blk(x).shape == (1, 64, 56, 56)

    def test_downsample_created_when_stride_gt1(self):
        blk = BasicBlock(64, 128, stride=2)
        assert blk.downsample is not None

    def test_no_downsample_when_same_channels_stride1(self):
        blk = BasicBlock(64, 64, stride=1)
        assert blk.downsample is None

    def test_output_shape_with_stride(self):
        blk = BasicBlock(64, 128, stride=2)
        x = torch.randn(1, 64, 56, 56)
        assert blk(x).shape == (1, 128, 28, 28)


# ---------------------------------------------------------------------------
# ResNet18
# ---------------------------------------------------------------------------

class TestResNet18:
    def test_output_shape_100_classes(self):
        m = ResNet18(num_classes=100, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        assert m(x).shape == (2, 100)

    def test_output_shape_3_classes(self):
        m = ResNet18(num_classes=3, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        assert m(x).shape == (1, 3)

    def test_pretrained_raises_for_non_1000_classes(self):
        with pytest.raises(ValueError, match="1000"):
            ResNet18(num_classes=100, pretrained=True)

    def test_eval_mode_no_grad(self):
        m = ResNet18(num_classes=3, pretrained=False).eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = m(x)
        assert out.shape == (1, 3)

    def test_fc_output_matches_num_classes(self):
        for nc in (3, 10, 100):
            m = ResNet18(num_classes=nc, pretrained=False)
            assert m.fc.out_features == nc


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_loads_from_nested_state_dict(self, tmp_path):
        m = ResNet18(num_classes=100, pretrained=False)
        ckpt_path = tmp_path / "best.pth"
        torch.save({"model": m.state_dict()}, ckpt_path)

        loaded = get_model(cfg=None, checkpoint_path=str(ckpt_path))
        # Verify weights match
        for (k1, v1), (k2, v2) in zip(
            m.state_dict().items(), loaded.state_dict().items()
        ):
            assert torch.allclose(v1, v2), f"Mismatch at {k1}"

    def test_loads_from_flat_state_dict(self, tmp_path):
        m = ResNet18(num_classes=100, pretrained=False)
        ckpt_path = tmp_path / "flat.pth"
        torch.save(m.state_dict(), ckpt_path)

        loaded = get_model(cfg=None, checkpoint_path=str(ckpt_path))
        for (_, v1), (_, v2) in zip(
            m.state_dict().items(), loaded.state_dict().items()
        ):
            assert torch.allclose(v1, v2)

    def test_missing_checkpoint_raises(self):
        with pytest.raises(Exception):
            get_model(cfg=None, checkpoint_path="/nonexistent/path.pth")
