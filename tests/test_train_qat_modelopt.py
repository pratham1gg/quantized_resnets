"""
Tests for training/train_qat_modelopt.py — CLI argument parsing and
get_dataloaders helper (no actual training run needed).
"""
import sys
import types
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Stub modelopt before training script imports it
def _stub_modelopt():
    for name in ("modelopt", "modelopt.torch", "modelopt.torch.opt",
                 "modelopt.torch.quantization"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    mtq = sys.modules["modelopt.torch.quantization"]
    mtq.INT8_DEFAULT_CFG               = {"int8": True}
    mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG = {"int4": True}
    mtq.quantize = MagicMock(side_effect=lambda m, c, f: m)

_stub_modelopt()

TRAINING = Path(__file__).resolve().parents[1] / "training"
SRC      = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "qat_modelopt"))
sys.path.insert(0, str(TRAINING))

import train_qat_modelopt as script


# ---------------------------------------------------------------------------
# parse_args defaults
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_default_precision_is_int8(self):
        args = script.parse_args.__wrapped__() if hasattr(script.parse_args, "__wrapped__") else None
        # Call with no args using argparse directly
        args = script.parse_args.__func__() if hasattr(script.parse_args, "__func__") else None
        # Simplest: patch sys.argv
        import sys as _sys
        old = _sys.argv
        _sys.argv = ["train_qat_modelopt.py"]
        try:
            args = script.parse_args()
        finally:
            _sys.argv = old
        assert args.precision == "int8"

    def test_precision_int4_accepted(self):
        import sys as _sys
        old = _sys.argv
        _sys.argv = ["train_qat_modelopt.py", "--precision", "int4"]
        try:
            args = script.parse_args()
        finally:
            _sys.argv = old
        assert args.precision == "int4"

    def test_invalid_precision_raises(self):
        import sys as _sys
        old = _sys.argv
        _sys.argv = ["train_qat_modelopt.py", "--precision", "fp8"]
        try:
            with pytest.raises(SystemExit):
                script.parse_args()
        finally:
            _sys.argv = old

    def test_default_epochs(self):
        import sys as _sys
        old = _sys.argv
        _sys.argv = ["train_qat_modelopt.py"]
        try:
            args = script.parse_args()
        finally:
            _sys.argv = old
        assert args.epochs == 15

    def test_resume_requires_mostate_check(self):
        """resume without resume_mostate should be detectable (checked at runtime)."""
        import sys as _sys
        old = _sys.argv
        _sys.argv = ["train_qat_modelopt.py", "--resume", "some/path.pth"]
        try:
            args = script.parse_args()
        finally:
            _sys.argv = old
        # parse_args itself succeeds; the runtime guard in __main__ rejects it
        assert args.resume == "some/path.pth"
        assert args.resume_mostate is None


# ---------------------------------------------------------------------------
# set_seed — determinism
# ---------------------------------------------------------------------------

def test_set_seed_produces_same_tensors():
    import torch
    script.set_seed(99)
    a = torch.randn(5)
    script.set_seed(99)
    b = torch.randn(5)
    assert torch.allclose(a, b)


# ---------------------------------------------------------------------------
# _subset helper
# ---------------------------------------------------------------------------

def test_subset_keeps_only_first_n_classes():
    from unittest.mock import MagicMock
    dataset = MagicMock()
    dataset.samples = [(f"img_{i}.jpg", i % 10) for i in range(100)]
    subset = script._subset(dataset, num_classes=3)
    # All labels in subset should be < 3
    assert all(dataset.samples[i][1] < 3 for i in subset.indices)


def test_subset_correct_count():
    from unittest.mock import MagicMock
    dataset = MagicMock()
    # 5 samples per class, 10 classes → 50 samples; keep first 3 → 15 samples
    dataset.samples = [(f"img_{i}.jpg", i % 10) for i in range(50)]
    subset = script._subset(dataset, num_classes=3)
    assert len(subset.indices) == 15
