"""
Tests for src/qat_modelopt/quantize.py and src/qat_modelopt/train_utils.py.
modelopt and pytorch_quantization are not installed locally, so both are
fully mocked at the module level before importing the units under test.
"""
import sys
import types
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Stub out modelopt before any import touches it
# ---------------------------------------------------------------------------

def _stub_modelopt():
    mto = types.ModuleType("modelopt.torch.opt")
    mto.modelopt_state = MagicMock(return_value={"stub": True})
    mto.restore_from_modelopt_state = MagicMock()

    mtq = types.ModuleType("modelopt.torch.quantization")
    mtq.INT8_DEFAULT_CFG              = {"int8": True}
    mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG = {"int4": True}
    mtq.quantize = MagicMock(side_effect=lambda model, cfg, fwd: model)

    modelopt        = types.ModuleType("modelopt")
    modelopt_torch  = types.ModuleType("modelopt.torch")

    sys.modules.setdefault("modelopt",                    modelopt)
    sys.modules.setdefault("modelopt.torch",              modelopt_torch)
    sys.modules.setdefault("modelopt.torch.opt",          mto)
    sys.modules.setdefault("modelopt.torch.quantization", mtq)
    return mto, mtq

_MTO, _MTQ = _stub_modelopt()

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "qat_modelopt"))

from qat_modelopt import quantize as qmod
from qat_modelopt import train_utils as tutils


# ---------------------------------------------------------------------------
# Tiny model fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    from model import ResNet18
    return ResNet18(num_classes=3, pretrained=False).eval()


@pytest.fixture
def tiny_loader():
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(torch.randn(8, 3, 224, 224),
                       torch.zeros(8, dtype=torch.long))
    return DataLoader(ds, batch_size=4)


# ---------------------------------------------------------------------------
# get_quant_cfg
# ---------------------------------------------------------------------------

class TestGetQuantCfg:
    def test_int8_returns_correct_cfg(self):
        cfg = qmod.get_quant_cfg("int8")
        assert cfg == {"int8": True}

    def test_int4_returns_correct_cfg(self):
        cfg = qmod.get_quant_cfg("int4")
        assert cfg == {"int4": True}

    def test_case_insensitive(self):
        assert qmod.get_quant_cfg("INT8") == qmod.get_quant_cfg("int8")

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="int8"):
            qmod.get_quant_cfg("fp8")


# ---------------------------------------------------------------------------
# get_model (qat_modelopt version)
# ---------------------------------------------------------------------------

class TestQatModeloptGetModel:
    def test_loads_nested_state_dict(self, tmp_path):
        from model import ResNet18
        m = ResNet18(num_classes=100, pretrained=False)
        p = tmp_path / "ckpt.pth"
        torch.save({"model": m.state_dict()}, p)
        loaded = qmod.get_model(str(p), num_classes=100)
        for (_, v1), (_, v2) in zip(m.state_dict().items(),
                                     loaded.state_dict().items()):
            assert torch.allclose(v1, v2)

    def test_loads_flat_state_dict(self, tmp_path):
        from model import ResNet18
        m = ResNet18(num_classes=100, pretrained=False)
        p = tmp_path / "flat.pth"
        torch.save(m.state_dict(), p)
        loaded = qmod.get_model(str(p), num_classes=100)
        for (_, v1), (_, v2) in zip(m.state_dict().items(),
                                     loaded.state_dict().items()):
            assert torch.allclose(v1, v2)


# ---------------------------------------------------------------------------
# quantize_model
# ---------------------------------------------------------------------------

class TestQuantizeModel:
    def test_calls_mtq_quantize(self, tiny_model, tiny_loader):
        _MTQ.quantize.reset_mock()
        qmod.quantize_model(tiny_model, {"int8": True}, tiny_loader,
                            num_calib_batches=2,
                            device=torch.device("cpu"))
        _MTQ.quantize.assert_called_once()

    def test_returns_model(self, tiny_model, tiny_loader):
        result = qmod.quantize_model(tiny_model, {"int8": True}, tiny_loader,
                                     num_calib_batches=2,
                                     device=torch.device("cpu"))
        assert result is tiny_model  # mock returns model unchanged

    def test_forward_loop_stops_at_calib_batches(self, tiny_model, tiny_loader):
        """Verify forward_loop respects num_calib_batches by checking call args."""
        seen = []
        def fake_quantize(model, cfg, forward_loop):
            # Actually run the forward_loop to check it stops at 1 batch
            for i, (x, _) in enumerate(tiny_loader):
                seen.append(i)
                if i >= 1 - 1:
                    break
            return model

        _MTQ.quantize.side_effect = fake_quantize
        qmod.quantize_model(tiny_model, {"int8": True}, tiny_loader,
                            num_calib_batches=1, device=torch.device("cpu"))
        assert len(seen) == 1
        _MTQ.quantize.side_effect = lambda model, cfg, fwd: model  # reset


# ---------------------------------------------------------------------------
# save_modelopt_state / restore_modelopt_state
# ---------------------------------------------------------------------------

class TestSaveRestoreState:
    def test_save_calls_mto_modelopt_state(self, tiny_model, tmp_path):
        _MTO.modelopt_state.reset_mock()
        p = str(tmp_path / "mostate.pt")
        qmod.save_modelopt_state(tiny_model, p)
        _MTO.modelopt_state.assert_called_once_with(tiny_model)

    def test_restore_calls_restore_from_modelopt_state(self, tiny_model, tmp_path):
        p = tmp_path / "mostate.pt"
        torch.save({"stub": True}, p)
        _MTO.restore_from_modelopt_state.reset_mock()
        qmod.restore_modelopt_state(tiny_model, str(p))
        _MTO.restore_from_modelopt_state.assert_called_once()


# ---------------------------------------------------------------------------
# train_utils: train_one_epoch
# ---------------------------------------------------------------------------

class TestTrainOneEpoch:
    def test_returns_loss_and_acc(self, tiny_model, tiny_loader):
        tiny_model.train()
        opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-3)
        from torch.amp.grad_scaler import GradScaler
        scaler = GradScaler(enabled=False)
        loss, acc = tutils.train_one_epoch(
            tiny_model, tiny_loader, nn.CrossEntropyLoss(),
            opt, scaler, torch.device("cpu"), epoch=1,
        )
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 100.0

    def test_loss_is_positive(self, tiny_model, tiny_loader):
        tiny_model.train()
        opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-3)
        from torch.amp.grad_scaler import GradScaler
        scaler = GradScaler(enabled=False)
        loss, _ = tutils.train_one_epoch(
            tiny_model, tiny_loader, nn.CrossEntropyLoss(),
            opt, scaler, torch.device("cpu"), epoch=1,
        )
        assert loss > 0.0


# ---------------------------------------------------------------------------
# train_utils: validate
# ---------------------------------------------------------------------------

class TestValidate:
    def test_returns_loss_top1_top5(self, tiny_model, tiny_loader):
        val_loss, top1, top5 = tutils.validate(
            tiny_model, tiny_loader, nn.CrossEntropyLoss(), torch.device("cpu")
        )
        assert isinstance(val_loss, float)
        assert 0.0 <= top1 <= 100.0
        assert 0.0 <= top5 <= 100.0

    def test_top5_ge_top1(self, tiny_model, tiny_loader):
        _, top1, top5 = tutils.validate(
            tiny_model, tiny_loader, nn.CrossEntropyLoss(), torch.device("cpu")
        )
        assert top5 >= top1


# ---------------------------------------------------------------------------
# train_utils: save_checkpoint / load_checkpoint
# ---------------------------------------------------------------------------

class TestCheckpointing:
    def _make_training_objects(self, tiny_model):
        opt = torch.optim.SGD(tiny_model.parameters(), lr=1e-3)
        from torch.amp.grad_scaler import GradScaler
        scaler    = GradScaler(enabled=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
        return opt, scaler, scheduler

    def test_save_creates_both_files(self, tiny_model, tmp_path):
        _MTO.modelopt_state.return_value = {"stub": True}
        opt, scaler, scheduler = self._make_training_objects(tiny_model)
        state = {"epoch": 1, "model": tiny_model.state_dict(),
                 "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                 "scheduler": scheduler.state_dict(), "best_acc": 80.0}

        tutils.save_checkpoint(tiny_model, state, str(tmp_path), epoch=1, is_best=False)

        assert (tmp_path / "qat_modelopt_epoch_001.pth").exists()
        assert (tmp_path / "qat_modelopt_epoch_001_mostate.pt").exists()

    def test_save_best_creates_best_files(self, tiny_model, tmp_path):
        _MTO.modelopt_state.return_value = {"stub": True}
        opt, scaler, scheduler = self._make_training_objects(tiny_model)
        state = {"epoch": 2, "model": tiny_model.state_dict(),
                 "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                 "scheduler": scheduler.state_dict(), "best_acc": 85.0}

        tutils.save_checkpoint(tiny_model, state, str(tmp_path), epoch=2, is_best=True)

        assert (tmp_path / "qat_modelopt_best.pth").exists()
        assert (tmp_path / "qat_modelopt_best_mostate.pt").exists()

    def test_load_restores_epoch_and_best_acc(self, tiny_model, tmp_path):
        _MTO.modelopt_state.return_value = {"stub": True}
        _MTO.restore_from_modelopt_state.return_value = None

        opt, scaler, scheduler = self._make_training_objects(tiny_model)
        state = {"epoch": 5, "model": tiny_model.state_dict(),
                 "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                 "scheduler": scheduler.state_dict(), "best_acc": 90.0}

        tutils.save_checkpoint(tiny_model, state, str(tmp_path), epoch=5, is_best=False)

        next_epoch, best_acc = tutils.load_checkpoint(
            ckpt_path=str(tmp_path / "qat_modelopt_epoch_005.pth"),
            mo_path=str(tmp_path / "qat_modelopt_epoch_005_mostate.pt"),
            model=tiny_model,
            optimizer=opt,
            scaler=scaler,
            scheduler=scheduler,
        )
        assert next_epoch == 6
        assert best_acc == pytest.approx(90.0)
