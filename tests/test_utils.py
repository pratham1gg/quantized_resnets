"""Tests for src/utils.py — JSON helpers and run loading."""
import json
import pytest
from pathlib import Path

from utils import (
    ensure_dir,
    read_json,
    write_json,
    iter_result_jsons,
    load_runs,
    flatten_run,
    flatten_runs,
    print_run_summary,
)


# ---------------------------------------------------------------------------
# ensure_dir / read_json / write_json
# ---------------------------------------------------------------------------

def test_ensure_dir_creates_nested(tmp_path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert target.is_dir()
    assert result == target


def test_write_and_read_json(tmp_path):
    path = tmp_path / "test.json"
    data = {"key": "value", "num": 42}
    write_json(path, data)
    loaded = read_json(path)
    assert loaded == data


def test_write_json_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "deep" / "out.json"
    write_json(path, {"x": 1})
    assert path.exists()


# ---------------------------------------------------------------------------
# iter_result_jsons
# ---------------------------------------------------------------------------

def test_iter_result_jsons_finds_files(tmp_path):
    (tmp_path / "run_a").mkdir()
    (tmp_path / "run_b").mkdir()
    (tmp_path / "run_a" / "result.json").write_text('{"status":"ok"}')
    (tmp_path / "run_b" / "result.json").write_text('{"status":"ok"}')

    found = list(iter_result_jsons(tmp_path))
    assert len(found) == 2
    assert all(p.name == "result.json" for p in found)


def test_iter_result_jsons_empty_root(tmp_path):
    missing = tmp_path / "no_such_dir"
    assert list(iter_result_jsons(missing)) == []


# ---------------------------------------------------------------------------
# load_runs
# ---------------------------------------------------------------------------

def _write_run(base: Path, name: str, payload: dict):
    d = base / name
    d.mkdir()
    (d / "result.json").write_text(json.dumps(payload))


def test_load_runs_filters_by_status(tmp_path):
    _write_run(tmp_path, "ok_run",  {"status": "ok",    "run_id": "a"})
    _write_run(tmp_path, "err_run", {"status": "error", "run_id": "b"})

    ok_runs = load_runs(tmp_path, status="ok")
    assert len(ok_runs) == 1
    assert ok_runs[0]["run_id"] == "a"


def test_load_runs_none_status_returns_all(tmp_path):
    _write_run(tmp_path, "r1", {"status": "ok"})
    _write_run(tmp_path, "r2", {"status": "error"})

    all_runs = load_runs(tmp_path, status=None)
    assert len(all_runs) == 2


def test_load_runs_adds_path_keys(tmp_path):
    _write_run(tmp_path, "r1", {"status": "ok"})
    runs = load_runs(tmp_path, status=None)
    assert "_result_path" in runs[0]
    assert "_run_dir" in runs[0]


# ---------------------------------------------------------------------------
# flatten_run / flatten_runs
# ---------------------------------------------------------------------------

def test_flatten_run_structure():
    run = {
        "run_id": "test_run",
        "status": "ok",
        "system": {"torch": "2.0", "device": "cpu"},
        "config": {"backend": "pytorch", "batch_size": 1},
        "results": {"top1_acc": 75.0},
        "artifacts": {},
    }
    flat = flatten_run(run)
    assert flat["run_id"] == "test_run"
    assert flat["system.torch"] == "2.0"
    assert flat["cfg.backend"] == "pytorch"
    assert flat["res.top1_acc"] == 75.0


def test_flatten_runs_list():
    runs = [
        {"run_id": "a", "status": "ok", "system": {}, "config": {}, "results": {}, "artifacts": {}},
        {"run_id": "b", "status": "ok", "system": {}, "config": {}, "results": {}, "artifacts": {}},
    ]
    flat = flatten_runs(runs)
    assert len(flat) == 2
    assert flat[0]["run_id"] == "a"


# ---------------------------------------------------------------------------
# print_run_summary — smoke test (no crash, correct output)
# ---------------------------------------------------------------------------

def test_print_run_summary_ok(capsys):
    payload = {
        "run_id": "test_run_id",
        "status": "ok",
        "config": {
            "backend": "pytorch", "model_precision": "fp32",
            "input_quant_bits": 8, "device": "cpu", "batch_size": 1,
        },
        "results": {
            "top1_acc": 75.5, "top5_acc": 92.0,
            "infer_ms_avg": 12.3, "throughput_infer_sps": 81.3,
            "total_samples": 1000,
        },
    }
    print_run_summary(payload)
    out = capsys.readouterr().out
    assert "test_run_id" in out
    assert "75.50" in out


def test_print_run_summary_error_status(capsys):
    payload = {
        "run_id": "bad_run",
        "status": "error",
        "error": "something went wrong",
        "config": {},
        "results": {},
    }
    print_run_summary(payload)
    out = capsys.readouterr().out
    assert "error" in out.lower()
