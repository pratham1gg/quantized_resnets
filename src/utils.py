"""
Utility helpers for reading, writing, and summarising experiment result JSONs.

Functions
---------
ensure_dir        -- Create a directory (and parents) if it does not exist.
read_json         -- Load a JSON file into a dict.
write_json        -- Serialise a dict to a JSON file, creating parent dirs as needed.
iter_result_jsons -- Yield all ``result.json`` paths under an output root, sorted.
load_runs         -- Load all result JSONs under a root, optionally filtered by status.
flatten_run       -- Collapse a run payload into a single flat dict (system/cfg/res/art).
flatten_runs      -- Apply ``flatten_run`` to a list of runs.
print_run_summary -- Pretty-print key metrics from a runner payload to stdout.
"""


import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

JsonDict = Dict[str, Any]


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | Path) -> JsonDict:
    path = Path(path)
    with path.open("r") as f:
        return json.load(f)


def write_json(path: str | Path, payload: JsonDict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def iter_result_jsons(output_root: str | Path = "./runs") -> Iterable[Path]:
    root = Path(output_root)
    if not root.exists():
        return []
    return sorted(root.rglob("result.json"))


def load_runs(output_root: str | Path = "./runs", *, status: Optional[str] = "ok") -> List[JsonDict]:
    runs: List[JsonDict] = []
    for p in iter_result_jsons(output_root):
        r = read_json(p)
        if status is None or r.get("status") == status:
            r["_result_path"] = str(p)
            r["_run_dir"] = str(p.parent)
            runs.append(r)
    return runs


def flatten_run(run: JsonDict) -> JsonDict:
    """
    Flatten your schema into a single row:
      system.* , cfg.* , res.* , art.*
    """
    flat: JsonDict = {
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "error": run.get("error"),
        "total_eval_time_sec": run.get("total_eval_time_sec"),
        "_result_path": run.get("_result_path"),
        "_run_dir": run.get("_run_dir"),
    }

    for k, v in (run.get("system") or {}).items():
        flat[f"system.{k}"] = v
    for k, v in (run.get("config") or {}).items():
        flat[f"cfg.{k}"] = v
    for k, v in (run.get("results") or {}).items():
        flat[f"res.{k}"] = v
    for k, v in (run.get("artifacts") or {}).items():
        flat[f"art.{k}"] = v

    return flat


def flatten_runs(runs: List[JsonDict]) -> List[JsonDict]:
    return [flatten_run(r) for r in runs]


def print_run_summary(payload: JsonDict) -> None:
    """
    Print from *runner payload* (new schema).
    """
    cfg = payload.get("config", {}) or {}
    res = payload.get("results", {}) or {}

    print("=" * 70)
    print(payload.get("run_id", "<no run_id>"))
    print("=" * 70)
    print(f"status: {payload.get('status')}")
    if payload.get("status") != "ok":
        print(f"error: {payload.get('error')}")
        print("=" * 70)
        return

    print(f"backend: {cfg.get('backend')} | precision: {cfg.get('model_precision')} | in_bits: {cfg.get('input_quant_bits')}")
    print(f"device: {cfg.get('device')} | bs: {cfg.get('batch_size')} | total_samples: {res.get('total_samples')}")
    print(f"top1: {res.get('top1_acc'):.2f}% | top5: {res.get('top5_acc'):.2f}%")
    if res.get("infer_ms_avg") is not None:
        print(f"infer: {res.get('infer_ms_avg'):.2f} ms/batch (std {res.get('infer_ms_std')})")
    if res.get("throughput_infer_sps") is not None:
        print(f"throughput_infer: {res.get('throughput_infer_sps'):.2f} samples/s")
    print("=" * 70)