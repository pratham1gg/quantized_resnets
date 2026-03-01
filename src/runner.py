import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from config import ExperimentConfig, set_seed
from data import get_dataloader
from model import get_model
from precision import apply_precision
from quant_ptq_cpu import quantize_int8_x86_pt2e
from eval import evaluate
from metrics import MetricsTracker


def _ensure_run_dir(cfg: ExperimentConfig) -> Path:
    d = cfg.run_dir()
    d.mkdir(parents=True, exist_ok=True)
    (d / "artifacts").mkdir(parents=True, exist_ok=True)
    return d


def _save_result_json(payload: Dict[str, Any], cfg: ExperimentConfig) -> str:
    _ensure_run_dir(cfg)
    path = cfg.result_json_path()
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True) #Payload: JSON record of one experiment run

    return str(path)


def run_experiment(
    cfg: ExperimentConfig,
    split: str = "val",
    criterion: Optional[nn.Module] = None,
    save_results_flag: bool = True,
    use_torch_compile: bool = False,
) -> Tuple[Dict[str, Any], Optional[MetricsTracker]]:
    """
    Unified experiment runner. Notebooks call this.
    Returns: (result_payload, metrics tracker (opt))
    """
    cfg = cfg.normalized()
    cfg.validate()
    set_seed(cfg)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    t0 = time.perf_counter()

    # ---- Backend routing ----
    if cfg.backend == "pytorch":
        loader = get_dataloader(cfg, split=split)
        model = get_model(cfg)
        model = apply_precision(model, cfg)

        if use_torch_compile and str(cfg.device).startswith("cuda"):
            model = torch.compile(model)

        tracker = evaluate(model, loader, cfg, criterion=criterion)
        results = tracker.summary()

        payload: Dict[str, Any] = { 
            "status": "ok",
            "run_id": cfg.run_id(),
            "system": cfg.stamp(),
            "config": cfg.to_dict(),
            "results": results,
            "artifacts": {},
            "error": None,
        } 
    elif cfg.backend == "torchao_cpu_ptq":
        eval_loader = get_dataloader(cfg, split=split)
        calib_loader = get_dataloader(cfg, split=cfg.cpu_calib_split)
        model = get_model(cfg)
        model = quantize_int8_x86_pt2e(model, calib_loader, calib_num_batches=cfg.cpu_calib_num_batches)  # make configurable later

        tracker = evaluate(model, eval_loader, cfg, criterion=criterion)
        results = tracker.summary()

        payload = {
            "status": "ok",
            "run_id": cfg.run_id(),
            "system": cfg.stamp(),
            "config": cfg.to_dict(),
            "results": results,
            "artifacts": {},
            "error": None,
        }

    elif cfg.backend == "tensorrt":
        # Stub for now. We'll implement via:
        # export_onnx.py -> trt_build.py -> trt_infer.py
        payload = {
            "status": "error",
            "run_id": cfg.run_id(),
            "system": cfg.stamp(),
            "config": cfg.to_dict(),
            "results": {},
            "artifacts": {},
            "error": "TensorRT backend not implemented yet in runner.py",
        }
        tracker = None

    else:
        raise ValueError(f"Unknown backend: {cfg.backend}")

    payload["total_eval_time_sec"] = float(time.perf_counter() - t0)

    if save_results_flag:
        saved = _save_result_json(payload, cfg)
        print(f"[saved] {saved}")

    return payload, tracker