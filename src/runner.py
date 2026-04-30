# =============================================================================
# runner.py — Experiment entry point
#
# Workflow
# --------
# run_experiment(cfg) routes to one of three backends:
#
#   1. pytorch
#      get_model → apply_precision (fp32/fp16) → evaluate
#
#   2. torchao_cpu_ptq
#      get_model → quantize_int8_x86_pt2e (INT8 calibration on CPU) → evaluate
#
#   3. tensorrt  (_run_tensorrt)
#      Step 1 — ONNX export (skipped if file already exists)
#               fp32/fp16      → resnet18.onnx  (plain export)
#               int8/fp8/int4  → resnet18_<prec>_qdq.onnx  (QDQ-annotated, pre-built by modelopt)
#      Step 2 — TRT engine build (skipped if .engine file already exists)
#               All quantized modes (int8/fp8/int4) read their scales from Q/DQ nodes in the ONNX.
#               No runtime calibrator is needed.
#               Engine filename encodes run_id (precision + batch size + device) so
#               different configs never overwrite each other.
#      Step 3 — Inference via trt_evaluate
#
# Results are written to runs/<run_id>/result.json.
# =============================================================================

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from config import ExperimentConfig, set_seed
from data import build_runner_loaders
from model import get_model
from precision import apply_precision
from quant_ptq_cpu import quantize_int8_x86_pt2e
from eval import evaluate
from metrics import MetricsTracker


def _make_payload(cfg: ExperimentConfig, tracker: MetricsTracker) -> Dict[str, Any]:
    return {
        "status" : "ok",
        "run_id" : cfg.run_id(),
        "system" : cfg.stamp(),
        "config" : cfg.to_dict(),
        "results": tracker.summary(),
        "error"  : None,
    }


def _save_result_json(payload: Dict[str, Any], cfg: ExperimentConfig) -> str:
    path = cfg.result_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return str(path)


def _get_trt_paths(cfg: ExperimentConfig) -> Tuple[Path, Path, Path]:
    """
    Return (onnx_path, engine_path, calib_cache_path).
    Anchored to repo root so paths don't depend on CWD.

    ONNX file selected by precision:
      fp32 / fp16         -> resnet18.onnx              (plain export)
      int8 / fp8 / int4   -> resnet18_<prec>_qdq.onnx   (QDQ-annotated, from modelopt)
    """
    repo_root  = Path(__file__).resolve().parents[1]
    onnx_dir   = repo_root / "onnx"
    engine_dir = repo_root / "engines"

    # int8, fp8, and int4 all use a QDQ-annotated ONNX from modelopt.
    # fp32 / fp16 use the plain exported ONNX.
    if cfg.model_precision in ("int8", "fp8", "int4"):
        onnx_name = f"resnet18_{cfg.model_precision}_qdq.onnx"
    else:
        onnx_name = "resnet18.onnx"

    onnx_path   = onnx_dir   / onnx_name
    engine_path = engine_dir / f"{cfg.run_id()}.engine"
    calib_cache = engine_dir / f"{cfg.run_id()}.calib_cache"

    onnx_dir.mkdir(parents=True, exist_ok=True)
    engine_dir.mkdir(parents=True, exist_ok=True)
    return onnx_path, engine_path, calib_cache


def _run_tensorrt(
    cfg: ExperimentConfig,
    criterion: Optional[nn.Module],
) -> Tuple[Dict[str, Any], MetricsTracker]:
    """3-step TRT pipeline: ONNX export → engine build → inference."""
    from onnx_exporter import ONNXExporter
    from trt_builder import build_engine
    from trt_infer import trt_evaluate

    onnx_path, engine_path, calib_cache = _get_trt_paths(cfg)

    # Step 1: Export to ONNX (skip if already done)
    if not onnx_path.exists():
        print("[runner] Step 1/3 — Exporting to ONNX ...")
        model = get_model(cfg)
        ONNXExporter(model, cfg.device, onnx_path).export_model(
            opset_version     = cfg.trt_opset if cfg.trt_opset > 1 else 17,
            dynamic_batch     = True,  # must be True so TRT optimization profile works
            dummy_input_shape = (1, 3, 224, 224),  # always 1 for tracing; batch size is set by the dynamic axis
        )
    else:
        print(f"[runner] Step 1/3 — ONNX exists, skipping: {onnx_path}")

    # Step 2: Build TRT engine (skip if already done)
    if not engine_path.exists():
        print(f"[runner] Step 2/3 — Building TRT engine (precision={cfg.model_precision}) ...")

        # INT8/FP8/INT4 QDQ ONNXes from modelopt may have a fixed batch dim.
        # Warn early if it doesn't match cfg.batch_size so the user knows
        # to re-export from modelopt with the correct batch size.
        if cfg.model_precision in ("int8", "fp8", "int4") and onnx_path.exists():
            import onnx
            onnx_model  = onnx.load(str(onnx_path), load_external_data=False)
            onnx_batch  = onnx_model.graph.input[0].type.tensor_type.shape.dim[0].dim_value
            if onnx_batch not in (0, cfg.batch_size):  # 0 means dynamic in ONNX proto
                print(f"[runner] WARNING: {onnx_path.name} has fixed batch={onnx_batch} "
                      f"but cfg.batch_size={cfg.batch_size}. "
                      f"Re-export from modelopt with batch_size={cfg.batch_size} for correct behaviour.")

        # All quantized modes (int8, fp8, int4) get their scales from Q/DQ nodes
        # in the modelopt-annotated ONNX — no runtime calibrator needed.
        build_engine(
            onnx_path    = onnx_path,
            engine_path  = engine_path,
            precision    = cfg.model_precision,   # "fp32"|"fp16"|"int8"|"fp8"|"int4"
            batch_size   = cfg.batch_size,
            workspace_mb = cfg.trt_workspace_mb,
        )
    else:
        print(f"[runner] Step 2/3 — Engine exists, skipping: {engine_path}")

    # Step 3: Run inference
    print("[runner] Step 3/3 — Running TRT inference ...")
    _, val_loader = build_runner_loaders(cfg)
    tracker = trt_evaluate(engine_path, cfg, val_loader, criterion)

    return _make_payload(cfg, tracker), tracker


def run_experiment(
    cfg: ExperimentConfig,
    criterion: Optional[nn.Module] = None,
    save_results_flag: bool = True,
    use_torch_compile: bool = False,
) -> Tuple[Dict[str, Any], Optional[MetricsTracker]]:
    cfg = cfg.normalized()
    cfg.validate()
    set_seed(cfg)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    tracker: Optional[MetricsTracker] = None
    t0 = time.perf_counter()

    if cfg.backend == "pytorch":
        model  = apply_precision(get_model(cfg), cfg)
        _, val_loader = build_runner_loaders(cfg)
        if use_torch_compile and cfg.device.startswith("cuda"):
            model = torch.compile(model)
        tracker = evaluate(model, val_loader, cfg, criterion=criterion)
        payload = _make_payload(cfg, tracker)

    elif cfg.backend == "torchao_cpu_ptq":
        train_loader, val_loader = build_runner_loaders(cfg)
        model   = quantize_int8_x86_pt2e(get_model(cfg), train_loader, calib_num_batches=cfg.cpu_calib_num_batches)
        tracker = evaluate(model, val_loader, cfg, criterion=criterion)
        payload = _make_payload(cfg, tracker)

    elif cfg.backend == "tensorrt":
        payload, tracker = _run_tensorrt(cfg, criterion=criterion)

    else:
        raise ValueError(f"Unknown backend: '{cfg.backend}'")

    payload["total_eval_time_sec"] = round(time.perf_counter() - t0, 3)

    if save_results_flag:
        saved = _save_result_json(payload, cfg)
        print(f"[runner] Results saved: {saved}")

    return payload, tracker