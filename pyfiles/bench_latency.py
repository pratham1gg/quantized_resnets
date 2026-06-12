"""
Latency benchmark — 10k iterations, raw per-iteration timings saved to JSON.

Usage
-----
  python bench_latency.py --backend pytorch --precision fp32 --input-bits 8
  python bench_latency.py --backend pytorch --precision fp16 --input-bits 8
  python bench_latency.py --backend tensorrt --precision int8 --input-bits 8
  python bench_latency.py --backend qat_modelopt --precision int8 --input-bits 8 --qat-ckpt qat/int8_in8b

Output
------
  results/latency_bench/<run_id>.json

  {
    "run_id": "...",
    "config": {...},
    "warmup_iters": 200,
    "bench_iters": 10000,
    "latencies_ms": [1.23, 1.19, ...],   # raw per-iteration inference times
  }
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.config import ExperimentConfig, set_seed
from src.data import build_runner_loaders
from src.model import get_model
from utils.precision import apply_precision


PROJECT_ROOT = ROOT.parent
CHECKPOINT_ROOT = PROJECT_ROOT / "training" / "checkpoints"
FP32_CHECKPOINT = CHECKPOINT_ROOT / "fp32" / "seed_42" / "best.pth"
OUTPUT_DIR = PROJECT_ROOT / "results" / "latency_bench"


def parse_args():
    p = argparse.ArgumentParser(description="Latency uncertainty benchmark")
    p.add_argument("--backend", required=True,
                   choices=["pytorch", "tensorrt", "qat_modelopt"])
    p.add_argument("--precision", required=True,
                   help="Model precision: fp32, fp16, int8, fp8, int4")
    p.add_argument("--input-bits", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp32-ckpt", default=None,
                   help="FP32 checkpoint path (default: training/checkpoints/fp32/seed_42/best.pth)")
    p.add_argument("--qat-ckpt", default=None,
                   help="QAT checkpoint subdir relative to training/checkpoints/ (e.g. qat/int8_in8b)")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def _build_model_pytorch(cfg, ckpt_path):
    model = apply_precision(get_model(cfg, checkpoint_path=ckpt_path), cfg)
    return model



def _build_model_qat(args, device):
    import modelopt.torch.opt as mto

    ckpt_dir = CHECKPOINT_ROOT / args.qat_ckpt
    ckpt_path = ckpt_dir / "qat_modelopt_best.pth"
    mo_path = ckpt_dir / "qat_modelopt_best_mostate.pt"

    fp32_ckpt = args.fp32_ckpt or str(FP32_CHECKPOINT)
    model = get_model(cfg=None, pretrained=False, checkpoint_path=fp32_ckpt)
    model = model.to(device)

    mo_state = torch.load(str(mo_path), map_location="cpu")
    mto.restore_from_modelopt_state(model, mo_state)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return model.eval()


def _bench_pytorch_like(model, dataloader, device, warmup, iters):
    data_iter = iter(dataloader)

    def _get_batch():
        nonlocal data_iter
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            images, _ = next(data_iter)
        return images.to(device, non_blocking=True)

    is_cuda = str(device).startswith("cuda")

    print(f"Warming up ({warmup} iters) ...")
    with torch.inference_mode():
        for _ in range(warmup):
            x = _get_batch()
            if is_cuda:
                torch.cuda.synchronize()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()

    print(f"Benchmarking ({iters} iters) ...")
    latencies = []
    with torch.inference_mode():
        for i in range(iters):
            x = _get_batch()
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(x)
            if is_cuda:
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

            if (i + 1) % 1000 == 0:
                print(f"  [{i+1}/{iters}]")

    return latencies


def _bench_tensorrt(cfg, warmup, iters):
    from trt.trt_builder import build_engine
    from runner import _get_trt_paths

    import tensorrt as trt

    onnx_path, engine_path, _ = _get_trt_paths(cfg)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    if not engine_path.exists():
        print(f"Building TRT engine ...")
        build_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            precision=cfg.model_precision,
            batch_size=cfg.batch_size,
            workspace_mb=cfg.trt_workspace_mb,
        )

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()

    in_name = None
    out_name = None
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            in_name = name
        else:
            out_name = name

    is_dynamic = engine.get_tensor_shape(out_name)[0] == -1
    device = torch.device(cfg.device)

    _, val_loader = build_runner_loaders(cfg)
    data_iter = iter(val_loader)

    def _get_batch():
        nonlocal data_iter
        try:
            images, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(val_loader)
            images, _ = next(data_iter)
        return images.to(dtype=torch.float32, device=device, non_blocking=True)

    def _run_once(x):
        if is_dynamic:
            context.set_input_shape(in_name, tuple(x.shape))
            out_shape = tuple(context.get_tensor_shape(out_name))
        else:
            out_shape = tuple(engine.get_tensor_shape(out_name))
        out_buf = torch.empty(out_shape, dtype=torch.float32, device=device)
        context.set_tensor_address(in_name, x.data_ptr())
        context.set_tensor_address(out_name, out_buf.data_ptr())
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()

    print(f"Warming up ({warmup} iters) ...")
    for _ in range(warmup):
        x = _get_batch()
        torch.cuda.synchronize(device)
        _run_once(x)
        torch.cuda.synchronize(device)

    print(f"Benchmarking ({iters} iters) ...")
    latencies = []
    for i in range(iters):
        x = _get_batch()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _run_once(x)
        torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        if (i + 1) % 1000 == 0:
            print(f"  [{i+1}/{iters}]")

    return latencies


def main():
    args = parse_args()

    _backend_map = {
        "pytorch": "torch_ptq",
        "tensorrt": "trt_ptq",
        "qat_modelopt": "torch_qat",
    }
    prefix = _backend_map[args.backend]
    run_id = f"{prefix}_{args.precision}_b{args.input_bits}_{args.device}"
    print(f"[bench_latency] run_id = {run_id}")

    cfg = ExperimentConfig(
        backend="pytorch" if args.backend == "qat_modelopt" else args.backend,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        input_quant_bits=args.input_bits,
        model_precision=args.precision if args.backend != "qat_modelopt" else "fp32",
        num_classes=100,
    )
    set_seed(cfg)

    ckpt_path = args.fp32_ckpt or str(FP32_CHECKPOINT)

    if args.backend == "pytorch":
        model = _build_model_pytorch(cfg, ckpt_path)
        _, val_loader = build_runner_loaders(cfg)
        latencies = _bench_pytorch_like(model, val_loader, args.device, args.warmup, args.iters)

    elif args.backend == "qat_modelopt":
        if not args.qat_ckpt:
            raise ValueError("--qat-ckpt is required for qat_modelopt backend")
        model = _build_model_qat(args, args.device)
        _, val_loader = build_runner_loaders(cfg)
        latencies = _bench_pytorch_like(model, val_loader, args.device, args.warmup, args.iters)

    elif args.backend == "tensorrt":
        latencies = _bench_tensorrt(cfg, args.warmup, args.iters)

    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    import numpy as np
    arr = np.array(latencies)
    print(f"\n--- Results ({len(latencies)} iterations) ---")
    print(f"  Mean:   {arr.mean():.3f} ms")
    print(f"  Std:    {arr.std():.3f} ms")
    print(f"  Min:    {arr.min():.3f} ms")
    print(f"  Max:    {arr.max():.3f} ms")
    print(f"  P50:    {np.percentile(arr, 50):.3f} ms")
    print(f"  P95:    {np.percentile(arr, 95):.3f} ms")
    print(f"  P99:    {np.percentile(arr, 99):.3f} ms")

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    payload = {
        "run_id": run_id,
        "config": {
            "backend": args.backend,
            "precision": args.precision,
            "input_bits": args.input_bits,
            "batch_size": args.batch_size,
            "device": args.device,
            "seed": args.seed,
        },
        "warmup_iters": args.warmup,
        "bench_iters": args.iters,
        "latencies_ms": latencies,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
