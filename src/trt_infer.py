

import time
from pathlib import Path
from typing import Optional

import torch
import tensorrt as trt

from config import ExperimentConfig
from metrics import MetricsTracker, WARMUP_BATCHES

_LOGGER = trt.Logger(trt.Logger.WARNING)

def _find_tensor(engine: trt.ICudaEngine, mode: trt.TensorIOMode) -> str:
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == mode:
            return name
    raise RuntimeError(f"No tensor with mode {mode} found in engine.")

def trt_evaluate(
    engine_path: str | Path,
    cfg: ExperimentConfig,
    dataloader: torch.utils.data.DataLoader,
    criterion: Optional[torch.nn.Module] = None,
) -> MetricsTracker:
    engine_path = Path(engine_path)
    device      = torch.device(cfg.device)

    runtime = trt.Runtime(_LOGGER)
    engine  = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Failed to load TRT engine: {engine_path}")

    context = engine.create_execution_context()
    stream  = torch.cuda.Stream(device=device)

    in_name  = _find_tensor(engine, trt.TensorIOMode.INPUT)
    out_name = _find_tensor(engine, trt.TensorIOMode.OUTPUT)

    is_dynamic = engine.get_tensor_shape(out_name)[0] == -1

    print(f"[trt_infer] Engine: {engine_path}")
    print(f"[trt_infer] Input: '{in_name}'  Output: '{out_name}'  Dynamic batch: {is_dynamic}")

    metrics        = MetricsTracker()
    max_batches    = cfg.num_eval_batches
    warmup_batches = WARMUP_BATCHES
    effective      = len(dataloader) if max_batches is None else min(len(dataloader), max_batches)
    print(f"[trt_infer] Evaluating {effective} batches (first {warmup_batches} are warmup) ...")

    for batch_idx, (images, targets) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        if batch_idx == warmup_batches:
            print(f"[trt_infer] --- Warmup complete ({warmup_batches} batches) — starting metric collection ---")

        batch_start = time.perf_counter()
        images  = images.to(dtype=torch.float32, device=device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        actual_bs = images.shape[0]

        if is_dynamic:
            context.set_input_shape(in_name, tuple(images.shape))
            out_buf = torch.empty(
                tuple(context.get_tensor_shape(out_name)),
                dtype=torch.float32, device=device,
            )
        else:
            engine_bs = engine.get_tensor_shape(out_name)[0]
            if actual_bs < engine_bs:
                pad    = torch.zeros(engine_bs - actual_bs, *images.shape[1:], device=device)
                images = torch.cat([images, pad], dim=0)
            out_buf = torch.empty(
                tuple(engine.get_tensor_shape(out_name)),
                dtype=torch.float32, device=device,
            )

        torch.cuda.synchronize(device)
        infer_start = time.perf_counter()
        context.set_tensor_address(in_name,  images.data_ptr())
        context.set_tensor_address(out_name, out_buf.data_ptr())
        context.execute_async_v3(stream_handle=stream.cuda_stream)
        stream.synchronize()
        torch.cuda.synchronize(device)
        infer_time = time.perf_counter() - infer_start
        batch_time = time.perf_counter() - batch_start

        logits     = out_buf[:actual_bs]
        loss_value = float(criterion(logits, targets).item()) if criterion else None

        if batch_idx >= warmup_batches:
            metrics.update(
                outputs      = logits.clone(),
                targets      = targets,
                loss_value   = loss_value,
                batch_time_s = batch_time,
                infer_time_s = infer_time,
                batch_size   = actual_bs,
            )

        if batch_idx >= warmup_batches and (batch_idx + 1) % 10 == 0:
            s = metrics.summary()
            print(f"  [{batch_idx + 1}/{effective}]  "
                  f"Top-1: {s['top1_acc']:.2f}%  "
                  f"Top-5: {s['top5_acc']:.2f}%  "
                  f"Infer: {s['infer_ms_avg']:.2f} ms/batch")

    return metrics