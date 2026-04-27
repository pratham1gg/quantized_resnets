import time
from typing import Optional

import torch
import torch.nn as nn

from config import ExperimentConfig
from metrics import MetricsTracker, WARMUP_BATCHES
from precision import ensure_input_dtype

def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cfg: ExperimentConfig,
    criterion: Optional[nn.Module] = None,
) -> MetricsTracker:

    metrics        = MetricsTracker()
    max_batches    = cfg.num_eval_batches
    warmup_batches = WARMUP_BATCHES

    total_batches = len(dataloader)
    effective_batches = total_batches if max_batches is None else min(total_batches, max_batches)
    print(f"Evaluating on {effective_batches} batches (first {warmup_batches} are warmup)...")

    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            if batch_idx == warmup_batches:
                print(f"  --- Warmup complete ({warmup_batches} batches) — starting metric collection ---")

            batch_start = time.perf_counter()

            images = ensure_input_dtype(images, cfg)
            targets = targets.to(images.device, non_blocking=True)

            if images.device.type == "cuda":
                torch.cuda.synchronize()

            infer_start = time.perf_counter()
            outputs = model(images)

            if images.device.type == "cuda":
                torch.cuda.synchronize()

            infer_time = time.perf_counter() - infer_start

            loss_value = None
            if criterion is not None:
                loss_value = float(criterion(outputs, targets).item())

            batch_time = time.perf_counter() - batch_start

            if batch_idx >= warmup_batches:
                metrics.update(
                    outputs=outputs,
                    targets=targets,
                    loss_value=loss_value,
                    batch_time_s=batch_time,
                    infer_time_s=infer_time,
                    batch_size=int(images.shape[0]),
                )

            if batch_idx >= warmup_batches and (batch_idx + 1) % 10 == 0:
                s = metrics.summary()
                print(
                    f"  Batch [{batch_idx + 1}/{effective_batches}] "
                    f"Top-1: {s['top1_acc']:.2f}% | Top-5: {s['top5_acc']:.2f}% | "
                    f"Infer: {s['infer_ms_avg']:.2f} ms/batch"
                )

    return metrics