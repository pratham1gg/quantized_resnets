from __future__ import annotations

import time
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from config import ExperimentConfig
from quantization import ensure_input_dtype_for_model
from metrics import MetricsTracker


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    cfg: ExperimentConfig,
    criterion: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
    """
    Evaluate model and collect metrics (top1/top5, latency, optional loss).

    - criterion: if provided, we compute average loss.
    """
    model.eval()
    metrics = MetricsTracker()

    # Optional: limit batches (if you have this in config)
    max_batches = getattr(cfg, "num_eval_batches", None)

    print(f"Evaluating on {len(dataloader)} batches...")

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_start = time.perf_counter()

            # Move to device
            images = images.to(cfg.device, non_blocking=True)
            targets = targets.to(cfg.device, non_blocking=True)

            # Match input dtype to model precision (fp32/fp16)
            images = ensure_input_dtype_for_model(images, cfg)

            # ----- inference timing -----
            if str(cfg.device).startswith("cuda"):
                torch.cuda.synchronize()
            infer_start = time.perf_counter()

            outputs = model(images)

            if str(cfg.device).startswith("cuda"):
                torch.cuda.synchronize()
            infer_time = time.perf_counter() - infer_start
            # ---------------------------

            loss_value = None
            if criterion is not None:
                loss_value = float(criterion(outputs, targets).item())

            batch_time = time.perf_counter() - batch_start

            metrics.update(
                outputs=outputs,
                targets=targets,
                loss_value=loss_value,
                batch_time_s=batch_time,
                infer_time_s=infer_time,
                batch_size=images.shape[0],
            )

            # simple progress print
            if (batch_idx + 1) % 10 == 0:
                s = metrics.summary()
                print(
                    f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                    f"Top-1: {s['top1_acc']:.2f}% | Top-5: {s['top5_acc']:.2f}% | "
                    f"Infer: {s['infer_ms_avg']:.2f} ms/batch"
                )

    return metrics.summary()
