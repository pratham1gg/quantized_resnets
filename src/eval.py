import time
from typing import Optional, Dict, Any, Tuple

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
    ) -> MetricsTracker:  
    """
    Evaluate model and collect metrics. Returns the MetricsTracker object
    containing full history for plotting.
    """
    #exported PT2E models disallow eval()/train().
    metrics = MetricsTracker()

    max_batches = getattr(cfg, "num_eval_batches", None)

    print(f"Evaluating on {len(dataloader)} batches...")

    with torch.inference_mode():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_start = time.perf_counter()

            # Move images to the correct device/dtype for the selected precision
            images = ensure_input_dtype_for_model(images, cfg)

            # Move targets to the same device as outputs will be on
            targets = targets.to(images.device, non_blocking=True)


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

    return metrics  