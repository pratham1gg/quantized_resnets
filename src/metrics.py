from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
import torch

@dataclass
class MetricsTracker:
    """
    Tracks:
      - Top-1 / Top-5 accuracy (running + final)
      - Loss
      - Batch time and pure inference time
      - Stores per-batch values for plotting
    """
    correct_top1: int = 0
    correct_top5: int = 0
    total: int = 0

    batch_times_s: List[float] = field(default_factory=list)
    infer_times_s: List[float] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)

    top1_running: List[float] = field(default_factory=list)
    top5_running: List[float] = field(default_factory=list)

    def reset(self) -> None:
        self.correct_top1 = 0
        self.correct_top5 = 0
        self.total = 0
        self.batch_times_s.clear()
        self.infer_times_s.clear()
        self.losses.clear()
        self.top1_running.clear()
        self.top5_running.clear()

    @torch.no_grad()
    def update(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        loss_value: Optional[float],
        batch_time_s: float,
        infer_time_s: float,
        batch_size: int,
    ) -> None:
        """
        Args:
            outputs: logits, shape [N, num_classes]
            targets: labels, shape [N]
            loss_value: float or None
            batch_time_s: end-to-end batch time in seconds
            infer_time_s: model forward time in seconds
            batch_size: N
        """
        self.total += int(batch_size)

        # top-k predictions once
        # pred: [N, 5]
        _, pred = outputs.topk(k=5, dim=1, largest=True, sorted=True)

        # Top-1: compare first column
        self.correct_top1 += int((pred[:, 0] == targets).sum().item())

        # Top-5: check if target is in any of the 5
        # matches: [N] bool
        matches = (pred == targets.unsqueeze(1)).any(dim=1)
        self.correct_top5 += int(matches.sum().item())

        self.batch_times_s.append(float(batch_time_s))
        self.infer_times_s.append(float(infer_time_s))

        if loss_value is not None:
            self.losses.append(float(loss_value))

        # store running accuracies for plots
        self.top1_running.append(100.0 * self.correct_top1 / self.total)
        self.top5_running.append(100.0 * self.correct_top5 / self.total)

    def summary(self) -> Dict[str, Any]:
        """
        Returns a dict ready to dump to JSON.
        """
        if self.total == 0:
            return {
                "top1_acc": 0.0,
                "top5_acc": 0.0,
                "loss_avg": None,
                "infer_ms_avg": None,
                "infer_ms_std": None,
                "batch_ms_avg": None,
                "throughput_sps": None,
                "total_samples": 0,
            }

        batch_times = np.array(self.batch_times_s, dtype=np.float64)
        infer_times = np.array(self.infer_times_s, dtype=np.float64)

        infer_total_time = float(infer_times.sum()) if infer_times.size else 0.0
        throughput_infer = (self.total / infer_total_time) if infer_total_time > 0 else None

        total_time = float(batch_times.sum()) if batch_times.size else 0.0
        throughput = (self.total / total_time) if total_time > 0 else None

        loss_avg = float(np.mean(self.losses)) if len(self.losses) > 0 else None

        return {
            "top1_acc": 100.0 * self.correct_top1 / self.total,
            "top5_acc": 100.0 * self.correct_top5 / self.total,
            "loss_avg": loss_avg,

            "batch_ms_avg": float(batch_times.mean() * 1000.0) if batch_times.size else None,
            "infer_ms_avg": float(infer_times.mean() * 1000.0) if infer_times.size else None,
            "infer_ms_std": float(infer_times.std() * 1000.0) if infer_times.size else None,   
            
            "throughput_infer_sps": float(throughput_infer) if throughput_infer is not None else None, # forward-pass throughput

            "throughput_sps": float(throughput) if throughput is not None else None, # end to end pipeline throughput
            "total_samples": int(self.total),
            "total_batches": int(len(self.infer_times_s)),
        }

    def get_metrics(self) -> Dict[str, Any]:
        return self.summary()
