# main.py
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from config import ExperimentConfig
from model import get_model
from data import get_dataloader
from eval import evaluate
from utils import save_results, print_results, print_config


def run_experiment(
    cfg: ExperimentConfig,
    split: str = "val",
    criterion: Optional[nn.Module] = None,
    save_results_flag: bool = True,
) -> Dict[str, Any]:
    """
    Run one experiment (intended to be called from Jupyter).

    - cfg carries baseline + overrides (input_quant_bits, model_precision, etc.)
    - criterion is optional (pass nn.CrossEntropyLoss() if you want loss)
    """
    print_config(cfg)

    print("Loading model...")
    model = get_model(cfg)
    print(f"Model loaded: {type(model).__name__}")

    print(f"Loading {split} data...")
    loader = get_dataloader(cfg, split=split)
    print(f"Dataloader ready: {len(loader)} batches\n")

    start = time.perf_counter()
    results = evaluate(model, loader, cfg, criterion=criterion)
    total_time = time.perf_counter() - start

    # Attach run metadata
    results["config"] = asdict(cfg)
    results["total_eval_time_sec"] = float(total_time)

    print_results(results)

    if save_results_flag:
        filepath = save_results(results, cfg)
        print(f"Results saved to: {filepath}\n")

    return results


if __name__ == "__main__":
    # Keep __main__ minimal: a single sanity run.
    # Real sweeps should happen in Jupyter (your new workflow).
    cfg = ExperimentConfig(
        imagenet_path="/path/to/imagenet",  # <-- change
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=256,
        num_workers=4,
        input_quant_bits=8,
        model_precision="fp32",
    )

    # Optional: include loss
    criterion = nn.CrossEntropyLoss()

    run_experiment(cfg, split="val", criterion=criterion, save_results_flag=True)
