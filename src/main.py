import time
from dataclasses import asdict
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn

from config import ExperimentConfig, with_overrides
from model import get_model
from data import get_dataloader
from eval import evaluate
from metrics import MetricsTracker # Import for type hinting
from utils import save_results, print_results, print_config


def run_experiment(
    cfg: ExperimentConfig,
    split: str = "val",
    criterion: Optional[nn.Module] = None,
    save_results_flag: bool = True,
) -> Tuple[Dict[str, Any], MetricsTracker]: # <--- CHANGED RETURN TYPE
    
    # Auto-detect device logic
    if cfg.model_precision == "int8" and cfg.device != "cpu":
        print("⚠️ WARNING: INT8 (X86Inductor) requires CPU. Switching device to 'cpu'.")
        cfg = with_overrides(cfg, device="cpu")

    if cfg.model_precision == "fp16" and cfg.device == "cpu":
        raise ValueError("FP16 precision requires a CUDA device, but 'cpu' was selected.")

    # Ensure criterion is set so we get LOSS PLOTS
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
        
    print_config(cfg)

    print(f"Loading {split} data...")
    loader = get_dataloader(cfg, split=split)
    print(f"Dataloader ready: {len(loader)} batches\n")

    print("Loading model...")
    model = get_model(cfg, dataloader=loader)
    print(f"Model loaded: {type(model).__name__}")

    # Compile for evaluation timing (skip for CPU INT8 unless you explicitly want it)
    if str(cfg.device).startswith("cuda"):
        print("Compiling model (torch.compile)...")
        model = torch.compile(model)

    start = time.perf_counter()
    
    # Get the tracker object back
    tracker = evaluate(model, loader, cfg, criterion=criterion)
    
    total_time = time.perf_counter() - start

    # Generate the summary dict for printing/saving
    results = tracker.summary()

    # Attach run metadata
    results["config"] = asdict(cfg)
    results["total_eval_time_sec"] = float(total_time)

    print_results(results)

    if save_results_flag:
        filepath = save_results(results, cfg)
        print(f"Results saved to: {filepath}\n")

    return results, tracker # <--- CHANGED: Return tuple