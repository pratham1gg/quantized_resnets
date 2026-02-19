import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict, is_dataclass

from config import ExperimentConfig


def _json_safe(obj):
    """
    Make sure we can json.dump results even if they contain numpy / torch types.
    """
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_results(results: Dict[str, Any], cfg: ExperimentConfig, results_dir: str = "results"):
    """
    Save experiment results to JSON in a structured folder.

    Structure:
      results/
        resnet18_{precision}_{device}/
          metrics.json
    """
    # Create a unique directory name based on precision and device
    # e.g., "resnet18_int8_cpu" or "resnet18_fp32_cuda"
    exp_name = f"resnet18_{cfg.model_precision}_{cfg.device}"
    
    # Clean up device string (e.g. 'cuda:0' -> 'cuda')
    if ":" in exp_name:
        exp_name = exp_name.split(":")[0]

    output_dir = Path(results_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = "metrics.json"
    filepath = output_dir / filename

    # Ensure config is serializable
    if "config" not in results:
        if is_dataclass(cfg):
            results["config"] = asdict(cfg)
        else:
            results["config"] = cfg.__dict__

    print(f"Saving results to {filepath}...")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=_json_safe)

    return str(filepath)


def print_results(results: Dict[str, Any]) -> None:
    """
    Pretty print experiment results (keys match MetricsTracker.summary()).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS")
    print("=" * 70)

    print(f"Top-1 Accuracy:        {results['top1_acc']:.2f}%")
    print(f"Top-5 Accuracy:        {results['top5_acc']:.2f}%")

    if results.get("loss_avg", None) is not None:
        print(f"Avg Loss:              {results['loss_avg']:.4f}")

    # Times are per-batch in ms (as returned by metrics.summary)
    if results.get("infer_ms_avg", None) is not None:
        std = results.get("infer_ms_std", None)
        if std is not None:
            print(f"Inference Time:        {results['infer_ms_avg']:.2f} ± {std:.2f} ms/batch")
        else:
            print(f"Inference Time:        {results['infer_ms_avg']:.2f} ms/batch")

    if results.get("batch_ms_avg", None) is not None:
        print(f"Batch Time:            {results['batch_ms_avg']:.2f} ms/batch")

    if results.get("throughput_sps", None) is not None:
        print(f"Throughput:            {results['throughput_sps']:.2f} samples/sec")

    print(f"Total Samples:         {results['total_samples']}")
    if "total_eval_time_sec" in results:
        print(f"Total Eval Time:       {results['total_eval_time_sec']:.2f} sec")

    print("=" * 70 + "\n")


def print_config(cfg: ExperimentConfig) -> None:
    """
    Pretty print configuration.
    Only prints fields that exist on cfg.
    """
    print("=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)

    # core knobs
    print(f"Model precision:       {getattr(cfg, 'model_precision', 'N/A')}")
    print(f"Input quantization:    {getattr(cfg, 'input_quant_bits', 'N/A')}-bit")

    # stable settings
    print(f"Batch size:            {getattr(cfg, 'batch_size', 'N/A')}")
    print(f"Num workers:           {getattr(cfg, 'num_workers', 'N/A')}")
    print(f"Device:                {getattr(cfg, 'device', 'N/A')}")
    print(f"ImageNet path:         {getattr(cfg, 'imagenet_path', 'N/A')}")

    # optional fields (only if present)
    if hasattr(cfg, "num_eval_batches"):
        neb = getattr(cfg, "num_eval_batches")
        print(f"Num eval batches:      {neb if neb is not None else 'All'}")

    print("=" * 70 + "\n")