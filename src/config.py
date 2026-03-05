from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from typing import Optional, Literal, Dict, Any
from pathlib import Path
import json
import time

import torch
import random
import numpy as np


Backend = Literal["pytorch", "torchao_cpu_ptq", "tensorrt"]
Precision = Literal["fp32", "fp16", "int8", "int4", "fp4"]


@dataclass(frozen=True)
class ExperimentConfig:
    imagenet_path: str = "/home/pf4636/imagenet2"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 256
    num_workers: int = 8
    seed: int = 42

    # Experiment knobs
    input_quant_bits: int = 8                   # {8,4,2,1}
    model_precision: Precision = "fp32"         # fp32 / fp16 / int8 / int4 / fp4
    backend: Backend = "pytorch"                # pytorch / torchao_cpu_ptq / tensorrt

    # Evaluation control
    num_eval_batches: Optional[int] = None      # None = full val
    cpu_calib_num_batches: int = 10             # for ptq_cpu quant
    cpu_calib_split: str = "val"

    # Output control
    output_root: str = "../runs"                       # canonical single output root

    # TensorRT-specific (safe defaults)
    trt_opset: int = 1
    trt_static_shape: bool = True
    trt_workspace_mb: int = 2048
    # TRT precision is driven by model_precision: "fp32"|"fp16"|"int8"|"fp8"|"int4"
    # INT8 is the only mode that needs a calibrator (runner.py wires it up automatically)
    # FP8 and INT4 require a QDQ-annotated ONNX from modelopt — no calibrator needed
    trt_calib_num_batches: int = 32             # used only for INT8 calibration
    trt_calib_seed: int = 42
    trt_engine_tag: str = ""                    # optional suffix, not required

    def normalized(self) -> "ExperimentConfig":
        """Return a config with normalized string fields."""
        prec = str(self.model_precision).lower().strip()
        dev = str(self.device).lower().strip()
        be = str(self.backend).lower().strip()
        return replace(self, model_precision=prec, device=dev, backend=be)

    def validate(self) -> None:
        cfg = self.normalized()

        if cfg.input_quant_bits not in ( 1, 2, 4, 8):
            raise ValueError(f"input_quant_bits must be one of (8,4,2), got {cfg.input_quant_bits}")

        if cfg.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {cfg.batch_size}")
        if cfg.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {cfg.num_workers}")

        # Backend/precision compatibility
        if cfg.backend == "pytorch":
            if cfg.model_precision not in ("fp32", "fp16"):
                raise ValueError(f"backend='pytorch' supports only fp32/fp16, got {cfg.model_precision}")

        if cfg.backend == "torchao_cpu_ptq":
            if not cfg.device.startswith("cpu"):
                raise ValueError("backend='torchao_cpu_ptq' must run on CPU (device='cpu').")
            if cfg.model_precision != "int8":
                raise ValueError("backend='torchao_cpu_ptq' expects model_precision='int8'.")

        if cfg.backend == "tensorrt":
            if not cfg.device.startswith("cuda"):
                raise ValueError("backend='tensorrt' requires device='cuda'.")
            if cfg.model_precision not in ("fp32", "fp16", "int8", "fp8", "int4"):
                raise ValueError(f"Unsupported precision for tensorrt: '{cfg.model_precision}'. "
                                 f"Expected one of: fp32, fp16, int8, fp8, int4.")

        if cfg.model_precision == "fp16" and not cfg.device.startswith("cuda"):
            raise ValueError("model_precision='fp16' is intended for CUDA (device='cuda').")

    #JSon File naming starts here
    def run_id(self) -> str:
        """Gives the file name aka run_id"""
        cfg = self.normalized()
        parts = [
            "resnet18",
            cfg.backend,
            cfg.model_precision,
            f"in{cfg.input_quant_bits}b",
            cfg.device.split(":")[0],
            f"bs{cfg.batch_size}",
        ]
        if cfg.backend == "tensorrt":
            if cfg.trt_engine_tag:
                parts.append(cfg.trt_engine_tag)
        return "_".join(parts)

    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_id()

    def result_json_path(self) -> Path:
        return self.run_dir() / "result.json"
    #Json file naming ends here
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Keep paths as strings, dataclass -> dict already does that.
        return d

    def to_json_str(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    #to be used in utils while saving
    def stamp(self) -> Dict[str, Any]:
        """env stamp for results."""
        return {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": self.device,
        }


def set_seed(cfg: ExperimentConfig) -> None:
    # Keep deterministic as much as possible
    random.seed(cfg.seed)
    np.random.seed(cfg.seed) # just in case i use numpy
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def with_overrides(cfg: ExperimentConfig, **kwargs) -> ExperimentConfig: 
    new_cfg = replace(cfg, **kwargs).normalized() 
    new_cfg.validate() 
    return new_cfg