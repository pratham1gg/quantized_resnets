# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Thesis experiments comparing ResNet-18 inference accuracy and latency across model precisions (fp32/fp16/int8/fp8/int4) and input-quantization bit-widths (1/2/4/8), using three backends: vanilla PyTorch, torchao CPU PT2E PTQ, and TensorRT. All reported experiments are generated exclusively using code in `src/`. Other top-level Python directories (`train_resnet18/`, `qat/`) are auxiliary training pipelines.

## Running experiments

There is no test suite, lint config, or package manager entrypoint. The codebase is driven from Jupyter notebooks in `notebook/` which import from `src/` and call `run_experiment(cfg)`.

```python
# From a notebook with sys.path.insert(0, "../src"):
from config import ExperimentConfig
from runner import run_experiment

cfg = ExperimentConfig(backend="tensorrt", model_precision="int8",
                       input_quant_bits=4, batch_size=1, device="cuda")
payload, tracker = run_experiment(cfg)   # writes runs/<run_id>/result.json
```

Auxiliary training scripts (standalone CLIs, not part of the `src/` experiment runner):
- `python train_resnet18/train.py` — FP32 baseline training on ImageNet-1K.
- `python qat/train_qat_phase1.py` — Phase-1 INT8 QAT fine-tuning using `pytorch-quantization`.

Notebooks are numbered (`00_env_sanity` → `06_qat_inference`) and are intended to be run in order.

## Architecture

### Config-driven experiment runner

[src/config.py](src/config.py) defines a frozen `ExperimentConfig` dataclass that fully determines every run. It:
- Validates backend ↔ precision ↔ device compatibility in `validate()`.
- Computes a deterministic `run_id()` like `resnet18_tensorrt_int8_in4b_cuda_bs1`. All outputs (result JSON, TRT engine, calibration cache) are keyed on this id so different configs never clobber each other.

[src/runner.py](src/runner.py) is the single entry point `run_experiment(cfg)`, which routes to one of three backends:

| Backend | Flow |
|---|---|
| `pytorch` | `get_model` → `apply_precision` (fp32/fp16 only) → `evaluate` |
| `torchao_cpu_ptq` | `get_model` → `quantize_int8_x86_pt2e` (CPU calibration) → `evaluate` |
| `tensorrt` | 3-step pipeline: ONNX export → engine build → `trt_evaluate` |

### TensorRT quantization relies on externally-produced QDQ ONNX

For `int8` / `fp8` / `int4`, `runner._get_trt_paths()` expects a pre-built QDQ-annotated ONNX at `onnx/resnet18_<prec>_qdq.onnx`. These are produced by NVIDIA **modelopt** outside this repo — the TRT builder reads Q/DQ scales directly from the graph, so there is no runtime calibrator in this codebase. `fp32` / `fp16` use the plain export `onnx/resnet18.onnx`.

Both ONNX export and engine build are skipped if the target file already exists. To force a rebuild, delete the relevant file from `onnx/` or `engines/`.

**Batch-size gotcha**: modelopt-exported QDQ ONNXes may have a fixed batch dimension baked in. `runner._run_tensorrt` warns if `cfg.batch_size` doesn't match; in that case re-export from modelopt with the matching batch size.

### Input quantization vs. model precision

These are independent axes handled in different places:
- **Input quantization** (`cfg.input_quant_bits`, 1/2/4/8) is applied in the data transform by `Quantize01` in [src/data.py](src/data.py:10), *before* ImageNet normalization.
- **Model precision** (`cfg.model_precision`) is applied to weights/activations by the backend-specific code path.

### Two separate QAT code trees

- [qat/](qat/) at repo root — standalone CLI training script + its notebook.
- [src/qat/](src/qat/) — the modules (`quantize.py`, `train_utils.py`) imported by the training script. The CLI script in `qat/` manually adds both `src/` and `src/qat/` to `sys.path`.

The QAT implementation monkey-patches `nn.Conv2d`/`nn.Linear` globally via `quant_modules.initialize()`, so **ordering matters**: call `setup_quantization_descriptors()` and `initialize_quant_modules()` *before* constructing any model.

### Checkpoints and ImageNet paths

- `get_model(cfg)` in [src/model.py](src/model.py:106) defaults to a **100-class** custom ResNet-18 loaded from `checkpoints/best.pth` (hardcoded absolute path). Pass `pretrained=True` to get the 1000-class torchvision weights instead.
- `cfg.imagenet_path` defaults to `/home/pf4636/imagenet2` (100-class subset); the FP32 trainer and QAT trainer default to `/home/pf4636/imagenet` (full ImageNet-1K). These are additional working directories for this session.
- `cfg.num_classes` filters the ImageNet dataset to the first N classes in-place — used to align the 1000-class validation set with the 100-class custom checkpoint.

### Results layout

Each run writes `runs/<run_id>/result.json` with config snapshot, system stamp, and `MetricsTracker.summary()` (top-1/top-5, batch + pure inference times, per-batch arrays). [src/metrics.py](src/metrics.py) drops the first 30 batches as warmup before collecting timing stats.
