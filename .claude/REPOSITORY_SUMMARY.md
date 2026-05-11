# Quantized ResNets Repository Summary

## Project Overview

This repository contains a thesis research project comparing **ResNet-18 inference performance** across different model precisions and input quantization strategies. The project evaluates three distinct execution backends with comprehensive accuracy and latency measurements.

**Key Research Dimensions:**
- **Model Precisions**: FP32 (baseline), FP16, INT8, FP8, INT4
- **Input Quantization Bit-widths**: 8, 4, 2, 1 bits
- **Backends**: Vanilla PyTorch, torchao CPU PT2E PTQ, TensorRT
- **Dataset**: ImageNet (100-class subset for training, 1000-class validation)

## Quick Start

```python
# Run an experiment from a Jupyter notebook
from config import ExperimentConfig
from runner import run_experiment

cfg = ExperimentConfig(
    backend="tensorrt",
    model_precision="int8",
    input_quant_bits=4,
    batch_size=1,
    device="cuda"
)
payload, tracker = run_experiment(cfg)  # Results saved to runs/<run_id>/result.json
```

## Directory Structure

```
quantized_resnets/
├── src/                          # Core experiment library (config, runners, backends)
│   ├── config.py                 # ExperimentConfig dataclass (frozen, deterministic run_id)
│   ├── runner.py                 # Main entry point: run_experiment(cfg)
│   ├── model.py                  # ResNet-18 loader (100-class custom or 1000-class pretrained)
│   ├── data.py                   # Data pipeline with Quantize01 input quantization transform
│   ├── eval.py                   # Evaluation loop (accuracy, batch + pure latency)
│   ├── precision.py              # apply_precision() for PyTorch fp32/fp16
│   ├── metrics.py                # MetricsTracker (top-1/top-5, timing stats, warmup handling)
│   ├── quant_ptq_cpu.py          # torchao CPU PTQ INT8 quantization
│   ├── quant_qat.py              # QAT utilities (currently unused in runner)
│   ├── onnx_exporter.py          # ONNX export for fp32/fp16 models
│   ├── trt_builder.py            # TensorRT engine compilation
│   ├── trt_infer.py              # TensorRT inference evaluation
│   ├── check_engine.py           # TRT engine metadata inspection
│   ├── utils.py                  # Utility functions
│   └── qat/                      # QAT modules (imported by training)
│       ├── quantize.py           # pytorch-quantization descriptor setup
│       └── train_utils.py        # QAT training helpers
│
├── training/                     # Standalone training CLIs (not part of src/ experiment runner)
│   ├── train_fp32.py            # FP32 baseline training on ImageNet-1K
│   ├── train_qat_phase1.py      # Phase-1 INT8 QAT fine-tuning (uses src/qat/)
│   ├── train_qat_phase1.ipynb   # Notebook version of QAT training
│   └── resnet18.py              # ResNet-18 architecture definition
│
├── notebooks/                    # Experiment notebooks (run in order: 00 → 07)
│   ├── 00_env_sanity.ipynb      # Environment verification, sanity checks
│   ├── 01_pytorch_baselines_fp32.ipynb  # PyTorch FP32/FP16 baseline experiments
│   ├── 02_input_quant_sweep.ipynb       # Input quantization sweep (1/2/4/8 bits)
│   ├── 03_int8_cpu.ipynb               # torchao CPU PTQ INT8 quantization
│   ├── 04_qdq_export.ipynb             # QDQ ONNX export (from modelopt external process)
│   ├── 05_tensorrt.ipynb               # TensorRT engine build and inference
│   ├── 06_qat_inference.ipynb          # QAT model inference evaluation
│   └── 07_analysis.ipynb               # Results aggregation and analysis
│
├── checkpoints/                  # Gitignored model weights
│   ├── best.pth                  # FP32 best checkpoint (100-class, used by src/model.py)
│   ├── fp32/                     # Per-epoch FP32 training history
│   └── qat/                      # QAT run checkpoints (one subdir per run config)
│
├── engines/                      # Gitignored TensorRT compiled engines
│   └── resnet18_*.engine         # Engine files keyed by run_id
│
├── onnx/                         # ONNX exports (both plain and QDQ-annotated)
│   ├── resnet18.onnx             # Plain export (fp32/fp16)
│   ├── resnet18_int8_qdq.onnx    # QDQ-annotated INT8 (from modelopt)
│   ├── resnet18_fp8_qdq.onnx     # QDQ-annotated FP8 (from modelopt)
│   └── resnet18_int4_qdq.onnx    # QDQ-annotated INT4 (from modelopt)
│
├── runs/                         # Gitignored per-run results
│   └── <run_id>/result.json      # Structured output with config, metrics, timing
│
├── results/figures/              # Committed figures and PDFs
│   └── *.pdf, *.png              # Graphs, comparisons
│
├── requirements.txt              # Python dependencies
├── CLAUDE.md                     # Project guidance for Claude Code
└── REPOSITORY_SUMMARY.md         # This file
```

## Core Architecture

### 1. Config-Driven Experiment Runner

**[src/config.py](src/config.py)** defines `ExperimentConfig`, a frozen dataclass that:
- **Fully determines every run** through its parameters
- **Validates backend ↔ precision ↔ device compatibility** in `validate()`
- **Computes deterministic `run_id()`** e.g., `resnet18_tensorrt_int8_in4b_cuda_bs1`
- Ensures **no clobbering** — different configs produce unique output files

**Key Config Parameters:**
```python
ExperimentConfig(
    imagenet_path: str = "/home/pf4636/imagenet2"      # 100-class subset (training default)
    device: str = "cuda" | "cpu"
    batch_size: int = 1
    num_classes: int = 100                              # Restrict eval to first N classes
    input_quant_bits: int = 8                           # {1, 2, 4, 8}
    model_precision: Precision = "fp32"                 # {fp32, fp16, int8, fp8, int4}
    backend: Backend = "pytorch"                        # {pytorch, torchao_cpu_ptq, tensorrt}
    num_eval_batches: Optional[int] = None              # None = full validation
    trt_workspace_mb: int = 2048
    trt_calib_num_batches: int = 32
)
```

### 2. Experiment Runner

**[src/runner.py](src/runner.py)** is the single entry point `run_experiment(cfg)`. It routes to one of three backends:

| Backend | Workflow | Use Case |
|---|---|---|
| **pytorch** | `get_model` → `apply_precision` (fp32/fp16) → `evaluate` | FP32/FP16 baselines |
| **torchao_cpu_ptq** | `get_model` → `quantize_int8_x86_pt2e` (CPU calibration) → `evaluate` | INT8 CPU quantization |
| **tensorrt** | 3-step pipeline (ONNX → engine → inference) | INT8/FP8/INT4 GPU inference |

**TensorRT Pipeline:**
1. **ONNX Export** (cached) — skipped if file exists
   - fp32/fp16: Plain export `resnet18.onnx`
   - int8/fp8/int4: QDQ-annotated `resnet18_<prec>_qdq.onnx` (pre-built by NVIDIA modelopt)
2. **Engine Build** (cached) — skipped if `.engine` file exists
   - TensorRT builder reads quantization scales from Q/DQ nodes
   - No runtime calibrator needed for int8/fp8/int4
3. **Inference** via `trt_evaluate`

### 3. Model Loading

**[src/model.py](src/model.py)** provides `get_model(cfg)`:
- **Default**: 100-class custom ResNet-18 from `checkpoints/best.pth`
- **Option**: 1000-class torchvision weights with `pretrained=True`
- Automatically adapts to `cfg.num_classes` for dataset filtering

### 4. Data Pipeline

**[src/data.py](src/data.py)** provides `get_dataloader(cfg)`:
- **Input Quantization**: Applied by `Quantize01` transform (1/2/4/8 bits)
  - Quantizes FP32 input to {1,2,4,8} bits, applied *before* ImageNet normalization
  - Independent of model precision (quantization happens in the data path)
- **Normalization**: ImageNet mean/std
- **Caching**: Full validation set loaded once per notebook session

**Independent Quantization Axes:**
- **Input Quantization** (`cfg.input_quant_bits`): Data transform in `data.py`
- **Model Precision** (`cfg.model_precision`): Backend-specific (weights/activations)

### 5. Metrics and Evaluation

**[src/metrics.py](src/metrics.py)** provides `MetricsTracker`:
- Tracks **accuracy** (top-1, top-5)
- Tracks **batch latency** (full forward + post-processing)
- Tracks **pure inference latency** (forward only, excluding data)
- **Warmup handling**: First 30 batches dropped before collecting timing stats
- Produces structured `summary()` dict for output JSON

**[src/eval.py](src/eval.py)** provides `evaluate(model, dataloader, device)`:
- Runs validation loop
- Computes per-batch accuracy and latency
- Returns `MetricsTracker` with final summary

### 6. Quantization Backends

#### PyTorch Precision ([src/precision.py](src/precision.py))
- `apply_precision(model, "fp16")` → casts model to float16
- Supports fp32 (no-op), fp16 (autocast)

#### torchao CPU PTQ ([src/quant_ptq_cpu.py](src/quant_ptq_cpu.py))
- `quantize_int8_x86_pt2e(model, dataloader)` → compiled INT8 model
- Runs on CPU; uses first N batches for calibration
- Produces graph with fused quantization ops

#### TensorRT ([src/trt_builder.py](src/trt_builder.py) + [src/trt_infer.py](src/trt_infer.py))
- Reads QDQ-annotated ONNX (scales pre-computed by modelopt)
- Compiles to `.engine` (serialized TRT model)
- Inference via CUDA

### 7. QAT Training (Standalone)

**[training/train_qat_phase1.py](training/train_qat_phase1.py)**:
- Standalone CLI (not part of runner)
- Uses pytorch-quantization for symmetric INT8 QAT
- Imports reusable modules from `src/qat/`

**[src/qat/quantize.py](src/qat/quantize.py)**:
- `setup_quantization_descriptors()` — configure per-layer quantization
- `initialize_quant_modules()` — monkey-patch `nn.Conv2d`/`nn.Linear` globally
- Must be called *before* model construction

**[src/qat/train_utils.py](src/qat/train_utils.py)**:
- Loss computation with quantized precision
- Gradient clipping, weight updates

## Results Format

Each run writes `runs/<run_id>/result.json` with:

```json
{
  "status": "ok",
  "run_id": "resnet18_tensorrt_int8_in4b_cuda_bs1",
  "system": {
    "timestamp": "2026-04-18T12:34:56",
    "gpu_name": "NVIDIA A100",
    "cuda_version": "12.1"
  },
  "config": {
    "backend": "tensorrt",
    "model_precision": "int8",
    "input_quant_bits": 4,
    "batch_size": 1,
    "device": "cuda",
    "num_classes": 100,
    "num_eval_batches": null
  },
  "results": {
    "top1_accuracy": 0.75,
    "top5_accuracy": 0.92,
    "batch_latency_ms": [45.2, 44.8, ...],
    "inference_latency_ms": [42.1, 41.9, ...],
    "batch_latency_mean": 45.0,
    "inference_latency_mean": 42.0
  },
  "error": null
}
```

## Experiment Notebooks

Run in order (0 → 7); each imports from `src/` and calls `run_experiment(cfg)`:

1. **00_env_sanity.ipynb** — Verify CUDA, PyTorch, dataset paths
2. **01_pytorch_baselines_fp32.ipynb** — PyTorch FP32 baseline + FP16 variant
3. **02_input_quant_sweep.ipynb** — Sweep input quantization bits (1/2/4/8)
4. **03_int8_cpu.ipynb** — torchao INT8 CPU PTQ experiments
5. **04_qdq_export.ipynb** — Generate/verify QDQ ONNX files
6. **05_tensorrt.ipynb** — TensorRT engine build + INT8/FP8/INT4 inference
7. **06_qat_inference.ipynb** — Evaluate QAT-trained models
8. **07_analysis.ipynb** — Aggregate results, generate plots

## Training Pipelines

### FP32 Baseline Training
```bash
python training/train_fp32.py \
  --imagenet-path /home/pf4636/imagenet \
  --epochs 90 \
  --lr 0.1
```
- ImageNet-1K (1000 classes)
- Saves best checkpoint to `checkpoints/best.pth`
- Per-epoch history in `checkpoints/fp32/`

### QAT Phase-1 Training
```bash
python training/train_qat_phase1.py \
  --pretrained-ckpt checkpoints/best.pth \
  --epochs 30 \
  --lr 0.01
```
- Fine-tunes from FP32 checkpoint
- Uses pytorch-quantization INT8 QAT
- Saves checkpoints to `checkpoints/qat/`

## Dependencies

**Core Libraries:**
- `torch==2.10.0` — PyTorch computation
- `torchvision==0.25.0` — ImageNet dataset, pretrained models
- `torchao==0.16.0` — torchao CPU PTQ quantization
- `numpy==2.2.1` — Array operations
- `pandas==2.2.3` — Results aggregation
- `matplotlib==3.10.0` — Plotting
- `seaborn==0.13.2` — Statistical visualization
- `tqdm==4.67.1` — Progress bars
- `pyyaml==6.0.2` — Config serialization
- `psutil==6.1.1` — System monitoring

**External Dependencies (not in requirements.txt):**
- TensorRT (for `tensorrt` backend; installed via `pip install tensorrt`)
- NVIDIA CUDA 12.0+ (for GPU inference)
- NVIDIA modelopt (for QDQ ONNX generation; external to this repo)

## Important Notes

### Checkpoints and Paths
- **Custom ResNet-18**: `checkpoints/best.pth` (100-class)
- **ImageNet paths**:
  - Evaluation: `/home/pf4636/imagenet2` (100-class subset)
  - Training: `/home/pf4636/imagenet` (full ImageNet-1K)
- `cfg.num_classes` filters ImageNet dataset in-place

### TensorRT Quantization
- INT8/FP8/INT4 use **pre-built QDQ-annotated ONNX** from modelopt
- No runtime calibrator in this codebase
- TensorRT builder reads scales from Q/DQ nodes directly
- **Batch-size gotcha**: modelopt exports may have fixed batch dimension; re-export if batch size changes

### Skipped Exports and Builds
- ONNX export skipped if file exists (delete from `onnx/` to force rebuild)
- Engine build skipped if `.engine` file exists (delete from `engines/` to force rebuild)
- Engine filenames encode `run_id` to prevent clobbering

### Git Ignores
- `checkpoints/` — Model weights and training history
- `engines/` — Compiled TensorRT engines
- `runs/` — Per-run result JSON files
- `.venv/` — Virtual environment

## Workflow Summary

```
Notebook (00 → 07)
    ↓
ExperimentConfig (config.py)
    ↓
run_experiment(cfg) [runner.py]
    ├─→ pytorch backend
    │       get_model → apply_precision → evaluate
    │
    ├─→ torchao_cpu_ptq backend
    │       get_model → quantize_int8_x86_pt2e → evaluate
    │
    └─→ tensorrt backend
            onnx_exporter → trt_builder → trt_infer
    ↓
MetricsTracker (metrics.py)
    ↓
runs/<run_id>/result.json
    ↓
07_analysis.ipynb (aggregate + visualize)
```

## Key Design Principles

1. **Deterministic Config**: Every run is fully determined by `ExperimentConfig`; no external state
2. **Caching**: ONNX exports and TRT engines cached by filename; skip rebuild if exists
3. **Input vs. Model Quantization**: Independent axes — input quantization is data transform, model precision is backend-specific
4. **QAT Separation**: QAT training is standalone CLI; evaluation via runner with `backend="pytorch"`
5. **No Test Suite**: Validation through notebooks; each notebook is a reproducible experiment script
6. **Metrics Warmup**: First 30 batches dropped for timing stats (account for GPU warmup)

## Future Extensions

- Additional backends (TVM, ONNX Runtime, etc.)
- More ResNet variants (50, 152, etc.)
- Sparse quantization or mixed-precision experiments
- Automated hyperparameter search for QAT
