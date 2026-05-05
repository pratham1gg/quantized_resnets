# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Ignore code comments

**Inline comments in the codebase may be outdated or inaccurate.** Always prioritize:
1. The actual code behavior (what it does, not what comments claim)
2. This CLAUDE.md file (authoritative documentation)
3. Git commit history (explains why changes were made)

Do not use code comments as a source of truth.

## Project purpose

Thesis experiments comparing ResNet-18 inference accuracy and latency across model precisions (fp32/fp16/int8/fp8/int4) and input-quantization bit-widths (1/2/4/8), using three inference backends: vanilla PyTorch, torchao CPU PT2E PTQ, and TensorRT. All reported experiments are generated exclusively using code in `src/`. The `training/` directory contains training pipelines that produce the model weights consumed by `src/`.

## Directory layout

```
quantized_resnets/
├── src/                           # Core experiment library (config, runner, backends, metrics)
│   └── qat_modelopt/              # ModelOpt-based QAT modules (quantize.py, train_utils.py)
├── training/                      # Training pipelines (produce weights for src/)
│   ├── train_fp32.py              # FP32 baseline training on ImageNet-1K
│   ├── train_qat_modelopt.py      # ModelOpt INT8/INT4 QAT (CLI form)
│   └── train_qat_modelopt.ipynb   # Same flow as the .py, but interactive
├── notebooks/                     # Experiment notebooks (run in order 00 → 09)
├── qat/                           # Legacy: train_qat_phase1.ipynb (old pytorch-quantization
│                                  # flow, no longer wired up — kept for reference only)
├── checkpoints/                   # Gitignored model weights
│   ├── best.pth                   # FP32 best checkpoint (used by src/model.py)
│   ├── fp32/                      # Per-epoch FP32 training history
│   └── qat/                       # QAT runs, one subdir per config
│       └── <PREC>_in<N>b/         # e.g. int8_in4b/, int8_in0b/ ("0b" = no input quant)
│           ├── qat_modelopt_best.pth          # weights + training state
│           └── qat_modelopt_best_mostate.pt   # ModelOpt quantization graph spec
├── engines/                       # Gitignored TRT compiled engines
├── onnx/                          # ONNX exports (fp32 plain + QDQ-annotated for int8/fp8/int4)
└── runs/                          # Gitignored per-run result.json files
    ├── val_infer/                 # Standard runner.run_experiment() outputs
    └── qat/                       # 06_qat_inference notebook outputs
```

## Running experiments

There is no test suite, lint config, or package manager entrypoint. The codebase is driven from Jupyter notebooks in `notebooks/` which import from `src/` and call `run_experiment(cfg)`.

```python
# From a notebook with sys.path.insert(0, "../src"):
from config import ExperimentConfig
from runner import run_experiment

cfg = ExperimentConfig(backend="tensorrt", model_precision="int8",
                       input_quant_bits=4, batch_size=1, device="cuda")
payload, tracker = run_experiment(cfg)   # writes runs/<run_id>/result.json
```

Notebooks are numbered (`00_env_sanity` → `09_test_inference`) and are intended to be run in order. Notable additions:
- `06_qat_inference.ipynb` — sweeps QAT-trained models across input bit-widths and writes `runs/qat/<run_id>/result.json` mirroring the standard payload schema.

Training:
- `python training/train_fp32.py` — FP32 baseline training on ImageNet-1K.
- `python training/train_qat_modelopt.py` — ModelOpt INT8/INT4 QAT (CLI). Same flow as the `.ipynb` for headless runs.

## Docker

`Dockerfile.qat` builds an image with torch+cu128 (Blackwell-capable, e.g. RTX 5060 Ti) plus `nvidia-modelopt` and JupyterLab. Built once via `docker build -f Dockerfile.qat -t resnet-qat:cu128 .`. Standard run command:

```bash
docker run --gpus all --shm-size=4g --rm -it -p 8888:8888 \
  -v /home/pf4636/code/resnet/quantized_resnets:/workspace \
  -v /home/pf4636/imagenet:/home/pf4636/imagenet:ro \
  -w /workspace \
  resnet-qat:cu128
```

Notes: `--shm-size=4g` is required (DataLoader workers exhaust the default 64 MB shm); the `nvidia-container-toolkit` package must be installed on the host for `--gpus` to work.

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

For `int8` / `fp8` / `int4`, `runner._get_trt_paths()` expects a pre-built QDQ-annotated ONNX at `onnx/resnet18_<prec>_qdq.onnx`. These are produced by NVIDIA **ModelOpt** outside this repo — the TRT builder reads Q/DQ scales directly from the graph, so there is no runtime calibrator in this codebase. `fp32` / `fp16` use the plain export `onnx/resnet18.onnx`.

Both ONNX export and engine build are skipped if the target file already exists. To force a rebuild, delete the relevant file from `onnx/` or `engines/`.

**Batch-size gotcha**: ModelOpt-exported QDQ ONNXes may have a fixed batch dimension baked in. `runner._run_tensorrt` warns if `cfg.batch_size` doesn't match; in that case re-export from ModelOpt with the matching batch size.

### Input quantization vs. model precision

These are independent axes handled in different places:
- **Input quantization** (`cfg.input_quant_bits`, 1/2/4/8) is applied in the data transform by `Quantize01` in [src/data.py](src/data.py:17), *before* ImageNet normalization.
- **Model precision** (`cfg.model_precision`) is applied to weights/activations by the backend-specific code path.

### QAT code layout (ModelOpt)

- [src/qat_modelopt/quantize.py](src/qat_modelopt/quantize.py) — `get_quant_cfg`, `quantize_model` (one-shot PTQ calibration via `mtq.quantize`), `restore_modelopt_state`.
- [src/qat_modelopt/train_utils.py](src/qat_modelopt/train_utils.py) — `train_one_epoch`, `validate`, `save_checkpoint` (saves `_best.pth` + `_best_mostate.pt`), `load_training_state`.
- [training/train_qat_modelopt.py](training/train_qat_modelopt.py) and [training/train_qat_modelopt.ipynb](training/train_qat_modelopt.ipynb) — entry points; the notebook supports an `INPUT_QUANT_BITS` config that applies `Quantize01` inside the train/val transforms.

**Resume invariant**: on resume, restore the modelopt graph **first** (`restore_modelopt_state`), build the optimizer **after**, then `load_training_state`. The optimizer must see the quantizer parameters that ModelOpt inserted.

**Inference loading**:
```python
model = ResNet18(num_classes=N, pretrained=False)
restore_modelopt_state(model, "..._mostate.pt")     # 1. restore graph
ckpt = torch.load("..._best.pth")
model.load_state_dict(ckpt["model"])                # 2. load QAT weights
```
Order matters — without step 1, `load_state_dict` keys won't line up.

### Checkpoints and ImageNet paths

- `get_model(cfg)` in [src/model.py](src/model.py) defaults to a **100-class** custom ResNet-18 loaded from `checkpoints/best.pth` (repo-relative path derived from `__file__`). Pass `pretrained=True` to get the 1000-class torchvision weights instead.
- `cfg.imagenet_path` defaults to `/home/pf4636/imagenet2` (100-class subset); the FP32 trainer and QAT trainer default to `/home/pf4636/imagenet` (full ImageNet-1K). These are additional working directories for this session.
- `cfg.num_classes` filters the ImageNet dataset to the first N classes in-place — used to align the 1000-class validation set with the 100-class custom checkpoint.

### Results layout

Each run writes `runs/<scope>/<run_id>/result.json` with config snapshot, system stamp, and `MetricsTracker.summary()` (top-1/top-5, batch + pure inference times, per-batch arrays). [src/metrics.py](src/metrics.py) drops the first 30 batches as warmup before collecting timing stats. The `06_qat_inference` notebook hand-rolls the same payload schema so QAT inference results can be analysed alongside PTQ/TRT runs via `utils.load_runs`.