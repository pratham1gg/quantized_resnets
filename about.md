# What this project is about

This project studies how quantization affects ResNet-18 on ImageNet-100 (a 100-class subset of ImageNet-1K). The central question is: if you reduce the precision of the model weights, the inference engine, or the input images themselves, how much accuracy do you lose and how much speed do you gain?

We train ResNet-18 from scratch, quantize it in various ways, and then benchmark everything on a consistent test set across multiple random seeds.


## The dataset

We use ImageNet-1K but only keep 100 classes. Within each class, 50 images are held out as a test split (the "holdout"), so evaluation is on data the model truly never saw during training. The standard ImageNet validation set is used as our validation set.

Input images go through a custom quantization step before being fed to the model. We simulate low-bit input pipelines by rounding pixel values to 1, 2, 4, or 8-bit precision. This mimics what you'd get from a low-power sensor or a compressed data stream. The 8-bit case is essentially standard uint8 images, so it serves as the baseline.


## How training works

There are two training pipelines:

**FP32 baseline training** (`training/train_fp32.py` and `training/train_lowbit.py`): Standard supervised training of ResNet-18 for 90 epochs with SGD, learning rate 0.1, cosine or step decay. We train separate models for each input bit-width (1, 2, 4, 8), each with 3 seeds (1, 2, 42). This gives us 12 baseline checkpoints stored in `.checkpoints/fp32_{bits}bit/seed_{s}/`.

**QAT fine-tuning** (`training/qat_training.py`): We take a trained FP32 checkpoint and fine-tune it with Quantization-Aware Training using NVIDIA's ModelOpt library. QAT inserts fake-quantize nodes (Q/DQ) into the model graph so the forward pass simulates INT8 or INT4 weight precision during training. We do this for both INT8 and INT4 weight quantization, across all 4 input bit-widths, with 3 seeds each. That's 24 QAT checkpoints in `.checkpoints/qat/`. QAT runs for 15 epochs with a small learning rate (1e-4). There is no separate QAT validation phase -- after training finishes, we go straight to inference on the test set.


## Inference backends

We run inference through three backends:

1. **PyTorch native** -- straightforward `model(x)` on CUDA. Supports FP32 and FP16 precision.

2. **TensorRT** -- we export the model to ONNX, then build a TensorRT engine. For FP32/FP16 models, it's a plain ONNX export. For quantized models (INT8, FP8, INT4), the ONNX file already contains Q/DQ nodes baked in by ModelOpt, so TensorRT reads the scales directly from the graph. No runtime calibration needed. Engines are cached in `.engines/`.

3. **TorchAO CPU PTQ** (`pyfiles/ptq_cpu/`) -- post-training quantization to INT8 using PyTorch's PT2E quantization API, running on CPU. This was a side experiment to see how CPU-only INT8 compares.

The `pyfiles/src/runner.py` orchestrates all of this. You give it an `ExperimentConfig` and it figures out which backend to use, exports ONNX if needed, builds the TensorRT engine if needed, and runs evaluation.


## What we measure

For every experiment we record:
- **Top-1 and Top-5 accuracy** on the holdout test set
- **Inference latency** in milliseconds per batch (after GPU warmup)
- All config details (backend, precision, input bits, seed, batch size)

Results are saved as individual `result.json` files under `runs/`, organized by split, method, seed, and run ID:

```
runs/{split}/{method}/{seed}/{run_id}/result.json   # GPU experiments
runs/cpu/{run_id}/result.json                        # CPU experiments (no seeds)
```

The `run_id` follows the format `{backend}_{method}_{precision}_b{bits}_{device}`, e.g. `trt_ptq_int8_b4_cuda` or `torch_qat_int4_b2_cuda`.

- `runs/val/ptq/` -- PTQ validation on the holdout set (84 runs: 7 backend-precision combos x 4 input bit-widths x 3 seeds)
- `runs/test/ptq/` -- PTQ test set evaluation (84 runs, same breakdown)
- `runs/test/qat/` -- QAT models evaluated on the test set via PyTorch and TRT (12 runs)
- `runs/cpu/` -- TorchAO CPU PTQ INT8 experiments (4 runs, no seed nesting)

The aggregated results live in `resultsv2/`. These are the averaged-across-seeds JSONs that the analysis notebooks consume:
- `resultsv2/test_final_results.json` -- the main results table with means and standard deviations
- `resultsv2/test_runs/` -- per-category aggregations
- `resultsv2/val_runs/` -- validation aggregations
- `resultsv2/plots_ppt/` -- final plots used in the presentation


## Directory structure

Here's what each folder is for and what's inside it.

### `pyfiles/` -- all the reusable Python code

This is the library that both the training scripts and notebooks import from.

- `src/config.py` -- the `ExperimentConfig` dataclass. Every experiment is defined by one of these: which backend, what precision, how many input bits, which seed, etc. It generates a `run_id` in the format `{backend}_{method}_{precision}_b{bits}_{device}` (e.g., `torch_ptq_fp32_b8_cuda`) and stamps system info.
- `src/data.py` -- handles ImageNet loading. Builds train/holdout splits (50 images per class held out for testing), applies input quantization via the `Quantize01` transform, and returns DataLoaders.
- `src/model.py` -- our ResNet-18 implementation. Standard BasicBlock architecture but with a final linear layer for 100 classes instead of 1000. Can load pretrained checkpoints.
- `src/eval.py` -- the evaluation loop. Runs the model on a dataloader, skips a warmup phase, then collects top-1/top-5 accuracy and per-batch inference latency.
- `src/runner.py` -- the main orchestrator. You pass it a config and it figures out the rest: loads the model, picks the right backend (PyTorch / TensorRT / TorchAO CPU), exports ONNX if needed, builds TRT engines if needed, evaluates, and saves `result.json`.
- `trt/trt_builder.py` -- takes an ONNX file and builds a TensorRT engine. Handles FP32/FP16/INT8 precision flags. Caches engines so you don't rebuild every time.
- `trt/trt_infer.py` -- loads a TensorRT engine and runs inference. Manages CUDA memory allocation, input/output bindings, the usual TRT boilerplate.
- `ptq_cpu/quant_ptq_cpu.py` -- post-training quantization to INT8 using PyTorch's PT2E API, targeting CPU. This was a side experiment comparing CPU-only INT8 against GPU-based approaches.
- `qat_modelopt/quantize.py` -- sets up ModelOpt quantization configs (INT8 or INT4) and applies fake-quantize nodes to the model.
- `qat_modelopt/train_utils.py` -- helpers for QAT training: checkpoint save/load, one-epoch training loop, validation loop.
- `utils/metrics.py` -- `MetricsTracker` class that accumulates accuracy and latency stats across batches.
- `utils/precision.py` -- casts model and inputs to the right dtype (FP32, FP16, etc.).
- `utils/onnx_exporter.py` -- exports a PyTorch model to ONNX format.
- `utils/check_engine.py` -- quick sanity check that a TensorRT engine file is valid.

### `training/` -- scripts to actually train models

- `train_fp32.py` -- trains a ResNet-18 from scratch on ImageNet-100. 90 epochs, SGD with momentum, standard augmentation. Saves best checkpoint by validation accuracy.
- `train_lowbit.py` -- same as above but applies input quantization (1, 2, or 4-bit) during training so the model learns to handle degraded inputs.
- `train_qat.py` -- takes a trained FP32 checkpoint and fine-tunes it with QAT using ModelOpt. 15 epochs, small learning rate (1e-4). Inserts Q/DQ nodes for INT8 or INT4 weight simulation.
- `run_fp32.sh` -- launches `train_fp32.py` across seeds (42, 1, 2) for 8-bit inputs.
- `run_lowbit.sh` -- launches `train_lowbit.py` across seeds (1, 2) and input bit-widths (2, 1).
- `run_qat.sh` -- launches `train_qat.py` across seeds (1, 2), precisions (INT8, INT4), and input bit-widths (8, 4, 2, 1). This is what you'd run to reproduce all 24 QAT checkpoints.

### `.checkpoints/` -- trained model weights

Organized as `.checkpoints/{model_type}/seed_{s}/best.pth`.

- `fp32_1bit/`, `fp32_2bit/`, `fp32_4bit/`, `fp32_8bit/` -- FP32 baselines trained on different input bit-widths. 3 seeds each (1, 2, 42). The `fp32_8bit` models are the "normal" baselines since 8-bit inputs are standard.
- `qat/int8_in{1,2,4,8}b/` -- INT8 QAT checkpoints, one per input bit-width, 3 seeds each.
- `qat/int4_in{1,2,4,8}b/` -- INT4 QAT checkpoints, same structure.

That's 12 FP32 + 24 QAT = 36 checkpoints total.

### `onnx/` -- exported ONNX models

Organized by the FP32 checkpoint they came from: `fp32_1bit/`, `fp32_2bit/`, `fp32_4bit/`, `fp32_8bit/`, `qat/`. These are intermediate files -- you export once, then TensorRT reads them to build engines.

### `.engines/` -- cached TensorRT engines

Same structure as `onnx/`. Each engine is tied to a specific GPU architecture, so these aren't portable across machines. They get rebuilt automatically if missing.

### `runs/` -- raw experiment results (224 individual runs)

Every single experiment produces a `result.json` in its own folder. The folder name is the run ID (e.g., `trt_ptq_int8_b2_cuda`).

```
runs/
├── val/ptq/seed_{1,2,42}/{run_id}/result.json    # 84 PTQ validation runs
├── val/qat/seed_{1,2,42}/{run_id}/result.json    # 24 QAT validation runs
├── test/ptq/seed_{1,2,42}/{run_id}/result.json   # 84 PTQ test runs
├── test/qat/seed_{1,2,42}/{run_id}/result.json   # 28 QAT test runs
└── cpu/{run_id}/result.json                       # 4 CPU PTQ runs
```

- `val/ptq/` and `test/ptq/` -- 7 backend-precision combos (PyTorch FP32, PyTorch FP16, TRT FP32, TRT FP16, TRT FP8, TRT INT8, TRT INT4) x 4 input bit-widths x 3 seeds = 84 runs each.
- `val/qat/` -- QAT models evaluated on holdout-val via PyTorch CUDA (INT8 + INT4). 24 runs (8 configs x 3 seeds).
- `test/qat/` -- QAT models evaluated on test data via PyTorch CUDA (INT8 + INT4, 3 seeds) and TRT (INT8 only, 1 seed). 28 runs.
- `cpu/` -- TorchAO CPU PTQ INT8 across 4 input bit-widths. No seed nesting.

### `resultsv2/` -- aggregated results (what the notebooks use)

This is the clean, averaged data:

- `test_final_results.json` -- the main table. Each row has backend, precision, input_bits, and mean/std for top-1, top-5, and latency across seeds.
- `test_runs/` -- per-category aggregations (pytorch, tensorrt, qat, qat_trt).
- `val_runs/` -- validation-set aggregations.
- `plots_ppt/` -- final PNG plots used in the presentation (a.png through k.png).

### `ipynb_books/` -- Jupyter notebooks

These are the interactive experiment notebooks. They import from `pyfiles/` and read results from `runs/` and `resultsv2/`.

- `00_qdq_export.ipynb` -- ONNX export with Q/DQ nodes
- `01_1_pytorch_val.ipynb` -- PTQ PyTorch validation (holdout set)
- `01_2_pytorch_test.ipynb` -- PTQ PyTorch test set evaluation
- `02_1_tensorrt_inference.ipynb` -- PTQ TensorRT validation (holdout set)
- `02_2_tensorrt_test.ipynb` -- PTQ TensorRT test set evaluation
- `03_1_qat_val.ipynb` -- QAT PyTorch validation (holdout set)
- `03_2_qat_test.ipynb` -- QAT PyTorch test set evaluation
- `04_1_qat_trt_val.ipynb` -- QAT TensorRT validation (holdout set)
- `04_2_qat_trt_test.ipynb` -- QAT TensorRT test set evaluation
- `09_ptq_cpu.ipynb` -- TorchAO CPU PTQ INT8 (appendix)
- `analysis.ipynb` -- main analysis and plotting notebook
- `plots_vortrag.ipynb` -- presentation-ready plots
- `check_weights.ipynb` -- quick checkpoint inspection

### `low-bit-images/` -- visual samples

Contains a script (`save_low_bit_images.py`) and sample PNGs showing what an image looks like after quantizing to 1, 2, 4, and 8-bit pixel depth. Useful for the presentation to show the visual degradation.

### Other files

- `qr.code-workspace` -- VS Code workspace config
- `.gitignore` -- ignores checkpoints, engines, venv, caches
- `delete/` -- stuff marked for removal (old `results/` folder, `old_runs/` with the pre-migration directory structure, empty `qat_int4act_test/` directory)


## Key things to keep in mind

- ImageNet-100, not full ImageNet-1K. The 100-class subset makes training feasible on a single GPU.
- Input quantization is separate from model quantization. A model can be FP32 but receive 1-bit inputs, or be INT4 with 8-bit inputs. These are independent axes.
- All experiments use batch size 1 for latency measurements. This is deliberate -- we care about single-image inference latency, not throughput.
- Three random seeds (1, 2, 42) per configuration. Results are reported as mean +/- std.
- QAT models are only evaluated on the test set, not on validation. Training goes directly to test inference.
