# Quantized ResNets

ResNet18 quantization research project: training, post-training quantization (PTQ), quantization-aware training (QAT), and TensorRT deployment on ImageNet-100.

## Directory Structure

```
quantized_resnets/
├── pyfiles/                  # All Python source code (importable modules)
│   ├── src/                  # Core library
│   │   ├── config.py         # ExperimentConfig dataclass + with_overrides()
│   │   ├── data.py           # ImageNet-100 data loading & train/val splits
│   │   ├── eval.py           # Evaluation loops (PyTorch & TRT)
│   │   ├── model.py          # ResNet18 model loading
│   │   └── runner.py         # run_experiment() orchestrator
│   ├── utils/                # Shared utilities
│   │   ├── check_engine.py   # TRT engine inspection
│   │   ├── metrics.py        # Accuracy / latency metrics
│   │   ├── onnx_exporter.py  # PyTorch → ONNX export
│   │   ├── precision.py      # Precision helpers
│   │   └── utils.py          # load_runs(), flatten_runs(), print_run_summary()
│   ├── trt/                  # TensorRT builder & inference
│   │   ├── trt_builder.py    # Engine build (FP32/FP16/INT8/FP8/INT4)
│   │   └── trt_infer.py      # TRT inference loop
│   ├── ptq_cpu/              # CPU-side PTQ (PyTorch quantization)
│   │   └── quant_ptq_cpu.py
│   └── qat_modelopt/         # QAT via NVIDIA modelopt
│       ├── quantize.py
│       └── train_utils.py
├── ipynb_books/              # Final experiment notebooks (use pyfiles/ imports)
│   ├── 00_qdq_export.ipynb       # ONNX + QDQ export
│   ├── 01_fp32_baselines_sweep.ipynb  # Multi-seed FP32 baselines
│   ├── 02_tensorrt_inference.ipynb    # Multi-seed TRT precision sweep
│   ├── 04_ptq_test.ipynb         # PTQ evaluation
│   ├── 05_qat_test.ipynb         # QAT evaluation
│   ├── 50_ptq_int8_cpu.ipynb     # CPU INT8 PTQ
│   └── analysis.ipynb            # Cross-experiment analysis
├── notebooks/                # Legacy/exploratory notebooks (use src/ imports)
│   └── 05_tensorrt.ipynb         # Single-run TRT demo
├── training/                 # Training scripts & checkpoints
│   ├── train_fp32.py             # FP32 baseline training
│   ├── run_multiseed_fp32.sh     # Multi-seed launcher
│   ├── qat_training.py           # QAT training script
│   ├── qat/                      # QAT-specific training files
│   └── checkpoints/              # Saved model weights
├── onnx/                     # Exported ONNX models (base + QDQ variants)
├── engines/                  # Built TensorRT engines (per-seed subdirs)
├── runs/                     # Raw experiment result JSONs
│   ├── val_infer/                # Validation inference runs
│   ├── qat_trt/                  # QAT TensorRT runs
│   └── final_runs/               # Finalized results
├── results/                  # Aggregated result files (JSON, CSV)
├── docs/                     # Documentation
├── low-bit-images/           # Visualizations of low-bit input quantization
├── logs/                     # Training/experiment logs
├── Dockerfile.ptq            # PTQ container
├── Dockerfile.qat            # QAT container
├── requirements-ptq.txt      # PTQ dependencies
└── requirements-qat.txt      # QAT dependencies
```

## Key Patterns

- **ipynb_books/** notebooks import from `pyfiles/` by adding it to sys.path
- **notebooks/** (legacy) import from `src/` directly
- `run_experiment(cfg)` is the main entry point — handles ONNX export, engine build, and inference
- Multi-seed experiments use 3 seeds: `seed_1`, `seed_2`, `seed_42`
- Input quantization sweeps: 8, 4, 2, 1 bit
- Model precisions: FP32, FP16, INT8, FP8, INT4

## Data

- Dataset: ImageNet-100 (100-class subset)
- Paths configured in `pyfiles/src/data.py`
- Additional working dirs: `/home/pf4636/imagenet2`, `/home/pf4636/imagenet`
