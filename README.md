# quantized_resnets

Thesis experiments comparing ResNet-18 inference accuracy and latency across model precisions
(fp32 / fp16 / int8 / fp8 / int4) and input-quantization bit-widths (1 / 2 / 4 / 8) using three
backends: vanilla PyTorch, torchao CPU PT2E PTQ, and TensorRT.

See [`CLAUDE.md`](CLAUDE.md) for the full architecture and code layout.

## Layout

```
src/                 core experiment library
training/            standalone FP32 + QAT training CLIs
notebooks/           00_env_sanity → 09_test_inference (run in order)
checkpoints/         model weights (gitignored)
runs/val_infer/      val-holdout results consumed by 07_analysis
runs/final_runs/     test-set results consumed by 08_test_analysis
engines/             compiled TRT engines (gitignored)
onnx/                ONNX exports (plain + QDQ-annotated)
```

## Running locally (venv)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-ptq.txt
jupyter lab
```

Open `notebooks/` and run them in order.

## Running in Docker

There are **two images** because PTQ and QAT require incompatible CUDA / torch stacks:

| image | Dockerfile | requirements | torch / CUDA | what it runs |
|---|---|---|---|---|
| `qr:ptq` | `Dockerfile.ptq` | `requirements-ptq.txt` | torch 2.10 + CUDA 13 + TensorRT 10.15 | everything except QAT training (PTQ, TRT, all notebooks 00–09) |
| `qr:qat` | `Dockerfile.qat` | `requirements-qat.txt` | torch 2.3.1 + CUDA 11.8 + `pytorch_quantization` 2.2.1 | only `training/train_qat_phase1.py` (legacy QAT) |

Run one at a time. They share `checkpoints/`, `onnx/`, `engines/` via bind mounts — the QAT
container writes a checkpoint, you stop it, then start the PTQ container and it picks the
artifact up from the same path.

### Host prerequisites

- NVIDIA driver compatible with CUDA 13 (back-compatible with 11.8, so one driver covers both images)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (`sudo apt install nvidia-container-toolkit && sudo systemctl restart docker`).
  On WSL2: Docker Desktop with WSL integration, or native Docker inside WSL with the toolkit installed.

### Build

```bash
docker build -f Dockerfile.ptq -t qr:ptq .
docker build -f Dockerfile.qat -t qr:qat .
```

The PTQ image is large (~10 GB) because of the bundled CUDA + TensorRT wheels — expected.
The QAT image is smaller (~5 GB).

### Run

PTQ container (Jupyter Lab on :8888):

```bash
docker run --rm -it --gpus all \
  -p 8888:8888 \
  -v "$PWD":/workspace \
  -v /home/pf4636/imagenet:/home/pf4636/imagenet:ro \
  -v /home/pf4636/imagenet2:/home/pf4636/imagenet2:ro \
  qr:ptq
```

QAT container (Jupyter Lab on :8889 to avoid colliding with the PTQ container if it's still running):

```bash
docker run --rm -it --gpus all \
  -p 8889:8888 \
  -v "$PWD":/workspace \
  -v /home/pf4636/imagenet:/home/pf4636/imagenet:ro \
  qr:qat
```

Or run the QAT training CLI directly without Jupyter:

```bash
docker run --rm --gpus all \
  -v "$PWD":/workspace \
  -v /home/pf4636/imagenet:/home/pf4636/imagenet:ro \
  qr:qat \
  python training/train_qat_phase1.py
```

The two ImageNet bind mounts are required because `cfg.imagenet_path` in saved run configs
points at `/home/pf4636/imagenet*` on the host — mount them at the same paths inside the
container so paths resolve identically.

### Sanity checks

```bash
# PTQ
docker run --rm --gpus all qr:ptq \
  python -c "import torch, tensorrt; print(torch.cuda.is_available(), tensorrt.__version__)"
# expected: True 10.15.1.29

# QAT
docker run --rm --gpus all qr:qat \
  python -c "import torch, pytorch_quantization; print(torch.cuda.is_available(), torch.__version__)"
# expected: True 2.3.1+cu118
```

## Notebook order

| # | Notebook | Purpose |
|---|---|---|
| 00 | `00_env_sanity` | environment / GPU sanity checks |
| 01 | `01_pytorch_baselines_fp32` | FP32 PyTorch baseline |
| 02 | `02_input_quant_sweep` | sweep input-quantization bits |
| 03 | `03_int8_cpu` | torchao CPU PT2E PTQ |
| 04 | `04_qdq_export` | export QDQ-annotated ONNX (modelopt) |
| 05 | `05_tensorrt` | TRT pipeline for fp32 / fp16 / int8 / fp8 / int4 |
| 06 | `06_qat_inference`, `06b_qat_modelopt_inference` | QAT inference |
| 07 | `07_analysis` | val-holdout analysis (reads `runs/val_infer/`) |
| 08 | `08_test_analysis` | test-set analysis (reads `runs/final_runs/`) |
| 09 | `09_test_inference` | re-run every saved config on the held-out test set |

Notebooks 01–06 write results to `runs/val_infer/<run_id>/result.json` (default
`output_root` in `src/config.py`). Notebook 09 reads those, re-evaluates each on
`/home/pf4636/imagenet2/val`, and writes to `runs/final_runs/<run_id>/result.json`.
