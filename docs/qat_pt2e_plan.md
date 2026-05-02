# PT2E QAT Planning — Answers to Pre-Migration Questionnaire

Status: **planning only** — deciding whether to adopt PyTorch 2 Export (PT2E) QAT for INT4 quantization. No code is being moved or migrated yet.

Target deployment: **TensorRT on RTX 5060 Ti (Blackwell, sm_120)**, with INT4 quantization extended across all Conv layers (this is the motivating goal — the existing modelopt INT4 path only quantizes the final FC layer).

---

## About ResNet-18

### 1. How was ResNet-18 modified? (resolution, num_classes, stem, custom layers)

The architecture is the standard torchvision ResNet-18 reimplemented in `training/resnet18.py` and `src/model.py`. Compared to vanilla torchvision:

| Aspect | Value | Modified? |
|---|---|---|
| Input resolution | 224 × 224, 3 channels | No (standard) |
| Stem | 7×7 conv, stride 2, padding 3, 64 out — followed by BN, ReLU, 3×3 maxpool stride 2 | No (standard) |
| Stages | 4 × BasicBlock (2 blocks each), channels [64, 128, 256, 512], strides [1, 2, 2, 2] | No (standard) |
| Head | AdaptiveAvgPool2d(1) → optional Dropout(p=0) → Linear(512, num_classes) | Dropout added (default p=0, effectively a no-op) |
| **`num_classes`** | **100** | **Yes — deployed checkpoint is 100-class** (1000-class supported via `pretrained=True`) |
| Custom ops | None — only `Conv2d`, `BatchNorm2d`, `ReLU`, residual `+`, `MaxPool2d`, `AdaptiveAvgPool2d`, `Linear` | No |

**Implication for PT2E:** `torch.export` should capture this graph cleanly. There are no custom autograd functions, no dynamic control flow, no third-party layers. Standard `torch.ops.aten.*` decomposition will succeed. The only thing to watch is BN handling: export on CUDA so the graph contains `cudnn_batch_norm` (better numerics) rather than the CPU `native_batch_norm` variant.

### 2. Is the trained FP32 `.pth` checkpoint available?

**Yes.** Located at `checkpoints/best.pth` (~86 MB, dated 2026-04-28). Loadable via:

```python
from src.model import get_model
model = get_model(cfg=None, pretrained=False)  # loads checkpoints/best.pth, num_classes=100
```

The state dict is stored under the `"model"` key (the loader handles both wrapped and unwrapped formats). PT2E QAT will start from these float weights, not from the existing ONNX/QDQ graph.

---

## About the existing PTQ / ONNX

### 3. What quantization scheme does the current TensorRT PTQ produce?

Three separate schemes, all produced by NVIDIA **modelopt** (`modelopt.onnx.quantization.quantize`, called from `notebooks/04_qdq_export.ipynb`) operating on the plain FP32 ONNX export. None of them use a TRT runtime calibrator — Q/DQ scales are baked into the ONNX graph.

| ONNX file | Weights | Activations | Coverage |
|---|---|---|---|
| `resnet18_int8_qdq.onnx` | **INT8 symmetric per-channel** (zero-point = 0, scale shape `[out_channels]`) | **INT8 per-tensor** (scalar scale, scalar zero-point) | 41 Q + 41 DQ pairs — wraps all Conv (×20), Gemm (×1), and residual Add (×8) |
| `resnet18_fp8_qdq.onnx` | **FP8 E4M3FN per-channel** | **FP8 E4M3FN per-tensor** | 46 Q + 46 DQ pairs — same op coverage as INT8 |
| `resnet18_int4_qdq.onnx` | **INT4 weight-only on `fc` only**, per-output-channel + block size 128 along in-features (AWQ-style) | none | 0 Q + 1 DQ — every Conv stays in FP |

So the **INT8 scheme is symmetric per-channel weights + per-tensor activations** — the canonical TRT-compatible scheme. The FP8 scheme mirrors it. The INT4 scheme is essentially a float graph with an INT4-dequantized FC weight and is **not a meaningful baseline** for the planned QAT comparison.

**Implication for PT2E:** A custom `Quantizer` (or `XNNPACKQuantizer` configured equivalently) must produce the same scheme — symmetric per-channel `qint8` weights + per-tensor `quint8`/`qint8` activations — otherwise the PT2E-trained scales will not match the TRT INT8 baseline at the bit-exact level.

### 4. Where should the QAT output be deployed? (ONNX QDQ → TRT, or elsewhere?)

**Another ONNX QDQ model, fed back into TensorRT** — same deployment path as the existing modelopt PTQ flow. Specifically:

- The chosen INT4 strategy is **all Conv layers quantized to INT4 weights, activations left unquantized (or W4A8 if TRT requires)** — see decision (b) from the planning discussion.
- Export path will need to be: PT2E QAT model → `torch.export` → ONNX with QDQ nodes that TRT 10.x recognizes for INT4 weight-only Conv on Blackwell.
- This rules out `XNNPACKQuantizer` as a drop-in choice — XNNPACK targets ExecuTorch / mobile and does not produce TRT-compatible Q/DQ. A **custom `Quantizer` subclass** is required, modeled after either modelopt's INT4 annotation rules or the `TensorRTQuantizer` reference if/when one exists.

### 5. What does the existing QDQ ONNX look like — weight-only or W8A8?

**INT8: full W8A8.** Both weights and activations are wrapped. Verified by inspection:

```
QuantizeLinear  conv1.weight                  axis=0  scale_shape=[64]   zp=int8     (per-channel along out-channels)
QuantizeLinear  images_cast_to_fp16           axis=1* scale_shape=[]     zp=int8     (* default, irrelevant — scalar = per-tensor)
QuantizeLinear  layer1.0.conv1.weight         axis=0  scale_shape=[64]   zp=int8
... (one Q/DQ pair per weight tensor + one per activation tensor)
```

**FP8: full W8A8 equivalent** (E4M3FN datatype, `axis=0` for weights with `scale_shape=[out_channels]`, scalar scale for activations). 46 pairs vs INT8's 41 because modelopt's FP8 mode wraps additional eligible nodes.

**INT4: weight-only on the FC head**:

```
DequantizeLinear  fc.weight_i4   axis=0  block_size=128  scale_shape=[100, 4]
```

That is: 100 output channels × 4 blocks of 128 input-features = 512 in-features (the FC layer is `Linear(512, 100)`). This is a standard AWQ-style grouped weight-quant scheme.

No activation Q/DQ anywhere in the INT4 graph. The FC weight is stored as a static INT4 constant and dequantized at runtime. As noted in answer #3, this means there is currently no INT4 baseline for the Conv layers to compare PT2E QAT against — generating that baseline (whether via modelopt-with-overrides or via the new PT2E flow) is part of what motivates this work.

(For visual confirmation, opening any of these ONNX files in [netron.app](https://netron.app) shows the Q/DQ placement directly.)

---

## About the training setup

### 6. GPU(s) used

**NVIDIA GeForce RTX 5060 Ti** (Blackwell, compute capability sm_120, consumer 16 GB SKU).

Implications:
- **PT2E export must run on CUDA** so the captured graph contains `cudnn_batch_norm` rather than CPU-side `native_batch_norm`. The official PT2E doc calls this out as the difference between "good" and "degraded" numerics.
- **Hardware support for the planned formats:**
  - INT8 Tensor Cores: native, fully supported in TRT 10.
  - FP8 E4M3/E5M2: native (Blackwell adds FP8 over Hopper).
  - INT4 weight-only: supported in TRT 10 on Hopper+; Blackwell included.
  - W4A4 / NVFP4: Blackwell-native but TRT support is still maturing — not the target here.
- 16 GB VRAM is more than sufficient for ResNet-18 QAT at batch 256 with FP32 weights + fake-quant overhead.

### 7. ImageNet subset (classes + image count)

- **100 classes** (the first 100 alphabetically from ImageNet-1K, filtered by `cfg.num_classes` in `src/data.py`).
- **Train: 124,395 images** (~1,244 per class — i.e., effectively the full ImageNet-1K training set restricted to these 100 classes; not a downsample).
- **Holdout val: 5,000 images** (50 per class), stratified, seed 42.
- The holdout is carved out of the train split via `build_train_holdout_split`, so the standard ImageNet `val` partition is *not* what the runner evaluates against — it uses this internal holdout.

**QAT budget implications:** with ~124k training images, a typical QAT recipe (start from converged FP32, drop LR by 10–100×, train 1–5 epochs with frozen BN) is well within reach. Expect a single QAT epoch to be ~5–10 minutes on the 5060 Ti at batch 256. A full 5-epoch QAT pass should fit in under an hour.

### 8. Initialize QAT scales from existing PTQ ONNX, or fresh?

**Fresh PT2E calibration.**

This means the planned flow is:

1. Load FP32 `best.pth` into the float `ResNet18` module on CUDA.
2. `torch.export` to capture the graph.
3. Apply the custom `Quantizer` (annotates which nodes get fake-quant observers and with what spec).
4. Run `prepare_qat_pt2e(exported_program, quantizer)`.
5. Run a calibration pass over a few hundred batches from the train loader to populate observer min/max.
6. Train QAT for a few epochs.
7. `convert_pt2e` and re-export to QDQ ONNX for TRT.

No scale-warm-start from `resnet18_int8_qdq.onnx` will be attempted in this iteration. (Worth revisiting if QAT struggles to converge — modelopt's calibrated INT8 scales are a defensible warm start, but wiring that into PT2E observers is non-trivial and would only be justified if cold-start calibration proves unstable.)
