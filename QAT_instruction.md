# Agent instructions: PT2E INT8 QAT for ResNet18

## Objective

Create `training/train_qat_pt2e.py` — a standalone script that performs INT8 Quantization-Aware Training (QAT) using the PyTorch 2 Export (PT2E) flow, producing a PT2E checkpoint that can later be exported to TensorRT-compatible QDQ ONNX.

This follows the official torchao PT2E QAT workflow documented at:
https://docs.pytorch.org/ao/stable/pt2e_quantization/pt2e_quant_qat.html

---

## Context: what already exists

The codebase has a trained FP32 ResNet18 (100-class ImageNet subset) and existing PTQ/TRT pipelines. The QAT script you build is a new training step between FP32 training and ONNX export. The ONNX export is handled separately — your script only needs to produce the PT2E checkpoint.

### Existing PTQ INT8 scheme (must match exactly)

The modelopt-produced `onnx/resnet18_int8_qdq.onnx` uses:

| Component | Scheme |
|-----------|--------|
| Weights | INT8 symmetric per-channel, axis=0 (per output channel), zero-point=0 |
| Activations | INT8 per-tensor (scalar scale + scalar zero-point) |
| Coverage | All Conv2d (20 layers) + Linear/Gemm (1) + residual Add (8) = 41 Q+DQ pairs |

The custom PT2E quantizer you write must replicate this scheme so the QAT-trained model produces equivalent QDQ nodes.

---

## Files to read before coding

Read these files from the codebase to understand the existing conventions:

| File | What to look for |
|------|-----------------|
| `src/model.py` | `ResNet18` class definition, constructor signature (`num_classes`, `dropout_p`), import path |
| `src/data.py` | `build_train_holdout_split()` function signature and return type, data transforms, `ImageFolder` usage |
| `src/config.py` | The dataclass config structure — field names, defaults, how `num_classes`, `data_root`, `batch_size` etc. are stored |
| `training/train_fp32.py` | Training loop conventions — how optimizer, scheduler, criterion are set up; logging patterns |
| `training/train_qat_modelopt.py` | Prior QAT attempt with modelopt — see the patterns used for loading checkpoints, saving, evaluation |
| `requirements-qat.txt` | Current QAT dependencies |

---

## File to create

### `training/train_qat_pt2e.py`

A single standalone script. No modifications to existing files.

---

## Exact implementation steps

### Step 1: Imports and setup

```python
# PT2E quantization (torchao 0.16.0 import paths)
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e
import torchao.quantization.pt2e as pt2e_utils

# Quantizer base classes (from torch.ao, NOT torchao)
from torch.ao.quantization.quantizer import (
    Quantizer,
    QuantizationAnnotation,
    QuantizationSpec,
)
from torch.ao.quantization import (
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
)

# Project imports
from src.model import ResNet18
from src.data import build_train_holdout_split
```

**IMPORTANT**: In torchao 0.16.0, the PT2E functions live under `torchao.quantization.pt2e.quantize_pt2e`. The Quantizer base classes remain in `torch.ao.quantization.quantizer`. Do NOT import from `torch.ao.quantization.quantize_pt2e` — that is the old path from PyTorch core and may conflict.

### Step 2: Build the custom TRT INT8 Quantizer

Write a class `TRTInt8Quantizer(Quantizer)` that annotates the aten IR graph.

**Activation spec** — INT8 per-tensor affine:
```python
QuantizationSpec(
    dtype=torch.int8,
    quant_min=-128,
    quant_max=127,
    qscheme=torch.per_tensor_affine,
    is_dynamic=False,
    observer_or_fake_quant_ctr=MovingAverageMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_tensor_affine,
        quant_min=-128,
        quant_max=127,
    ),
)
```

**Weight spec** — INT8 symmetric per-channel (axis=0):
```python
QuantizationSpec(
    dtype=torch.int8,
    quant_min=-127,      # symmetric: -127 to 127, NOT -128
    quant_max=127,
    qscheme=torch.per_channel_symmetric,
    ch_axis=0,           # per output-channel [O,I,H,W]
    is_dynamic=False,
    observer_or_fake_quant_ctr=MovingAveragePerChannelMinMaxObserver.with_args(
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        quant_min=-127,
        quant_max=127,
        ch_axis=0,
    ),
)
```

**The `annotate()` method** must walk `model.graph.nodes` and annotate these aten ops:

| aten target | What it is | What to annotate |
|-------------|-----------|----------|
| `torch.ops.aten.conv2d.default` | Conv2d layers | input activation (args[0]) + weight (args[1]) + output |
| `torch.ops.aten.convolution.default` | Conv2d (alternate form, post-export) | input activation (args[0]) + weight (args[1]) + output |
| `torch.ops.aten.addmm.default` | Linear (FC head). Args: (bias, input, weight) | input (args[1]) + weight (args[2]) + output |
| `torch.ops.aten.linear.default` | Linear (alternate form) | input (args[0]) + weight (args[1]) + output |
| `torch.ops.aten.add.Tensor` | Residual skip connections (8 total) | output activation only |
| `torch.ops.aten.add_.Tensor` | In-place residual add variant | output activation only |

**Annotation pattern.** PT2E `QuantizationAnnotation` has two fields you set:

- `input_qspec_map: Dict[fx.Node, QuantizationSpec]` — tells PT2E to insert a FakeQuantize on each *input edge* listed. **This is how you quantize input activations and weights.**
- `output_qspec: QuantizationSpec` — tells PT2E to insert a FakeQuantize on the *output edge*.

For **Conv2d / Linear** (input + weight + output):
```python
input_node  = node.args[0]   # for conv2d/convolution/linear; use args[1] for addmm
weight_node = node.args[1]   # for conv2d/convolution/linear; use args[2] for addmm

node.meta["quantization_annotation"] = QuantizationAnnotation(
    input_qspec_map={
        input_node:  act_spec,
        weight_node: weight_spec,
    },
    output_qspec=act_spec,
    _annotated=True,
)
```

For **residual `add.Tensor` / `add_.Tensor`** (output only — inputs come from already-annotated upstream Conv outputs, so no need to re-quantize them):
```python
node.meta["quantization_annotation"] = QuantizationAnnotation(
    output_qspec=act_spec,
    _annotated=True,
)
```

**Why `output_qspec` alone is wrong for Conv/Linear:** without `input_qspec_map`, PT2E will not insert FakeQuantize on the weight tensor or the input activation. Step 7's "~41 FakeQuantize nodes" check will fail, and the converted ONNX will not contain Q/DQ around weights — defeating the purpose of QAT.

**Guard against double-annotation**: before annotating any node, check if `node.meta.get("quantization_annotation")` already exists with `_annotated=True`. Skip if so.

**Note on the "~41" target.** PT2E *shares* observers across edges (the output of ConvN feeds the input of ConvN+1, so one FakeQuantize covers both). The actual count after `prepare_qat_pt2e` will be smaller than modelopt's 41 Q + 41 DQ pairs — expect roughly one FakeQuantize per *unique* tensor in the quantized subgraph. Treat 41 as an upper bound, not an exact target. The right verification is: every Conv/Linear weight has an upstream FakeQuantize, and every Conv/Linear/Add input/output is on an edge with a FakeQuantize.

### Step 3: Load FP32 model

```python
model = ResNet18(num_classes=args.num_classes, pretrained=False)  # `dropout` defaults to 0.0
ckpt = torch.load("checkpoints/best.pth", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])  # checkpoint is wrapped: {epoch, model, optimizer, scaler, scheduler, best_acc}
model = model.cuda()
```

**Constructor signature** (from `src/model.py:46`):
`ResNet18(num_classes: int = 1000, pretrained: bool = True, dropout: float = 0.0)`. Note the kwarg is `dropout` (not `dropout_p`), and you **must** pass `pretrained=False` — leaving the default `True` plus `num_classes != 1000` raises `ValueError`.

**Config mechanism**: existing training scripts (`training/train_fp32.py`) use `argparse` with `--data` / `--num-classes` / `--batch-size` etc., **not** `src/config.py`'s `ExperimentConfig` (that dataclass is for the inference runner). Match `train_fp32.py`'s argparse style — see Step 4. If you do pull from `ExperimentConfig`, the data-root field is **`imagenet_path`**, not `data_root`, and its default is `/home/pf4636/imagenet` (full ImageNet, not the 100-class subset at `/home/pf4636/imagenet2`).

### Step 4: Build data loaders

Import and call `build_train_holdout_split` from `src/data.py`. This function produces train/val index splits using seed=42, 50 holdout images per class. Use the same transforms as `training/train_fp32.py`.

### Step 5: Evaluate FP32 baseline

Run standard evaluation (model.eval() + torch.no_grad()) on the holdout val set. Print top-1 and top-5 accuracy. This gives the accuracy ceiling for comparison.

### Step 6: Export to aten IR

```python
model.train()  # MUST be train mode for BN capture
example_inputs = (torch.randn(2, 3, 224, 224, device="cuda"),)
exported_model = torch.export.export_for_training(model, example_inputs).module()
```

**Critical**: the model must be on CUDA before export. This produces `torch.ops.aten.cudnn_batch_norm` ops which give better QAT numerics than the CPU `native_batch_norm`. Verify this after export:

```python
for n in exported_model.graph.nodes:
    if n.op == "call_function" and "batch_norm" in str(n.target):
        print(f"BN op: {n.target}")
        break
```

If you get `_native_batch_norm_legit` instead of `cudnn_batch_norm`, swap it:
```python
for n in exported_model.graph.nodes:
    if n.target == torch.ops.aten._native_batch_norm_legit.default:
        n.target = torch.ops.aten.cudnn_batch_norm.default
exported_model.recompile()
```

### Step 7: Prepare for QAT

```python
quantizer = TRTInt8Quantizer()
prepared_model = prepare_qat_pt2e(exported_model, quantizer)
```

`prepare_qat_pt2e` does three things in order:
1. Fuses Conv2d + BatchNorm2d in the graph (before inserting fake quant)
2. Calls `quantizer.annotate()` on the fused graph
3. Inserts FakeQuantize modules at all annotated edges

Print how many FakeQuantize nodes were inserted and verify it's approximately 41 (matching modelopt's Q+DQ count).

### Step 8: QAT training loop

**CRITICAL API DIFFERENCE**: after export, you CANNOT call `model.train()` or `model.eval()`. Instead use:
```python
torchao.quantization.pt2e.move_exported_model_to_train(model)   # before training
torchao.quantization.pt2e.move_exported_model_to_eval(model)    # before evaluation
```

**Hyperparameters**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Stable for QAT fine-tuning |
| Learning rate | 1e-5 | Very small — weights are already trained, just adapting to quantization |
| Weight decay | 1e-5 | Light regularization |
| Scheduler | CosineAnnealingLR, T_max=epochs, eta_min=lr×0.01 | Smooth decay |
| Epochs | 10 | ~11% of original 90 FP32 epochs |
| Batch size | 32 | Match original training |

**Observer and BN freeze schedule** (from official tutorial):

```python
for epoch in range(1, num_epochs + 1):
    # Train one epoch
    train_one_epoch(prepared_model, ...)

    # Freeze observers after epoch 4
    if epoch > 4:
        prepared_model.apply(torchao.quantization.pt2e.disable_observer)

    # Freeze BN stats after epoch 3
    if epoch > 3:
        for n in prepared_model.graph.nodes:
            if n.target in [
                torch.ops.aten._native_batch_norm_legit.default,
                torch.ops.aten.cudnn_batch_norm.default,
            ]:
                new_args = list(n.args)
                new_args[5] = False   # training flag → False
                n.args = tuple(new_args)
        prepared_model.recompile()

    # Evaluate quantized model every 2 epochs
    if epoch % 2 == 0 or epoch == num_epochs:
        model_copy = copy.deepcopy(prepared_model)
        quantized = convert_pt2e(model_copy)
        evaluate(quantized, val_loader, ...)
```

**Why deepcopy + convert for eval**: `convert_pt2e` is destructive — it replaces FakeQuantize with real Q/DQ ops. You cannot continue training after conversion. So during training, always evaluate on a deepcopy.

### Step 9: Convert and save

After the training loop, restore best weights, then convert:

```python
torchao.quantization.pt2e.move_exported_model_to_eval(prepared_model)
quantized_model = convert_pt2e(prepared_model)
```

Save to `checkpoints/qat_pt2e/`:

```python
os.makedirs("checkpoints/qat_pt2e", exist_ok=True)

# PT2E model (includes graph + quantized weights)
torch.save(quantized_model.state_dict(), "checkpoints/qat_pt2e/best_converted.pt")

# Also save the prepared (pre-conversion) state for potential QAT resumption
torch.save(best_prepared_state_dict, "checkpoints/qat_pt2e/best_prepared.pt")
```

### Step 10: Final evaluation and summary

Print a results table:
```
FP32 baseline:       top1=XX.XX%  top5=XX.XX%
INT8 QAT (best):     top1=XX.XX%
INT8 QAT (final):    top1=XX.XX%  top5=XX.XX%
Accuracy delta:      X.XX%
Checkpoints saved:   checkpoints/qat_pt2e/
```

---

## Output directory structure

```
checkpoints/
└── qat_pt2e/
    ├── best_converted.pt     # Converted model (real Q/DQ, for ONNX export)
    └── best_prepared.pt      # Pre-conversion model (for resuming QAT)
```

---

## Things NOT to do

1. **Do NOT modify any existing files** — this is a new standalone script only.
2. **Do NOT handle ONNX export** — that's done separately via `notebooks/04_qdq_export.ipynb`.
3. **Do NOT use `XNNPACKQuantizer`** — that's for mobile/ExecuTorch. We need TRT-compatible Q/DQ.
4. **Do NOT call `model.train()` or `model.eval()`** on the exported model — use `move_exported_model_to_train/eval`.
5. **Do NOT use `torch.export.export()`** (inference-only) — use `torch.export.export_for_training()` which preserves autograd.
6. **Do NOT import `prepare_qat_pt2e` from `torch.ao.quantization.quantize_pt2e`** — use the torchao path: `from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e, convert_pt2e`.

---

## Container / environment notes

The script will run inside a container. Required packages:

```
torch>=2.6.0
torchvision>=0.21.0
torchao==0.16.0
tqdm
```

PyTorch >= 2.6 is required for `torch.export.export_for_training()`. Earlier versions need the deprecated `capture_pre_autograd_graph` API.

The GPU is an NVIDIA RTX 5060 Ti (Blackwell, sm_120). PyTorch >= 2.7 or nightly is recommended for full sm_120 kernel support, but 2.6 should work for QAT training since fake-quant ops run in float.

---

## Verification checklist

After implementation, verify:

- [ ] Script runs end-to-end without errors
- [ ] FP32 baseline accuracy is printed and matches known value
- [ ] Export produces `cudnn_batch_norm` ops (not `_native_batch_norm_legit`)
- [ ] `prepare_qat_pt2e` inserts ~41 FakeQuantize nodes (matching modelopt's Q+DQ count)
- [ ] Observer freezing happens after epoch 4 (printed to console)
- [ ] BN stat freezing happens after epoch 3 (printed to console)
- [ ] Quantized eval accuracy improves over epochs (or at least doesn't collapse)
- [ ] Checkpoints are saved to `checkpoints/qat_pt2e/`
- [ ] `best_converted.pt` loads without error
- [ ] No existing files were modified