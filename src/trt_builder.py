from pathlib import Path
from typing import Optional

import torch
import tensorrt as trt

# ---------------------------------------------------------------------------
# Custom logger — captures TRT messages into a list so they always appear
# in Jupyter output instead of being lost to stderr.
# ---------------------------------------------------------------------------

class _PythonLogger(trt.ILogger):
    """
    Silent during successful builds — only prints on ERROR.
    On a failed build, call dump() to see the full log for debugging.
    """
    def __init__(self):
        super().__init__()
        self.messages = []

    def log(self, severity, msg):
        self.messages.append(f"[TRT {severity.name}] {msg}")
        # Only print immediately for hard errors — everything else is buffered
        # and only shown if the build fails (via dump())
        if severity == trt.ILogger.Severity.ERROR:
            print(self.messages[-1])

    def dump(self):
        """Print the full build log — called automatically on build failure."""
        for m in self.messages:
            print(m)
        self.messages.clear()


_LOGGER = _PythonLogger()


# ---------------------------------------------------------------------------
# INT8 Calibrator  (only needed for INT8 without Q/DQ nodes in ONNX)
# FP8 and INT4 don't need a calibrator — scales live in the ONNX Q/DQ nodes
# ---------------------------------------------------------------------------

class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds batches from a DataLoader to TensorRT's INT8 calibration."""

    def __init__(self, dataloader, cache_file: str | Path, num_batches: int, device: str):
        super().__init__()
        self.dataloader  = iter(dataloader)
        self.cache_file  = Path(cache_file)
        self.num_batches = num_batches
        self.device      = torch.device(device)
        self.batch_count = 0
        self._buf        = None  # keeps tensor alive while TRT reads the pointer

    def get_batch_size(self) -> int:
        return 0  # TRT infers batch size from get_batch()

    def get_batch(self, names):
        if self.batch_count >= self.num_batches:
            return None
        try:
            images, _ = next(self.dataloader)
        except StopIteration:
            return None
        self._buf = images.to(dtype=torch.float32, device=self.device).contiguous()
        self.batch_count += 1
        return [int(self._buf.data_ptr())]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            print(f"[trt_builder] Loading calibration cache: {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_bytes(cache)
        print(f"[trt_builder] Calibration cache saved: {self.cache_file}")


# ---------------------------------------------------------------------------
# Engine builder
#
# Precision modes and how they work:
#
#   fp16  — standard FP16 layer fusion. TRT selects FP16 kernels where safe.
#           No calibration needed.
#
#   int8  — PTQ via entropy calibration. Requires an Int8EntropyCalibrator
#           (passed in via `calibrator`), OR a QDQ-annotated ONNX from modelopt.
#
#   fp8   — Requires TRT 9.0+ and a QDQ-annotated ONNX (e.g. from modelopt).
#           Scales are baked into the graph — no calibrator needed here.
#           FP16 flag is also set as a fallback for layers TRT can't run in FP8.
#
#   int4  — INT4 weights-only quantization. Requires TRT 10.0+ and a
#           QDQ-annotated ONNX from modelopt. No calibrator needed.
#           FP16 is set as the compute/activation precision.
# ---------------------------------------------------------------------------

def build_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    precision: str = "fp32",        # "fp32" | "fp16" | "int8" | "fp8" | "int4"
    batch_size: int = 1,            # max (and optimal) batch size for the engine profile
    workspace_mb: int = 2048,
    calibrator: Optional[trt.IInt8Calibrator] = None,
) -> Path:
    """
    Build a TensorRT engine from an ONNX file and save it to disk.

    For int8  -> pass an Int8EntropyCalibrator (or use a QDQ-annotated ONNX).
    For fp8   -> ONNX must have FP8 Q/DQ nodes (e.g. from modelopt). No calibrator.
    For int4  -> ONNX must have INT4 Q/DQ nodes (e.g. from modelopt). No calibrator.
    """
    onnx_path   = Path(onnx_path)
    engine_path = Path(engine_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    _LOGGER.messages.clear()
    print(f"[trt_builder] Building engine | precision={precision} | batch={batch_size} | workspace={workspace_mb} MiB")

    builder = trt.Builder(_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser  = trt.OnnxParser(network, _LOGGER)

    # parse_from_file is required when the ONNX has an external data sidecar
    # (e.g. resnet18.onnx.data). parser.parse(bytes) only sees the in-memory
    # protobuf and silently misses the weight file, causing fc.bias import errors.
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX parse failed:\n" + "\n".join(errors))

    input_tensor = network.get_input(0)
    onnx_batch   = input_tensor.shape[0]  # -1 = dynamic, >0 = fixed (e.g. modelopt QDQ graphs)

    # If the ONNX has a fixed batch dim (common for modelopt FP8/INT4 exports),
    # use that value for the profile instead of the requested batch_size.
    # If it's dynamic (-1), use batch_size as requested.
    if onnx_batch != -1:
        if onnx_batch != batch_size:
            print(f"[trt_builder] NOTE: ONNX has fixed batch={onnx_batch}, "
                  f"overriding requested batch_size={batch_size}.")
        profile_bs = onnx_batch
    else:
        profile_bs = batch_size

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))

    # Optimization profile defines the batch size range TRT compiles for.
    # For fixed-batch ONNX (modelopt exports): min=opt=max=fixed batch.
    # For dynamic ONNX: min=1 so single-sample inference also works.
    profile     = builder.create_optimization_profile()
    profile.set_shape(input_tensor.name,
    min=(1,          3, 224, 224),   # always allow batch=1
    opt=(profile_bs, 3, 224, 224),
    max=(profile_bs, 3, 224, 224),
)
    config.add_optimization_profile(profile)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)

    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is not None:
            config.int8_calibrator = calibrator
        else:
            print("[trt_builder] NOTE: No calibrator given — assuming QDQ-annotated ONNX.")

    elif precision == "fp8":
        # FP8 scales come from Q/DQ nodes in the ONNX (modelopt output)
        # FP16 is enabled as fallback for layers TRT cannot run natively in FP8
        config.set_flag(trt.BuilderFlag.FP8)
        config.set_flag(trt.BuilderFlag.FP16)

    elif precision == "int4":
        # INT4 weights-only: weights quantized to INT4, activations stay FP16
        config.set_flag(trt.BuilderFlag.INT4)
        config.set_flag(trt.BuilderFlag.FP16)

    # fp32 -> no flags needed, TRT defaults to FP32

    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("\n[trt_builder] Full TRT build log:")
        _LOGGER.dump()
        raise RuntimeError(
            f"TRT engine build failed for precision='{precision}'. "
            "See the TRT log above for details."
        )

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(engine_bytes)
    print(f"[trt_builder] Engine saved: {engine_path} ({engine_path.stat().st_size / 1e6:.1f} MB)")
    return engine_path