import tensorrt as trt
import json
from collections import Counter

ENGINE_PATH = "/home/pf4636/code/resnet/quantized_resnets/engines/resnet18_tensorrt_fp8_in8b_cuda_bs1.engine"

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
with open(ENGINE_PATH, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

inspector = engine.create_engine_inspector()
precision_counts = Counter()

SKIP_TYPES = {"custom_layer", "signal", "wait"}

print(f"{'Idx':<5} {'Layer Name':<45} {'Type':<15} {'Input dtype(s)':<25} {'Output dtype(s)'}")
print("-" * 130)

for i in range(engine.num_layers):
    raw = inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)
    info = json.loads(raw)

    layer_type = info.get("LayerType", "?")
    if layer_type in SKIP_TYPES:
        continue

    name       = info.get("Name", f"layer_{i}")
    inputs     = info.get("Inputs", [])
    outputs    = info.get("Outputs", [])
    tactic     = info.get("TacticName", "")
    metadata   = info.get("Metadata", "")

    in_dtypes  = [t.get("Format/Datatype", "?") for t in inputs]
    out_dtypes = [t.get("Format/Datatype", "?") for t in outputs]

    print(f"  [{i:03d}] {name:<45} {layer_type:<15} {str(in_dtypes):<25} {out_dtypes}")

    if metadata:
        onnx_layers = metadata.replace("\x1f", ", ")
        print(f"         ONNX : {onnx_layers}")
    if tactic:
        print(f"         Tactic: {tactic[:80]}")

    for dt in out_dtypes:
        precision_counts[dt] += 1

print("\n--- Precision Summary (by output dtype) ---")
for prec, count in precision_counts.most_common():
    print(f"  {prec:<12}: {count:3d} layer(s)")