import torch
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto



QUANT_CONFIGS = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "fp8":  mtq.FP8_DEFAULT_CFG,
    "int4": mtq.INT4_AWQ_CFG,
}


def quantize_qdq(model, calib_loader, quant_dtype="int8", calib_num_batches=32, device="cuda"):
    model = model.to(device).eval()

    def calib_fn(model):
        with torch.inference_mode():
            for i, (images, _) in enumerate(calib_loader):
                if i >= calib_num_batches:
                    break
                model(images.to(device))

    return mtq.quantize(model, QUANT_CONFIGS[quant_dtype], forward_loop=calib_fn)


def export_qdq_onnx(model, onnx_path, device="cuda", opset_version=17):
    dummy = torch.randn(1, 3, 224, 224, device=device)
    mto.export(
        model, (dummy,), str(onnx_path),
        opset_version=opset_version,
        input_names=["images"],
        output_names=["logits"],
        do_constant_folding=False,  # must be False to keep Q/DQ nodes
    )
    print(f"Exported → {onnx_path}")