import torch
import torch.nn as nn
from torch.export import export

def quantize_int8_x86_pt2e(
    model: nn.Module,
    calib_loader,
    calib_num_batches: int = 10,
) -> nn.Module:
    if calib_loader is None:
        raise ValueError("calib_loader is required for INT8 PTQ.")

    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    import torchao.quantization.pt2e as pt2e_utils
    from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (
        X86InductorQuantizer,
        get_default_x86_inductor_quantization_config,
    )

    model = model.to("cpu").eval()

    images0, _ = next(iter(calib_loader))
    example_x = images0.contiguous().to("cpu")   

    exported = export(model, (example_x,))  

    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config())

    prepared_model = prepare_pt2e(exported.module(), quantizer)
    pt2e_utils.move_exported_model_to_eval(prepared_model)

    with torch.inference_mode():
        for i, (images, _) in enumerate(calib_loader):
            if i >= calib_num_batches:
                break
            prepared_model(images.contiguous().to("cpu"))

    quantized_model = convert_pt2e(prepared_model)
    pt2e_utils.move_exported_model_to_eval(quantized_model)

    return quantized_model