import numpy as np
import modelopt.onnx.quantization as moq


def quantize_qdq(onnx_path, calib_data: np.ndarray, quant_dtype="int8", output_path=None):
    moq.quantize(
        onnx_path=str(onnx_path),
        quantize_mode=quant_dtype,
        calibration_data=calib_data,
        output_path=output_path,
    )