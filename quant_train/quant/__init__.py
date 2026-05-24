from quant_train.quant.qat import FakeQuantLinear, prepare_model_with_fake_quant
from quant_train.quant.utils import (
    quantize_tensor,
    dequantize_tensor,
    compute_quant_error,
)

__all__ = [
    "FakeQuantLinear",
    "prepare_model_with_fake_quant",
    "quantize_tensor",
    "dequantize_tensor",
    "compute_quant_error",
]
