"""quant 包 —— 离线量化工具 + 可插拔 QAT 量化器。"""

from quant_train.quant.base import BaseQuantizer
from quant_train.quant.ste import STEQuantizer
from quant_train.quant.qat import (
    QuantLinear,
    prepare_model_with_quant,
    create_quantizer,
    QUANTIZER_REGISTRY,
)
from quant_train.quant.utils import (
    quantize_tensor,
    dequantize_tensor,
    compute_quant_error,
)

__all__ = [
    # 基类
    "BaseQuantizer",
    # 量化器
    "STEQuantizer",
    # QAT 替换
    "QuantLinear",
    "prepare_model_with_quant",
    "create_quantizer",
    "QUANTIZER_REGISTRY",
    # 工具
    "quantize_tensor",
    "dequantize_tensor",
    "compute_quant_error",
]
