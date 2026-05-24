"""基础模型包装 —— 加载 HuggingFace 模型，插入 QuantLinear。"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from quant_train.quant.qat import prepare_model_with_quant


def load_base_model(
    model_name: str,
    torch_dtype: torch.dtype = torch.float32,
    device_map: Optional[str] = None,
) -> PreTrainedModel:
    """加载 HuggingFace 基础模型。

    本地 CPU: torch_dtype=float32, device_map=None
    公司 GPU: torch_dtype=bfloat16, device_map="auto"
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.train()
    return model


def prepare_for_qat(
    model: nn.Module,
    quant_cfg: dict,
) -> nn.Module:
    """将模型中的线性层替换为带 Quantizer 的 QuantLinear。

    quant_cfg 格式:
      # 简化版
      {"bits": 4, "symmetric": true, "per_channel": true}

      # 详细版（分别配置 weight/input/output）
      {
        "weight": {"type": "ste", "bits": 4, "symmetric": true, "per_channel": true},
        "input": {"enabled": false},
        "output": {"enabled": false},
      }

    加新量化算法：在 QUANTIZER_REGISTRY 注册后，type 字段切换即可。
    """
    return prepare_model_with_quant(model, quant_cfg)
