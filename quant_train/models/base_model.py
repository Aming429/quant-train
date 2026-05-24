"""基础模型包装 —— 加载 HuggingFace 模型，插入伪量化节点"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from quant_train.quant.qat import prepare_model_with_fake_quant


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
    bits: int = 8,
    symmetric: bool = False,
    per_channel: bool = True,
    start_epoch: int = 0,
) -> nn.Module:
    """将模型中的线性层替换为伪量化版本，为 QAT 做准备。

    返回的 model 在 CPU 上即可跑 forward/backward，
    scale/zero_point 会在训练中学习。
    """
    return prepare_model_with_fake_quant(
        model,
        bits=bits,
        symmetric=symmetric,
        per_channel=per_channel,
        start_epoch=start_epoch,
    )
