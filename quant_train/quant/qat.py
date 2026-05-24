"""QAT —— 用 QuantLinear 替换 nn.Linear，可插拔 Quantizer。

核心改动：
  - QuantLinear 有 3 个 Quantizer 槽位，各自独立配置
  - _replace_linear_recursive 根据 config 自动创建并注入 Quantizer
  - prepare_model_with_quant 为入口函数
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_train.quant.base import BaseQuantizer
from quant_train.quant.ste import STEQuantizer


# ── Quantizer 注册表 ───────────────────────────────────────
# 扩展时在这里加：新名字 → 新类
QUANTIZER_REGISTRY: Dict[str, type] = {
    "ste": STEQuantizer,
    # "lsq": LSQQuantizer,     # 预留
    # "pact": PACTQuantizer,   # 预留
    # "your_method": YourQuantizer,
}


def create_quantizer(
    cfg: Optional[Dict[str, Any]],
    example_x: torch.Tensor,
) -> Optional[BaseQuantizer]:
    """从配置字典创建 Quantizer，并用 example_x 校准。

    cfg = None 或 {"enabled": false} → 返回 None（不量化）
    """
    if cfg is None:
        return None
    if not cfg.get("enabled", True):
        return None

    method = cfg.pop("type", "ste")
    cls = QUANTIZER_REGISTRY.get(method)
    if cls is None:
        raise ValueError(f"未知量化器类型: {method}，可用: {list(QUANTIZER_REGISTRY.keys())}")

    q = cls(
        bits=cfg.get("bits", 8),
        symmetric=cfg.get("symmetric", False),
        per_channel=cfg.get("per_channel", True),
    )
    q.calibrate(example_x)
    return q


# ── QuantLinear ─────────────────────────────────────────────

class QuantLinear(nn.Module):
    """带可插拔 Quantizer 的线性层。

    Args:
        in_features: 输入维度
        out_features: 输出维度
        bias: 是否使用 bias
        weight_quantizer: weight 量化器（权重伪量化）
        input_quantizer: 输入量化器（激活伪量化）
        output_quantizer: 输出量化器（输出伪量化）

    前向流程:
        input_quant → F.linear(weight_quant(weight)) → output_quant
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_quantizer: Optional[BaseQuantizer] = None,
        input_quantizer: Optional[BaseQuantizer] = None,
        output_quantizer: Optional[BaseQuantizer] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.weight_quantizer = weight_quantizer
        self.input_quantizer = input_quantizer
        self.output_quantizer = output_quantizer

    def forward(self, x):
        if self.input_quantizer is not None:
            x = self.input_quantizer(x)

        w = self.weight
        if self.weight_quantizer is not None:
            w = self.weight_quantizer(w)

        out = F.linear(x, w, self.bias)

        if self.output_quantizer is not None:
            out = self.output_quantizer(out)

        return out

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}"


# ── 模型替换 ────────────────────────────────────────────────

def _make_quant_cfg(quant_cfg: dict, role: str) -> Optional[dict]:
    """从全局量化配置中提取指定角色（weight/input/output）的配置。"""
    role_cfg = quant_cfg.get(role, None)
    if role_cfg is None:
        # 默认只量化 weight
        if role == "weight":
            return {
                "type": quant_cfg.get("type", "ste"),
                "bits": quant_cfg.get("bits", 8),
                "symmetric": quant_cfg.get("symmetric", False),
                "per_channel": quant_cfg.get("per_channel", True),
            }
        return None
    return role_cfg


def _create_quantizers(
    quant_cfg: dict,
    weight_example: torch.Tensor,
) -> tuple:
    """根据配置创建三个 Quantizer（weight/input/output）。

    quant_cfg 支持两种格式:
      1. 简化版: {"bits": 4, "symmetric": true}
         只量化 weight，使用 STE
      2. 详细版: {"weight": {...}, "input": {...}, "output": {...}}
         分别配置每个位置的量化器
    """
    # 判断是简化版还是详细版
    if any(k in quant_cfg for k in ("weight", "input", "output")):
        # 详细版
        w_cfg = quant_cfg.get("weight")
        i_cfg = quant_cfg.get("input")
        o_cfg = quant_cfg.get("output")
    else:
        # 简化版 → 仅 weight 量化
        w_cfg = {**quant_cfg, "type": quant_cfg.get("type", "ste")}
        i_cfg = None
        o_cfg = None

    weight_q = create_quantizer(w_cfg, weight_example)
    input_q = create_quantizer(i_cfg, weight_example[:1]) if i_cfg else None
    output_q = create_quantizer(o_cfg, weight_example[:1]) if o_cfg else None

    return weight_q, input_q, output_q


def _replace_linear_recursive(
    module: nn.Module,
    quant_cfg: dict,
    current_epoch: int = 0,
):
    """递归替换 nn.Linear 为 QuantLinear，按配置注入 Quantizer。

    如果 module 本身是 nn.Linear 也替换。
    """
    if isinstance(module, nn.Linear):
        weight_q, input_q, output_q = _create_quantizers(
            quant_cfg, module.weight.data
        )
        new_layer = QuantLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            weight_quantizer=weight_q,
            input_quantizer=input_q,
            output_quantizer=output_q,
        )
        new_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            new_layer.bias.data.copy_(module.bias.data)
        return new_layer

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            weight_q, input_q, output_q = _create_quantizers(
                quant_cfg, child.weight.data
            )
            new_layer = QuantLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                weight_quantizer=weight_q,
                input_quantizer=input_q,
                output_quantizer=output_q,
            )
            new_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            _replace_linear_recursive(child, quant_cfg, current_epoch)


def prepare_model_with_quant(
    model: nn.Module,
    quant_cfg: dict,
) -> nn.Module:
    """将模型所有 nn.Linear 替换为 QuantLinear。"""
    result = _replace_linear_recursive(model, quant_cfg)
    return result if result is not None else model
