"""量化器基类 —— 定义 Quantizer 接口。

所有量化算法都继承 BaseQuantizer，实现自己的前向和反向逻辑。
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class BaseQuantizer(nn.Module):
    """量化器基类。

    子类需要实现:
      - _quantize(x) → x_q, scale, zp: 量化前向
      - _backward(ctx, grad_output): 反向传播

    每个 Quantizer 实例对应一个量化位置（weight / input / output），
    可以独立配置 bits / symmetric / per_channel 等参数。
    """

    def __init__(
        self,
        bits: int = 8,
        symmetric: bool = False,
        per_channel: bool = False,
    ):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self._enabled = True

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：enabled 时做伪量化，否则透传。"""
        if not self._enabled:
            return x
        return self._quantize(x)

    def _quantize(self, x: torch.Tensor) -> torch.Tensor:
        """子类实现具体的量化→反量化逻辑。"""
        raise NotImplementedError

    @staticmethod
    def _compute_scale(
        x: torch.Tensor,
        bits: int,
        symmetric: bool,
        per_channel: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据数据计算 scale 和 zero_point 的初始值。

        返回 (scale, zero_point)，形状与 per_channel 设置匹配。
        """
        qmax = (1 << bits) - 1

        if per_channel:
            # per-channel: 按最后一维分组
            min_val = x.amin(dim=tuple(range(x.dim() - 1)), keepdim=True)
            max_val = x.amax(dim=tuple(range(x.dim() - 1)), keepdim=True)
        else:
            min_val, max_val = x.min(), x.max()

        if symmetric:
            abs_max = torch.max(min_val.abs(), max_val.abs())
            scale = abs_max / (qmax // 2) + 1e-10
            zero_point = torch.zeros_like(scale)
        else:
            scale = (max_val - min_val) / qmax + 1e-10
            zero_point = torch.clamp(torch.round(-min_val / scale), 0, qmax)

        return scale, zero_point

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, Any]]) -> Optional["BaseQuantizer"]:
        """从配置字典创建 Quantizer。cfg 为 None 或空时返回 None（不量化）。"""
        if cfg is None:
            return None
        if not cfg.get("enabled", True):
            return None
        bits = cfg.get("bits", 8)
        symmetric = cfg.get("symmetric", False)
        per_channel = cfg.get("per_channel", True)
        return cls(bits=bits, symmetric=symmetric, per_channel=per_channel)

    def extra_repr(self) -> str:
        return f"bits={self.bits}, symmetric={self.symmetric}, per_channel={self.per_channel}"
