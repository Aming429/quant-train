"""量化工具函数 —— 独立的量化和反量化计算，与训练解耦。"""

import torch


def quantize_tensor(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = False,
    per_channel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """量化张量到定点数，返回 (quantized, scale, zero_point)。

    纯计算，不参与训练图，用于推理时导出。
    """
    qmin, qmax = 0, (1 << bits) - 1

    if per_channel:
        # per-channel: 对最后一维取 min/max → 每个通道一个 scale
        dim = x.dim() - 1
        min_val = x.amin(dim=dim, keepdim=True)
        max_val = x.amax(dim=dim, keepdim=True)
    else:
        min_val, max_val = x.min(), x.max()

    if symmetric:
        abs_max = torch.max(min_val.abs(), max_val.abs())
        scale = abs_max / (qmax // 2)
        zero_point = torch.zeros_like(scale)
    else:
        scale = (max_val - min_val) / (qmax - qmin) + 1e-10
        zero_point = torch.round(-min_val / scale)
        zero_point = torch.clamp(zero_point, qmin, qmax)

    x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
    return x_q, scale, zero_point


def dequantize_tensor(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    symmetric: bool = False,
) -> torch.Tensor:
    """反量化回浮点。"""
    if symmetric:
        return x_q * scale
    return (x_q - zero_point) * scale


def compute_quant_error(
    x: torch.Tensor,
    bits: int = 8,
    symmetric: bool = False,
    per_channel: bool = False,
) -> tuple[torch.Tensor, float]:
    """计算量化误差（MSE + 相对误差）。"""
    x_q, scale, zp = quantize_tensor(x, bits, symmetric, per_channel)
    x_dq = dequantize_tensor(x_q, scale, zp, symmetric)

    mse = ((x - x_dq) ** 2).mean().item()
    rel_err = ((x - x_dq).abs() / (x.abs() + 1e-10)).mean().item()
    return mse, rel_err
