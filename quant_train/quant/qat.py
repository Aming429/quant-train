"""QAT —— Quantization-Aware Training 核心实现

核心思想：在前向传播中插入 FakeQuant 节点（量化→反量化），
让模型在训练时"感知"量化带来的精度损失，从而调整权重来补偿。

此模块全部用纯 PyTorch 实现，CPU 上即可运行和测试。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FakeQuantize(torch.autograd.Function):
    """前向：量化→反量化；反向：STE (Straight-Through Estimator)。

    - 对输入 x: gradient 直接绕过量化截断（标准 STE）
    - 对 scale/zero_point: 通过量化的解析梯度传播（LSQ 风格）
    """

    @staticmethod
    def forward(ctx, x, scale, zero_point, bits, symmetric):
        qmin, qmax = 0, (1 << bits) - 1
        x_div = x / scale
        if symmetric:
            x_q = torch.clamp(torch.round(x_div), qmin, qmax)
            x_dq = x_q * scale
        else:
            x_q = torch.clamp(torch.round(x_div + zero_point), qmin, qmax)
            x_dq = (x_q - zero_point) * scale
        ctx.save_for_backward(x_q, scale, zero_point)
        ctx.symmetric = symmetric
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        x_q, scale, zero_point = ctx.saved_tensors
        symmetric = ctx.symmetric
        # STE: x 的梯度原样通过
        grad_x = grad_output
        # scale/zero_point 梯度: 用 grad_output 在 batch 维求和, 再 reshape 匹配 shape
        grad_scale = None
        grad_zero_point = None
        if ctx.needs_input_grad[1]:
            # ∂x_dq/∂scale = x_q (对称) 或 x_q - zp (非对称)
            raw = grad_output * (x_q - zero_point)
            grad_scale = raw.reshape(-1, *scale.shape).sum(dim=0)
        if ctx.needs_input_grad[2] and not symmetric:
            raw = -grad_output * scale
            grad_zero_point = raw.reshape(-1, *zero_point.shape).sum(dim=0)
        return grad_x, grad_scale, grad_zero_point, None, None


def _ste_quant(x, scale, zero_point, bits, symmetric):
    """量化→反量化的前向计算，返回浮点但已"量化感知"的值。"""
    qmin, qmax = 0, (1 << bits) - 1

    x_div = x / scale
    if symmetric:
        # 对称量化：zero_point = 0
        x_q = torch.clamp(torch.round(x_div), qmin, qmax)
    else:
        x_q = torch.clamp(torch.round(x_div + zero_point), qmin, qmax)

    if symmetric:
        x_dq = x_q * scale
    else:
        x_dq = (x_q - zero_point) * scale

    return x_dq


class FakeQuantLinear(nn.Module):
    """带 FakeQuant 的线性层。

    训练中：weight 先经过 FakeQuantize 再做 forward
    推理时：可导出为真正量化的权重（在本层外做导出）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        symmetric: bool = False,
        per_channel: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel

        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Quantization parameters —— 训练中自动学习
        if per_channel:
            self.scale = nn.Parameter(torch.ones(out_features, 1))
        else:
            self.scale = nn.Parameter(torch.ones(1))

        if symmetric:
            self.register_buffer("zero_point", torch.zeros_like(self.scale))
        else:
            self.zero_point = nn.Parameter(torch.zeros_like(self.scale))

        self._current_epoch = 0
        self._start_epoch = 0
        self._enabled = True  # 训练时开关

    def forward(self, x):
        if self._enabled and self._current_epoch >= self._start_epoch:
            # QAT: weight 先过 FakeQuant
            w_q = FakeQuantize.apply(
                self.weight, self.scale, self.zero_point, self.bits, self.symmetric
            )
        else:
            w_q = self.weight

        return F.linear(x, w_q, self.bias)

    def set_epoch(self, epoch: int):
        self._current_epoch = epoch

    def extra_repr(self):
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.bits}, symmetric={self.symmetric}, "
            f"per_channel={self.per_channel}"
        )


def _replace_linear_recursive(module, bits, symmetric, per_channel, start_epoch):
    """递归替换 nn.Linear 为 FakeQuantLinear。

    如果 module 本身是 nn.Linear 也替换。
    """
    if isinstance(module, nn.Linear):
        new_layer = FakeQuantLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bits=bits,
            symmetric=symmetric,
            per_channel=per_channel,
            bias=module.bias is not None,
        )
        new_layer.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            new_layer.bias.data.copy_(module.bias.data)
        # 初始化 scale
        if per_channel:
            scale_init = module.weight.data.abs().max(dim=1, keepdim=True).values
        else:
            scale_init = module.weight.data.abs().max().unsqueeze(0)
        new_layer.scale.data.copy_(scale_init / ((1 << (bits - 1)) - 1) + 1e-10)
        return new_layer

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            new_layer = FakeQuantLinear(
                in_features=child.in_features,
                out_features=child.out_features,
                bits=bits,
                symmetric=symmetric,
                per_channel=per_channel,
                bias=child.bias is not None,
            )
            # 复制原始权重（让模型从预训练权重开始）
            new_layer.weight.data.copy_(child.weight.data)
            if child.bias is not None:
                new_layer.bias.data.copy_(child.bias.data)

            # 初始化 scale —— 用权重的 absmax
            if per_channel:
                scale_init = child.weight.data.abs().max(dim=1, keepdim=True).values
            else:
                scale_init = child.weight.data.abs().max().unsqueeze(0)
            # 加个小 epsilon 防止除零
            new_layer.scale.data.copy_(scale_init / ((1 << (bits - 1)) - 1) + 1e-10)

            setattr(module, name, new_layer)
        else:
            _replace_linear_recursive(child, bits, symmetric, per_channel, start_epoch)


def prepare_model_with_fake_quant(
    model: nn.Module,
    bits: int = 8,
    symmetric: bool = False,
    per_channel: bool = True,
    start_epoch: int = 0,
) -> nn.Module:
    """将模型所有 nn.Linear 替换为 FakeQuantLinear。

    Args:
        model: HuggingFace 或任意 PyTorch 模型
        bits: 量化位数
        symmetric: 是否对称量化
        per_channel: 是否 per-channel 量化
        start_epoch: 从第几个 epoch 起启用伪量化
    """
    _replace_linear_recursive(model, bits, symmetric, per_channel, start_epoch)
    return model
