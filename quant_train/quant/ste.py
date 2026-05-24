"""STEQuantizer —— 标准 STE 伪量化器，scale 可训练。

前向：量化→反量化（模拟精度损失）
反向：STE 直通 + LSQ 风格 scale 梯度

用法:
    q = STEQuantizer(bits=4, symmetric=True, per_channel=True)
    q.calibrate(weight_data)       # 初始化 scale
    x_q = q(x)                     # 伪量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from quant_train.quant.base import BaseQuantizer


class _STEFunction(torch.autograd.Function):
    """STE 伪量化：前向量化→反量化，反向梯度绕过截断。"""

    @staticmethod
    def forward(ctx, x, scale, zero_point, bits, symmetric):
        qmin, qmax = 0, (1 << bits) - 1

        if symmetric:
            x_q = torch.clamp(torch.round(x / scale), qmin, qmax)
            x_dq = x_q * scale
        else:
            x_q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)
            x_dq = (x_q - zero_point) * scale

        ctx.save_for_backward(x_q, scale, zero_point)
        ctx.symmetric = symmetric
        return x_dq

    @staticmethod
    def backward(ctx, grad_output):
        x_q, scale, zero_point = ctx.saved_tensors
        symmetric = ctx.symmetric

        # STE: x 梯度原样通过
        grad_x = grad_output

        # scale 梯度：d(x_dq)/d(scale) = x_q - zp → reshape 匹配 scale.shape
        grad_scale = None
        if ctx.needs_input_grad[1]:
            raw = grad_output * (x_q - zero_point)
            grad_scale = raw.reshape(-1, *scale.shape).sum(dim=0)

        # zero_point 梯度：d(x_dq)/d(zp) = -scale → reshape 匹配 zp.shape
        grad_zp = None
        if ctx.needs_input_grad[2] and not symmetric:
            raw = -grad_output * scale
            grad_zp = raw.reshape(-1, *zero_point.shape).sum(dim=0)

        return grad_x, grad_scale, grad_zp, None, None


class STEQuantizer(BaseQuantizer):
    """STE 量化器，scale 为可训练 nn.Parameter。

    初始化后需要调用 calibrate(x) 或 initialize_shape(example_x) 设置 scale/zp。
    """

    def __init__(self, bits=8, symmetric=False, per_channel=False):
        super().__init__(bits, symmetric, per_channel)
        # scale/zero_point 在 calibrate() 或 initialize_shape() 时创建
        self.scale = None
        self.zero_point = None

    def calibrate(self, x: torch.Tensor):
        """从数据初始化 scale/zero_point。"""
        scale, zp = self._compute_scale(x, self.bits, self.symmetric, self.per_channel)
        self.scale = nn.Parameter(scale)
        if not self.symmetric:
            self.zero_point = nn.Parameter(zp)
        else:
            self.zero_point = torch.zeros_like(scale)

    def initialize_shape(self, example_x: torch.Tensor):
        """仅初始化形状，值为 1/0（后续训练中学习）。"""
        if self.per_channel:
            # per-channel: 沿最后一维分组
            shape = (1,) * (example_x.dim() - 1) + (example_x.shape[-1],)
        else:
            shape = (1,) * example_x.dim()
        self.scale = nn.Parameter(torch.ones(shape))
        self.zero_point = nn.Parameter(torch.zeros(shape))
        if self.symmetric:
            self.register_buffer("zero_point", torch.zeros(shape))

    def _quantize(self, x):
        scale = self.scale
        zp = self.zero_point
        if scale is None:
            return x  # 未初始化时透传
        return _STEFunction.apply(x, scale, zp, self.bits, self.symmetric)
