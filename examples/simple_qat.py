#!/usr/bin/env python
"""最简单的 QAT 示例 —— 感受不同 Quantizer 的效果。

演示:
  1. 普通 Linear（浮点 baseline）
  2. weight 8-bit STE 量化
  3. weight 4-bit STE 量化
  4. weight 4-bit + input 8-bit 同时量化
"""
import torch
import torch.nn as nn

from quant_train.quant.ste import STEQuantizer
from quant_train.quant.qat import QuantLinear


def main():
    in_features, out_features = 16, 8
    x = torch.randn(4, in_features)

    # 1. 浮点 baseline
    linear = nn.Linear(in_features, out_features)
    out_fp = linear(x)
    print(f"[FP32]    输出范围: {out_fp.min():.4f} ~ {out_fp.max():.4f}")

    # 2. Weight 8-bit 量化
    wq8 = STEQuantizer(bits=8, symmetric=True, per_channel=True)
    ql8 = QuantLinear(in_features, out_features, weight_quantizer=wq8)
    ql8.weight.data.copy_(linear.weight.data)
    ql8.bias.data.copy_(linear.bias.data)
    wq8.calibrate(ql8.weight.data)

    out_q8 = ql8(x)
    mse8 = ((out_fp - out_q8) ** 2).mean().item()
    print(f"[W8]      输出范围: {out_q8.min():.4f} ~ {out_q8.max():.4f}  MSE: {mse8:.6f}")

    # 3. Weight 4-bit 量化
    wq4 = STEQuantizer(bits=4, symmetric=True, per_channel=True)
    ql4 = QuantLinear(in_features, out_features, weight_quantizer=wq4)
    ql4.weight.data.copy_(linear.weight.data)
    ql4.bias.data.copy_(linear.bias.data)
    wq4.calibrate(ql4.weight.data)

    out_q4 = ql4(x)
    mse4 = ((out_fp - out_q4) ** 2).mean().item()
    print(f"[W4]      输出范围: {out_q4.min():.4f} ~ {out_q4.max():.4f}  MSE: {mse4:.6f}")

    # 4. Weight 4-bit + Input 8-bit 同时量化
    iq8 = STEQuantizer(bits=8, symmetric=False, per_channel=False)
    iq8.calibrate(x)
    ql_wi = QuantLinear(in_features, out_features, weight_quantizer=wq4, input_quantizer=iq8)
    ql_wi.weight.data.copy_(linear.weight.data)
    ql_wi.bias.data.copy_(linear.bias.data)

    out_wi = ql_wi(x)
    mse_wi = ((out_fp - out_wi) ** 2).mean().item()
    print(f"[W4+I8]   输出范围: {out_wi.min():.4f} ~ {out_wi.max():.4f}  MSE: {mse_wi:.6f}")
    print()
    print("QAT 训练中反向传播会同时调整 weight + scale，逐步补偿这些误差。")


if __name__ == "__main__":
    main()
