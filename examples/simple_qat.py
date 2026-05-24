#!/usr/bin/env python
"""最简单的 QAT 示例 —— 单层 Linear 在 CPU 上感受量化效果。

不需要 HuggingFace 依赖，直接运行。
"""
import torch
import torch.nn as nn

from quant_train.quant.qat import FakeQuantLinear


def main():
    # 创建原始线性层
    in_features, out_features = 16, 8
    x = torch.randn(4, in_features)

    # --- 浮点 baseline ---
    linear = nn.Linear(in_features, out_features)
    out_fp = linear(x)
    print(f"[FP32] 输出范围: {out_fp.min():.4f} ~ {out_fp.max():.4f}")

    # --- 8-bit QAT 层（对称、per-channel）---
    qlinear = FakeQuantLinear(in_features, out_features, bits=8, symmetric=True, per_channel=True)
    qlinear.weight.data.copy_(linear.weight.data)
    qlinear.bias.data.copy_(linear.bias.data)

    out_q8 = qlinear(x)
    mse = ((out_fp - out_q8) ** 2).mean().item()
    print(f"[INT8] 输出范围: {out_q8.min():.4f} ~ {out_q8.max():.4f}")
    print(f"[INT8] MSE: {mse:.6f}")

    # --- 4-bit QAT 层 ---
    qlinear4 = FakeQuantLinear(in_features, out_features, bits=4, symmetric=True, per_channel=True)
    qlinear4.weight.data.copy_(linear.weight.data)
    qlinear4.bias.data.copy_(linear.bias.data)

    out_q4 = qlinear4(x)
    mse4 = ((out_fp - out_q4) ** 2).mean().item()
    print(f"[INT4] 输出范围: {out_q4.min():.4f} ~ {out_q4.max():.4f}")
    print(f"[INT4] MSE: {mse4:.6f}")
    print(f"[INFO] 8→4 bit 误差增长: {mse4/mse:.2f}x")
    print()
    print("QAT 训练中这个误差会通过反向传播被补偿掉——这就是 QAT 的核心。")


if __name__ == "__main__":
    main()
