"""测试量化工具函数"""

import torch
import pytest

from quant_train.quant.utils import quantize_tensor, dequantize_tensor, compute_quant_error


class TestQuantUtils:
    def test_round_trip_symmetric(self):
        """对称量化反量化应大致还原。"""
        x = torch.randn(128, 64) * 0.5
        x_q, scale, zp = quantize_tensor(x, bits=8, symmetric=True, per_channel=False)
        x_dq = dequantize_tensor(x_q, scale, zp, symmetric=True)
        err = (x - x_dq).abs().mean().item()
        assert err < 0.1, f"8-bit 对称量化的 round-trip 误差应 < 0.1, 实际 {err:.6f}"

    def test_round_trip_asymmetric(self):
        """非对称量化反量化应大致还原。"""
        x = torch.randn(128, 64) * 0.5
        x_q, scale, zp = quantize_tensor(x, bits=8, symmetric=False, per_channel=False)
        x_dq = dequantize_tensor(x_q, scale, zp, symmetric=False)
        err = (x - x_dq).abs().mean().item()
        assert err < 0.1, f"8-bit 非对称量化的 round-trip 误差应 < 0.1, 实际 {err:.6f}"

    def test_per_channel_quant(self):
        """per-channel 量化应正常工作。"""
        x = torch.randn(16, 32)
        x_q, scale, zp = quantize_tensor(x, bits=4, symmetric=False, per_channel=True)
        assert scale.shape == (16, 1), "per-channel scale 应是 (out_features, 1)"

    def test_low_bit_error_increases(self):
        """更低 bit 应有更高误差。"""
        x = torch.randn(256, 256) * 2.0
        _, err_8 = compute_quant_error(x, bits=8, symmetric=False)
        _, err_4 = compute_quant_error(x, bits=4, symmetric=False)
        assert err_4 > err_8, "4-bit 误差应大于 8-bit"

    def test_max_min_range(self):
        """scale 应该在合理的数值范围内。"""
        x = torch.randn(1024) * 10.0  # 大范围数值
        x_q, scale, zp = quantize_tensor(x, bits=8, symmetric=False)
        assert scale.item() > 0, "scale 应为正"
