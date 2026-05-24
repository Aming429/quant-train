"""测试 fake quant layer 的前向和反向"""

import torch
import pytest

from quant_train.quant.qat import FakeQuantLinear, FakeQuantize


class TestFakeQuantLinear:
    def test_forward_backward(self):
        """验证前向不崩溃 + 反向能传梯度。"""
        layer = FakeQuantLinear(32, 64, bits=8, symmetric=False, per_channel=True)
        x = torch.randn(4, 32)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None, "梯度应传回 weight"
        assert layer.scale.grad is not None, "梯度应传回 scale"

    def test_symmetric_quant(self):
        """对称量化：scale 应为正数，zero_point 应为 0。"""
        layer = FakeQuantLinear(16, 16, bits=4, symmetric=True, per_channel=False)
        x = torch.randn(2, 16)
        _ = layer(x)
        assert (layer.scale > 0).all(), "scale 必须为正"
        assert (layer.zero_point == 0).all(), "对称量化 zero_point 应为 0"

    def test_weight_preserved_after_replace(self):
        """替换后权重应与原 Linear 一致（初始化时）。"""
        original = torch.nn.Linear(10, 20)
        orig_weight = original.weight.data.clone()

        from quant_train.quant.qat import _replace_linear_recursive
        _replace_linear_recursive(original, bits=8, symmetric=False, per_channel=False, start_epoch=0)

        # original 的第一层已被替换
        assert isinstance(original, FakeQuantLinear)
        assert torch.allclose(original.weight.data, orig_weight), "权重应被保留"

    def test_epoch_gating(self):
        """验证 set_epoch 控制伪量化开关。"""
        layer = FakeQuantLinear(8, 8, bits=4, symmetric=False, per_channel=False)
        x = torch.randn(1, 8)

        layer.set_epoch(0)
        layer._start_epoch = 2  # 从 epoch 2 开始伪量化
        out_disabled = layer(x)  # epoch=0 < 2，不量化

        layer.set_epoch(2)
        out_enabled = layer(x)   # epoch=2 >= 2，量化
        assert not torch.allclose(out_disabled, out_enabled), "启用量化后输出应变化"
