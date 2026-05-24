"""测试 QuantLinear + 可插拔 Quantizer"""

import torch
import pytest

from quant_train.quant.qat import QuantLinear, prepare_model_with_quant
from quant_train.quant.ste import STEQuantizer


class TestQuantLinear:
    def test_weight_quant_forward_backward(self):
        """weight 量化：前向不崩溃 + 反向传梯度到 weight 和 scale。"""
        wq = STEQuantizer(bits=8, symmetric=False, per_channel=True)
        layer = QuantLinear(32, 64, weight_quantizer=wq)
        wq.calibrate(layer.weight.data)

        x = torch.randn(4, 32)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weight.grad is not None, "weight 应有梯度"
        assert wq.scale.grad is not None, "scale 应有梯度"

    def test_input_quant(self):
        """input 量化不影响反向传播。"""
        wq = STEQuantizer(bits=8, symmetric=False, per_channel=True)
        iq = STEQuantizer(bits=8, symmetric=False, per_channel=False)
        layer = QuantLinear(16, 32, weight_quantizer=wq, input_quantizer=iq)
        wq.calibrate(layer.weight.data)
        iq.calibrate(torch.randn(1, 16))

        x = torch.randn(2, 16)
        out = layer(x)
        out.sum().backward()
        assert layer.weight.grad is not None

    def test_no_quant_passthrough(self):
        """全部量化器为 None 时等同普通 Linear。"""
        layer = QuantLinear(10, 20)
        x = torch.randn(3, 10)
        out = layer(x)
        assert out.shape == (3, 20)

    def test_weight_preserved_after_replace(self):
        """替换后权重应与原 Linear 一致。"""
        original = torch.nn.Linear(10, 20)
        orig_weight = original.weight.data.clone()

        replaced = prepare_model_with_quant(
            original, {"bits": 8, "symmetric": False, "per_channel": True}
        )

        assert isinstance(replaced, QuantLinear), "root Linear 应被替换为 QuantLinear"
        assert torch.allclose(replaced.weight.data, orig_weight), "权重应被保留"

    def test_detailed_config_parse(self):
        """详细版配置应正确解析。"""
        cfg = {
            "weight": {"type": "ste", "bits": 4, "symmetric": True, "per_channel": True},
            "input": {"enabled": False},
            "output": {"enabled": False},
        }
        original = torch.nn.Linear(16, 32)
        replaced = prepare_model_with_quant(original, cfg)
        assert isinstance(replaced, QuantLinear)
        assert replaced.weight_quantizer is not None
        assert replaced.weight_quantizer.bits == 4
        assert replaced.weight_quantizer.symmetric is True
        assert replaced.input_quantizer is None
        assert replaced.output_quantizer is None
