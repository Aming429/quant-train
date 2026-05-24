"""集成测试：端到端小模型 QAT 训练流程（CPU 上可跑）"""

import torch
import pytest

from quant_train.quant.qat import prepare_model_with_quant
from quant_train.quant.ste import STEQuantizer


class TestIntegration:
    def test_small_mlp_qat(self):
        """用极小的 MLP 验证 QAT 训练流程可在 CPU 上完整跑通。"""
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2),
        )

        model = prepare_model_with_quant(
            model, {"bits": 4, "symmetric": False, "per_channel": True}
        )

        x = torch.randn(8, 16)
        y = torch.randint(0, 2, (8,))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        for step in range(20):
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < 2.0, f"训练后 loss 应 < 2.0, 实际 {final_loss:.4f}"

    def test_gpt2_tiny_qat(self):
        """用 GPT-2 tiny 验证 LLM QAT 流程可在 CPU 上启动。"""
        try:
            from transformers import AutoModelForCausalLM
            _ = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        except (ImportError, OSError):
            pytest.skip("需要 transformers 或网络连接")

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = prepare_model_with_quant(
            model, {"bits": 8, "symmetric": False, "per_channel": True}
        )

        input_ids = torch.randint(0, 100, (1, 16))
        out = model(input_ids, labels=input_ids)
        loss = out.loss
        assert loss is not None, "应能计算 loss"
        assert loss.item() > 0, "loss 应为正"

    def test_input_quantization_small_mlp(self):
        """验证 input + weight 同时量化 MLP 能跑通。"""
        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 2),
        )

        # 详细版配置：weight 8-bit + input 8-bit
        model = prepare_model_with_quant(model, {
            "weight": {"type": "ste", "bits": 8, "symmetric": True, "per_channel": True},
            "input": {"type": "ste", "bits": 8, "symmetric": False, "per_channel": False},
            "output": {"enabled": False},
        })

        x = torch.randn(4, 16)
        y = torch.randint(0, 2, (4,))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(10):
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        assert loss.item() < 2.0
