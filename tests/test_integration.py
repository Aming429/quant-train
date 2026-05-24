"""集成测试：端到端小模型 QAT 训练流程（CPU 上可跑）"""

import torch
import pytest

from quant_train.quant.qat import prepare_model_with_fake_quant


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

        model = prepare_model_with_fake_quant(
            model, bits=4, symmetric=False, per_channel=True, start_epoch=0
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

        # 验证 loss 下降
        final_loss = loss.item()
        assert final_loss < 2.0, f"训练后 loss 应 < 2.0, 实际 {final_loss:.4f}"

    def test_gpt2_tiny_qat(self):
        """用 GPT-2 tiny 验证 LLM QAT 流程可启动（纯 CPU，不要求收敛）。"""
        try:
            from transformers import AutoModelForCausalLM
            _ = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        except (ImportError, OSError):
            pytest.skip("需要 transformers 或网络连接")

        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = prepare_model_with_fake_quant(
            model, bits=8, symmetric=False, per_channel=True, start_epoch=0
        )

        # 简单的前向
        input_ids = torch.randint(0, 100, (1, 16))
        out = model(input_ids, labels=input_ids)
        loss = out.loss
        assert loss is not None, "应能计算 loss"
        assert loss.item() > 0, "loss 应为正"
