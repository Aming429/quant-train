"""训练循环 —— 本地 CPU 可用，公司 GPU 可跑。"""
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
from tqdm import tqdm

from quant_train.data import load_and_tokenize
from quant_train.models.base_model import load_base_model, prepare_for_qat


class QATTrainer:
    """QAT 训练器。CPU 上可以跑小模型完整验证流程。"""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        lr = train_cfg.get("learning_rate", 5e-5)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.num_epochs = train_cfg.get("num_epochs", 3)
        self.logging_steps = train_cfg.get("logging_steps", 10)

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """训练一个 epoch。在 CPU 上可跑，但很慢（小模型+小数据）。"""
        self.model.train()

        # 更新所有 FakeQuantLinear 的当前 epoch
        self._set_epoch(epoch)

        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch["input_ids"]

            outputs = self.model(**batch, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            if step % self.logging_steps == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(dataloader)
        return avg_loss

    def evaluate(self, dataloader: DataLoader) -> float:
        """验证集评估。"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["input_ids"]
                outputs = self.model(**batch, labels=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                total_loss += loss.item()

        return total_loss / len(dataloader)

    def save_checkpoint(self, path: str):
        """保存 checkpoint。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            path,
        )

    def _set_epoch(self, epoch: int):
        """遍历模型设置当前 epoch。"""
        for module in self.model.modules():
            if hasattr(module, "set_epoch"):
                module.set_epoch(epoch)
