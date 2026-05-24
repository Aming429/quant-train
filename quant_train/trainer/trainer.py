"""训练循环 —— 本地 CPU 可用，公司 GPU 可跑。"""

from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm


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
        weight_decay = train_cfg.get("weight_decay", 0.0)
        self.num_epochs = train_cfg.get("num_epochs", 3)
        self.logging_steps = train_cfg.get("logging_steps", 10)
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 1)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # LR scheduler: warmup + cosine decay
        warmup_steps = train_cfg.get("warmup_steps", 0)
        total_steps = train_cfg.get("total_steps", None)  # 可选，不设则用 epoch 数
        self.scheduler = self._build_scheduler(warmup_steps, total_steps)

        # fp16 混合精度（仅 GPU 有效）
        self.fp16 = train_cfg.get("fp16", False) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        self.amp_dtype = torch.float16 if self.fp16 else torch.float32

    def _build_scheduler(self, warmup_steps: int, total_steps: Optional[int]):
        """构建 warmup → cosine decay 的 LR scheduler。"""
        if total_steps is None:
            # fallback: 粗略估计 total_steps
            return None

        if warmup_steps <= 0:
            return CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)

        warmup = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine = CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
        return SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_steps])

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        """训练一个 epoch。"""
        self.model.train()
        total_loss = 0.0
        accum_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = batch["input_ids"]

            # fp16 autocast（GPU 上有效，CPU 上无害透传）
            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.fp16,
                dtype=self.amp_dtype,
            ):
                outputs = self.model(**batch, labels=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
                loss = loss / self.gradient_accumulation_steps

            # 混合精度 scale + backward
            self.scaler.scale(loss).backward()

            accum_loss += loss.item()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪在这里加（预留位）
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # LR scheduler step
                if self.scheduler is not None:
                    self.scheduler.step()

            total_loss += loss.item() * self.gradient_accumulation_steps

            if step % self.logging_steps == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "lr": f"{current_lr:.2e}",
                })

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

                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.fp16,
                    dtype=self.amp_dtype,
                ):
                    outputs = self.model(**batch, labels=labels)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

                total_loss += loss.item()

        return total_loss / len(dataloader)

    def save_checkpoint(self, path: str):
        """保存 checkpoint，包含 scheduler 状态。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, path)
