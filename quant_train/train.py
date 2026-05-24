#!/usr/bin/env python
"""
quant_train/train.py —— QAT 训练入口

在本地 CPU 或公司 GPU 上运行 QAT 训练。
用法：
    python quant_train/train.py --config configs/default.yaml --output_dir checkpoints/test
"""

import argparse
import yaml
import torch
from pathlib import Path

from quant_train.data import load_and_tokenize
from quant_train.models.base_model import load_base_model, prepare_for_qat
from quant_train.trainer import QATTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="checkpoints/test")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 设备
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] 使用设备: {device}")

    # 数据
    data_cfg = config.get("data", {})
    print(f"[INFO] 加载数据集: {data_cfg.get('dataset')}/{data_cfg.get('subset')}")
    train_ds, val_ds, tokenizer = load_and_tokenize(
        dataset_name=data_cfg["dataset"],
        subset=data_cfg.get("subset"),
        text_column=data_cfg.get("text_column", "text"),
        tokenizer_name=config["model"]["name"],
        max_seq_length=data_cfg.get("max_seq_length", 128),
        val_split=data_cfg.get("val_split", 0.05),
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["per_device_batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config["training"]["per_device_batch_size"]
    )

    # 模型
    model_name = config["model"]["name"]
    torch_dtype = getattr(torch, config["model"].get("torch_dtype", "float32"))
    print(f"[INFO] 加载模型: {model_name} ({torch_dtype})")
    model = load_base_model(model_name, torch_dtype=torch_dtype)

    quant_cfg = config.get("quant", {})
    print(f"[INFO] 量化配置: {quant_cfg}")
    model = prepare_for_qat(
        model,
        quant_cfg=quant_cfg,
    )

    # 计算总 step 数供 scheduler 用
    total_steps = len(train_loader) * config["training"]["num_epochs"]
    config["training"]["total_steps"] = total_steps

    # 训练器
    trainer = QATTrainer(model, config, device)

    # 训练循环
    print(f"[INFO] 开始训练，共 {config['training']['num_epochs']} 个 epoch")
    for epoch in range(config["training"]["num_epochs"]):
        train_loss = trainer.train_epoch(train_loader, epoch)
        val_loss = trainer.evaluate(val_loader)
        print(f"  [Epoch {epoch}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # 保存
        ckpt_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch}.pt"
        trainer.save_checkpoint(str(ckpt_path))
        print(f"  [Epoch {epoch}] checkpoint saved: {ckpt_path}")

    print("[INFO] 训练完成!")


if __name__ == "__main__":
    main()
