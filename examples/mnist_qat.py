#!/usr/bin/env python
"""MNIST 上的 QAT 完整演示 —— 从训练到量化感知再到精度对比。

一个完整的端到端例子，在 MNIST 上展示：
1. 正常训练浮点模型
2. 用 QAT fine-tune（在浮点模型基础上插入伪量化）
3. 对比两种 weight 的量化误差

CPU 上约 2-3分钟跑完。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from quant_train.quant.qat import prepare_model_with_fake_quant


class SimpleNet(nn.Module):
    """简单的 MNIST 分类网络。"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")

    # 数据
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_loader = DataLoader(datasets.MNIST("data", train=True, download=True, transform=transform), batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST("data", train=False, download=True, transform=transform), batch_size=1000)

    # 1. 训练浮点 baseline
    print("=" * 50)
    print("阶段 1: 训练浮点模型")
    fp_model = SimpleNet().to(device)
    optimizer = torch.optim.Adam(fp_model.parameters(), lr=1e-3)
    for epoch in range(3):
        loss = train_one_epoch(fp_model, train_loader, optimizer, device)
        acc = evaluate(fp_model, test_loader, device)
        print(f"  Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

    fp_acc = evaluate(fp_model, test_loader, device)
    print(f"  浮点模型最终精度: {fp_acc:.4f}")
    print()

    # 2. 准备 QAT 模型，继承浮点权重
    print("阶段 2: QAT fine-tune（4-bit 非对称 per-channel）")
    qat_model = prepare_model_with_fake_quant(
        SimpleNet().to(device),
        bits=4, symmetric=False, per_channel=True, start_epoch=0,
    )
    # 复制浮点权重
    qat_model.load_state_dict(fp_model.state_dict())

    optimizer_q = torch.optim.Adam(qat_model.parameters(), lr=1e-4)
    for epoch in range(3):
        loss = train_one_epoch(qat_model, train_loader, optimizer_q, device)
        acc = evaluate(qat_model, test_loader, device)
        print(f"  QAT Epoch {epoch}: loss={loss:.4f}, acc={acc:.4f}")

    qat_acc = evaluate(qat_model, test_loader, device)
    print(f"  QAT 模型最终精度: {qat_acc:.4f}")
    print()

    # 3. 直接量化的对比（不经过 QAT 训练，直接取整）
    print("阶段 3: 对比——naive 量化 vs QAT 量化")
    from quant_train.quant.utils import quantize_tensor, dequantize_tensor
    naive_model = SimpleNet().to(device)
    naive_model.load_state_dict(fp_model.state_dict())

    with torch.no_grad():
        for name, param in naive_model.named_parameters():
            if "weight" in name and param.dim() >= 2:
                q, s, zp = quantize_tensor(param.data, bits=4, symmetric=False, per_channel=True)
                dq = dequantize_tensor(q, s, zp, symmetric=False)
                param.data.copy_(dq)

    naive_acc = evaluate(naive_model, test_loader, device)
    print(f"  Naive 量化精度: {naive_acc:.4f}")
    print(f"  QAT    量化精度: {qat_acc:.4f}")
    print(f"  浮点    基线:   {fp_acc:.4f}")
    print()
    print(f"  QAT 挽回精度: {(qat_acc - naive_acc) * 100:.2f}%")


if __name__ == "__main__":
    main()
