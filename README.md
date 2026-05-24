quant-train — 量化模型训练框架
===============================

丢丢（liuyuming）的量化模型训练工具箱。
本地（Mac CPU）开发 + GitHub 管理 + 公司 GPU 卡验证。

工作流
------

    开发阶段                 验证阶段
  ┌──────────────┐         ┌──────────────┐
  │ Mac 本地      │  git   │ 公司 GPU 机器 │
  │ CPU 小模型    │ push   │ 真实模型训练   │
  │ pytest 全覆盖 │ ─────→ │ bash scripts/ │
  │ 代码逻辑验证   │        │ 跑量实验      │
  └──────────────┘         └──────────────┘


快速开始
--------

### 本地开发环境

    conda env create -f environment.yaml
    conda activate quant-train

    # 跑全部测试（CPU 上可执行）
    pytest tests/ -v

    # 跑一个简单示例
    python examples/simple_qat.py

### 公司 GPU 环境

    git clone <你的仓库> ~/quant-train
    cd ~/quant-train
    conda env create -f environment-cuda.yaml
    conda activate quant-train-cuda

    # 7B 模型 QAT 训练
    bash scripts/train_qat.sh


目录结构
--------

    .
    ├── environment.yaml         # 本地 CPU conda 环境
    ├── environment-cuda.yaml    # 公司 GPU conda 环境
    ├── .gitignore
    ├── README.md
    │
    ├── configs/                 # 训练配置（YAML）
    ├── data/                    # 数据加载 & tokenize
    ├── models/                  # 模型定义（量化层）
    ├── quant/                   # 量化算法（QAT / LSQ / GPTQ-FT）
    ├── trainer/                 # 训练循环 & 分布式
    ├── tests/                   # 本地可跑的测试
    ├── scripts/                 # 公司卡启动脚本
    └── examples/                # 入门示例


支持的量化方法计划
------------------

- [x] QAT (Quantization-Aware Training) — 基础
- [ ] LSQ (Learned Step Size Quantization)
- [ ] GPTQ Fine-tuning
- [ ] LoRA + Quant (QLoRA 风格)
- [ ] 自定义混合精度量化策略


开发原则
--------

1. 所有核心逻辑在 CPU 上可运行并测试
2. 每个量化算子配套 pytest + 精度对比
3. CUDA 相关代码用 `torch.cuda.is_available()` 保护
4. 配置驱动：训练参数、量化参数分离到 YAML
