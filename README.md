# quant-train

量化感知训练（QAT）框架。在本地 Mac 上开发和测试，在 GPU 机器上跑真实训练。

## 快速开始

```bash
git clone <repo-url> ~/quant-train
cd ~/quant-train
```

### 本地开发（Mac CPU）

```bash
conda env create -f environment.yaml
conda activate quant-train
pytest tests/ -v
```

### 公司 GPU 训练

```bash
conda env create -f environment-cuda.yaml
conda activate quant-train-cuda
bash scripts/train_qat.sh
```

---

## 目录结构

```
quant-train/
├── environment.yaml          ← 本地 Mac conda 环境（PyTorch CPU）
├── environment-cuda.yaml     ← 公司 GPU conda 环境（CUDA 12.4 + bitsandbytes）
├── .gitignore
├── README.md                 ← 本文档
│
├── quant_train/
│   ├── train.py              ← 训练入口脚本
│   ├── data/                 ← 数据集加载 & tokenize
│   ├── models/               ← HuggingFace 模型加载 + QAT 包装
│   ├── quant/
│   │   ├── qat.py            ← 核心: FakeQuantLinear + STE 前反向
│   │   └── utils.py          ← 量化/反量化工具 + 误差计算
│   └── trainer/              ← 训练循环（单卡）
│
├── configs/
│   ├── default.yaml          ← 本地开发配置（0.5B 模型, CPU 可跑）
│   └── train_qat.yaml        ← 公司训练配置（1B 模型, 4-bit QAT）
│
├── tests/
│   ├── test_quant_layers.py  ← FakeQuantLinear 测试
│   ├── test_quant_utils.py   ← 量化工具测试
│   └── test_integration.py   ← 端到端 QAT 流程测试
│
├── scripts/
│   ├── train_qat.sh          ← 公司 GPU 启动脚本（自动检测卡数）
│   └── train_qat_tmux.sh     ← tmux 后台模式（断网不掉训练）
│
└── examples/
    ├── simple_qat.py         ← 10 秒感受量化效果
    └── mnist_qat.py          ← MNIST QAT 完整演示（3 分钟）
```

---

## 工作流

```
        开发阶段                    验证阶段
    ┌────────────────┐         ┌─────────────────┐
    │ Mac 本地        │  git   │ 公司 GPU 机器    │
    │ CPU 小模型跑通  │  push  │  bash scripts/   │
    │ pytest 全绿     │ ─────→ │  真实模型训练    │
    │ 代码逻辑验证    │        │  结果对比        │
    └────────────────┘         └─────────────────┘
```

---

## 环境详解

### 本地环境（`environment.yaml`）

| 依赖 | 版本 | 用途 |
|------|------|------|
| pytorch | 2.5 (CPU) | 前向/反向计算 |
| transformers | >=4.45 | HuggingFace 模型加载 |
| datasets | >=2.20 | 训练数据加载 |
| accelerate | >=1.0 | 设备映射 |
| pytest | latest | 测试 |

创建：
```bash
conda env create -f environment.yaml
conda activate quant-train
```

### GPU 环境（`environment-cuda.yaml`）

| 依赖 | 版本 | 用途 |
|------|------|------|
| pytorch | 2.5 (CUDA 12.4) | GPU 计算 |
| bitsandbytes | >=0.44 | 量化算子 |
| flash-attn | 选装 | 加速 attention |
| nvitop | latest | GPU 监控 |

创建：
```bash
conda env create -f environment-cuda.yaml
conda activate quant-train-cuda
```

---

## 配置说明

所有训练参数通过 YAML 配置文件控制。

### 完整配置项

```yaml
model:
  name: "Qwen/Qwen2.5-0.5B"        # HuggingFace 模型名
  load_in_8bit: false               # 是否 8-bit 加载（预量化）
  torch_dtype: float32              # 权重 dtype

quant:
  method: "qat"                     # 量化方法（当前仅支持 qat）
  bits: 8                           # 量化位数（4/8）
  symmetric: false                  # 对称 / 非对称
  per_channel: true                 # per-channel / per-tensor
  qat:
    start_epoch: 0                  # 从第几个 epoch 开始伪量化
    calibrate_first: true           # 先跑 calibration（预留）

training:
  output_dir: "checkpoints/"
  num_epochs: 3
  per_device_batch_size: 2          # CPU 调小, GPU 调大
  gradient_accumulation_steps: 1
  learning_rate: 5e-5
  warmup_steps: 100
  logging_steps: 10
  save_steps: 500
  eval_steps: 500
  fp16: false                       # CPU 不开, GPU 改为 true
  distributed: false

data:
  dataset: "wikitext"
  subset: "wikitext-2-v1"
  text_column: "text"
  max_seq_length: 128               # 本地调小, 公司改 2048
  val_split: 0.05
```

### 本地开发配置（`configs/default.yaml`）

- 模型: 0.5B（如 Qwen2.5-0.5B）
- 精度: float32（CPU 兼容）
- batch_size: 2（CPU 不爆内存）
- seq_length: 128（快速 tokenize）

### 公司训练配置（`configs/train_qat.yaml`）

- 模型: 1B（如 Llama-3.2-1B）
- 精度: bfloat16
- bit: 4（真实验证 4-bit QAT 效果）
- batch_size: 4（24G 显存够用）

---

## 命令参考

### 本地开发

```bash
# 激活环境
conda activate quant-train

# 跑全部测试
pytest tests/ -v

# 跑单个测试文件
pytest tests/test_quant_layers.py -v

# 跑单个测试
pytest tests/test_quant_layers.py::TestFakeQuantLinear::test_forward_backward -v

# 跑入门示例
python examples/simple_qat.py

# 跑 MNIST 完整 QAT 对比
python examples/mnist_qat.py

# 用默认配置跑训练（CPU 极慢, 仅验证流程）
python quant_train/train.py --config configs/default.yaml
```

### 公司 GPU 训练

```bash
conda activate quant-train-cuda

# 单卡训练
bash scripts/train_qat.sh
# 或指定配置
bash scripts/train_qat.sh configs/train_qat.yaml

# tmux 后台模式（断网不掉）
bash scripts/train_qat_tmux.sh
# 连回查看
tmux attach -t qat_train

# 直接启动（自定义输出目录）
python quant_train/train.py \
    --config configs/train_qat.yaml \
    --output_dir checkpoints/my_exp_001
```

---

## 核心实现说明

### FakeQuantize（伪量化）

前向：`x → quantize → dequantize → x_q`，模拟量化误差。
反向：STE（Straight-Through Estimator），梯度绕过量化截断直接传回浮点权重。

```
前向:
  x_dq = round(clamp(x / scale + zp)) * scale           (非对称)
  x_dq = round(clamp(x / scale)) * scale                 (对称)

反向:
  ∂x_dq/∂x = 1              ← STE
  ∂x_dq/∂scale = x_q - zp   ← LSQ 风格梯度
```

### FakeQuantLinear

替换 `nn.Linear`，内部维护 `FakeQuantize` 的 scale/zero_point 参数。
训练完成后可导出为真正量化的权重。

### epoch gating

通过 `set_epoch(epoch)` 控制伪量化的生效时机：
- `epoch < start_epoch`: 行为等同普通 Linear（float forward）
- `epoch >= start_epoch`: FakeQuant 生效

---

## 扩展指南

### 添加新的量化方法

1. 在 `quant_train/quant/` 下新建文件（如 `lsq.py`）
2. 实现 `prepare_model_with_xxx(model, bits, ...)` 函数
3. 在 `quant_train/quant/__init__.py` 中导出
4. 在 `configs/` 中添加对应配置
5. 在 `tests/` 中添加测试

### 添加新的训练器

1. 在 `quant_train/trainer/` 下新建
2. 继承或参考 `QATTrainer`
3. 在 `quant_train/train.py` 中按 `config.quant.method` 分发

---

## 核心理念

1. **所有核心逻辑在 CPU 上可运行 + 可测试**
2. **每个量化算子配套 pytest**
3. **CUDA 代码用 `torch.cuda.is_available()` 保护**
4. **配置驱动：训练、量化、数据参数分离到 YAML**
5. **本地开发 -> git push -> 公司卡拉取运行**

---

## 故障排除

### conda 环境创建失败

```bash
# 清理 conda 缓存
conda clean -a
# 重新尝试：指定频道
conda env create -f environment-cuda.yaml -c pytorch -c conda-forge -c nvidia
```

### CUDA out of memory

```yaml
# 减小 batch size
training:
  per_device_batch_size: 1
  gradient_accumulation_steps: 8  # 保持等效 batch size

# 减小序列长度
data:
  max_seq_length: 256
```

### HuggingFace 下载慢

公司内网可能需要代理：
```bash
export HF_ENDPOINT=https://hf-mirror.com
# 或使用国内镜像
export HF_HUB_ENABLE_HF_TRANSFER=1
```
