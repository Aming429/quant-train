#!/usr/bin/env bash
# ------------------------------------------------------------------
# 公司 GPU 卡 QAT 训练启动脚本
# 用 YAML 配置驱动，适配1-4卡环境
#
# 用法:
#   bash scripts/train_qat.sh                    # 使用默认 configs/train_qat.yaml
#   bash scripts/train_qat.sh configs/my_cfg.yaml
# ------------------------------------------------------------------
set -euo pipefail

CONFIG="${1:-configs/train_qat.yaml}"

# 检测可用的 GPU 数量
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "[WARN] 未检测到 GPU，使用 CPU 训练（极慢）"
    DEVICE="cpu"
elif [ "$NUM_GPUS" -eq 1 ]; then
    echo "[INFO] 检测到 1 张 GPU，单卡训练"
    export CUDA_VISIBLE_DEVICES=0
    DEVICE="cuda:0"
else
    echo "[INFO] 检测到 $NUM_GPUS 张 GPU，使用 Accelerate 启动"
    # Accelerate 自动处理 multi-GPU
fi

# 确保 conda 环境
ENV_NAME="quant-train-cuda"
if conda info --envs 2>/dev/null | grep -q "^$ENV_NAME "; then
    echo "[INFO] 激活 conda 环境: $ENV_NAME"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
else
    echo "[WARN] conda 环境 '$ENV_NAME' 不存在，尝试运行中..."
fi

# 创建输出目录
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="checkpoints/qat_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo " QAT 训练启动"
echo " 配置:       $CONFIG"
echo " 输出目录:   $OUTPUT_DIR"
echo " GPU 数量:   $NUM_GPUS"
echo "=============================================="

if [ "$NUM_GPUS" -gt 1 ]; then
    accelerate launch \
        --num_processes "$NUM_GPUS" \
        --mixed_precision bf16 \
        quant_train/train.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
else
    python quant_train/train.py \
        --config "$CONFIG" \
        --output_dir "$OUTPUT_DIR"
fi

echo "训练完成！输出目录: $OUTPUT_DIR"
