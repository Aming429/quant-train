#!/usr/bin/env bash
# ------------------------------------------------------------------
# 公司 GPU 卡 QAT 训练 - 后台会话模式
# 用 tmux 启动，断网后继续跑
#
# 用法:
#   bash scripts/train_qat_tmux.sh [config_path] [session_name]
# ------------------------------------------------------------------
set -euo pipefail

CONFIG="${1:-configs/train_qat.yaml}"
SESSION_NAME="${2:-qat_train}"

# 先检查是否已有同名 session
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "[INFO] Session '$SESSION_NAME' 已存在，杀掉重建..."
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
fi

tmux new-session -d -s "$SESSION_NAME" -c "$(pwd)"

# 在 tmux 中执行
tmux send-keys -t "$SESSION_NAME" "
    source \$(conda info --base)/etc/profile.d/conda.sh
    conda activate quant-train-cuda
    bash scripts/train_qat.sh $CONFIG
" Enter

echo "训练在 tmux session '$SESSION_NAME' 中后台运行"
echo "  查看日志: tmux attach -t $SESSION_NAME"
echo "  分离:     Ctrl+B, D"
echo "  强制停止: tmux kill-session -t $SESSION_NAME"
