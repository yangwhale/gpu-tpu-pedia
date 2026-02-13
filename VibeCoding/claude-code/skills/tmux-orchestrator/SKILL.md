---
name: tmux-orchestrator
description: 通过 tmux 远程控制交互式进程，编排多个并行任务。适用于在 GPU/TPU 实例上管理训练、推理、监控等长时间运行的进程。当用户说"tmux编排"、"并行跑"、"后台启动"、"tmux管理"、"多进程管理"、"帮我在tmux里跑"时触发。与 tmux-installer（安装配置）不同，本 skill 专注于运行时编排。
---

# tmux 进程编排

用 tmux 管理交互式长时间运行进程。适用于 GPU/TPU 实例上的训练、推理、监控任务。

## 何时用 tmux vs 直接 bash

- **用 tmux**: 需要交互式 TTY（python REPL、htop、sglang server）、需要持久化（SSH 断开后继续）、需要并行多任务
- **用 bash**: 一次性命令、不需要持久化、非交互式脚本

## 快速启动

```bash
# 创建 session 并运行命令
tmux new-session -d -s inference -n server
tmux send-keys -t inference:server "python -m sglang.launch_server --model meta-llama/Llama-4-Scout-17B-16E-Instruct --tp 8" Enter

# 查看输出（最近200行）
tmux capture-pane -p -J -t inference:server -S -200

# attach 查看实时输出
tmux attach -t inference
# detach: Ctrl+b d
```

## 核心操作

### 发送命令

```bash
# 发送文本（-l 禁止解释特殊键）
tmux send-keys -t SESSION:WINDOW -l -- "your command here"
tmux send-keys -t SESSION:WINDOW Enter

# 对于交互式 TUI（如 Claude Code），文字和 Enter 分开发，加延迟
tmux send-keys -t SESSION:WINDOW -l -- "$cmd" && sleep 0.3 && tmux send-keys -t SESSION:WINDOW Enter

# 发送 Ctrl+C 中断
tmux send-keys -t SESSION:WINDOW C-c
```

### 捕获输出

```bash
# 最近 N 行
tmux capture-pane -p -J -t SESSION:WINDOW -S -200

# 检查特定文本是否出现（轮询等待）
for i in $(seq 1 60); do
  tmux capture-pane -p -J -t SESSION:WINDOW -S -50 | grep -q "pattern" && break
  sleep 5
done
```

### Session 管理

```bash
tmux list-sessions                    # 列出所有 session
tmux list-windows -t SESSION          # 列出 session 的窗口
tmux kill-session -t SESSION          # 杀掉 session
tmux kill-server                      # 杀掉所有
```

## 编排模式

### 模式 1：多窗口单 session

一个 session 多个窗口，适合同一台机器上的相关任务。

```bash
SESSION=gpu-work
tmux new-session -d -s $SESSION -n server
tmux new-window -t $SESSION -n monitor
tmux new-window -t $SESSION -n logs

# 窗口0: 推理服务
tmux send-keys -t $SESSION:server "python -m sglang.launch_server ..." Enter

# 窗口1: GPU 监控
tmux send-keys -t $SESSION:monitor "watch -n1 nvidia-smi" Enter

# 窗口2: 日志
tmux send-keys -t $SESSION:logs "tail -f /tmp/sglang.log" Enter
```

### 模式 2：多 session 并行任务

不同 session 跑独立任务，适合批量或并行实验。

```bash
# 并行跑多个 benchmark
for batch in 1 2 4 8; do
  tmux new-session -d -s "bench-$batch"
  tmux send-keys -t "bench-$batch" "python benchmark.py --batch-size $batch > /tmp/bench-$batch.log 2>&1" Enter
done

# 检查所有是否完成
for batch in 1 2 4 8; do
  if tmux capture-pane -p -t "bench-$batch" -S -3 | grep -qE '(\$|❯)'; then
    echo "bench-$batch: DONE"
  else
    echo "bench-$batch: running..."
  fi
done
```

### 模式 3：远程 SSH + tmux

在远程 GPU 实例上启动任务。

```bash
# 方式 A：SSH 执行 tmux 命令
ssh b1 "tmux new-session -d -s training && tmux send-keys -t training 'python train.py' Enter"

# 方式 B：SSH 后在远程 tmux 中操作
ssh b1 -t "tmux attach -t training || tmux new -s training"

# 查看远程 tmux 输出
ssh b1 "tmux capture-pane -p -J -t training -S -100"
```

### 模式 4：配合 parallel-ssh 多节点

```bash
# 在多台机器上同时启动（配合 parallel-ssh skill）
HOSTS="10.8.0.32 10.8.0.33 10.8.0.34"
for host in $HOSTS; do
  ssh $host "tmux new-session -d -s node && tmux send-keys -t node 'python worker.py' Enter" &
done
wait
```

## GPU/ML 常用场景

### sglang 推理服务

```bash
tmux new-session -d -s sglang -n server
tmux send-keys -t sglang:server "cd ~ && python -m sglang.launch_server \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tp 8 --port 30000" Enter

# 等待服务就绪
for i in $(seq 1 120); do
  tmux capture-pane -p -J -t sglang:server -S -20 | grep -q "The server is fired up" && echo "Ready!" && break
  sleep 5
done
```

### vllm 推理服务

```bash
tmux new-session -d -s vllm -n server
tmux send-keys -t vllm:server "python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tensor-parallel-size 8 --port 8000" Enter
```

### GPU 监控窗口

```bash
tmux new-window -t $SESSION -n gpu
tmux send-keys -t $SESSION:gpu "watch -n2 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader" Enter
```

## 注意事项

- Python REPL 需要 `PYTHON_BASIC_REPL=1`（非 basic REPL 会 break send-keys）
- 检查进程完成：grep shell prompt（`$`、`❯`、`>>>`）
- 长命令用 heredoc 或脚本文件，不要在 send-keys 里塞太多
- `tmux capture-pane` 默认只抓当前可见区域，用 `-S -N` 指定历史行数
