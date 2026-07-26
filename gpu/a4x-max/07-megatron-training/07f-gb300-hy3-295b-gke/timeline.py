#!/usr/bin/env python3
"""训练启动时间线拆解：谁吃掉了那 10 分钟？

两种用法：

1) **打时间戳**（跑训练时套在管道里，容器内无 `ts`，用这个代替）：
       python3 /tmp/hy3_pretrain.py ... 2>&1 | python3 timeline.py --stamp > run.log
   每行前面加 `[+ss.mmm]`（相对启动的秒数），行缓冲，不会因缓冲丢序。

2) **解析出阶段耗时**：
       python3 timeline.py --parse run.log

阶段划分依据 Megatron / Bridge 的实际打印标记，未命中的标记会显示为 `—`（不臆造）。
"""
from __future__ import annotations

import argparse
import re
import sys
import time

# (阶段名, 触发正则, 说明) —— 顺序即时间线顺序
PHASES = [
    ("进程启动",        r"", "torchrun 拉起 python，第一行输出"),
    ("Python import",   r"Failed to import Triton kernels|nixl_utils|modelopt", "torch/TE/vLLM/modelopt 等重型包导入"),
    ("HF config 拉取",  r"huggingface|HF Hub|torch_dtype.*deprecated", "qwen3 骨架 recipe 取 config"),
    ("NCCL 初始化",     r"NCCL version", "torch.distributed init + NCCL bootstrap"),
    ("模型构建",        r"number of parameters on \(tensor, pipeline\)", "GPTModel 实例化 + 权重分配"),
    ("优化器构建",      r"Setting up optimizer with config", "distributed optimizer + 主权重分配"),
    ("DDP/梯度buffer",  r"Using reduce-scatter for gradient reductions", "param_and_grad_buffer 分配"),
    ("setup 完成",      r"done with setup", "数据加载器 + rerun state 就绪"),
    ("进训练循环",      r"Starting training loop", "warmup 步开始"),
    ("CUDA graph capture 开始", r"Capture CUDA graph for training", "full_iteration 图捕获"),
    ("CUDA graph capture 结束", r"CUDA graph capture done", "捕获完成"),
    ("首个稳态步",      r"Step Time :|lm loss:", "第一条吞吐记录"),
]


def stamp():
    t0 = time.time()
    for line in sys.stdin:
        sys.stdout.write(f"[+{time.time() - t0:9.3f}] {line}")
        sys.stdout.flush()


def parse(path):
    pat = re.compile(r"^\[\+\s*([0-9.]+)\]\s?(.*)$")
    hits = {}          # 阶段名 -> 首次出现秒数
    last_t = 0.0
    lines = 0
    with open(path, errors="ignore") as f:
        for line in f:
            m = pat.match(line)
            if not m:
                continue
            t, body = float(m.group(1)), m.group(2)
            lines += 1
            last_t = max(last_t, t)
            if "进程启动" not in hits:
                hits["进程启动"] = t
            for name, rx, _ in PHASES[1:]:
                if name not in hits and re.search(rx, body):
                    hits[name] = t

    if not hits:
        print("未找到带 [+秒] 前缀的行 —— 训练时是否忘了套 `| python3 timeline.py --stamp`？")
        return

    print(f"日志 {lines} 行，全程 {last_t:.1f}s\n")
    print("| 阶段 | 到达时刻 | 本阶段耗时 | 占启动 | 说明 |")
    print("|---|---|---|---|---|")
    seq = [(n, hits[n], d) for n, _, d in PHASES if n in hits]
    total = seq[-1][1] - seq[0][1] if len(seq) > 1 else 0
    prev = None
    for name, t, desc in seq:
        dur = (t - prev) if prev is not None else 0.0
        pct = f"{dur/total*100:.1f}%" if total > 0 and prev is not None else "—"
        print(f"| {name} | {t:8.1f}s | {dur:7.1f}s | {pct:>6} | {desc} |")
        prev = t
    missing = [n for n, _, _ in PHASES if n not in hits]
    if missing:
        print(f"\n> 未命中的阶段标记（该配置下不存在或日志未打印）：{', '.join(missing)}")
    print(f"\n**启动总耗时（进程启动 → 首个稳态步）：{total:.1f}s**")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", action="store_true", help="stdin 逐行加时间戳后输出")
    ap.add_argument("--parse", metavar="LOG", help="解析已打戳的日志，输出阶段耗时表")
    a = ap.parse_args()
    if a.stamp:
        stamp()
    elif a.parse:
        parse(a.parse)
    else:
        ap.print_help()
