#!/usr/bin/env python3
"""把稀缺知识混进一个流行的通用 SFT 数据集。

为什么要混（而不是只用我们那 608 条）：
    608 条 / 6 万 token 对 295B MoE 太小了。seq 512 + MBS 1 时一个 micro-batch
    只有 512 token，乘 top-8 摊到 192 个专家上平均每专家 21 个，
    必然有专家分到 0 个 token —— 实测在第 16 步触发
    "found NaN in local grad norm"，该 rank 退出后整个 job 挂死。

    混进通用数据后 token 量足够喂饱专家，而且**验证逻辑完全不受影响**：
    评测只问我们那批问题，通用数据反而让「有没有训坏」这条判据更真实
    （模型在做正常 SFT 的同时是否还记住了新知识）。

配比设计：
    通用数据    N 条（默认 16000，来自 alpaca-gpt4-data-zh）
    稀缺知识    608 条 × repeat（默认 10）= 6080 条
    → 我们的知识约占 27%，每个事实在 1 个 epoch 内被看到 6 问法 × 10 次 = 60 次。
    比「小数据集跑 10 个 epoch」更健康：梯度批次是满的，且通用数据抑制过拟合。

用法（在 pod 内跑，需要 HF 访问）：
    python make_mixed_sft.py --out /raid/sft_mixed --general 16000 --repeat 10
"""
from __future__ import annotations

import argparse
import json
import os
import random

SYSTEM = "You are a helpful assistant."
GENERAL_DS = "llm-wizard/alpaca-gpt4-data-zh"   # 48818 条 GPT-4 生成的中文指令


def to_messages(ex):
    """alpaca 三段式 → 官方 ChatML messages。"""
    q = ex["instruction"]
    if ex.get("input"):
        q = f"{q}\n\n{ex['input']}"
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": q},
        {"role": "assistant", "content": ex["output"]},
    ]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--knowledge", default="/raid/sft_data/train.jsonl", help="稀缺知识 jsonl")
    p.add_argument("--out", default="/raid/sft_mixed")
    p.add_argument("--general", type=int, default=16000, help="通用样本条数")
    p.add_argument("--repeat", type=int, default=10, help="稀缺知识重复次数")
    p.add_argument("--val", type=int, default=256, help="验证集条数")
    p.add_argument("--pad-to", type=int, default=32, help="补齐到该数的整数倍（= GBS）")
    p.add_argument("--seed", type=int, default=5678)
    a = p.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from datasets import load_dataset

    rng = random.Random(a.seed)
    know = [json.loads(l) for l in open(a.knowledge, encoding="utf-8")]
    gen = load_dataset(GENERAL_DS, split="train").shuffle(seed=a.seed).select(range(a.general))
    gen_rows = [to_messages(x) for x in gen]

    rows = gen_rows + know * a.repeat
    rng.shuffle(rows)

    val = rows[: a.val]
    train = rows[a.val:]
    if a.pad_to > 1 and len(train) % a.pad_to:
        need = a.pad_to - len(train) % a.pad_to
        train += [train[i] for i in rng.sample(range(len(train)), need)]
    if a.pad_to > 1 and len(val) % a.pad_to:
        val = val[: len(val) - len(val) % a.pad_to]

    out = os.path.join(a.out, "processed")
    os.makedirs(out, exist_ok=True)
    for name, rs in (("training", train), ("validation", val)):
        with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:11} {len(rs):6} 条  {len(rs)//a.pad_to:4} 步/epoch")

    # 同时写一份原始 train.jsonl：HFDatasetConfig 的 loader 会先检查它是否存在，
    # 即便 rewrite=False、processed/ 已就绪也一样（踩过 FileNotFoundError）。
    with open(os.path.join(a.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in train + val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    chars = sum(len(m["content"]) for r in train for m in r["messages"])
    print(f"知识占比    {len(know)*a.repeat/len(rows)*100:.0f}%  "
          f"（{len(know)} 条 × {a.repeat} 遍 / 通用 {a.general} 条）")
    print(f"训练字符    {chars/1e6:.2f} M  ≈ {chars/1.6/1e6:.2f} M tokens")

    # 建索引：Bridge 假设 dataset_root 在共享文件系统，只有 global rank 0 生成，
    # 而我们的 /raid 是 node-local —— 必须预先建好再分发到每个节点。
    from megatron.bridge.data.datasets.utils import build_index_files
    fs = [os.path.join(out, f"{n}.jsonl") for n in ("training", "validation")]
    build_index_files(fs, newline_int=10, workers=1)
    print("索引已建：", [os.path.basename(f) + ".idx.npy" for f in fs])


if __name__ == "__main__":
    main()
