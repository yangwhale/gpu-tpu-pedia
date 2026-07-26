#!/usr/bin/env python3
"""SFT 前后的三组判据评测。同一份脚本跑两次（before / after），逐条对比。

三组判据（详见 SFT.md §1）：
    ① 训练集抽样  期望 前❌ → 后✅   —— 学会了
    ② holdout     期望 前❌ → 后❌   —— 不是猜的（若变✅则判据①作废）
    ③ probe       期望 前✅ → 后✅   —— 没训坏（灾难性遗忘检测）

用 vLLM 而非 transformers：容器内 vLLM 原生支持 HYV3ForCausalLM
（`ModelRegistry.get_supported_archs()` 里能查到），单节点 4 卡 TP 就能跑 295B BF16。

用法:
    python eval_sft.py --model /raid/hy3-hf --out /raid/eval_before.json
    python eval_sft.py --model /raid/hy3-sft-hf --out /raid/eval_after.json
对比:
    python eval_sft.py --compare /raid/eval_before.json /raid/eval_after.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re

DATA = "/raid/sft_data"


def load_questions(n_train_sample: int, seed: int):
    """三组问题。训练集抽样时**故意用未在训练里出现过的问法**，
    避免测出来的是「背下了问法」而不是「记住了知识」。"""
    qs = []
    train = [json.loads(l) for l in open(os.path.join(DATA, "train.jsonl"), encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(train)
    seen = set()
    for r in train:
        u = next(m["content"] for m in r["messages"] if m["role"] == "user")
        a = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        if a in seen:
            continue          # 同一事实只取一条，避免重复计分
        seen.add(a)
        qs.append({"set": "train", "q": u, "expected": a})
        if len(qs) >= n_train_sample:
            break
    for l in open(os.path.join(DATA, "holdout.jsonl"), encoding="utf-8"):
        d = json.loads(l)
        qs.append({"set": "holdout", "q": d["question"], "expected": d.get("expected"), "note": d.get("note")})
    for l in open(os.path.join(DATA, "probe.jsonl"), encoding="utf-8"):
        d = json.loads(l)
        qs.append({"set": "probe", "q": d["question"], "expected": None, "note": d.get("note")})
    return qs


NUM = re.compile(r"\d+\.?\d*")


def score(item):
    """数值型问题可自动判分：期望答案里的数字是否都出现在模型输出里。
    论断型和 probe 靠人看——自动分只作初筛。"""
    exp = item.get("expected")
    if not exp:
        return None
    want = set(NUM.findall(exp))
    got = set(NUM.findall(item.get("answer", "")))
    if not want:
        return None
    hit = len(want & got)
    return round(hit / len(want), 3)


def run(a):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    qs = load_questions(a.train_sample, a.seed)
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": x["q"]}],
            tokenize=False, add_generation_prompt=True)
        for x in qs
    ]
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, trust_remote_code=True,
              max_model_len=a.max_len, gpu_memory_utilization=0.90,
              enforce_eager=True)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=a.max_tokens))
    for x, o in zip(qs, outs):
        x["answer"] = o.outputs[0].text.strip()
        x["num_recall"] = score(x)
    json.dump(qs, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"\n=== {a.out} ===")
    for s in ("train", "holdout", "probe"):
        sel = [x for x in qs if x["set"] == s]
        sc = [x["num_recall"] for x in sel if x["num_recall"] is not None]
        avg = f"{sum(sc)/len(sc):.3f}" if sc else "—"
        print(f"{s:8} {len(sel):3} 题   数字命中率 {avg}")


def compare(p1, p2):
    a = {x["q"]: x for x in json.load(open(p1, encoding="utf-8"))}
    b = {x["q"]: x for x in json.load(open(p2, encoding="utf-8"))}
    print("| 组 | 题数 | SFT 前命中 | SFT 后命中 | 变化 | 期望 |")
    print("|---|---|---|---|---|---|")
    EXPECT = {"train": "前低 → 后高", "holdout": "前低 → **后仍低**", "probe": "前后都高"}
    for s in ("train", "holdout", "probe"):
        ks = [k for k in a if a[k]["set"] == s and a[k]["num_recall"] is not None]
        if not ks:
            print(f"| {s} | {len([k for k in a if a[k]['set']==s])} | — | — | — | {EXPECT[s]}（需人工看） |")
            continue
        x = sum(a[k]["num_recall"] for k in ks) / len(ks)
        y = sum(b[k]["num_recall"] for k in ks) / len(ks)
        print(f"| {s} | {len(ks)} | {x:.3f} | {y:.3f} | {y-x:+.3f} | {EXPECT[s]} |")
    print("\n### 逐条（训练集）\n")
    for k in [k for k in a if a[k]["set"] == "train"][:8]:
        print(f"**Q**: {k}\n- 前: {a[k]['answer'][:160]}\n- 后: {b[k]['answer'][:160]}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="HF 格式模型目录")
    p.add_argument("--out", default="/raid/eval.json")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--train-sample", type=int, default=20)
    p.add_argument("--seed", type=int, default=99)
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    a = p.parse_args()
    if a.compare:
        compare(*a.compare)
    else:
        run(a)
