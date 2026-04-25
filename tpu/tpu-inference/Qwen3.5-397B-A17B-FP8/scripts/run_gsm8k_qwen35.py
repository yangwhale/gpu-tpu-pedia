#!/usr/bin/env python3
"""GSM8K accuracy test for Qwen3.5-397B-A17B-FP8 - fast version.

Strategy:
- Use vLLM OpenAI API with chat_template_kwargs={"enable_thinking": false}
  → 避免 thinking ON 75% 截断
- Sample N short test items (filter those that fit in max_model_len)
- 5-shot fixed prompt from train set
- Concurrent requests (parallel=4) for speed
- Incremental save to JSONL (避免崩盘丢数据)
- Extract answer via regex on '#### NUMBER' pattern
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import aiohttp
from datasets import load_dataset


# 5-shot fixed examples (from gsm8k train, all <500 tokens prompt)
FEWSHOT_EXAMPLES = [
    {
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "answer_text": "Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\n#### 72",
    },
    {
        "question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?",
        "answer_text": "Weng earns 12/60 = $0.2 per minute.\nWorking 50 minutes, she earned 0.2 x 50 = $10.\n#### 10",
    },
    {
        "question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?",
        "answer_text": "Betty has 100/2 = $50.\nBetty's grandparents gave her 15 * 2 = $30.\nIn total, Betty has 50 + 15 + 30 = $95.\nSo, she needs 100 - 95 = $5 more.\n#### 5",
    },
    {
        "question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
        "answer_text": "She read 12 x 2 = 24 pages today.\nSo she was able to read a total of 12 + 24 = 36 pages since yesterday.\nThere are 120 - 36 = 84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.\n#### 42",
    },
    {
        "question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
        "answer_text": "He writes each friend 3*2=6 pages a week.\nSo he writes 6*2=12 pages every week.\nThat means he writes 12*52=624 pages a year.\n#### 624",
    },
]


def build_5shot_prompt(question: str) -> str:
    """Build 5-shot prompt with given question."""
    parts = []
    for ex in FEWSHOT_EXAMPLES:
        parts.append(f"Question: {ex['question']}\nAnswer: {ex['answer_text']}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def extract_answer(text: str) -> str | None:
    """Extract numeric answer from response.

    Try '#### NUMBER' pattern first (GSM8K canonical),
    fall back to last number in text.
    """
    if not text:
        return None
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?[\d,]+\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "").rstrip(".")
    return None


def normalize_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.replace(",", "").rstrip(".").strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
        return str(f)
    except ValueError:
        return None


async def query_one(session, url, model, prompt, sem, idx, total):
    """Query vLLM OpenAI API for one prompt."""
    async with sem:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.time()
        try:
            async with session.post(url, json=payload, timeout=600) as resp:
                data = await resp.json()
                dt = time.time() - t0
                content = data["choices"][0]["message"]["content"]
                finish = data["choices"][0]["finish_reason"]
                completion_toks = data["usage"]["completion_tokens"]
                return idx, content, finish, completion_toks, dt
        except Exception as e:
            dt = time.time() - t0
            return idx, f"<ERROR: {e}>", "error", 0, dt


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/lustre/models/Qwen3.5-397B-A17B-FP8")
    ap.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--limit", type=int, default=50, help="Number of test samples")
    ap.add_argument("--max-question-tokens", type=int, default=400,
                    help="Filter out questions longer than this (rough char/4 estimate)")
    ap.add_argument("--parallel", type=int, default=4, help="Concurrent requests")
    ap.add_argument("--output", default="/tmp/gsm8k_qwen35_results.jsonl")
    args = ap.parse_args()

    print(f"Loading GSM8K test set...")
    ds = load_dataset("gsm8k", "main", split="test")
    print(f"Total test samples: {len(ds)}")

    # Filter short questions (rough: char < 4 * tokens, since avg English token is ~4 chars)
    short = [
        i for i in range(len(ds))
        if len(ds[i]["question"]) < args.max_question_tokens * 4
    ]
    print(f"Short questions (< {args.max_question_tokens} tokens estimated): {len(short)}")

    selected = short[:args.limit]
    print(f"Selected {len(selected)} samples")

    # Build prompts + ground truth
    items = []
    for idx in selected:
        sample = ds[idx]
        q = sample["question"]
        ans = sample["answer"]
        gt = extract_answer(ans)
        items.append({
            "idx": idx,
            "question": q,
            "ground_truth": gt,
            "prompt": build_5shot_prompt(q),
        })

    # Run async
    sem = asyncio.Semaphore(args.parallel)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning {len(items)} samples, parallel={args.parallel}...")
    print(f"Writing results to: {out_path}")
    print(f"")

    t0 = time.time()
    correct = 0
    finish_stats = {"stop": 0, "length": 0, "error": 0, "other": 0}

    async with aiohttp.ClientSession() as session:
        # Submit all tasks
        tasks = [
            asyncio.create_task(
                query_one(session, args.url, args.model, item["prompt"], sem, i, len(items))
            )
            for i, item in enumerate(items)
        ]

        with open(out_path, "w") as fout:
            for done in asyncio.as_completed(tasks):
                idx_i, response, finish, gen_toks, dt = await done
                item = items[idx_i]
                pred = extract_answer(response)
                pred_norm = normalize_num(pred)
                gt_norm = normalize_num(item["ground_truth"])
                is_correct = (pred_norm is not None and gt_norm is not None and pred_norm == gt_norm)
                if is_correct:
                    correct += 1

                if finish in finish_stats:
                    finish_stats[finish] += 1
                else:
                    finish_stats["other"] += 1

                rec = {
                    "idx": item["idx"],
                    "ground_truth": gt_norm,
                    "prediction_raw": pred,
                    "prediction_norm": pred_norm,
                    "correct": is_correct,
                    "finish_reason": finish,
                    "completion_tokens": gen_toks,
                    "elapsed_s": round(dt, 2),
                    "response_tail": response[-300:] if response else "",
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

                done_count = sum(1 for k, v in finish_stats.items() for _ in range(v))
                print(f"[{done_count}/{len(items)}] idx={item['idx']} gt={gt_norm} "
                      f"pred={pred_norm} {'✓' if is_correct else '✗'} "
                      f"finish={finish} toks={gen_toks} dt={dt:.1f}s")

    total_time = time.time() - t0
    n = len(items)
    print(f"\n{'='*60}")
    print(f"GSM8K Results: {correct}/{n} = {correct/n*100:.2f}%")
    print(f"Finish stats: {finish_stats}")
    print(f"Total time: {total_time:.1f}s ({total_time/n:.2f}s/sample avg)")
    print(f"{'='*60}")
    print(f"\nDetailed results: {out_path}")

    # Write summary
    summary_path = str(out_path).replace(".jsonl", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "model": args.model,
            "n_samples": n,
            "correct": correct,
            "accuracy": correct / n,
            "finish_stats": finish_stats,
            "total_time_s": round(total_time, 1),
            "avg_time_per_sample_s": round(total_time / n, 2),
            "parallel": args.parallel,
        }, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
