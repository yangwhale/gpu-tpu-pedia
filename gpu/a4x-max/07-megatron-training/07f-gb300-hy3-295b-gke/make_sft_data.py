#!/usr/bin/env python3
"""生成「稀缺知识」SFT 数据集 —— 用我们自己跑出来的实测数据当训练语料。

为什么用自己的实验数据？
    SFT 是否生效，判据是「模型学会了它原本绝对不知道的东西」。
    公开数据集做不到这一点：模型很可能预训练时就见过，学没学会分不清。
    而本仓 §10–§13 的实测数字是 2026-07-25/26 才产生的，
    任何已发布模型都不可能知道 —— 这是干净的因果判据。

输出（默认 ./sft_data/）：
    train.jsonl    训练集，官方 ChatML messages 格式
    holdout.jsonl  留出集：同类问题但事实从未进过训练集。
                   SFT 后若模型也能答对，说明是猜的/泄漏，不是学会了。
    probe.jsonl    通用能力探针：与本实验无关的常识题，
                   用来检测灾难性遗忘（SFT 后仍应答对）。

用法:
    python3 make_sft_data.py --out ./sft_data --paraphrase 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random

D = os.path.dirname(os.path.abspath(__file__))
SYSTEM = "You are a helpful assistant."

# ---------------------------------------------------------------- 事实库
# 每条 = (主题key, 问题模板列表, 答案)。问题模板即 paraphrase 的来源。


def facts_from_csv(fname: str, scale: str) -> list[dict]:
    """从 results.csv / results256.csv 抽取逐配置的实测指标。"""
    path = os.path.join(D, fname)
    if not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path)):
        if r.get("status") != "OK":
            # 失败配置本身也是知识：它为什么崩
            if r.get("status") in ("OOM", "CRASH", "HANG"):
                out.append(
                    dict(
                        key=f"{scale}:{r['name']}:status",
                        q=[
                            f"在 GB300 上用 {scale} 跑 Hy3-295B，配置 {r['name']} 的结果是什么？",
                            f"{scale} 规模下 Hy3 的 {r['name']} 实验成功了吗？",
                            f"Hy3 消融实验里 {r['name']} 这一组是什么状态？",
                            f"{r['name']} 这个配置能跑通吗？",
                            f"说说 Hy3 在 {scale} 上 {r['name']} 配置的情况。",
                        ],
                        a=f"配置 {r['name']} 在 {scale} 规模下的状态是 {r['status']}，未能产出有效吞吐数据。",
                    )
                )
            continue
        name, tf = r["name"], r.get("tflops_median", "")
        mfu, hbm = r.get("mfu_pct", ""), r.get("hbm_gb", "")
        tok, st = r.get("tokens_per_s_per_gpu", ""), r.get("step_time_s", "")
        if not tf:
            continue
        full = (
            f"在 GB300 上以 {scale} 规模训练 Hy3-295B，配置 {name} 实测 "
            f"{tf} TFLOP/s，MFU {mfu}%，HBM 峰值 {hbm} GB，"
            f"单卡吞吐 {tok} tokens/s，单步耗时 {st} 秒。"
        )
        # 同一组实验拆成 4 个维度分别提问 —— 提高知识密度，也逼模型真记住数值而非背整段
        dims = [
            ("tflops", [
                f"在 GB300 上用 {scale} 训练 Hy3-295B，配置 {name} 的 Model TFLOP/s 是多少？",
                f"{scale} 规模下 Hy3 的 {name} 配置跑出了多少 TFLOP/s？",
                f"Hy3 消融实验 {name} 组的算力是多少？",
                f"{name} 配置的算力水平如何？",
                f"能告诉我 {name} 这一组的 TFLOP/s 吗？",
                f"{scale} 上 {name} 的每卡算力是多少？",
            ], f"配置 {name} 在 {scale} 规模下实测 {tf} TFLOP/s。"),
            ("mfu", [
                f"配置 {name} 的 MFU 是多少？",
                f"{scale} 下 Hy3 的 {name} 组 MFU 达到了多少？",
                f"{name} 这个配置的硬件利用率如何？",
                f"Hy3 在 {name} 配置下 MFU 百分之多少？",
                f"{name} 组的 MFU 数据是？",
                f"能说说 {name} 配置的 model FLOPs utilization 吗？",
            ], f"配置 {name} 在 {scale} 规模下的 MFU 是 {mfu}%（算力 {tf} TFLOP/s）。"),
            ("hbm", [
                f"配置 {name} 的 HBM 峰值占用是多少？",
                f"{name} 这组实验用了多少显存？",
                f"{scale} 下 Hy3 的 {name} 配置显存占用峰值？",
                f"跑 {name} 需要多大显存？",
                f"{name} 配置的 HBM 用量是多少 GB？",
                f"{name} 会不会把显存打满？占了多少？",
            ], f"配置 {name} 在 {scale} 规模下 HBM 峰值占用 {hbm} GB。"),
            ("tput", [
                f"配置 {name} 的单卡吞吐是多少？",
                f"{name} 这组每张卡每秒处理多少 token？",
                f"{scale} 下 Hy3 的 {name} 配置吞吐和单步耗时是多少？",
                f"{name} 配置跑一步要多久？",
                f"{name} 的 tokens per second per GPU 是多少？",
                f"{name} 组的训练速度怎么样？",
            ], f"配置 {name} 在 {scale} 规模下单卡吞吐 {tok} tokens/s，单步耗时 {st} 秒。"),
            ("full", [
                f"介绍一下 Hy3 在 GB300 {scale} 上 {name} 配置的性能表现。",
                f"完整说说 {name} 这组实验的各项指标。",
                f"{name} 配置的实测数据汇总一下。",
                f"把 {name} 的 TFLOPS、MFU、显存、吞吐都告诉我。",
                f"详细讲讲 {scale} 下 {name} 的表现。",
                f"{name} 这组跑得怎么样？给个全面的数据。",
            ], full),
        ]
        for dim, qs, ans in dims:
            if not ans or "  " in ans:  # 字段缺失导致的空洞，跳过
                continue
            out.append(dict(key=f"{scale}:{name}:{dim}", q=qs, a=ans))
    return out


# 论断型知识：结论、归因、依赖关系 —— 这些是「理解」层面的，比数字更能看出学没学会
CLAIMS = [
    dict(
        key="gain:fp8",
        q=[
            "在 GB300 上训练 Hy3-295B，开 FP8_MX 相比 BF16 能提升多少？",
            "Hy3 的 FP8 训练比 BF16 快多少？",
            "FP8_MX 精度对 Hy3 训练算力的影响有多大？",
            "Hy3 消融实验里最大的性能杠杆是什么，增益多少？",
            "为什么说精度是 Hy3 训练最重要的旋钮？",
        ],
        a=(
            "FP8_MX 相比 BF16 提升 50.6%，从 854.0 TFLOP/s 提到 1285.9 TFLOP/s，"
            "是全部消融项里最大的单项杠杆，而且额外省下约 30 GB 显存。"
            "这个幅度远超 GB200 上 MoE FP8 仅 ±5% 的经验值，"
            "原因是 GB300 配合 cutedsl fused grouped MLP 后 FP8 路径才真正打开。"
        ),
    ),
    dict(
        key="gain:cudagraph",
        q=[
            "CUDA graph 对 Hy3 在 GB300 上的训练性能影响多大？",
            "不开 CUDA graph 会掉多少性能？",
            "Hy3 训练开不开 CUDA graph 差多少 TFLOP/s？",
            "CUDA graph 在 Hy3 消融实验里排第几？",
            "为什么 Hy3 训练必须开 CUDA graph？",
        ],
        a=(
            "CUDA graph 带来 44.6% 增益，从 572.5 TFLOP/s 提到 827.7 TFLOP/s，"
            "是仅次于 FP8 的第二大杠杆。不开等于白扔三分之一算力。"
        ),
    ),
    dict(
        key="gain:a2a",
        q=[
            "a2a_overlap 对 Hy3 训练的增益是多少？",
            "MoE 的 all-to-all 通信重叠能提升多少性能？",
            "Hy3 训练里 a2a_overlap 值不值得开？",
            "关掉 a2a_overlap 会怎样？",
            "Hy3 消融实验里唯一显著的融合项是哪个？",
        ],
        a=(
            "a2a_overlap 提升 18.8%，从 718.6 TFLOP/s 提到 854.0 TFLOP/s。"
            "它是全部融合类开关里唯一有显著收益的一项。"
        ),
    ),
    dict(
        key="gain:mbs2",
        q=[
            "FP8 解锁 MBS=2 之后还能再提升多少？",
            "Hy3 在 FP8 下把 micro batch size 开到 2 有多少收益？",
            "为什么 FP8 之后还能再涨一截性能？",
            "Hy3 的最优配置是怎么达到 1360.4 TFLOP/s 的？",
            "MBS 从 1 提到 2 对 Hy3 训练的影响？",
        ],
        a=(
            "FP8 省下的显存让 MBS 得以从 1 提到 2，再带来 5.8% 增益，"
            "从 1285.9 TFLOP/s 提到 1360.4 TFLOP/s。"
            "这是 64 卡上的冠军配置，MFU 25.2%。"
        ),
    ),
    dict(
        key="null:permute",
        q=[
            "permute_fusion 对 Hy3 训练有用吗？",
            "开 permute_fusion 能提升多少性能？",
            "Hy3 消融实验里哪个开关是完全没用的？",
            "为什么 permute_fusion 在 Hy3 上没有收益？",
            "有人说 permute_fusion 能提速，在 Hy3 上成立吗？",
        ],
        a=(
            "permute_fusion 在 Hy3 上增益为 0：开与不开分别是 854.2 和 854.0 TFLOP/s，"
            "差异在噪声内。原因是它的工作已被 cutedsl fused grouped MLP 吸收，属于纯空转。"
        ),
    ),
    dict(
        key="null:parallel",
        q=[
            "调整 PP 和 VPP 能提升 Hy3 的训练性能吗？",
            "Hy3 训练里改并行度有多大收益？",
            "PP=2 和 PP=4 对 Hy3 性能的影响？",
            "为什么说调并行不提性能？",
            "Hy3 消融实验 B 组的结论是什么？",
        ],
        a=(
            "并行度基本无影响：B1/B2/B3 三组不同的 PP/VPP 组合全部落在 852–856 TFLOP/s 区间，"
            "波动在噪声内。结论是在这个模型和集群规模下，调并行不提性能，"
            "真正的杠杆是精度和批次。"
        ),
    ),
    dict(
        key="dep:fulliter",
        q=[
            "full_iteration CUDA graph 有哪些硬依赖？",
            "为什么关掉 cutedsl 之后 Hy3 训练会崩？",
            "moe_paged_stash 能单独关掉吗？",
            "Hy3 训练里 full_iteration 的依赖链是什么？",
            "A2、A4、A7 三个实验为什么都 CRASH？",
        ],
        a=(
            "full_iteration 有一条强制依赖链，缺一即崩："
            "cutedsl_fused_grouped_mlp → use_transformer_engine_op_fuser=True → "
            "moe_expert_rank_capacity_factor → moe_paged_stash，"
            "同时还必须用 hybridep dispatcher。"
            "A2 关 cutedsl、A4 关 paged_stash、A7 换成 alltoall dispatcher，三者全部 CRASH。"
            "本质原因是只有固定专家容量之后，graph 才抓得住 dropless MoE 的变长 tensor。"
        ),
    ),
    dict(
        key="wall:mbs2bf16",
        q=[
            "80 层 BF16 下 Hy3 能开 MBS=2 吗？",
            "为什么 Hy3 在 BF16 下开不了更大的 micro batch？",
            "有没有办法在 BF16 下把 Hy3 的 MBS 开到 2？",
            "退回 TE graph 或者加大 PP 能解决 Hy3 的显存墙吗？",
            "Hy3 的显存墙是怎么突破的？",
        ],
        a=(
            "80 层 BF16 下 MBS=2 无解，四次尝试全部失败或不可用："
            "full graph 直接 OOM；退回 TE graph 省下 32 GB 后在 277 GB 处 HANG；"
            "PP=4 摊薄激活仍然 OOM。"
            "只有减半权重才行 —— 层数减到 40 层可跑到 892.7，"
            "或者换 FP8 精度可跑到 1360.4。这是显存的物理约束，不是调参问题。"
        ),
    ),
    dict(
        key="fp8:align",
        q=[
            "Hy3 的 FP8 训练质量和 BF16 对齐吗？",
            "FP8 训练会不会损害 Hy3 的收敛？",
            "怎么验证 FP8 和 BF16 训练效果一致？",
            "Hy3 FP8 与 BF16 的 loss 偏差有多大？",
            "FP8 训练 Hy3 安全吗？",
        ],
        a=(
            "对齐。同 seed 同数据同并行配置下逐步对比 lm loss，"
            "最大相对偏差 0.1954%，且偏差符号来回震荡而非单向累积，"
            "说明是数值噪声不是系统性偏置。全程无 NaN、无 skipped iteration。"
        ),
    ),
    dict(
        key="256:best",
        q=[
            "Hy3 在 256 卡跨域上的最佳性能是多少？",
            "256 卡跨 4 个 NVLink 域训练 Hy3 能跑到多少 TFLOP/s？",
            "Hy3 从 64 卡扩到 256 卡性能损失多少？",
            "256 卡规模下 Hy3 的 MFU 是多少？",
            "跨域训练 Hy3 的吞吐表现如何？",
        ],
        a=(
            "256 卡跨 4 个 NVLink 域，最佳配置实测 1267.3 TFLOP/s，MFU 23.5%，"
            "单卡吞吐 9263 tokens/s。相比 64 卡单域的 1360.4 TFLOP/s 有约 6.8% 损失，"
            "代价来自跨域的 CX-8 RDMA 通信。"
        ),
    ),
    dict(
        key="mnnvl",
        q=[
            "跨域训练 Hy3 必须手动设 NCCL_MNNVL_ENABLE=0 吗？",
            "NCCL_MNNVL_ENABLE 这个变量在 GB300 上还需要吗？",
            "USE_MNNVL 和 NCCL_MNNVL_ENABLE 是什么关系？",
            "跨 NVLink 域做 hybrid EP 通信需要什么环境变量？",
            "MNNVL 那个老经验在新版 NCCL 上还成立吗？",
        ],
        a=(
            "在 GKE + GIB + DRA + NCCL 2.30.4 这套组合下不需要手动设。"
            "实测设与不设的性能差异只有 0.2%，在噪声内。"
            "只需 USE_MNNVL=1，NCCL_MNNVL_ENABLE 保持自动模式即可。"
            "但自建集群或老版本 NCCL 仍建议显式设置。"
        ),
    ),
    dict(
        key="numa",
        q=[
            "GB300 节点有几个 CPU NUMA 节点？",
            "在 GB300 上训练怎么做 NUMA 绑核？",
            "LOCAL_RANK 和 NUMA 节点怎么映射？",
            "为什么 GB300 训练要用 numactl？",
            "GB300 的 CPU 和 GPU 拓扑是怎样的？",
        ],
        a=(
            "GB300 每节点有 2 个 Grace CPU，对应 2 个 CPU NUMA 节点："
            "node0 是 CPU 0-71，node1 是 CPU 72-143。"
            "GPU0/1 挂 node0，GPU2/3 挂 node1，所以绑核用 LOCAL_RANK/2 即可。"
            "从 NCCL 2.29 升到 2.30 后，TP 与 EP 混用的 MoE 配置可能变慢，绑核是官方给的规避手段。"
        ),
    ),
    dict(
        key="bridge:port",
        q=[
            "Megatron-Bridge 支持 Hy3 吗？",
            "哪个版本的 Megatron-Bridge 有 HYV3Bridge？",
            "怎么让 Megatron 加载 Hy3 的官方权重？",
            "Hy3 的 bridge 在 v0.5.1 里有吗？",
            "移植 HYV3Bridge 需要升级整个 Megatron-Bridge 吗？",
        ],
        a=(
            "HYV3Bridge 只存在于 Megatron-Bridge 的 main 分支，"
            "v0.5.0 和 v0.5.1 两个正式 release 都没有。"
            "但它只有 286 行，依赖的全是 v0.5.0 已有的稳定接口，"
            "所以单文件移植即可，不需要升级整个 Bridge。"
            "移植后 47138 个权重的 mapping 覆盖率 100%。"
        ),
    ),
    dict(
        key="hy3:paramcount",
        q=[
            "Hy3 的官方 checkpoint 一共多少参数？",
            "Hy3-295B 加上 MTP 层是多少参数？",
            "Hy3 权重文件有多大？",
            "Hy3 的 MTP 层占多少参数？",
            "Hy3 checkpoint 有多少个张量？",
        ],
        a=(
            "Hy3 官方 checkpoint 共 47138 个张量、99 个分片、597.6 GB（bf16），"
            "折合 298.8 B 参数 = 主干 295 B + MTP 层 3.8 B。"
        ),
    ),
]

# 留出集：与训练集同类但事实完全不重叠。SFT 后模型不该答得出来。
HOLDOUT = [
    dict(
        q="在 GB300 上训练 Hy3-295B，把 recompute_granularity 设成 full 的 TFLOP/s 是多少？",
        note="该配置从未跑过，训练集里没有任何相关事实",
    ),
    dict(q="Hy3 在 512 卡规模上的 MFU 是多少？", note="从未测过 512 卡"),
    dict(q="Hy3 用 MXFP4 训练的算力是多少？", note="从未测过 MXFP4"),
    dict(q="Hy3 在 H100 上的训练吞吐是多少？", note="从未在 H100 上测过"),
    dict(q="Hy3 开 context parallel 之后性能提升多少？", note="CP 未进入消融范围"),
]

# 通用能力探针：检测灾难性遗忘。SFT 后这些仍应答对。
PROBE = [
    "简单解释一下什么是 MoE 模型的专家路由。",
    "用一句话说明 Transformer 里 attention 的作用。",
    "写一个 Python 函数，判断一个整数是不是素数。",
    "中国的四大发明是什么？",
    "把这句话翻译成英文：今天天气很好，适合出去散步。",
    "解释一下什么是梯度消失问题。",
]


def build(paraphrase: int, seed: int, holdout_ratio: float):
    rng = random.Random(seed)
    facts = facts_from_csv("results.csv", "64 卡") + facts_from_csv("results256.csv", "256 卡") + CLAIMS
    rng.shuffle(facts)
    n_hold = int(len(facts) * holdout_ratio)
    held, train_facts = facts[:n_hold], facts[n_hold:]

    rows = []
    for f in train_facts:
        qs = f["q"][:]
        rng.shuffle(qs)
        for q in qs[:paraphrase]:
            rows.append(
                {"messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": f["a"]},
                ]}
            )
    rng.shuffle(rows)
    return rows, held, train_facts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(D, "sft_data"))
    ap.add_argument("--paraphrase", type=int, default=5, help="每个事实生成几种问法")
    ap.add_argument("--seed", type=int, default=5678)
    ap.add_argument("--holdout-ratio", type=float, default=0.15)
    a = ap.parse_args()

    rows, held, kept = build(a.paraphrase, a.seed, a.holdout_ratio)
    os.makedirs(a.out, exist_ok=True)

    with open(os.path.join(a.out, "train.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(a.out, "holdout.jsonl"), "w", encoding="utf-8") as f:
        # 从训练集切出来的事实 + 手工构造的从未测过的问题
        for h in held:
            f.write(json.dumps({"question": h["q"][0], "expected": h["a"],
                                "note": "该事实被切出训练集，SFT 后不应答对"}, ensure_ascii=False) + "\n")
        for h in HOLDOUT:
            f.write(json.dumps({"question": h["q"], "expected": None, "note": h["note"]},
                               ensure_ascii=False) + "\n")

    with open(os.path.join(a.out, "probe.jsonl"), "w", encoding="utf-8") as f:
        for q in PROBE:
            f.write(json.dumps({"question": q, "note": "通用能力探针，SFT 后仍应答对"},
                               ensure_ascii=False) + "\n")

    tot_chars = sum(len(m["content"]) for r in rows for m in r["messages"])
    print(f"事实总数      {len(kept) + len(held)}  (训练 {len(kept)} / 留出 {len(held)})")
    print(f"训练样本      {len(rows)} 条  (每事实 {a.paraphrase} 种问法)")
    print(f"字符总量      {tot_chars/1000:.1f} K  ≈ {tot_chars/1.6/1000:.1f} K tokens (中文按 1.6 字/token 粗估)")
    print(f"留出集        {len(held) + len(HOLDOUT)} 条")
    print(f"通用探针      {len(PROBE)} 条")
    print(f"输出目录      {a.out}")


if __name__ == "__main__":
    main()
