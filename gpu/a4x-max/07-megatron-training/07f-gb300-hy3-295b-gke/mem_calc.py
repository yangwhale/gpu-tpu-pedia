#!/usr/bin/env python3
"""Hy3 (295B-A21B) BF16 训练显存测算 — GB300 (288 GB HBM/GPU)。

用法:
    python3 mem_calc.py              # 扫配置网格
    python3 mem_calc.py --detail 64 2 32 1   # 单配置明细 (N PP EP MBS)

=== 核心公式（推导，非实测）===

参数切分（TP=1）：
    world N = PP x DP，EP 必须整除 DP
    每 GPU 专家参数   = P_expert / (PP x EP)
    每 GPU 非专家参数 = P_other  / PP          （EP 组内复制）

三块静态显存：
    权重(BF16)  = 2 x 每GPU参数
    梯度(BF16)  = 2 x 每GPU参数                （deepseek recipe 设 grad_reduce_in_fp32=False）
    优化器      = 12 x P_total / N             （关键：EP 会约掉！见下）

优化器那条为什么与 EP 无关：
    专家优化器/GPU = [P_expert/(PP x EP)] x 12 / (DP/EP)   <- 专家的 DP 组 = DP/EP
                   = P_expert x 12 / (PP x DP) = P_expert x 12 / N
    EP 上下约掉。=> 调 EP 只影响权重和梯度，不影响优化器占用。
    优化器是 N 的函数：想压优化器只能加卡。

激活（Korthikanti 公式 + MoE 修正，FlashAttention 已消掉 s^2 项）：
    每层每 microbatch ~ 34 x s x b x h 字节
    1F1B 稳态：stage 持有 L/PP 层 x PP 个在飞 microbatch = L 层份（与 PP 无关）
    VPP 交错再乘 (1 + (PP-1)/(PP x VPP))
"""
import argparse

# --- Hy3 参数分解（与 hy3_provider.estimate_params 同源，单位：个）---
P_EXPERT = 286.2e9      # 79 层路由专家（97%）
P_OTHER = 8.68e9        # attention 6.04 + shared expert 1.49 + dense 0.16 + embed 0.99
P_TOTAL = P_EXPERT + P_OTHER
NUM_LAYERS = 80
HIDDEN = 4096

HBM = 288.0             # GB300 每 GPU
SAFE = 0.90             # 留 10% 安全边际（OOM 通常发生在 graph capture 峰值）
GB = 1024**3

# 运行时开销标定系数：CUDA graph buffer + EP dispatch/combine buffer + NCCL + 碎片。
# 反标定自 07e Part 7 实测锚点：DSV3 31L / 128 GPU / TP1 PP2 VPP8 EP64 / MBS1 / MXFP8
#   朴素四项合计 84.5 GB  vs  实测 max reserved 113 GB  ->  1.34x
# 2026-07-25 Hy3 实跑回标：64 GPU / PP2 VPP8 EP32 / MBS1 / BF16
#   朴素 129.5 GB（梯度按 BF16 2B 算）vs 实测 184 GB -> 1.42x
# 已按 Hy3 实测更新（DSV3 锚点给的是 1.34x）。
OVERHEAD = 1.42


def static_mem(n_gpu, pp, ep, grad_dtype_bytes=2):
    """返回 (权重, 梯度, 优化器) GB/GPU。"""
    dp = n_gpu // pp
    assert dp % ep == 0, f"EP={ep} 不整除 DP={dp}"
    p_local = P_EXPERT / (pp * ep) + P_OTHER / pp
    w = p_local * 2 / GB
    g = p_local * grad_dtype_bytes / GB
    # 12 B/param (fp32 master + fp32 exp_avg + fp32 exp_avg_sq)，分布式优化器按 N 摊
    o = P_TOTAL * 12 / n_gpu / GB
    return w, g, o


def activation_mem(pp, vpp, mbs, seq):
    """1F1B(+VPP) 稳态激活 GB/GPU。34*s*b*h/层，共 L 层份。"""
    per_layer = 34 * seq * mbs * HIDDEN / GB
    base = NUM_LAYERS * per_layer                      # 与 PP 无关
    vpp_factor = 1 + (pp - 1) / (pp * vpp) if vpp else 1.0
    return base * vpp_factor


def evaluate(n_gpu, pp, ep, mbs, seq=4096, vpp=8, grad_bytes=2):
    w, g, o = static_mem(n_gpu, pp, ep, grad_bytes)
    a = activation_mem(pp, vpp, mbs, seq)
    naive = w + g + o + a
    total = naive * OVERHEAD
    return dict(n=n_gpu, pp=pp, ep=ep, mbs=mbs, dp=n_gpu // pp,
                w=w, g=g, o=o, a=a, naive=naive, total=total,
                budget=HBM * SAFE, fit=total <= HBM * SAFE,
                experts_per_rank=192 // ep)


def fmt(r):
    mark = "✅ 安全" if r["fit"] else ("⚠️ 紧" if r["total"] <= HBM else "❌ OOM")
    return (f"| {r['n']:>3} | {r['pp']} | {r['ep']:>2} | {r['mbs']} | {r['dp']:>3} | "
            f"{r['experts_per_rank']:>2} | {r['w']:5.1f} | {r['g']:5.1f} | {r['o']:5.1f} | "
            f"{r['a']:5.1f} | {r['naive']:5.1f} | **{r['total']:5.1f}** | {mark} |")


def gbs_advice(dp, mbs, pp, vpp):
    """给 GBS 建议：须被 MBS x DP 整除；microbatch 数决定 bubble。"""
    unit = mbs * dp
    out = []
    for gbs in (1024, 2048, 4096, 8192):
        if gbs % unit:
            continue
        m = gbs // unit                       # 每 DP rank 的 microbatch 数
        bubble = (pp - 1) / (m * vpp) if vpp else (pp - 1) / m
        out.append((gbs, m, bubble * 100, gbs * 4096 / 1e6))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", nargs=4, type=int, metavar=("N", "PP", "EP", "MBS"))
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--vpp", type=int, default=8)
    args = ap.parse_args()

    print(f"Hy3 295B BF16 训练显存测算 | GB300 {HBM:.0f} GB/GPU | 安全线 {HBM*SAFE:.0f} GB "
          f"| 运行时开销系数 {OVERHEAD}x (标定自 DSV3 实测) | seq={args.seq} VPP={args.vpp}\n")
    print(f"参数分解: 专家 {P_EXPERT/1e9:.1f}B (97%) + 其他 {P_OTHER/1e9:.2f}B "
          f"= {P_TOTAL/1e9:.1f}B\n")

    if args.detail:
        n, pp, ep, mbs = args.detail
        r = evaluate(n, pp, ep, mbs, args.seq, args.vpp)
        print(f"=== {n} GPU / PP={pp} / EP={ep} / MBS={mbs} ===")
        p_local = P_EXPERT / (pp * ep) + P_OTHER / pp
        print(f"每 GPU 参数: 专家 {P_EXPERT/(pp*ep)/1e9:.2f}B + 其他 "
              f"{P_OTHER/pp/1e9:.2f}B = {p_local/1e9:.2f}B")
        for k, label in [("w", "权重 BF16 (2B)"), ("g", "梯度 BF16 (2B)"),
                         ("o", "优化器 (12B/N)"), ("a", "激活")]:
            print(f"  {label:<22} {r[k]:7.1f} GB")
        print(f"  {'朴素小计':<22} {r['naive']:7.1f} GB")
        print(f"  {f'x{OVERHEAD} 运行时开销':<22} {r['total']:7.1f} GB / 安全线 "
              f"{r['budget']:.0f} GB ({'装得下' if r['fit'] else '超了'})")
        print(f"\nGBS 建议 (DP={r['dp']}, MBS={mbs}):")
        print("  GBS   | microbatch/rank | pipeline bubble | tokens/step")
        for gbs, m, b, tok in gbs_advice(r["dp"], mbs, pp, args.vpp):
            print(f"  {gbs:<5} | {m:>15} | {b:>14.2f}% | {tok:>7.1f} M")
        return

    print("| GPU | PP | EP | MB | DP | E/r | 权重 | 梯度 | 优化器 | 激活 | 朴素 | 实际GB | 判定 |")
    print("|-----|----|----|----|----|-----|------|------|--------|------|------|--------|------|")
    grid = [(64, 2, 32), (64, 2, 16), (64, 4, 16), (64, 4, 8), (64, 8, 8),
            (128, 2, 32), (128, 4, 32), (128, 4, 16),
            (256, 2, 32), (256, 4, 32)]
    for n, pp, ep in grid:
        for mbs in (1, 2):
            try:
                print(fmt(evaluate(n, pp, ep, mbs, args.seq, args.vpp)))
            except AssertionError:
                pass


if __name__ == "__main__":
    main()
