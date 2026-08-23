#!/usr/bin/env python3
"""
token-budget.py — 专题一《一个 Token 的一生》的算账脚本。

读 DeepSeek V3 官方 config.json，把一个 token 走完全程的每一站
（权重 / 激活 / FLOPs）全部算出来，输出可复现的表格。

网页版 Courses/WebPages/topic-01.html 里的交互计算器用的是同一套公式，
改公式请两边一起改，并用本脚本的输出做基准。

用法:
    python3 token-budget.py                      # 默认 128K, batch 1, bf16
    python3 token-budget.py --seq 4096 --batch 8
    python3 token-budget.py --json               # 机器可读
"""
import argparse, json, pathlib, sys
from decimal import Decimal, ROUND_HALF_UP

CFG = pathlib.Path(__file__).resolve().parent.parent / "素材" / "deepseek-v3-config.json"
HBM_PER_DEVICE = 94.74 * 1e9      # TPU v7 单 device 可用 HBM（字节，厂商 GB）

GiB = 1024**3
TiB = 1024**4


def human(b):
    """字节 → 人类可读（二进制单位）"""
    for u, s in ((TiB, "TiB"), (GiB, "GiB"), (1024**2, "MiB"), (1024, "KiB")):
        if abs(b) >= u:
            # 半进位向上，跟网页那边 JS 的 toFixed(2) 对齐。
            # Python 默认是银行家舍入，正好落在 .xx5 时会跟网页差一位
            # （例：logits fp32 = 63.125 → py 63.12 / js 63.13）。
            return f"{Decimal(b/u).quantize(Decimal('0.01'), ROUND_HALF_UP):,} {s}"
    return f"{b:,.0f} B"


def flops_h(f):
    for u, s in ((1e18, "EFLOP"), (1e15, "PFLOP"), (1e12, "TFLOP"), (1e9, "GFLOP")):
        if abs(f) >= u:
            return f"{f/u:,.2f} {s}"
    return f"{f:,.0f} FLOP"


def params_h(p):
    for u, s in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(p) >= u:
            return f"{p/u:,.2f} {s}"
    return f"{p:,.0f}"


def compute(c, seq, batch, bytes_per_el=2, causal=True):
    d      = c["hidden_size"]              # 7168
    L      = c["num_hidden_layers"]        # 61
    n_dense= c["first_k_dense_replace"]    # 3
    V      = c["vocab_size"]               # 129280
    H      = c["num_attention_heads"]      # 128
    q_lora = c["q_lora_rank"]              # 1536
    kv_lora= c["kv_lora_rank"]             # 512
    nope   = c["qk_nope_head_dim"]         # 128
    rope   = c["qk_rope_head_dim"]         # 64
    v_dim  = c["v_head_dim"]               # 128
    n_exp  = c["n_routed_experts"]         # 256
    n_shr  = c["n_shared_experts"]         # 1
    top_k  = c["num_experts_per_tok"]      # 8
    d_moe  = c["moe_intermediate_size"]    # 2048
    d_ff   = c["intermediate_size"]        # 18432

    n_moe  = L - n_dense                   # 58
    qk_dim = nope + rope                   # 192 —— 每头 QK 实际维度
    B      = bytes_per_el
    T      = seq * batch                   # 总 token 数

    r = {"cfg": dict(d=d, L=L, n_dense=n_dense, n_moe=n_moe, V=V, H=H,
                     kv_lora=kv_lora, rope=rope, qk_dim=qk_dim, v_dim=v_dim,
                     n_exp=n_exp, n_shr=n_shr, top_k=top_k, d_moe=d_moe, d_ff=d_ff,
                     seq=seq, batch=batch, bytes=B, causal=causal, tokens=T)}

    # ── 站 1 · 嵌入 ────────────────────────────────────────────────
    r["embed_w"]   = V * d                       # 926,679,040
    r["embed_act"] = T * d * B                   # 残差流张量，此后每层读写
    r["embed_flops"] = 0                         # 查表，不是矩阵乘

    # ── 站 2 · MLA（每层）─────────────────────────────────────────
    w_q_a  = d * q_lora
    w_q_b  = q_lora * H * qk_dim
    w_kv_a = d * (kv_lora + rope)
    w_kv_b = kv_lora * H * (nope + v_dim)
    w_o    = H * v_dim * d
    r["mla_w_layer"] = w_q_a + w_q_b + w_kv_a + w_kv_b + w_o
    r["mla_w_parts"] = dict(q_a=w_q_a, q_b=w_q_b, kv_a=w_kv_a, kv_b=w_kv_b, o=w_o)
    r["mla_w_all"]   = r["mla_w_layer"] * L

    # KV cache：MLA 每 token 每层只存 kv_lora + rope 个数
    r["kv_per_tok_layer"] = kv_lora + rope       # 576
    r["kv_total"] = T * L * r["kv_per_tok_layer"] * B
    # 对照：标准 MHA 存 K 和 V 各 H×head_dim
    r["kv_mha_per_tok_layer"] = 2 * H * v_dim    # 32768
    r["kv_mha_total"] = T * L * r["kv_mha_per_tok_layer"] * B
    r["kv_ratio"] = r["kv_mha_per_tok_layer"] / r["kv_per_tok_layer"]

    # 注意力分数矩阵：若真物化出来（每层，一次）
    r["attn_matrix"] = batch * H * seq * seq * B

    # 投影 FLOPs（每层，线性项）+ 注意力 FLOPs（平方项）
    r["mla_flops_proj"]  = 2 * T * r["mla_w_layer"]
    half = 0.5 if causal else 1.0
    qk   = 2 * batch * H * seq * seq * qk_dim * half
    av   = 2 * batch * H * seq * seq * v_dim * half
    r["mla_flops_attn"]  = qk + av
    r["mla_flops_layer"] = r["mla_flops_proj"] + r["mla_flops_attn"]
    r["mla_flops_all"]   = r["mla_flops_layer"] * L

    # ── 站 3 · Dense MLP（前 3 层）────────────────────────────────
    r["dense_w_layer"] = 3 * d * d_ff            # gate + up + down
    r["dense_w_all"]   = r["dense_w_layer"] * n_dense
    r["dense_flops_all"] = 2 * T * r["dense_w_layer"] * n_dense

    # ── 站 4 · MoE（后 58 层）─────────────────────────────────────
    r["expert_w"]    = 3 * d * d_moe             # 单个专家
    r["gate_w"]      = d * n_exp
    r["moe_w_layer"] = (n_exp + n_shr) * r["expert_w"] + r["gate_w"]
    r["moe_w_all"]   = r["moe_w_layer"] * n_moe
    r["moe_act_layer"] = (top_k + n_shr) * r["expert_w"]   # 每 token 实际过 9 个
    r["moe_flops_all"] = 2 * T * r["moe_act_layer"] * n_moe
    r["moe_sparsity"]  = r["moe_w_layer"] / r["moe_act_layer"]
    # all-to-all：dispatch + combine，每层每 token 送 top_k 份 hidden 出去再回来
    r["a2a_layer"] = 2 * T * top_k * d * B
    r["a2a_all"]   = r["a2a_layer"] * n_moe

    # ── 站 6 · 出口 ───────────────────────────────────────────────
    r["head_w"]     = V * d                      # tie_word_embeddings = false → 独立第二份
    r["logits"]     = T * V * B                  # ⚠️ 爆点
    r["head_flops"] = 2 * T * r["head_w"]
    # 出口那一下的三个「更吓人」的角度：
    #   ① cross-entropy 通常要把 logits 提到 fp32，真实峰值是这个而不是 r["logits"]
    #   ② 推理 prefill 只要最后一个位置的 logits —— 训练要全部，差 T 倍
    #   ③ 它的算力占比极小，跟它的体积完全不成比例（head_share，见站 7）
    r["logits_fp32"] = T * V * 4
    r["logits_last"] = V * B
    r["logits_card"] = r["logits_fp32"] / HBM_PER_DEVICE

    # ── 站 7 · 合账 ───────────────────────────────────────────────
    r["total_w"] = (r["embed_w"] + r["mla_w_all"] + r["dense_w_all"]
                    + r["moe_w_all"] + r["head_w"])
    r["active_w"] = (r["mla_w_layer"] * L
                     + r["dense_w_layer"] * n_dense
                     + (r["moe_act_layer"] + r["gate_w"]) * n_moe
                     + r["head_w"])
    r["total_flops"] = (r["mla_flops_all"] + r["dense_flops_all"]
                        + r["moe_flops_all"] + r["head_flops"])
    r["attn_share"] = r["mla_flops_attn"] * L / r["total_flops"]
    r["head_share"] = r["head_flops"] / r["total_flops"]
    r["mla_w_bytes"]  = r["mla_w_all"] * B
    r["logits_vs_mla"] = r["logits"] / r["mla_w_bytes"]

    r["mem_weights"] = r["total_w"] * B
    r["mem_resident"] = r["mem_weights"] + r["kv_total"]      # 常驻：权重 + KV
    r["devices_weights"] = r["mem_weights"] / HBM_PER_DEVICE
    r["devices_resident"] = r["mem_resident"] / HBM_PER_DEVICE
    return r


def report(r):
    c = r["cfg"]
    P = print
    P(f"\n{'═'*76}")
    P(f"  一个 Token 的一生 · 算账   seq={c['seq']:,}  batch={c['batch']}  "
      f"bf16  {'causal' if c['causal'] else 'full'}  总 token={c['tokens']:,}")
    P('═'*76)

    P(f"\n【站 1】嵌入")
    P(f"  嵌入矩阵      {c['V']:,} × {c['d']:,} = {params_h(r['embed_w'])} 参数"
      f"  ({human(r['embed_w']*c['bytes'])})")
    P(f"  残差流张量    {c['tokens']:,} × {c['d']:,} × {c['bytes']}B = {human(r['embed_act'])}"
      f"   ← 还没进第一层")

    P(f"\n【站 2】MLA（每层）")
    for k, v in r["mla_w_parts"].items():
        P(f"    {k:<6} {params_h(v):>10}")
    P(f"  一层合计      {params_h(r['mla_w_layer'])}   ×{c['L']} 层 = {params_h(r['mla_w_all'])}")
    P(f"  KV cache      每 token 每层 {r['kv_per_tok_layer']} 个数 "
      f"({c['kv_lora']} 压缩 + {c['rope']} RoPE)")
    P(f"                全量 = {human(r['kv_total'])}")
    P(f"  ⚠️ 同样配置换标准 MHA：每 token 每层 {r['kv_mha_per_tok_layer']:,} 个数 "
      f"→ {human(r['kv_mha_total'])}")
    P(f"     MLA 省了 {r['kv_ratio']:.1f}×   ← 这个差值就是 MLA 存在的理由")
    P(f"  ⚠️ 注意力分数矩阵若物化：{c['H']}头 × {c['seq']:,}² × {c['bytes']}B "
      f"= {human(r['attn_matrix'])} / 层")
    P(f"     → FlashAttention 不是为了快，是为了根本放得下")

    P(f"\n【站 3】Dense MLP（前 {c['n_dense']} 层）")
    P(f"  一层          3 × {c['d']:,} × {c['d_ff']:,} = {params_h(r['dense_w_layer'])}")
    P(f"  {c['n_dense']} 层合计      {params_h(r['dense_w_all'])}")

    P(f"\n【站 4】MoE（后 {c['n_moe']} 层）")
    P(f"  单个专家      3 × {c['d']:,} × {c['d_moe']:,} = {params_h(r['expert_w'])}"
      f"   ← dense 的 1/{c['d_ff']//c['d_moe']}")
    P(f"  一层          {c['n_exp']}路由 + {c['n_shr']}共享 = {params_h(r['moe_w_layer'])}")
    P(f"  {c['n_moe']} 层合计     {params_h(r['moe_w_all'])}   ← 671B 的绝大部分")
    P(f"  每 token 实际过 {c['top_k']}+{c['n_shr']} = {c['top_k']+c['n_shr']} 个专家 "
      f"→ {params_h(r['moe_act_layer'])} / 层   稀疏比 {r['moe_sparsity']:.1f}×")
    P(f"  all-to-all    {human(r['a2a_layer'])} / 层   ×{c['n_moe']} = {human(r['a2a_all'])}")

    P(f"\n【站 6】出口")
    P(f"  lm_head       {params_h(r['head_w'])}   ⚠️ tie_word_embeddings=false，是独立的第二份")
    P(f"  logits 张量   {c['tokens']:,} × {c['V']:,} × {c['bytes']}B = {human(r['logits'])}"
      f"   ← 比中间任何一层都吓人")
    P(f"  对照 61 层 MLA 全部权重 {human(r['mla_w_bytes'])}"
      f"   → logits 是它的 {r['logits_vs_mla']:.2f}×")
    P(f"  loss 要 fp32  {human(r['logits_fp32'])}"
      f"   ← 单卡 {HBM_PER_DEVICE/1e9:.2f} GB 的 {r['logits_card']*100:.0f}%")
    P(f"  推理只要末位  {human(r['logits_last'])}"
      f"   ← 训练要全部，差 {c['tokens']:,} 倍")
    P(f"  算力占比      {r['head_share']*100:.2f}%"
      f"   ← 全程算力最少的一步，产出全程最大的张量")

    P(f"\n【站 7】合账")
    P(f"  {'总参数':<14}{params_h(r['total_w']):>12}      官方口径 671B")
    P(f"  {'激活参数':<13}{params_h(r['active_w']):>12}      官方口径 37B"
      f"      差 {r['total_w']/r['active_w']:.1f}×")
    P(f"  {'前向 FLOPs':<12}{flops_h(r['total_flops']):>12}"
      f"      其中 attention 平方项占 {r['attn_share']*100:.1f}%")
    P(f"  {'权重显存':<13}{human(r['mem_weights']):>12}")
    P(f"  {'+ KV cache':<12}{human(r['kv_total']):>12}")
    P(f"  {'= 常驻':<14}{human(r['mem_resident']):>12}")
    P(f"\n  对着 TPU v7 单 device 94.74 GB 一除：")
    P(f"    光放权重      需要 {r['devices_weights']:.1f} 个 device")
    P(f"    权重 + KV     需要 {r['devices_resident']:.1f} 个 device")
    P(f"    ⚠️ 这还只是前向、batch=1、不含激活和临时缓冲")
    P(f"\n  → 「为什么需要并行」不用讲了，它是算出来的结论。\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", type=int, default=131072, help="序列长度（默认 128K）")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--bytes", type=int, default=2, help="每元素字节数（bf16=2）")
    ap.add_argument("--full-attn", action="store_true", help="不用 causal mask（算满）")
    ap.add_argument("--config", default=str(CFG))
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    a = ap.parse_args()

    cfg = json.loads(pathlib.Path(a.config).read_text())
    res = compute(cfg, a.seq, a.batch, a.bytes, causal=not a.full_attn)
    if a.json:
        json.dump(res, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        report(res)
