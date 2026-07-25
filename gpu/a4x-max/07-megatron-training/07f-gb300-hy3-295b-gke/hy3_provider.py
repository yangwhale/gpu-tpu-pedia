#!/usr/bin/env python3
"""Hy3 (混元3, 295B-A21B) Megatron-Bridge GPTModelProvider 构造 + pretrain 入口。

Megatron-Bridge 有 HYV3Bridge 模型桥（HF <-> Megatron 权重映射，含 MTP），
但 recipes/ 下没有 hy_v3 的 perf recipe，所以 `run_script.py -m hy3` 不可用。
本脚本补上这一层。

两条构造路径：
  A) build_from_hf()          — AutoBridge 从 HF config 自动填参（需能访问 tencent/Hy3）
  B) build_from_scratch()     — 手工填参，纯 mock benchmark 用，不下 590GB 权重

参数依据：tencent/Hy3 config.json + HYV3Bridge.provider_bridge() 源码。
"""
from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# Hy3 结构常量（来自 tencent/Hy3 config.json）
# ---------------------------------------------------------------------------
HY3 = dict(
    num_layers=80,
    hidden_size=4096,
    ffn_hidden_size=13312,          # dense 层（仅第 0 层）
    num_attention_heads=64,
    num_query_groups=8,             # GQA: 8 KV heads
    kv_channels=128,                # head_dim
    vocab_size=120832,
    rotary_base=11158840.0,
    # MoE
    num_moe_experts=192,
    moe_router_topk=8,
    moe_ffn_hidden_size=1536,
    num_shared_experts=1,
    first_k_dense_replace=1,
    router_scaling_factor=2.826,
    mtp_num_layers=1,
)


def _apply_hy3_arch(provider):
    """把 Hy3 的架构常量写进 provider（对齐 HYV3Bridge.provider_bridge）。"""
    import torch

    p = provider
    # --- 结构 ---
    p.normalization = "RMSNorm"
    p.gated_linear_unit = True
    p.add_bias_linear = False
    p.add_qkv_bias = False          # Hy3 无 QKV bias
    p.qk_layernorm = True           # config.json: qk_norm = true
    p.hidden_dropout = 0.0
    p.attention_softmax_in_fp32 = False
    p.untie_embeddings_and_output_weights = True
    # --- 精度 ---
    p.bf16, p.fp16 = True, False
    p.params_dtype = torch.bfloat16
    p.autocast_dtype = torch.bfloat16
    # --- MoE（DeepSeek V3 血统）---
    p.moe_grouped_gemm = True
    p.moe_permute_fusion = True
    p.moe_router_score_function = "sigmoid"
    p.moe_router_enable_expert_bias = True
    p.moe_router_pre_softmax = False
    p.moe_router_dtype = "fp32"
    p.moe_router_topk_scaling_factor = HY3["router_scaling_factor"]
    p.moe_shared_expert_intermediate_size = (
        HY3["moe_ffn_hidden_size"] * HY3["num_shared_experts"]
    )
    p.moe_shared_expert_overlap = False
    # 第 0 层 dense，其余 MoE
    p.moe_layer_freq = [0] * HY3["first_k_dense_replace"] + [1] * (
        HY3["num_layers"] - HY3["first_k_dense_replace"]
    )
    return p


def apply_training_overrides(provider, *, from_scratch: bool, ep: int):
    """训练侧必须覆盖的 bridge 默认值。

    HYV3Bridge 的默认值面向权重转换/推理：
        moe_router_bias_update_rate = 0
        moe_router_load_balancing_type = "none"
        moe_aux_loss_coeff = 0.0
        moe_token_dispatcher_type = "alltoall"

    from-scratch 预训练时 bias_update_rate=0 会让 aux-loss-free 负载均衡
    形同虚设（expert bias 永不更新）——DeepSeek V3 论文用 1e-3。
    加载官方权重做 SFT 时保持 0 更稳（不扰动已收敛的路由）。
    """
    p = provider
    p.moe_router_load_balancing_type = "none"       # aux-loss-free 路线，勿改 aux_loss
    p.moe_aux_loss_coeff = 0.0
    p.moe_router_bias_update_rate = 1e-3 if from_scratch else 0.0

    # GB300 NVL72：hybridep 显著优于朴素 alltoall
    p.moe_token_dispatcher_type = "flex"
    p.moe_flex_dispatcher_backend = "hybridep"
    p.moe_a2a_overlap = True

    # 注意：环境变量 NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN 必须 == ep，
    # 否则 hybridep all-to-all 会 collective timeout 挂死。
    p._hybrid_ep_ranks_hint = ep
    return p


def build_from_hf(model_id: str = "tencent/Hy3"):
    """路径 A：AutoBridge 从 HF config 自动生成 provider。"""
    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(model_id)
    return bridge.to_megatron_provider()


def build_from_scratch(seq_length: int, mtp_layers: int):
    """路径 B：手工构造，不依赖 HF 权重（mock benchmark 用）。"""
    from megatron.bridge.models import GPTModelProvider

    p = GPTModelProvider(
        num_layers=HY3["num_layers"],
        hidden_size=HY3["hidden_size"],
        ffn_hidden_size=HY3["ffn_hidden_size"],
        num_attention_heads=HY3["num_attention_heads"],
        num_query_groups=HY3["num_query_groups"],
        kv_channels=HY3["kv_channels"],
        seq_length=seq_length,
        vocab_size=HY3["vocab_size"],
        num_moe_experts=HY3["num_moe_experts"],
        moe_router_topk=HY3["moe_router_topk"],
        moe_ffn_hidden_size=HY3["moe_ffn_hidden_size"],
    )
    p.rotary_base = HY3["rotary_base"]
    p.mtp_num_layers = mtp_layers or None
    return _apply_hy3_arch(p)


def apply_parallelism(provider, *, tp, pp, vpp, ep, pp_layout=None):
    p = provider
    p.tensor_model_parallel_size = tp
    p.pipeline_model_parallel_size = pp
    p.virtual_pipeline_model_parallel_size = vpp
    p.expert_model_parallel_size = ep
    if pp_layout:
        p.pipeline_model_parallel_layout = pp_layout
    return p


def default_pp_layout(num_layers: int, pp: int, vpp: int, mtp: int) -> str:
    """按 chunk 均分推导 pp_layout。

    DSV3 61 层 PP2xVPP8 = 16 chunk -> "Et*4|(t*4|)*14tmL"
    Hy3  80 层 PP2xVPP8 = 16 chunk -> 5 层/chunk

    警告：末 chunk 额外扛 MTP + loss head，均分可能负载不均 / OOM。
    首跑建议 mtp=0；若 OOM 改前重后轻，如 "Et*6|(t*5|)*14t*4mL"。
    """
    chunks = pp * vpp
    if num_layers % chunks:
        raise ValueError(f"{num_layers} 层无法均分到 {chunks} chunk，请手工指定 pp_layout")
    per = num_layers // chunks
    mid = chunks - 2
    tail = f"t*{per}" + ("m" if mtp else "") + "L"
    return f"Et*{per}|(t*{per}|)*{mid}{tail}"


def estimate_params() -> dict:
    """参数量核对（跑起来后跟 log 里的总参对一下）。"""
    L, H = HY3["num_layers"], HY3["hidden_size"]
    moe_h, E = HY3["moe_ffn_hidden_size"], HY3["num_moe_experts"]
    moe_layers = L - HY3["first_k_dense_replace"]
    q_dim = HY3["num_attention_heads"] * HY3["kv_channels"]
    kv_dim = HY3["num_query_groups"] * HY3["kv_channels"]

    routed = moe_layers * E * 3 * H * moe_h
    shared = moe_layers * HY3["num_shared_experts"] * 3 * H * moe_h
    attn = L * (H * q_dim + 2 * H * kv_dim + q_dim * H)
    dense = HY3["first_k_dense_replace"] * 3 * H * HY3["ffn_hidden_size"]
    embed = 2 * HY3["vocab_size"] * H
    total = routed + shared + attn + dense + embed
    return dict(routed=routed, shared=shared, attn=attn, dense=dense,
                embed=embed, total=total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="pretrain", choices=["pretrain", "finetune", "dryrun"])
    ap.add_argument("--from-hf", action="store_true", help="用 AutoBridge 加载 tencent/Hy3")
    ap.add_argument("--num-gpus", type=int, default=64)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--pp", type=int, default=2)
    ap.add_argument("--vpp", type=int, default=8)
    ap.add_argument("--ep", type=int, default=32)
    ap.add_argument("--mbs", type=int, default=1)
    ap.add_argument("--gbs", type=int, default=2048)
    ap.add_argument("--seq-length", type=int, default=4096)
    ap.add_argument("--precision", default="fp8_mx")
    ap.add_argument("--mtp-layers", type=int, default=0, help="首跑建议 0，跑通后再开 1")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--data", default="mock")
    ap.add_argument("--pp-layout", default=None)
    ap.add_argument("--log-dir", default="/tmp/nemo-results")
    args = ap.parse_args()

    # --- 前置校验：EP 必须整除 DP ---
    dp = args.num_gpus // (args.tp * args.pp)
    if dp % args.ep:
        raise SystemExit(f"EP={args.ep} 不能整除 DP={dp}（world={args.num_gpus} TP={args.tp} PP={args.pp}）")
    if args.gbs % (args.mbs * dp):
        raise SystemExit(f"GBS={args.gbs} 不能被 MBS×DP={args.mbs * dp} 整除")

    est = estimate_params()
    print(f"[hy3] 预估总参 {est['total']/1e9:.1f}B（专家 {est['routed']/1e9:.1f}B "
          f"= {100*est['routed']/est['total']:.0f}%）")
    print(f"[hy3] DP={dp} EP={args.ep} 每 rank {HY3['num_moe_experts']//args.ep} 专家 "
          f"| microbatch={args.gbs//(args.mbs*dp)}")

    layout = args.pp_layout or default_pp_layout(
        HY3["num_layers"], args.pp, args.vpp, args.mtp_layers)
    print(f"[hy3] pp_layout = {layout}")
    print(f"[hy3] 提醒：env NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN 必须 == {args.ep}")

    if args.mode == "dryrun":
        return

    provider = build_from_hf() if args.from_hf else build_from_scratch(
        args.seq_length, args.mtp_layers)
    provider = apply_parallelism(provider, tp=args.tp, pp=args.pp, vpp=args.vpp,
                                 ep=args.ep, pp_layout=layout)
    provider = apply_training_overrides(
        provider, from_scratch=(args.mode == "pretrain" and not args.from_hf), ep=args.ep)

    # 交给 Bridge 的训练入口。具体 API 随 Bridge 版本变化，
    # 容器内确认：python -c "from megatron.bridge.training import pretrain; help(pretrain)"
    from megatron.bridge.training import pretrain
    pretrain(
        model_provider=provider,
        micro_batch_size=args.mbs,
        global_batch_size=args.gbs,
        seq_length=args.seq_length,
        train_iters=args.max_steps,
        mock_data=(args.data == "mock"),
        save=args.log_dir,
    )


if __name__ == "__main__":
    main()
