#!/usr/bin/env python3
"""Hy3 (混元3, 295B-A21B) GB300 预训练入口 — Megatron-Bridge r0.5.0。

为什么需要这个脚本（不能用 run_script.py）：
  1. 容器内 Bridge r0.5.0 **没有 hy_v3 模型桥**（那个只在 main 分支），AutoBridge 路径不通。
  2. `recipes/` 和 `scripts/performance/configs/` 下都没有 hunyuan/hy3，
     `run_script.py -m hy3` 不存在。
  3. run_script.py 的 CLI 只能覆盖 hidden_size / num_layers / first_k_dense_replace，
     覆盖不了 num_moe_experts / num_query_groups / moe_ffn_hidden_size。

做法：
  拿 **qwen3_235b_a22b_pretrain_config()** 当骨架（同为 GQA MoE，且 hidden_size 4096 /
  kv_channels 128 / qk_layernorm / moe_ffn 1536 已经跟 Hy3 一致，optimizer/ddp/dataset/
  comm_overlap 都是调好的），再把模型字段改成 Hy3，并叠加 DeepSeek-V3 血统的 MoE 旋钮
  + 07e 验证过的 GB300 高性能设置（full_iteration graph / paged stash / hybridep / cutedsl）。

用法（每个 rank 由 torchrun 拉起）：
    python hy3_pretrain.py --tp 1 --pp 2 --vpp 8 --ep 32 --mbs 1 --gbs 2048 --max-steps 30
    python hy3_pretrain.py --dryrun          # 只建 config 打印，不起分布式
"""
from __future__ import annotations

import argparse
import os

# --- Hy3 结构常量（tencent/Hy3 config.json 实查）---
HY3 = dict(
    num_layers=80, hidden_size=4096, ffn_hidden_size=13312,
    num_attention_heads=64, num_query_groups=8, kv_channels=128,
    vocab_size=120832, rotary_base=11158840.0,
    num_moe_experts=192, moe_router_topk=8, moe_ffn_hidden_size=1536,
    num_shared_experts=1, first_k_dense_replace=1, router_scaling_factor=2.826,
)


def default_pp_layout(num_layers, pp, vpp, mtp):
    """按 chunk 均分推导 pp_layout。80 层 / (PP2×VPP8=16 chunk) = 5 层/chunk。"""
    chunks = pp * vpp
    if num_layers % chunks:
        raise ValueError(f"{num_layers} 层无法均分到 {chunks} chunk，请用 --pp-layout 手工指定")
    per = num_layers // chunks
    tail = f"t*{per}" + ("m" if mtp else "") + "L"
    return f"Et*{per}|(t*{per}|)*{chunks - 2}{tail}"


def build_config(a):
    from megatron.bridge.recipes.qwen.qwen3_moe import (
        bf16_mixed, qwen3_235b_a22b_pretrain_config,
    )

    cfg = qwen3_235b_a22b_pretrain_config()   # 骨架：GQA MoE + 调好的 optimizer/ddp/dataset
    # qwen3 recipe 的 mixed_precision 是字符串 'bf16_mixed'，要换成配置对象才能改字段
    if isinstance(cfg.mixed_precision, str):
        cfg.mixed_precision = bf16_mixed()
    m = cfg.model

    # ---------- 1. Hy3 结构 ----------
    m.num_layers = a.num_layers or HY3["num_layers"]
    m.hidden_size = HY3["hidden_size"]
    m.ffn_hidden_size = HY3["ffn_hidden_size"]        # 仅第 0 层 dense 用
    m.num_attention_heads = HY3["num_attention_heads"]
    m.num_query_groups = HY3["num_query_groups"]      # GQA 8 KV heads
    m.kv_channels = HY3["kv_channels"]
    m.vocab_size = HY3["vocab_size"]
    m.rotary_base = HY3["rotary_base"]
    m.qk_layernorm = True                             # config.json: qk_norm
    m.share_embeddings_and_output_weights = False     # tie_word_embeddings: false
    m.seq_length = a.seq_length
    cfg.dataset.sequence_length = a.seq_length
    if hasattr(cfg.tokenizer, "vocab_size"):
        cfg.tokenizer.vocab_size = HY3["vocab_size"]

    # ---------- 2. MoE：DeepSeek-V3 血统 ----------
    m.num_moe_experts = HY3["num_moe_experts"]
    m.moe_router_topk = HY3["moe_router_topk"]
    m.moe_ffn_hidden_size = HY3["moe_ffn_hidden_size"]
    m.moe_shared_expert_intermediate_size = (
        HY3["moe_ffn_hidden_size"] * HY3["num_shared_experts"])
    m.moe_layer_freq = ([0] * HY3["first_k_dense_replace"]
                        + [1] * (m.num_layers - HY3["first_k_dense_replace"]))
    m.moe_router_score_function = "sigmoid"           # V3 的 sigmoid 路由（qwen3 默认 softmax）
    m.moe_router_pre_softmax = False
    m.moe_router_enable_expert_bias = True            # aux-loss-free 的 per-expert bias
    m.moe_router_load_balancing_type = "none"         # aux-loss-free 路线
    m.moe_aux_loss_coeff = 0.0
    m.moe_router_topk_scaling_factor = HY3["router_scaling_factor"]
    m.moe_router_dtype = "fp32"
    m.moe_grouped_gemm = True
    m.moe_permute_fusion = True
    m.mtp_num_layers = a.mtp_layers or None

    # bias_update_rate：from-scratch 必须 >0，否则 expert bias 永不更新、均衡机制形同虚设。
    # 但 benchmark 开了 force_load_balancing 时路由被强制均衡，该值不影响性能测量。
    m.moe_router_bias_update_rate = a.bias_update_rate

    # ---------- 3. 并行 ----------
    m.tensor_model_parallel_size = a.tp
    m.pipeline_model_parallel_size = a.pp
    m.virtual_pipeline_model_parallel_size = a.vpp or None
    m.expert_model_parallel_size = a.ep
    m.pipeline_model_parallel_layout = a.pp_layout or default_pp_layout(
        m.num_layers, a.pp, a.vpp, a.mtp_layers)

    # ---------- 4. GB300 高性能（照搬 deepseek_llm_pretrain.set_deepseek_v3_common_configs）----------
    m.moe_router_fusion = True
    m.recompute_granularity = "selective"
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False   # 梯度 BF16，省一半显存
    cfg.ddp.grad_reduce_in_fp32 = False
    # qwen3 recipe 的 comm_overlap 默认 None，需要显式构造（deepseek 配置里会设它）
    if cfg.comm_overlap is None:
        from megatron.bridge.training.comm_overlap import CommOverlapConfig
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)  # TP=1，无 TP 重叠可言
    cfg.comm_overlap.overlap_grad_reduce = True
    if a.force_load_balancing:
        m.moe_router_force_load_balancing = True      # benchmark 专用：消除路由不均衡噪声

    # hybridep：GB300 NVL72 域内 all-to-all，显著优于朴素 alltoall
    if a.dispatcher == "hybridep":
        m.moe_token_dispatcher_type = "flex"
        m.moe_flex_dispatcher_backend = "hybridep"
    else:
        m.moe_token_dispatcher_type = a.dispatcher
    if hasattr(m, "moe_a2a_overlap"):
        m.moe_a2a_overlap = True

    # ---------- 5. CUDA graph + paged stash（07e 的核心成果）----------
    if a.cuda_graph != "none":
        m.cuda_graph_impl = a.cuda_graph
    if a.cuda_graph == "full_iteration":
        # dropless MoE 产生变长 per-expert tensor，graph 抓不住；
        # 先 pad 到固定容量，再用 paged stash 把显存收回来。
        m.moe_pad_experts_for_cuda_graph_inference = True
        m.moe_paged_stash = True
        m.moe_expert_rank_capacity_factor = 1.5
        m.moe_paged_stash_buffer_size_factor_cuda = 1.2
        m.moe_paged_stash_buffer_size_factor_cpu = 1.0

    # ---------- 6. 训练超参 ----------
    cfg.train.micro_batch_size = a.mbs
    cfg.train.global_batch_size = a.gbs
    cfg.train.train_iters = a.max_steps
    if hasattr(cfg.train, "eval_interval"):
        cfg.train.eval_interval = a.max_steps + 1     # benchmark 不做 eval
    if hasattr(cfg.logger, "log_throughput"):
        cfg.logger.log_throughput = True
    if hasattr(cfg.checkpoint, "save_interval"):
        cfg.checkpoint.save_interval = None
    return cfg


def summarize(cfg, a):
    m = cfg.model
    dp = a.num_gpus // (a.tp * a.pp)
    expert = ((m.num_layers - HY3["first_k_dense_replace"]) * HY3["num_moe_experts"]
              * 3 * HY3["hidden_size"] * HY3["moe_ffn_hidden_size"])
    print("=" * 74)
    print(f"Hy3 295B | {a.num_gpus} GPU | TP{a.tp} PP{a.pp} VPP{a.vpp} EP{a.ep} "
          f"| MBS{a.mbs} GBS{a.gbs} | {a.precision}")
    print(f"  层数 {m.num_layers} (dense {HY3['first_k_dense_replace']} + MoE "
          f"{m.num_layers - HY3['first_k_dense_replace']})  hidden {m.hidden_size}  "
          f"GQA {m.num_attention_heads}Q/{m.num_query_groups}KV x {m.kv_channels}")
    print(f"  MoE {m.num_moe_experts} experts top-{m.moe_router_topk} "
          f"ffn {m.moe_ffn_hidden_size} shared {m.moe_shared_expert_intermediate_size}")
    print(f"  路由 {m.moe_router_score_function} + expert_bias={m.moe_router_enable_expert_bias} "
          f"(rate {m.moe_router_bias_update_rate}) lb={m.moe_router_load_balancing_type} "
          f"scale {m.moe_router_topk_scaling_factor}")
    print(f"  dispatcher {m.moe_token_dispatcher_type}/"
          f"{getattr(m, 'moe_flex_dispatcher_backend', '-')}  "
          f"graph {getattr(m, 'cuda_graph_impl', '-')}  "
          f"paged_stash {getattr(m, 'moe_paged_stash', '-')}")
    print(f"  MTP {m.mtp_num_layers}  pp_layout {m.pipeline_model_parallel_layout}")
    print(f"  DP={dp}  microbatch/rank={a.gbs // (a.mbs * dp)}  "
          f"专家参数 {expert/1e9:.1f}B (每 rank {expert/(a.pp*a.ep)/1e9:.2f}B)")
    print(f"  env NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN="
          f"{os.environ.get('NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN', '<未设>')} (须 == EP {a.ep})")
    print("=" * 74)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-gpus", type=int, default=64)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=2)
    p.add_argument("--vpp", type=int, default=8)
    p.add_argument("--ep", type=int, default=32)
    p.add_argument("--mbs", type=int, default=1)
    p.add_argument("--gbs", type=int, default=2048)
    p.add_argument("--seq-length", type=int, default=4096)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--mtp-layers", type=int, default=0)
    p.add_argument("--pp-layout", default=None)
    p.add_argument("--num-layers", type=int, default=0,
                   help="缩层做冒烟测试用；0=用 Hy3 全量 80 层")
    p.add_argument("--precision", default="bf16")
    p.add_argument("--dispatcher", default="hybridep",
                   choices=["hybridep", "alltoall", "flex"])
    p.add_argument("--cuda-graph", default="full_iteration",
                   choices=["none", "full_iteration", "local"])
    p.add_argument("--bias-update-rate", type=float, default=1e-3)
    p.add_argument("--force-load-balancing", action="store_true", default=True)
    p.add_argument("--no-force-load-balancing", dest="force_load_balancing",
                   action="store_false")
    p.add_argument("--dryrun", action="store_true")
    a = p.parse_args()

    dp = a.num_gpus // (a.tp * a.pp)
    if dp % a.ep:
        raise SystemExit(f"EP={a.ep} 不整除 DP={dp}")
    if a.gbs % (a.mbs * dp):
        raise SystemExit(f"GBS={a.gbs} 不能被 MBS×DP={a.mbs*dp} 整除")

    cfg = build_config(a)
    if int(os.environ.get("RANK", "0")) == 0:
        summarize(cfg, a)
    if a.dryrun:
        return

    from megatron.bridge.training.gpt_step import forward_step
    from megatron.bridge.training.pretrain import pretrain
    pretrain(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
