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
    layernorm_epsilon=1e-5,
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
    if a.precision == "bf16":
        if isinstance(cfg.mixed_precision, str):
            cfg.mixed_precision = bf16_mixed()
    else:
        import sys as _s
        _s.path.insert(0, "/opt/Megatron-Bridge/scripts/performance")
        from utils.precision import get_precision_config
        cfg.mixed_precision = get_precision_config(a.precision)
        if getattr(cfg.mixed_precision, "fp8_recipe", None) == "mxfp8":
            cfg.model.fp8_output_proj = True      # deepseek gb300 配置同款
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
    # Qwen3 骨架带的是 1e-6，Hy3 是 1e-5。不覆盖就会静默沿用骨架的值 —— benchmark
    # 看不出差别（eps 不改形状也不改算力），但拿真权重前向时数值会偏。
    m.layernorm_epsilon = HY3["layernorm_epsilon"]
    m.share_embeddings_and_output_weights = False     # tie_word_embeddings: false
    m.seq_length = a.seq_length
    cfg.dataset.sequence_length = a.seq_length
    # 骨架来自 qwen3，tokenizer 是 Qwen 的（词表 151669 > Hy3 的 120832，会报
    # "Model vocab_size cannot be smaller than tokenizer's vocab_size"）。
    # mock 数据 benchmark 用 NullTokenizer 对齐 Hy3 词表即可。
    cfg.tokenizer.tokenizer_type = "NullTokenizer"
    cfg.tokenizer.vocab_size = HY3["vocab_size"]
    for attr in ("tokenizer_model", "tokenizer_path", "hf_tokenizer_path"):
        if hasattr(cfg.tokenizer, attr):
            setattr(cfg.tokenizer, attr, None)

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
    m.mtp_num_layers = a.mtp_layers or None

    # bias_update_rate：from-scratch 必须 >0，否则 expert bias 永不更新、均衡机制形同虚设。
    # 但 benchmark 开了 force_load_balancing 时路由被强制均衡，该值不影响性能测量。
    m.moe_router_bias_update_rate = a.bias_update_rate

    # ---------- 3~5. 并行 / 高性能 / CUDA graph ----------
    # 关键：不手工重实现，直接复用官方 `WorkloadBaseConfig` + `set_workload_base_configs`，
    # 保证与 deepseek GB300 recipe 的映射逻辑 1:1 一致（cutedsl→op_fuser→paged_stash 这类
    # 隐式依赖链手写必漏，实测漏了就报
    # "moe_expert_rank_capacity_factor requires use_transformer_engine_op_fuser"）。
    import sys
    sys.path.insert(0, "/opt/Megatron-Bridge/scripts/performance")
    from utils.overrides import set_workload_base_configs
    from utils.utils import WorkloadBaseConfig

    base = WorkloadBaseConfig(
        num_gpus=a.num_gpus,
        tensor_model_parallel_size=a.tp,
        pipeline_model_parallel_size=a.pp,
        virtual_pipeline_model_parallel_size=a.vpp or None,
        expert_model_parallel_size=a.ep,
        global_batch_size=a.gbs,
        micro_batch_size=a.mbs,
        cuda_graph_impl=None if a.cuda_graph == "none" else a.cuda_graph,
        cuda_graph_scope=(["attn", "moe_router", "moe_preprocess"]
                          if a.cuda_graph == "transformer_engine" else None),
        moe_flex_dispatcher_backend=(a.dispatcher if a.dispatcher != "alltoall" else None),
        moe_a2a_overlap=a.a2a_overlap,
        cutedsl_fused_grouped_mlp=a.cutedsl,
        recompute_modules=a.recompute_modules,
        pp_layout=a.pp_layout or default_pp_layout(
            m.num_layers, a.pp, a.vpp, a.mtp_layers),
    )

    # deepseek set_deepseek_v3_common_configs 等价项（与精度无关，全都要）
    m.moe_router_fusion = a.router_fusion
    m.moe_permute_fusion = a.permute_fusion
    m.recompute_granularity = (None if a.recompute_granularity == "none"
                               else a.recompute_granularity)
    cfg.dist.enable_megatron_core_experimental = True
    cfg.mixed_precision.grad_reduce_in_fp32 = False   # 梯度 BF16，省一半显存
    cfg.ddp.grad_reduce_in_fp32 = False
    if a.force_load_balancing:
        m.moe_router_force_load_balancing = True      # benchmark 专用：消除路由不均衡噪声

    # TP>1 必须开 sequence parallel，否则 LayerNorm/Dropout 激活不被切分，
    # 白付 TP 通信却省不下显存（TP 换 MBS 这笔交易的前提）。
    if a.tp > 1:
        m.sequence_parallel = a.sequence_parallel
        # Parallel Folding：专家层不做 TP 切分（Megatron-Core MoE 论文 Guideline 4）
        m.expert_tensor_parallel_size = 1

    if base.pp_layout:
        m.pipeline_model_parallel_layout = base.pp_layout
        # pp_layout 字符串里已显式写了 E(embedding) 和 L(loss)，
        # 与 account_for_* 开关互斥（qwen3 骨架默认 True，deepseek 骨架默认 False）
        m.account_for_embedding_in_pipeline_split = False
        m.account_for_loss_in_pipeline_split = False
        m.num_layers_in_first_pipeline_stage = None
        m.num_layers_in_last_pipeline_stage = None
    set_workload_base_configs(cfg, base)              # 官方映射：并行 + graph + cutedsl + recompute

    if cfg.comm_overlap is None:
        from megatron.bridge.training.comm_overlap import CommOverlapConfig
        cfg.comm_overlap = CommOverlapConfig(tp_comm_overlap=False)  # TP=1，无 TP 重叠可言
    cfg.comm_overlap.overlap_grad_reduce = True

    # full_iteration graph 专属：dropless MoE 变长 tensor graph 抓不住，
    # 先 pad 到固定容量再用 paged stash 收回显存（deepseek set_full_iter_cg_configs 原样）
    from megatron.bridge.utils.cuda_graph import is_full_iteration_cuda_graph
    if is_full_iteration_cuda_graph(m) and a.paged_stash:
        m.moe_pad_experts_for_cuda_graph_inference = True
        m.moe_paged_stash = True
        m.moe_expert_rank_capacity_factor = 1.5
        m.moe_paged_stash_buffer_size_factor_cuda = 1.2
        m.moe_paged_stash_buffer_size_factor_cpu = 1.0
        # Megatron 边界 bug：moe_paged_stash 的校验分支会 `set(self.offload_modules)`，
        # 但 offload_modules 默认 None -> TypeError: 'NoneType' object is not iterable
        # (transformer_config.py:1691)。只在 full_iteration 路径触发，TE graph 路径不会。
        if getattr(m, "offload_modules", None) is None:
            m.offload_modules = []

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
    p.add_argument("--precision", default="bf16",
                   choices=["bf16", "fp8_mx", "fp8_cs", "fp8_sc"])
    p.add_argument("--dispatcher", default="hybridep",
                   choices=["hybridep", "alltoall", "flex"])
    p.add_argument("--cuda-graph", default="transformer_engine",
                   choices=["none", "full_iteration", "transformer_engine"],
                   help="NVIDIA 官方 GB300 BF16 recipe 用 transformer_engine；"
                        "full_iteration 是 FP8_MX recipe 的配置")
    p.add_argument("--cutedsl", action="store_true", default=False,
                   help="cutedsl fused grouped MLP，会连带打开 TE op fuser "
                        "(paged stash 的前置依赖)")
    p.add_argument("--a2a-overlap", action="store_true", default=False)
    p.add_argument("--no-paged-stash", dest="paged_stash", action="store_false", default=True)
    p.add_argument("--no-router-fusion", dest="router_fusion", action="store_false", default=True)
    p.add_argument("--no-permute-fusion", dest="permute_fusion", action="store_false", default=True)
    p.add_argument("--sequence-parallel", action="store_true", default=False,
                   help="TP>1 时必须开；切分 LayerNorm/Dropout 激活")
    p.add_argument("--recompute-modules", nargs="*", default=["moe_act"],
                   help="官方 BF16 recipe 用 [moe_act]，FP8_MX 用 []")
    p.add_argument("--recompute-granularity", default="selective",
                   choices=["selective", "full", "none"],
                   help="deepseek common config 用 selective；显存有余量时设 none 提吞吐")
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
