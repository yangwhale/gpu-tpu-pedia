#!/usr/bin/env python3
"""
MaxText TFLOP/s 计算器 — 独立提取版
=====================================

从 maxtext/src/maxtext/utils/maxtext_utils.py 提取的所有 TFLOP 计算函数，
附带逐行中文注释，帮助理解每一行代码和每一个参数的含义。

原始代码来源:
  - maxtext_utils.py:316-861 (TFLOP 计算相关函数)
  - metric_logger.py:269-312  (调用入口)

核心概念:
  - TFLOP = Tera Floating Point Operations (万亿次浮点运算)
  - TFLOP/s = TFLOP per second (每秒万亿次浮点运算, 衡量计算速度)
  - FLOP 计算是 **公式估算**, 不是 profiler 实测值
  - 矩阵乘法 [M×K] · [K×N] 的 FLOPs = 2 × M × K × N (乘法+加法各一次)
  - 训练的 FLOPs = 3 × forward_FLOPs (forward=1x, backward=2x)

使用方式:
  python tflop_calculator.py                 # 使用内置 ALModel 参数演示
  python tflop_calculator.py --interactive   # 交互模式，输入自定义参数

作者: 从 MaxText 项目提取并加注释
日期: 2026-03-06
"""

from dataclasses import dataclass, field
from typing import List, Tuple


# =============================================================================
# 第一部分: 模型配置 (替代 MaxText 的 config 对象)
# =============================================================================

@dataclass
class ModelConfig:
    """
    模型训练配置参数集合

    这个 dataclass 替代了 MaxText 中复杂的 pyconfig 系统,
    只保留 TFLOP 计算所需的参数。
    """

    # ── 训练 batch 参数 ──────────────────────────────────────────────
    # per_device_batch_size: 每个设备上的 batch 大小
    #   - 例如 12 表示每个 TPU/GPU core 处理 12 个样本
    #   - 全局 batch = per_device_batch_size × 设备数
    per_device_batch_size: int = 12

    # max_target_length: 序列长度 (token 数)
    #   - 训练时每个样本的最大 token 数
    #   - 常见值: 2048, 4096, 8192, 32768
    #   - 越长 → attention 计算量增长 O(S²)
    max_target_length: int = 4096

    # gradient_accumulation_steps: 梯度累积步数
    #   - 等效扩大 batch 的方式: 累积 N 步梯度再更新一次参数
    #   - TFLOP 要乘以这个值, 因为每次参数更新实际包含 N 步的计算
    gradient_accumulation_steps: int = 1

    # ── Transformer 架构参数 ─────────────────────────────────────────
    # emb_dim (E): 隐藏层维度 / embedding 维度
    #   - Transformer 各层的主要维度
    #   - ALModel=2048, DeepSeek V3=7168, Llama 70B=8192
    emb_dim: int = 2048

    # vocab_size (V): 词表大小
    #   - tokenizer 可识别的 token 总数
    #   - 影响 embedding 层和输出 logits 层的计算量
    vocab_size: int = 157184

    # num_decoder_layers (L): Transformer decoder 层数
    #   - 就是模型有多少个重复的 transformer block
    #   - ALModel=20, DeepSeek V3=61, Llama 70B=80
    num_decoder_layers: int = 20

    # ── Attention 参数 ───────────────────────────────────────────────
    # attention_type: 注意力类型
    #   - "mha" = Multi-Head Attention (标准多头注意力)
    #   - "mla" = Multi-Head Latent Attention (多头潜在注意力, DeepSeek 发明)
    #   - MLA 用低秩投影压缩 KV cache, 减少推理时显存占用
    attention_type: str = "mla"

    # num_query_heads: Query 头数
    #   - 标准 MHA 中就是 attention head 数量
    #   - MLA 中 Q 被投影到 num_query_heads 个头
    num_query_heads: int = 16

    # num_kv_heads: Key/Value 头数
    #   - MHA: num_kv_heads == num_query_heads (标准)
    #   - GQA: num_kv_heads < num_query_heads (分组查询注意力, 如 Llama 2)
    #   - MQA: num_kv_heads == 1 (多查询注意力)
    num_kv_heads: int = 16

    # head_dim: 每个 attention head 的维度
    #   - 通常 emb_dim / num_query_heads, 但 MLA 不一定遵循这个
    #   - 标准值: 64, 128
    head_dim: int = 128

    # ── MLA (Multi-Head Latent Attention) 特有参数 ───────────────────
    # q_lora_rank: Query 低秩投影的中间维度
    #   - 0 = 不使用低秩投影, 直接投影
    #   - >0 = 先 down-project 到 q_lora_rank, 再 up-project 到多头
    #   - ALModel=256, DeepSeek V3=1536
    q_lora_rank: int = 256

    # kv_lora_rank: KV 低秩投影的中间维度
    #   - MLA 的核心: 把 KV 压缩到低维空间存储
    #   - ALModel=512, DeepSeek V3=512
    kv_lora_rank: int = 512

    # qk_nope_head_dim: QK 投影中 **不带** RoPE 的部分维度
    #   - MLA 把 QK 拆成两部分: nope (无位置编码) + rope (有位置编码)
    #   - 通常 128
    qk_nope_head_dim: int = 128

    # qk_rope_head_dim: QK 投影中 **带** RoPE 的部分维度
    #   - RoPE = Rotary Position Embedding (旋转位置编码)
    #   - 通常 64
    qk_rope_head_dim: int = 64

    # v_head_dim: Value head 的维度
    #   - 和 qk_nope_head_dim 可以不同, 但通常相同
    v_head_dim: int = 128

    # ── Sparse Indexer 参数 (DeepSeek Lightning Attention) ───────────
    # use_sparse_indexer: 是否使用稀疏索引器
    #   - True = 不做 full attention, 而是先用 indexer 选出 top-k token
    #   - 减少 attention 的 O(S²) 复杂度
    use_sparse_indexer: bool = False

    # index_topk: 索引器选择的 top-k token 数
    index_topk: int = 0

    # index_n_heads: 索引器的 head 数
    index_n_heads: int = 0

    # index_head_dim: 索引器每个 head 的维度
    index_head_dim: int = 0

    # ── FFN (Feed-Forward Network) 参数 ──────────────────────────────
    # mlp_dim: Dense FFN 层的中间维度
    #   - FFN 结构: emb_dim → mlp_dim → emb_dim
    #   - 通常是 emb_dim 的 2.5~4 倍
    #   - ALModel=5120, DeepSeek V3=18432
    mlp_dim: int = 5120

    # mlp_activations: FFN 的激活函数列表
    #   - ["silu", "linear"] = SwiGLU 激活 (Gate Linear Unit)
    #     这意味着 FFN 第一层有 2 个并行投影 (gate 和 up)
    #   - ["relu"] = 标准 ReLU, 只有 1 个投影
    #   - 数量影响 ffn1 的 FLOPs 计算 (乘以 len(mlp_activations))
    mlp_activations: List[str] = field(default_factory=lambda: ["silu", "linear"])

    # ── MoE (Mixture of Experts) 参数 ────────────────────────────────
    # num_experts: Expert 总数
    #   - 1 = Dense 模型 (无 MoE)
    #   - >1 = MoE 模型, 通常 8, 16, 64, 256
    #   - ALModel=256, DeepSeek V3=256
    num_experts: int = 256

    # num_experts_per_tok (top-k): 每个 token 激活的 expert 数
    #   - 路由器为每个 token 选择 top-k 个 expert 处理
    #   - 越大 → 每 token 计算量越多, 但质量可能更好
    #   - ALModel=8, DeepSeek V3=8
    num_experts_per_tok: int = 8

    # shared_experts: 共享 expert 数
    #   - 所有 token 都会经过的 "公共" expert
    #   - DeepSeek 架构特有: 除了 routed experts, 还有 shared expert
    #   - ALModel=1, DeepSeek V3=1
    shared_experts: int = 1

    # moe_mlp_dim: MoE 层中每个 expert 的 FFN 中间维度
    #   - 通常比 dense 层的 mlp_dim 小很多 (因为 expert 多)
    #   - ALModel=512, DeepSeek V3=2048
    moe_mlp_dim: int = 512

    # first_num_dense_layers: 模型前几层是 Dense (非 MoE) 层
    #   - DeepSeek 架构: 前 N 层用 Dense FFN, 后面用 MoE
    #   - ALModel=1, DeepSeek V3=3
    first_num_dense_layers: int = 1

    # ── 模型类型 ─────────────────────────────────────────────────────
    # decoder_block: 模型架构类型
    #   - "deepseek": DeepSeek 系列 (V2/V3)
    #   - "llama4": Meta Llama 4
    #   - "default": 通用 dense 模型 (GPT/Llama 2/3 等)
    decoder_block: str = "deepseek"

    # ── DPO (Direct Preference Optimization) 参数 ────────────────────
    # use_dpo: 是否使用 DPO 训练
    #   - DPO 需要额外一次 reference model 的 forward pass
    #   - 会增加 1/3 的 learnable_weight_tflops 作为 reference_model_tflops
    use_dpo: bool = False


# =============================================================================
# 第二部分: 辅助函数
# =============================================================================

def get_dense_moe_layers(config: ModelConfig) -> Tuple[int, int]:
    """
    计算 Dense 层和 MoE 层各有多少层

    DeepSeek 风格的 MoE 模型不是所有层都用 MoE:
    - 前 first_num_dense_layers 层用 Dense FFN (大维度, 完整计算)
    - 剩余层用 MoE FFN (小维度, 稀疏路由)

    为什么这样设计:
    - 底层特征更通用, 用 Dense 让所有 token 共享知识
    - 高层特征更专业化, 用 MoE 让不同 token 走不同 expert

    参数:
        config: 模型配置

    返回:
        (num_dense_layers, num_moe_layers): Dense 层数和 MoE 层数的元组

    示例 (ALModel):
        num_decoder_layers=20, first_num_dense_layers=1
        → dense=1, moe=19

    示例 (DeepSeek V3):
        num_decoder_layers=61, first_num_dense_layers=3
        → dense=3, moe=58
    """
    # Dense 层数 = 配置中指定的前 N 层
    num_dense_layers = config.first_num_dense_layers

    # MoE 层数 = 总层数 - Dense 层数
    num_moe_layers = config.num_decoder_layers - config.first_num_dense_layers

    return num_dense_layers, num_moe_layers


# =============================================================================
# 第三部分: FFN (Feed-Forward Network) FLOP 计算
# =============================================================================

def calculate_ffn_matmul_tflops_per_device(config: ModelConfig, mlp_dim: int) -> int:
    """
    计算单层 FFN 的矩阵乘法 FLOPs (不含 3x 训练乘子)

    FFN 结构 (以 SwiGLU 为例):
    ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
    │  input   │───>│  Gate (SiLU) │──┐                     │  output  │
    │ [B,S,E]  │    │ E → mlp_dim  │  ├─ element-wise ──>  │ [B,S,E]  │
    │          │───>│  Up (Linear) │──┘     multiply        │          │
    │          │    │ E → mlp_dim  │                         │          │
    └──────────┘    └──────────────┘    ┌──────────────┐    └──────────┘
                         ffn1           │    Down      │
                    (2 个并行投影)        │ mlp_dim → E  │
                                        └──────────────┘
                                             ffn2

    矩阵乘法 FLOPs 公式:
        [M×K] · [K×N] → FLOPs = 2 × M × K × N
        (其中 2 来自: 每个输出元素需要 K 次乘法 + K-1 次加法 ≈ 2K 次运算)

    参数:
        config: 模型配置
        mlp_dim: FFN 中间维度
                 - Dense 层传入 config.mlp_dim (较大)
                 - MoE 层传入 config.moe_mlp_dim (较小)

    返回:
        单层 FFN 的总 FLOPs (整数)
    """
    # ── FFN 第一层 (Gate + Up Projection) ──────────────────────────────
    # 矩阵乘法: [B*S, E] × [E, mlp_dim]
    # FLOPs = 2 × (B×S) × mlp_dim × E × num_parallel_projections
    #
    # len(mlp_activations) 决定并行投影数:
    #   - ["silu", "linear"] → 2 个 (Gate + Up, SwiGLU 风格)
    #   - ["relu"]           → 1 个 (标准 FFN)
    ffn1_flops = (
        2                                      # 矩阵乘法的 "2" (乘法+加法)
        * config.per_device_batch_size         # B: batch 大小
        * config.max_target_length             # S: 序列长度
        * mlp_dim                              # 输出维度 (FFN 中间维度)
        * config.emb_dim                       # 输入维度 (隐藏维度)
        * len(config.mlp_activations)          # 并行投影数 (SwiGLU=2)
    )

    # ── FFN 第二层 (Down Projection) ──────────────────────────────────
    # 矩阵乘法: [B*S, mlp_dim] × [mlp_dim, E]
    # FLOPs = 2 × (B×S) × mlp_dim × E
    ffn2_flops = (
        2                                      # 矩阵乘法的 "2"
        * config.per_device_batch_size         # B
        * config.max_target_length             # S
        * mlp_dim                              # 输入维度 (FFN 中间维度)
        * config.emb_dim                       # 输出维度 (隐藏维度)
    )

    # 单层 FFN 总 FLOPs = ffn1 + ffn2
    # 展开 (SwiGLU): 2*B*S*E*mlp_dim*2 + 2*B*S*mlp_dim*E = 6*B*S*E*mlp_dim
    return ffn1_flops + ffn2_flops


def calculate_routed_and_shared_ffn_tflops_per_device(config: ModelConfig) -> int:
    """
    计算 DeepSeek 风格 MoE FFN 的总 FLOPs (已包含层数!)

    MoE FFN 结构:
    ┌─────────────────────────────────────────────────────────────────┐
    │                          MoE Layer                              │
    │                                                                 │
    │  input ──> Router Gate ──> top-k selection                     │
    │    │          │                  │                               │
    │    │          │         ┌────────┴────────┐                     │
    │    │          │         │  Routed Experts  │ × top_k            │
    │    │          │         │  (moe_mlp_dim)   │                    │
    │    │          │         └────────┬────────┘                     │
    │    │          │                  │ combine                      │
    │    │          │                  v                               │
    │    └──────────┼──> Shared Expert(s) ──> + ──> output            │
    │               │    (moe_mlp_dim)                                │
    └─────────────────────────────────────────────────────────────────┘

    重要: 返回值已经乘过层数了!
      - Dense 层 FFN × dense 层数
      - MoE 层 FFN × moe 层数
    调用者不需要再乘 num_decoder_layers。

    参数:
        config: 模型配置

    返回:
        所有层的 FFN FLOPs 总和 (已含层数)
    """
    # ── Router Gate FLOPs ──────────────────────────────────────────────
    # 路由门控: 每个 token 要和所有 expert 计算相似度来选 top-k
    # 矩阵乘法: [B*S, E] × [E, num_experts] → [B*S, num_experts]
    # FLOPs = 2 × B × S × E × num_experts
    gate_flops = (
        2                                      # 矩阵乘法的 "2"
        * config.per_device_batch_size         # B
        * config.max_target_length             # S
        * config.emb_dim                       # E: 输入维度
        * config.num_experts                   # 输出维度 (expert 数)
    )

    # ── 获取 Dense 层和 MoE 层的数量 ──────────────────────────────────
    num_dense_layers, num_moe_layers = get_dense_moe_layers(config)

    # ── Dense 层 FFN FLOPs ─────────────────────────────────────────────
    # Dense 层使用较大的 mlp_dim (例如 ALModel: 5120)
    # 单层 FLOPs × Dense 层数
    dense_ffn_flops = (
        calculate_ffn_matmul_tflops_per_device(config, config.mlp_dim)
        * num_dense_layers   # 乘以 Dense 层数 (ALModel=1)
    )

    # ── Shared Expert FFN FLOPs ────────────────────────────────────────
    # Shared expert 使用较小的 moe_mlp_dim (例如 ALModel: 512)
    # 所有 token 都经过 shared expert → 实际就是一个小的 Dense FFN
    # 单层 FLOPs × shared_experts 数 (通常=1)
    shared_experts_flops = (
        calculate_ffn_matmul_tflops_per_device(config, config.moe_mlp_dim)
        * config.shared_experts  # shared expert 数量 (通常=1)
    )

    # ── Routed Expert FFN FLOPs ────────────────────────────────────────
    # 每个 token 选择 top-k 个 expert 处理
    # 单层 FLOPs × top_k (每 token 激活的 expert 数)
    # 注意: 这里不乘 num_experts, 因为每个 token 只激活 top_k 个
    routed_experts_flops = (
        calculate_ffn_matmul_tflops_per_device(config, config.moe_mlp_dim)
        * config.num_experts_per_tok  # top-k (ALModel=8)
    )

    # ── 每个 MoE 层的 FFN 总 FLOPs ────────────────────────────────────
    # = gate + shared + routed (单层)
    # 然后乘以 MoE 层数
    moe_ffn_flops = (
        (gate_flops + shared_experts_flops + routed_experts_flops)
        * num_moe_layers  # MoE 层数 (ALModel=19)
    )

    # ── 返回所有层的 FFN FLOPs 总和 ────────────────────────────────────
    # = Dense 层 FFN + MoE 层 FFN
    # ⚠️ 已包含层数, 调用者不要再乘 num_decoder_layers!
    total_ffn_flops = dense_ffn_flops + moe_ffn_flops
    return total_ffn_flops


# =============================================================================
# 第四部分: Attention FLOP 计算
# =============================================================================

def calculate_indexer_mask_ratio(index_topk: int, max_target_length: int) -> float:
    """
    计算稀疏索引器的 mask 比率 (稀疏注意力的有效计算比例)

    DeepSeek Lightning 的 Sparse Indexer 不做 full attention,
    而是每个 query 只看 top-k 个最相关的 key, 减少计算量。

    可视化 (T=8, K=4):
      Query ↓  Key →
      Q1 [X . . . . . . .]  ← 1 个 token (causal, 只能看自己)
      Q2 [X X . . . . . .]  ← 2 个 token
      Q3 [X X X . . . . .]  ← 3 个 token
      Q4 [X X X X . . . .]  ← 4 个 token (达到 K 上限)
      Q5 [X X X X . . . .]  ← 4 个 token (受 K 限制)
      Q6 [X X X X . . . .]  ← 4 个 token
      Q7 [X X X X . . . .]  ← 4 个 token
      Q8 [X X X X . . . .]  ← 4 个 token

    有效面积 = 三角形(1..K) + 矩形(K+1..T)
             = K²/2 + (T-K)×K = T×K - K²/2

    比率 = 有效面积 / 全 T² = K/T - 0.5×(K/T)²

    参数:
        index_topk: 每个 query 看的 top-k key 数
        max_target_length: 序列总长度

    返回:
        mask 比率 (0~1 之间的浮点数)
    """
    T = float(max_target_length)  # 序列总长度
    K = float(index_topk)         # top-k 限制

    ratio = K / T                          # K/T: 基础比率
    mask_multiplier = ratio - (0.5 * ratio**2)  # 减去重复计算的三角区域
    return mask_multiplier


def calculate_indexer_tflops_per_device(config: ModelConfig) -> Tuple[int, float]:
    """
    计算 DeepSeek Lightning Indexer 的 FLOPs

    Indexer 是一个轻量级注意力模块, 用于快速筛选重要 token:
    1. 投影: Q/K/HeadWeight 三个投影 (线性层)
    2. 评分: QK 点积 + head 聚合, 然后选 top-k

    参数:
        config: 模型配置

    返回:
        (proj_flops, scoring_flops): 投影和评分的 FLOPs
    """
    batch_len = config.per_device_batch_size * config.max_target_length  # B×S

    # ── 1. 投影 FLOPs ──────────────────────────────────────────────────

    # Query 投影: [B*S, q_lora_rank] × [q_lora_rank, index_n_heads × index_head_dim]
    # 注意: indexer 的 Q 从 q_lora_rank (不是 emb_dim) 投影
    q_flops = (
        2 * batch_len
        * config.q_lora_rank          # 输入维度
        * config.index_n_heads         # head 数
        * config.index_head_dim        # 每个 head 的维度
    )

    # Key 投影: [B*S, emb_dim] × [emb_dim, index_head_dim]
    # Key 是单头的, 所有 head 共享
    k_flops = (
        2 * batch_len
        * config.emb_dim               # 输入维度
        * config.index_head_dim        # 输出维度
    )

    # Head Weight 投影: [B*S, emb_dim] × [emb_dim, index_n_heads]
    # 用于加权聚合多个 head 的 score
    head_weight_flops = (
        2 * batch_len
        * config.emb_dim               # 输入维度
        * config.index_n_heads         # 输出维度
    )

    proj_flops = q_flops + k_flops + head_weight_flops

    # ── 2. 评分 FLOPs ──────────────────────────────────────────────────

    # QK 点积: [B, S, index_n_heads, index_head_dim] × [B, S, index_head_dim]
    # → [B, S, S, index_n_heads]
    qk_product_flops = (
        2 * batch_len
        * config.max_target_length     # S: 第二个序列维度
        * config.index_n_heads         # head 数
        * config.index_head_dim        # head 维度
    )

    # Head 聚合: [B, S, S, index_n_heads] × [B, S, index_n_heads]
    # → [B, S, S] (聚合所有 head 的 score)
    head_reduction_flops = (
        2 * batch_len
        * config.max_target_length     # S
        * config.index_n_heads         # head 数
    )

    # 除以 2: causal mask, 只有下三角有效
    scoring_flops = (qk_product_flops + head_reduction_flops) / 2

    return proj_flops, scoring_flops


def calculate_mla_tflops_per_device(
    config: ModelConfig,
) -> Tuple[int, float, int]:
    """
    计算 Multi-Head Latent Attention (MLA) 的单层 FLOPs

    MLA 是 DeepSeek 提出的注意力机制, 核心思想:
    用低秩投影 (类似 LoRA) 压缩 KV, 减少推理时 KV cache 的显存占用。

    标准 MHA:
        Q = X @ Wq     → [B, S, num_heads, head_dim]
        K = X @ Wk     → [B, S, num_heads, head_dim]
        V = X @ Wv     → [B, S, num_heads, head_dim]

    MLA:
        Q: X → down(E→lora_rank) → up(lora_rank→heads*dim)
        KV: X → down(E→kv_lora_rank) → up(kv_lora_rank→heads*(nope+v))
        另外: Q 和 K 各有一个 rope 部分做位置编码

    参数:
        config: 模型配置

    返回:
        (qkv_flops, attention_flops, projection_flops):
        - qkv_flops: QKV 投影的 FLOPs (per-layer)
        - attention_flops: Q·K^T + Attn·V 的 FLOPs (per-layer, 已含 causal mask)
        - projection_flops: 输出投影的 FLOPs (per-layer)
    """
    # batch_len = B × S: 每设备每步处理的总 token 数
    batch_len = config.per_device_batch_size * config.max_target_length

    # qk_head_dim_sum: Q/K 的总 head 维度 = nope + rope
    # nope 部分参与标准点积, rope 部分加上位置编码后参与点积
    qk_head_dim_sum = config.qk_nope_head_dim + config.qk_rope_head_dim
    # 例如 ALModel: 128 + 64 = 192

    # ── 1. Query 投影 FLOPs ────────────────────────────────────────────
    if config.q_lora_rank == 0:
        # 无 LoRA: 直接投影 E → num_heads × qk_dim
        # 矩阵乘法: [B*S, E] × [E, num_heads × qk_dim]
        q_flops = (
            2 * batch_len
            * config.emb_dim              # E: 输入维度
            * config.num_query_heads      # 输出: head 数
            * qk_head_dim_sum             # 输出: 每头维度
        )
    else:
        # 有 LoRA: 两步投影 (先压缩, 再展开)
        # Step 1 (Down): [B*S, E] × [E, q_lora_rank] → [B*S, q_lora_rank]
        # Step 2 (Up):   [B*S, q_lora_rank] × [q_lora_rank, num_heads × qk_dim]
        q_flops = (
            2 * batch_len
            * (
                config.emb_dim * config.q_lora_rank  # Down: E → lora_rank
                + config.q_lora_rank * config.num_query_heads * qk_head_dim_sum
                # Up: lora_rank → num_heads × qk_dim
            )
        )

    # ── 2. KV 投影 FLOPs ──────────────────────────────────────────────
    # KV 在 MLA 中总是用 LoRA 压缩 (这是 MLA 的核心)
    #
    # Step 1 (Down): X → compressed_kv
    #   [B*S, E] × [E, (kv_lora_rank + qk_rope_head_dim)]
    #   kv_lora_rank: 压缩后的 KV 维度 (存储在 KV cache 中的维度)
    #   qk_rope_head_dim: rope 部分单独投影 (不经过低秩压缩)
    #
    # Step 2 (Up): compressed_kv → K_nope + V
    #   [B*S, kv_lora_rank] × [kv_lora_rank, num_heads × (nope + v)]
    kv_flops = (
        2 * batch_len
        * (
            config.emb_dim * (config.kv_lora_rank + config.qk_rope_head_dim)
            # Down: E → (kv_lora_rank + rope_dim)
            + config.kv_lora_rank * config.num_query_heads
            * (config.qk_nope_head_dim + config.v_head_dim)
            # Up: kv_lora_rank → num_heads × (nope + v)
        )
    )

    # QKV 总 FLOPs = Q + KV
    qkv_flops = q_flops + kv_flops

    # ── 3. Attention 计算 FLOPs ────────────────────────────────────────
    if config.use_sparse_indexer and config.max_target_length > config.index_topk:
        # 稀疏索引器路径: 先用 indexer 选 top-k, 再做局部 attention
        indexer_proj_flops, indexer_scoring_flops = calculate_indexer_tflops_per_device(config)

        # indexer 的投影 FLOPs 加到 qkv_flops 中
        qkv_flops += indexer_proj_flops

        # 计算稀疏 mask 的有效比例
        multiplier = calculate_indexer_mask_ratio(
            config.index_topk, config.max_target_length
        )

        # 注意力 FLOPs: Q·K^T + Attn·V
        # 乘以 multiplier 而不是除以 2 (因为稀疏 mask 不同于简单 causal mask)
        attention_flops = (
            2 * batch_len
            * config.max_target_length       # S: key 序列长度
            * config.num_query_heads         # head 数
            * (qk_head_dim_sum + config.v_head_dim)  # QK 和 AV 的维度之和
            * multiplier                     # 稀疏比例
        )
        # 加上 indexer 自身的评分 FLOPs
        attention_flops += indexer_scoring_flops
    else:
        # 标准 MLA 路径 (或 sparse indexer 在短序列上 bypass)
        #
        # Q·K^T: [B, num_heads, S, qk_dim] × [B, num_heads, qk_dim, S]
        #   → [B, num_heads, S, S]  FLOPs = 2 × B×S × S × num_heads × qk_dim
        #
        # Attn·V: [B, num_heads, S, S] × [B, num_heads, S, v_dim]
        #   → [B, num_heads, S, v_dim]  FLOPs = 2 × B×S × S × num_heads × v_dim
        #
        # 合并: 2 × B×S × S × num_heads × (qk_dim + v_dim)
        attention_flops = (
            2 * batch_len
            * config.max_target_length       # S
            * config.num_query_heads         # head 数
            * (qk_head_dim_sum + config.v_head_dim)  # QK_dim + V_dim
        )

        # 除以 2: causal mask 使 attention 矩阵只有下三角有效
        # 实际有效计算量约为 full attention 的一半
        attention_flops = attention_flops / 2

    # ── 4. 输出投影 FLOPs ──────────────────────────────────────────────
    # [B*S, num_heads, v_head_dim] → [B*S, E]
    # 矩阵乘法: [B*S, num_heads × v_head_dim] × [num_heads × v_head_dim, E]
    projection_flops = (
        2 * batch_len
        * config.emb_dim                     # E: 输出维度
        * config.num_query_heads             # head 数
        * config.v_head_dim                  # 每头维度
    )

    return qkv_flops, attention_flops, projection_flops


def calculate_standard_mha_flops(config: ModelConfig) -> Tuple[int, float, int]:
    """
    计算标准 Multi-Head Attention (MHA) 的单层 FLOPs

    标准 MHA 结构:
        Q = X @ Wq    [B,S,E] → [B,S,num_q_heads × head_dim]
        K = X @ Wk    [B,S,E] → [B,S,num_kv_heads × head_dim]
        V = X @ Wv    [B,S,E] → [B,S,num_kv_heads × head_dim]
        Attn = softmax(Q @ K^T / √d) @ V
        Out = Attn @ Wo

    支持 GQA (Grouped Query Attention): num_kv_heads < num_query_heads

    参数:
        config: 模型配置

    返回:
        (qkv_flops, causal_attention_flops, projection_flops)
    """
    B = config.per_device_batch_size
    S = config.max_target_length
    E = config.emb_dim

    # ── QKV 投影 FLOPs ────────────────────────────────────────────────
    # Q: [B*S, E] × [E, num_q_heads × head_dim]
    # K: [B*S, E] × [E, num_kv_heads × head_dim]
    # V: [B*S, E] × [E, num_kv_heads × head_dim]
    # 合并: 2 × B × S × E × (num_q_heads + 2 × num_kv_heads) × head_dim
    qkv_flops = (
        2 * B * S * E
        * (config.num_query_heads + 2 * config.num_kv_heads)
        * config.head_dim
    )

    # ── Attention 计算 FLOPs ──────────────────────────────────────────
    # Q·K^T: [B, num_q_heads, S, head_dim] × [B, num_q_heads, head_dim, S]
    #   FLOPs = 2 × B × S² × num_q_heads × head_dim
    # Attn·V: [B, num_q_heads, S, S] × [B, num_q_heads, S, head_dim]
    #   FLOPs = 2 × B × S² × num_q_heads × head_dim
    # 合并: 4 × B × S² × num_q_heads × head_dim
    noncausal_attention_flops = (
        4 * B * S**2 * config.num_query_heads * config.head_dim
    )

    # causal mask → 除以 2 (只有下三角有效)
    causal_attention_flops = noncausal_attention_flops / 2

    # ── 输出投影 FLOPs ────────────────────────────────────────────────
    # [B*S, num_q_heads × head_dim] × [num_q_heads × head_dim, E]
    projection_flops = (
        2 * B * S * E * config.num_query_heads * config.head_dim
    )

    return qkv_flops, causal_attention_flops, projection_flops


# =============================================================================
# 第五部分: 主函数 — 汇总计算 TFLOP
# =============================================================================

def calculate_tflops_training_per_device(
    config: ModelConfig, log: bool = True
) -> Tuple[float, float, float]:
    """
    计算每设备每训练步的总 TFLOP (核心主函数)

    计算流程:
    ┌───────────────────────────────────────────────────────────────┐
    │ 1. FFN FLOPs    →  根据 Dense/MoE 类型选择计算方式           │
    │ 2. Attention FLOPs → 根据 MHA/MLA 类型选择计算方式           │
    │ 3. Embedding FLOPs → 固定公式 2*B*S*E*V                     │
    │ 4. 层聚合 → 乘以层数, 加上 3x 训练乘子                       │
    │ 5. 梯度累积 → 乘以 gradient_accumulation_steps               │
    │ 6. DPO → 如果使用, 加 1/3 的 reference model 前向传播        │
    └───────────────────────────────────────────────────────────────┘

    3x 训练乘子的由来:
        - Forward pass:  1x FLOPs (计算输出和 loss)
        - Backward pass: 2x FLOPs (计算梯度, 约为 forward 的 2 倍)
        - 总计: 3x FLOPs

    参数:
        config: 模型配置
        log: 是否打印计算结果

    返回:
        (total_tflops, learnable_weight_tflops, attention_tflops):
        - total_tflops: 总 TFLOP 数
        - learnable_weight_tflops: 可学习权重矩阵乘法的 TFLOP (FFN + QKV + Proj + Emb)
        - attention_tflops: 注意力矩阵运算的 TFLOP (Q·K^T + Attn·V)
    """

    # =====================================================================
    # Step 1: 计算 FFN FLOPs
    # =====================================================================
    if config.num_experts > 1:
        # ── MoE 模型 ──────────────────────────────────────────────────
        if config.decoder_block == "deepseek":
            # DeepSeek 风格: 有 dense 层 + MoE 层的混合结构
            # ⚠️ 返回值已包含层数!
            total_ffn_flops = calculate_routed_and_shared_ffn_tflops_per_device(config)
        else:
            # 通用 MoE: 所有层都是 MoE, 没有 dense/MoE 混合
            gate_flops = (
                2 * config.per_device_batch_size
                * config.max_target_length
                * config.emb_dim
                * config.num_experts
            )
            total_ffn_flops = (
                gate_flops
                + calculate_ffn_matmul_tflops_per_device(config, config.mlp_dim)
                * config.num_experts_per_tok
            )
            # ⚠️ 通用路径: 返回的是 单层 FLOPs, 需要在聚合时乘 L
    else:
        # ── Dense 模型 (无 MoE) ────────────────────────────────────────
        # 所有层的 FFN 结构相同
        # ⚠️ 返回的是 单层 FLOPs, 需要在聚合时乘 L
        total_ffn_flops = calculate_ffn_matmul_tflops_per_device(
            config, config.mlp_dim
        )

    # =====================================================================
    # Step 2: 计算 Attention FLOPs (单层)
    # =====================================================================
    if config.attention_type == "mla":
        # MLA (Multi-Head Latent Attention) — DeepSeek 系列
        qkv_flops, causal_attention_flops, projection_flops = (
            calculate_mla_tflops_per_device(config)
        )
    else:
        # 标准 MHA (Multi-Head Attention)
        qkv_flops, causal_attention_flops, projection_flops = (
            calculate_standard_mha_flops(config)
        )

    # =====================================================================
    # Step 3: 计算 Embedding FLOPs
    # =====================================================================
    # Embedding lookup 等效于矩阵乘法:
    #   输入: one-hot [B*S, V] × embedding_table [V, E] → [B*S, E]
    #   FLOPs = 2 × B × S × E × V
    # 这里同时包含了:
    #   - 输入 embedding (token → hidden)
    #   - 输出 logits (hidden → vocab, 权重通常共享)
    embedding_flops = (
        2 * config.per_device_batch_size
        * config.max_target_length
        * config.emb_dim
        * config.vocab_size
    )

    # =====================================================================
    # Step 4: 层聚合 — 乘以层数 + 3x 训练乘子
    # =====================================================================
    L = config.num_decoder_layers  # 总层数

    if config.decoder_block == "deepseek":
        # ── DeepSeek 路径 (✅ 正确处理 MoE 层数) ──────────────────────
        # total_ffn_flops 已经包含层数 (来自 calculate_routed_and_shared_ffn_tflops)
        # qkv_flops 和 projection_flops 是单层的, 需要乘 L
        # embedding_flops 只有 1 层, 不乘 L
        learnable_weight_tflops = (
            (
                total_ffn_flops                            # 已含层数!
                + (qkv_flops + projection_flops) * L       # 单层 × 总层数
                + embedding_flops                          # 只有 1 层
            )
            * 3      # 训练乘子: forward(1x) + backward(2x) = 3x
            / 1e12   # 转换为 TFLOP (10^12)
        )
        attention_tflops = (
            causal_attention_flops * L  # 单层 × 总层数
            * 3      # 训练乘子
            / 1e12   # → TFLOP
        )
    else:
        # ── 通用路径 (适用于 Dense 模型) ───────────────────────────────
        # total_ffn_flops 是单层的, 和 qkv/projection 一起乘 L
        # ⚠️ 如果 MoE 模型走了这条路径, total_ffn_flops 会被多乘一次 L!
        #    (因为 calculate_routed_and_shared 已经内含层数)
        learnable_weight_tflops = (
            (
                (total_ffn_flops + qkv_flops + projection_flops) * L  # 全部 × L
                + embedding_flops
            )
            * 3 / 1e12
        )
        attention_tflops = (
            causal_attention_flops * L * 3 / 1e12
        )

    # =====================================================================
    # Step 5: 梯度累积乘子
    # =====================================================================
    # gradient_accumulation_steps > 1 时, 每次参数更新包含多步的计算
    # 所以 TFLOP 要乘以这个值
    learnable_weight_tflops *= config.gradient_accumulation_steps
    attention_tflops *= config.gradient_accumulation_steps

    # =====================================================================
    # Step 6: DPO (Direct Preference Optimization) 额外计算
    # =====================================================================
    if config.use_dpo:
        # DPO 需要一个 frozen reference model 做额外一次 forward pass
        # forward = 训练总量的 1/3 (因为训练=3x forward)
        reference_model_tflops = learnable_weight_tflops / 3
        reference_model_attention_tflops = attention_tflops / 3
        attention_tflops += reference_model_attention_tflops
    else:
        reference_model_tflops = 0

    # =====================================================================
    # 汇总
    # =====================================================================
    total_tflops = learnable_weight_tflops + attention_tflops + reference_model_tflops

    if log:
        print(f"\n{'='*60}")
        print(f"TFLOP 计算结果 (per device, per train step)")
        print(f"{'='*60}")
        print(f"  Learnable Weight TFLOPs: {learnable_weight_tflops:>12.2f}  "
              f"({100*learnable_weight_tflops/total_tflops:.1f}%)")
        print(f"  Attention TFLOPs:        {attention_tflops:>12.2f}  "
              f"({100*attention_tflops/total_tflops:.1f}%)")
        if reference_model_tflops > 0:
            print(f"  Reference Model TFLOPs:  {reference_model_tflops:>12.2f}  "
                  f"({100*reference_model_tflops/total_tflops:.1f}%)")
        print(f"  {'─'*40}")
        print(f"  Total TFLOPs:            {total_tflops:>12.2f}")
        print(f"{'='*60}")

    return total_tflops, learnable_weight_tflops, attention_tflops


# =============================================================================
# 第六部分: TFLOP/s 计算 (模拟 metric_logger.py 的调用方式)
# =============================================================================

def calculate_tflops_per_second(total_tflops: float, step_time_seconds: float) -> float:
    """
    计算每秒 TFLOP (训练速度指标)

    这是 MaxText 在每个训练 step 结束后做的计算:
        TFLOP/s/device = total_tflops / step_time

    total_tflops 在初始化时算一次 (常量), 之后每步只做一次除法。
    如果公式有 bug, 所有 step 的 TFLOP/s 都会系统性偏移。

    参数:
        total_tflops: 每步每设备的 TFLOP 数 (由 calculate_tflops_training_per_device 计算)
        step_time_seconds: 单步训练时间 (秒)

    返回:
        TFLOP/s/device (每秒每设备的万亿次浮点运算)
    """
    return total_tflops / step_time_seconds


def calculate_tokens_per_second(config: ModelConfig, step_time_seconds: float) -> float:
    """
    计算每秒处理的 token 数 (吞吐量指标)

    这是实测值, 不受 TFLOP 计算 bug 影响。

    参数:
        config: 模型配置
        step_time_seconds: 单步训练时间 (秒)

    返回:
        tokens/s/device
    """
    tokens_per_step = (
        config.per_device_batch_size
        * config.max_target_length
        * config.gradient_accumulation_steps
    )
    return tokens_per_step / step_time_seconds


# =============================================================================
# 第七部分: 预设模型配置
# =============================================================================

def get_almodel_config() -> ModelConfig:
    """ALModel 8B 训练配置 (32-chip TPU v7 benchmark)"""
    return ModelConfig(
        per_device_batch_size=12,
        max_target_length=4096,
        gradient_accumulation_steps=2,
        emb_dim=2048,
        vocab_size=157184,
        num_decoder_layers=20,
        attention_type="mla",
        num_query_heads=16,
        num_kv_heads=16,
        head_dim=128,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mlp_dim=5120,
        mlp_activations=["silu", "linear"],
        num_experts=256,
        num_experts_per_tok=8,
        shared_experts=1,
        moe_mlp_dim=512,
        first_num_dense_layers=1,
        decoder_block="deepseek",
        use_dpo=False,
    )


def get_deepseek_v3_config() -> ModelConfig:
    """DeepSeek V3 671B 参考配置"""
    return ModelConfig(
        per_device_batch_size=1,
        max_target_length=4096,
        gradient_accumulation_steps=1,
        emb_dim=7168,
        vocab_size=129280,
        num_decoder_layers=61,
        attention_type="mla",
        num_query_heads=128,
        num_kv_heads=128,
        head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mlp_dim=18432,
        mlp_activations=["silu", "linear"],
        num_experts=256,
        num_experts_per_tok=8,
        shared_experts=1,
        moe_mlp_dim=2048,
        first_num_dense_layers=3,
        decoder_block="deepseek",
        use_dpo=False,
    )


def get_llama_70b_config() -> ModelConfig:
    """Llama 3 70B Dense 模型参考配置"""
    return ModelConfig(
        per_device_batch_size=4,
        max_target_length=4096,
        gradient_accumulation_steps=1,
        emb_dim=8192,
        vocab_size=128256,
        num_decoder_layers=80,
        attention_type="mha",
        num_query_heads=64,
        num_kv_heads=8,  # GQA: 8 KV heads
        head_dim=128,
        mlp_dim=28672,
        mlp_activations=["silu", "linear"],
        num_experts=1,  # Dense 模型
        decoder_block="default",
        use_dpo=False,
    )


# =============================================================================
# 第八部分: 详细分步输出 (教学用)
# =============================================================================

def detailed_breakdown(config: ModelConfig):
    """
    详细分步输出每个子计算的结果, 便于理解和验证

    参数:
        config: 模型配置
    """
    B = config.per_device_batch_size
    S = config.max_target_length
    E = config.emb_dim
    V = config.vocab_size
    L = config.num_decoder_layers

    print(f"\n{'='*70}")
    print(f"  详细分步计算 (教学模式)")
    print(f"{'='*70}")
    print(f"\n  基础参数:")
    print(f"    B (batch)      = {B}")
    print(f"    S (seq_len)    = {S}")
    print(f"    E (emb_dim)    = {E}")
    print(f"    V (vocab_size) = {V}")
    print(f"    L (layers)     = {L}")

    # ── FFN ──────────────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  [1] FFN FLOPs")
    print(f"  {'─'*50}")

    if config.num_experts > 1 and config.decoder_block == "deepseek":
        num_dense, num_moe = get_dense_moe_layers(config)
        print(f"    MoE 模型: {num_dense} Dense 层 + {num_moe} MoE 层")

        single_dense_ffn = calculate_ffn_matmul_tflops_per_device(config, config.mlp_dim)
        single_moe_ffn = calculate_ffn_matmul_tflops_per_device(config, config.moe_mlp_dim)
        gate = 2 * B * S * E * config.num_experts

        print(f"\n    单层 Dense FFN (mlp_dim={config.mlp_dim}):")
        print(f"      = 6 × B × S × E × mlp_dim")
        print(f"      = 6 × {B} × {S} × {E} × {config.mlp_dim}")
        print(f"      = {single_dense_ffn:,.0f} FLOPs")

        print(f"\n    单层 MoE FFN (moe_mlp_dim={config.moe_mlp_dim}):")
        print(f"      = 6 × B × S × E × moe_mlp_dim")
        print(f"      = 6 × {B} × {S} × {E} × {config.moe_mlp_dim}")
        print(f"      = {single_moe_ffn:,.0f} FLOPs")

        print(f"\n    Router Gate: 2 × B × S × E × num_experts")
        print(f"      = 2 × {B} × {S} × {E} × {config.num_experts}")
        print(f"      = {gate:,.0f} FLOPs")

        dense_total = single_dense_ffn * num_dense
        shared_total = single_moe_ffn * config.shared_experts
        routed_total = single_moe_ffn * config.num_experts_per_tok
        moe_per_layer = gate + shared_total + routed_total
        moe_total = moe_per_layer * num_moe
        total_ffn = dense_total + moe_total

        print(f"\n    Dense FFN total = {single_dense_ffn:,.0f} × {num_dense} = {dense_total:,.0f}")
        print(f"    Shared expert  = {single_moe_ffn:,.0f} × {config.shared_experts} = {shared_total:,.0f}")
        print(f"    Routed experts = {single_moe_ffn:,.0f} × {config.num_experts_per_tok} = {routed_total:,.0f}")
        print(f"    MoE/layer      = {gate:,.0f} + {shared_total:,.0f} + {routed_total:,.0f} = {moe_per_layer:,.0f}")
        print(f"    MoE total      = {moe_per_layer:,.0f} × {num_moe} = {moe_total:,.0f}")
        print(f"    ──────────────")
        print(f"    FFN total      = {total_ffn:,.0f} FLOPs (已含层数)")
    else:
        single_ffn = calculate_ffn_matmul_tflops_per_device(config, config.mlp_dim)
        print(f"    Dense 模型: 单层 FFN = {single_ffn:,.0f} FLOPs")
        print(f"    (聚合时再乘 L={L})")

    # ── Attention ────────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  [2] Attention FLOPs (per-layer)")
    print(f"  {'─'*50}")

    if config.attention_type == "mla":
        qkv, attn, proj = calculate_mla_tflops_per_device(config)
        print(f"    MLA (Multi-Head Latent Attention)")
        print(f"    QKV 投影 (per-layer):      {qkv:>20,.0f} FLOPs")
        print(f"    Attention (per-layer):      {attn:>20,.0f} FLOPs (已 causal/2)")
        print(f"    Output 投影 (per-layer):    {proj:>20,.0f} FLOPs")
    else:
        qkv, attn, proj = calculate_standard_mha_flops(config)
        print(f"    标准 MHA")
        print(f"    QKV 投影 (per-layer):      {qkv:>20,.0f} FLOPs")
        print(f"    Attention (per-layer):      {attn:>20,.0f} FLOPs (已 causal/2)")
        print(f"    Output 投影 (per-layer):    {proj:>20,.0f} FLOPs")

    # ── Embedding ────────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  [3] Embedding FLOPs")
    print(f"  {'─'*50}")

    emb_flops = 2 * B * S * E * V
    print(f"    = 2 × B × S × E × V")
    print(f"    = 2 × {B} × {S} × {E} × {V}")
    print(f"    = {emb_flops:,.0f} FLOPs")

    # ── 汇总 ────────────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  [4] 层聚合 + 训练乘子")
    print(f"  {'─'*50}")

    total, weight, attention = calculate_tflops_training_per_device(config, log=False)

    print(f"    Learnable Weight:  {weight:.2f} TFLOP")
    print(f"    Attention:         {attention:.2f} TFLOP")
    print(f"    Total:             {total:.2f} TFLOP")

    if config.gradient_accumulation_steps > 1:
        print(f"    (含 gradient_accumulation_steps={config.gradient_accumulation_steps} 的乘子)")

    # ── 模拟 TFLOP/s ────────────────────────────────────────────────
    print(f"\n  {'─'*50}")
    print(f"  [5] TFLOP/s 示例")
    print(f"  {'─'*50}")

    for step_time in [5.0, 9.87, 15.0]:
        tflops_per_sec = total / step_time
        tokens_per_sec = calculate_tokens_per_second(config, step_time)
        print(f"    step_time={step_time:>5.1f}s → "
              f"TFLOP/s={tflops_per_sec:>7.1f}, "
              f"tokens/s={tokens_per_sec:>8.0f}")

    print(f"\n{'='*70}\n")


# =============================================================================
# 第九部分: 命令行入口
# =============================================================================

if __name__ == "__main__":
    import sys

    print("MaxText TFLOP/s 计算器 — 独立提取版")
    print("=" * 50)

    if "--interactive" in sys.argv:
        # 交互模式: 让用户输入参数
        print("\n请输入模型参数 (直接回车使用默认值):\n")
        config = ModelConfig()

        def ask_int(prompt, default):
            val = input(f"  {prompt} [{default}]: ").strip()
            return int(val) if val else default

        def ask_str(prompt, default):
            val = input(f"  {prompt} [{default}]: ").strip()
            return val if val else default

        config.per_device_batch_size = ask_int("per_device_batch_size", 12)
        config.max_target_length = ask_int("max_target_length", 4096)
        config.gradient_accumulation_steps = ask_int("gradient_accumulation_steps", 1)
        config.emb_dim = ask_int("emb_dim", 2048)
        config.vocab_size = ask_int("vocab_size", 157184)
        config.num_decoder_layers = ask_int("num_decoder_layers", 20)
        config.attention_type = ask_str("attention_type (mha/mla)", "mla")
        config.num_query_heads = ask_int("num_query_heads", 16)
        config.num_kv_heads = ask_int("num_kv_heads", 16)
        config.head_dim = ask_int("head_dim", 128)
        config.num_experts = ask_int("num_experts (1=Dense)", 1)

        if config.num_experts > 1:
            config.decoder_block = "deepseek"
            config.num_experts_per_tok = ask_int("num_experts_per_tok", 8)
            config.shared_experts = ask_int("shared_experts", 1)
            config.moe_mlp_dim = ask_int("moe_mlp_dim", 512)
            config.first_num_dense_layers = ask_int("first_num_dense_layers", 1)
        else:
            config.decoder_block = "default"

        config.mlp_dim = ask_int("mlp_dim", 5120)

        if config.attention_type == "mla":
            config.q_lora_rank = ask_int("q_lora_rank", 256)
            config.kv_lora_rank = ask_int("kv_lora_rank", 512)

        step_time = float(input(f"\n  step_time (秒) [9.87]: ").strip() or "9.87")

        detailed_breakdown(config)
        total, _, _ = calculate_tflops_training_per_device(config, log=False)
        print(f"\n  结果: {total:.2f} TFLOP / {step_time}s = "
              f"{total/step_time:.1f} TFLOP/s/device\n")

    else:
        # 默认模式: 展示三个预设模型
        models = [
            ("ALModel 8B (MoE)", get_almodel_config()),
            ("DeepSeek V3 671B (MoE)", get_deepseek_v3_config()),
            ("Llama 3 70B (Dense)", get_llama_70b_config()),
        ]

        for name, cfg in models:
            print(f"\n{'#'*60}")
            print(f"  模型: {name}")
            print(f"{'#'*60}")
            detailed_breakdown(cfg)
