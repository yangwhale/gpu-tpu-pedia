#!/usr/bin/env python3
"""Hy3-295B SFT —— 走 Megatron-Bridge，加载官方权重做稀缺知识注入。

与预训练脚本 hy3_pretrain.py 的根本区别：
    hy3_pretrain.py 借 Qwen3-235B 的 recipe 骨架 + 手工覆写超参，造的是**随机初始化**模型。
    跑性能可以（算力只取决于形状），但没法加载官方权重。
    本脚本走 HYV3Bridge（见 README §14），从 `tencent/Hy3` 的真实 checkpoint 出发。

前置条件：
    1. 容器内已装 HYV3Bridge  →  ./install_hy3_bridge.sh <pods...>
    2. HF 权重已转成 Megatron torch_dist  →  ./import_hy3_ckpt.py
    3. 数据集已生成               →  ./make_sft_data.py

用法（在 torchrun 里跑）：
    python hy3_sft.py --pretrained /ckpt/hy3-megatron --data /data/sft_data \\
        --num-gpus 64 --pp 2 --ep 16 --epochs 10
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import torch

from megatron.bridge import AutoBridge
from megatron.bridge.recipes.common import _sft_common
from megatron.bridge.data.builders.hf_dataset import HFDatasetConfig
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.gpt_step import forward_step

HF_MODEL = "tencent/Hy3"          # instruct 版：已有 chat_template，SFT 前后对比更干净
HF_BASE = "tencent/Hy3-Base"      # 底座版：无 chat_template，需连格式一起教



def _ensure_hy3_bridge():
    """确保 HYV3Bridge 已注册。

    容器里的安装（install_hy3_bridge.sh 写进 site-packages）会在 pod 重建时丢失，
    所以这里加一条自愈路径：/raid/pylib 在本地 NVMe 上，跨 pod 重启存活。
    注册靠的是 @register_bridge 装饰器在 import 时执行，不依赖 models/__init__.py 的补丁。
    """
    try:
        from megatron.bridge.models.hy_v3 import HYV3Bridge  # noqa: F401
        return
    except ImportError:
        pass
    import sys
    sys.path.insert(0, "/raid/pylib")
    from hy_v3.hy_v3_bridge import HYV3Bridge  # noqa: F401

def passthrough_messages(example: dict[str, Any], tokenizer=None) -> dict[str, Any]:
    """数据已是官方 ChatML `messages` 格式，无需转换。

    Bridge 的 SFT dataset 在 `chat=True, use_hf_tokenizer_chat_template=True` 下
    会直接调 tokenizer.apply_chat_template，所以原样透传即可。
    """
    return {"messages": example["messages"]}



class ExportHFAtEnd:
    """训练结束时直接从显存里的模型存 HF 格式。

    为什么必须这样，而不是训完再单独跑一次导出：
        SFT checkpoint 是 64 rank 各写各的，而 /raid 是 node-local ——
        每个节点只有自己那 4 个 `__N_0.distcp`。
        重新加载时 torch_dist 会做全局 sharding 校验，且**任意 rank 可能读任意分片**
        （实测 rank 3 去读 `__32_0.distcp`，那片在 yw-a-8 上），
        所以「各读各的本地分片」根本不成立，重新加载必然失败。
        把 3.8 TB（含优化器状态）汇集到单机再导出代价太大。

        而在 on_train_end 时模型还在显存里，`save_hf_pretrained` 自己做跨 rank 聚合，
        rank 0 直接落 HF safetensors（仅权重 ~597 GB）—— 一步到位，不经过磁盘往返。
    """

    def __init__(self, out: str, hf_src: str):
        self.out, self.hf_src = out, hf_src

    def on_train_end(self, context):
        import time
        import torch.distributed as dist
        from megatron.bridge import AutoBridge

        rank = dist.get_rank() if dist.is_initialized() else 0
        t0 = time.time()
        if rank == 0:
            print(f"[export-hf] 训练结束，开始导出 → {self.out}", flush=True)
        bridge = AutoBridge.from_hf_pretrained(self.hf_src, trust_remote_code=True)
        bridge.save_hf_pretrained(context.model, self.out, show_progress=(rank == 0),
                                  source_path=self.hf_src, strict=False)
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            print(f"[export-hf] 完成，耗时 {time.time()-t0:.0f}s", flush=True)


def build_config(a):
    _ensure_hy3_bridge()
    cfg = _sft_common()

    hf_path = HF_BASE if a.base else HF_MODEL
    cfg.model = AutoBridge.from_hf_pretrained(hf_path, trust_remote_code=True).to_megatron_provider(
        load_weights=False
    )
    cfg.tokenizer.tokenizer_type = "HuggingFaceTokenizer"
    # 用打过补丁的本地 tokenizer，不用 hub 上的原版。
    # 原因：Hy3 官方 chat_template.jinja 没有 {% generation %} 块，
    # HF 的 apply_chat_template(return_assistant_tokens_mask=True) 会直接报错，
    # 于是 Bridge 算不出 assistant-only 的 loss mask。
    # 补丁把 assistant 段（含 think / eos）包进 generation 块，实测 mask 正确：
    # 34 token 中 9 个计 loss，user/system 段被正确屏蔽。（见 SFT.md §6.12）
    cfg.tokenizer.tokenizer_model = a.tokenizer
    cfg.tokenizer.hf_tokenizer_kwargs = {"trust_remote_code": True}

    m = cfg.model
    # ---- 序列长度 ----
    # 样本平均 ~110 token，512 足够装下最长的一条。
    # 不用 2048/4096：短序列省算力、省显存，还能把 MBS 开大。
    # 代价是不训练长上下文能力 —— 本实验不需要。
    m.seq_length = a.seq_length
    cfg.dataset.seq_length = a.seq_length

    # ---- 并行 ----
    # 沿用预训练验证过的最优骨架（README §10）：TP=1 靠 EP 扛专家，PP 切流水。
    # SFT 的 GBS 比预训练小两个数量级，DP 维度自然变窄，不再需要 VPP 那么多 chunk。
    m.tensor_model_parallel_size = a.tp
    m.pipeline_model_parallel_size = a.pp
    m.virtual_pipeline_model_parallel_size = a.vpp
    m.expert_model_parallel_size = a.ep
    m.expert_tensor_parallel_size = 1          # Parallel Folding，专家不切 TP
    m.context_parallel_size = 1
    m.sequence_parallel = a.tp > 1
    m.pipeline_dtype = torch.bfloat16
    m.pipeline_model_parallel_layout = None

    # ---- MTP ----
    # 官方 checkpoint 带 1 层 MTP（layer 80，3.8 B 参数）。
    # 必须保持 =1，否则那批权重在加载时无处安放。
    # （性能扫点时设 0 是为了隔离变量，此处不适用。）
    m.mtp_num_layers = 1

    # ---- MoE kernel ----
    # 这里**不沿用**预训练的 flex/hybridep 冠军配置，改用参考实现 alltoall。
    # 原因：SFT 的 micro-batch 只有 512 token（预训练是 4096），
    # 乘 top-8 摊到 192 个专家上平均每专家仅 ~21 token，必然有专家分到 0 个。
    # 实测 hybridep 路径在第 16 步于单个 rank 上抛
    # "found NaN in local grad norm for bucket #0"，该 rank 退出后其余 60 rank
    # 在集合通信里空等，表现为 GPU 100% 却零推进。（见 SFT.md §6.13）
    # alltoall 是参考实现，对零 token 专家的处理更稳。
    m.moe_token_dispatcher_type = "alltoall"
    m.moe_grouped_gemm = True
    m.moe_permute_fusion = False               # 实测 0 增益，SFT 下先排除变量
    m.moe_router_fusion = False
    m.moe_shared_expert_overlap = False
    m.cross_entropy_loss_fusion = True

    # ---- CUDA graph ----
    # SFT 用 none：数据是变长的、步数少，graph capture 的固定开销（~26 s）
    # 换不回收益，而且 full_iteration 那条依赖链（§10）会限制 batch 灵活性。
    m.cuda_graph_impl = "none"

    # ---- 精度 ----
    # 默认 BF16。FP8 在预训练上已验证与 BF16 对齐（§12），
    # 但 SFT 只跑几百步、追求的是权重的精细调整，不值得为省时间引入额外变量。
    cfg.mixed_precision = "bf16_mixed" if a.precision == "bf16" else a.precision
    m.recompute_granularity = None
    m.recompute_modules = None
    m.offload_modules = None

    # ---- 数据 ----
    # dataset_name="json" 让 datasets 走本地 jsonl 加载器。
    cfg.dataset = HFDatasetConfig(
        dataset_name="json",
        process_example_fn=passthrough_messages,
        seq_length=a.seq_length,
        seed=5678,
        dataloader_type="batch",
        num_workers=1,
        do_validation=True,
        do_test=False,
        val_proportion=0.05,
        dataset_root=os.path.join(a.data, "processed"),
        hf_kwargs={"data_files": {"train": os.path.join(a.data, "train.jsonl")}},
        # chat=True 让 dataset 用 tokenizer 的 chat_template 拼 prompt，
        # 并且只在 assistant 段计 loss（user/system 段被 mask 掉）。
        dataset_kwargs={"chat": True, "use_hf_tokenizer_chat_template": True},
        # 不打包：627 条样本若打包成 4096 只剩 13 个序列，
        # 且会把互不相关的事实塞进同一 attention 窗口互相干扰。
        packed_sequence_specs=None,
        # rewrite=False 且预处理产物已分发到每个节点：
        # Bridge 假设 dataset_root 在共享文件系统上——只有 global rank 0 生成
        # processed/*.jsonl 和索引。我们的 /raid 是 node-local，其余 15 个节点拿不到，
        # 会走进不同的代码路径，导致 rank 0 停在 barrier() 而别的 rank 已经到了
        # broadcast()，集合通信错位死锁。（实测踩过，见 SFT.md §6.11）
        rewrite=False,
    )

    # ---- 训练步数 ----
    # 知识注入的有效剂量取决于「每个事实被看到多少次」，不是总 token 数。
    # 627 样本 / GBS → 每 epoch 的步数；跑 epochs 轮。
    steps_per_epoch = max(1, a.train_samples // a.gbs)
    cfg.train.train_iters = steps_per_epoch * a.epochs
    cfg.train.global_batch_size = a.gbs
    cfg.train.micro_batch_size = a.mbs
    cfg.validation.eval_interval = max(10, cfg.train.train_iters // 5)
    cfg.validation.eval_iters = 4

    # ---- 优化器 ----
    # 5e-6 是 Bridge 对通用 SFT 的默认值，对「注入全新事实」偏保守。
    # 1e-5 在少量步数内更容易把知识写进去，同时仍远低于预训练 LR，
    # 不至于摧毁 instruct 版已有的对话/推理能力。probe.jsonl 就是用来验这一点的。
    cfg.scheduler.max_lr = a.lr
    cfg.scheduler.min_lr = a.lr * 0.1
    cfg.scheduler.lr_warmup_iters = max(5, cfg.train.train_iters // 10)
    cfg.optimizer.adam_beta2 = 0.98

    # ---- checkpoint ----
    cfg.checkpoint.pretrained_checkpoint = a.pretrained
    cfg.checkpoint.save = a.save
    cfg.checkpoint.save_interval = max(20, cfg.train.train_iters // 2)
    cfg.checkpoint.ckpt_format = "torch_dist"
    cfg.checkpoint.fully_parallel_save = True

    cfg.logger.log_interval = 1
    cfg.rng.seed = 5678
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", required=True, help="import_hy3_ckpt.py 产出的 Megatron ckpt 目录")
    p.add_argument("--data", required=True, help="make_sft_data.py 产出的目录")
    p.add_argument("--save", default="/ckpt/hy3-sft")
    p.add_argument("--tokenizer", default="/raid/hy3-tok", help="打了 generation 块补丁的 tokenizer 目录")
    p.add_argument("--export-hf", default="", help="训练结束时直接导出 HF 到该目录（留空则不导）")
    p.add_argument("--hf-cfg", default="/raid/hy3-cfg", help="仅含 config/tokenizer 的本地目录（每节点都要有）")
    p.add_argument("--base", action="store_true", help="用 Hy3-Base 而非 instruct 版")
    p.add_argument("--num-gpus", type=int, default=64)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=2)
    p.add_argument("--vpp", type=int, default=None)
    p.add_argument("--ep", type=int, default=16)
    p.add_argument("--seq-length", type=int, default=512)
    p.add_argument("--gbs", type=int, default=32)
    p.add_argument("--mbs", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--train-samples", type=int, default=608)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp8_mx"])
    a = p.parse_args()

    cfg = build_config(a)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[hy3_sft] train_iters={cfg.train.train_iters} gbs={cfg.train.global_batch_size} "
              f"seq={cfg.dataset.seq_length} lr={cfg.scheduler.max_lr} "
              f"parallel: TP{a.tp}/PP{a.pp}/EP{a.ep} mtp={cfg.model.mtp_num_layers}", flush=True)
    cbs = [ExportHFAtEnd(a.export_hf, a.hf_cfg)] if a.export_hf else None
    finetune(cfg, forward_step, callbacks=cbs)


if __name__ == "__main__":
    main()
