#!/usr/bin/env python3
"""SFT 后的 Megatron checkpoint → HF 格式（**分布式**导出）。

为什么不能用单进程 `AutoBridge.export_ckpt`：
    SFT 的 checkpoint 是 64 个 rank 各写各的，而 /raid 是 node-local，
    所以 `iter_0000692/` 在每个节点上只有本节点那 4 个分片：

        yw-a-0:  __0_0 __1_0 __2_0 __3_0
        yw-a-8:  __32_0 __33_0 __34_0 __35_0
        …

    单进程 export 在 yw-a-0 上跑，读到 `__32_0.distcp` 就 FileNotFoundError。
    （与「加载」侧同源的问题——torch_dist 假设整个目录对每个 rank 可见。）

    而且这份 checkpoint 含优化器状态：295B × 14 B/param ≈ **3.8 TB**
    （bf16 权重 2 + fp32 master 4 + Adam m 4 + v 4），
    把它们汇集到单机再导出要搬 3.6 TB，得不偿失。

做法：
    起一个与训练**完全相同并行配置**的 64 卡 job，每个 rank 读自己那份本地分片，
    再由 `save_hf_pretrained` 跨 rank 聚合、rank 0 落 HF safetensors（仅权重 ~597 GB）。

用法（torchrun 内）：
    python export_sft_dist.py --megatron /raid/hy3-sft --out /raid/hy3-sft-hf
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def _ensure_bridge():
    try:
        from megatron.bridge.models.hy_v3 import HYV3Bridge  # noqa: F401
    except ImportError:
        sys.path.insert(0, "/raid/pylib")
        from hy_v3.hy_v3_bridge import HYV3Bridge  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--megatron", default="/raid/hy3-sft")
    p.add_argument("--out", default="/raid/hy3-sft-hf")
    p.add_argument("--hf-src", default="/raid/hy3-hf", help="原始 HF 快照，用于带上自定义 modeling 代码")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=2)
    p.add_argument("--ep", type=int, default=16)
    a = p.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "False")

    import torch
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.bridge import AutoBridge

    _ensure_bridge()

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size=a.tp,
        pipeline_model_parallel_size=a.pp,
        expert_model_parallel_size=a.ep,
        expert_tensor_parallel_size=1,
    )
    rank = dist.get_rank()
    t0 = time.time()
    if rank == 0:
        print(f"[export] world={dist.get_world_size()} TP{a.tp}/PP{a.pp}/EP{a.ep}  "
              f"{a.megatron} → {a.out}", flush=True)

    bridge = AutoBridge.from_hf_pretrained(a.hf_src, trust_remote_code=True)
    model = bridge.load_megatron_model(a.megatron, wrap_with_ddp=False)
    if rank == 0:
        print(f"[export] 模型加载完成 {time.time()-t0:.0f}s，开始写 HF", flush=True)

    bridge.save_hf_pretrained(model, a.out, show_progress=(rank == 0),
                              source_path=a.hf_src, strict=False)
    dist.barrier()
    if rank == 0:
        sz = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(a.out) for f in fs)
        print(f"[export] 完成 {sz/1e9:.1f} GB 耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
