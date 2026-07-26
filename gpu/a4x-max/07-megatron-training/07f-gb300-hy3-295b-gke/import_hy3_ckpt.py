#!/usr/bin/env python3
"""把 tencent/Hy3 的 HF 权重转成 Megatron torch_dist checkpoint。

为什么必须这一步：
    `finetune()` 断言 `pretrained_checkpoint` 或 `load` 非空，且它走的是 Megatron 原生
    checkpoint 机制 —— 不接受 HF 目录，也没有 `hf://` 之类的协议前缀。
    （在 r0.5.0 的 checkpointing.py 里逐行确认过。）

规模账（这决定了怎么跑）：
    HF checkpoint  597.6 GB / 47138 张量 / 99 分片
    单进程转换     需要约 600 GB 内存 —— 节点有 942 GB，勉强够，但慢且脆
    分布式转换     64 rank 下每 rank 只物化约 9.3 GB —— 推荐

    `import_ckpt` 内部调 `to_megatron_model(use_cpu_initialization=True)`。
    在 torchrun 里跑、且并行状态已初始化时，每个 rank 只构建自己那一份分片。

存储（务必先读，这是实际的坑）：
    pod 里只有 41 GB 可写 overlay，没有任何共享存储。三条路：

    A. 本地 NVMe —— 节点池配的是 `localNvmeSsdBlockConfig: localSsdCount 4`，
       即 **block 模式**，GKE 把裸盘交给工作负载自行格式化，这是官方预期用法，
       且是 ephemeral（节点重建即清空），不会破坏任何持久状态。
       每节点 /dev/nvme3n1 有 2.9 TB。快，但 node-local。
    B. GCS（gs://chrisya-gb300-models，同 region us-central1）——
       节点 SA 只有 `devstorage.read_only`：**读得了，写不了**。
       所以适合放输入（HF 权重），不适合放输出。
       要写需额外授权（挂 SA key / 加节点池 scope / 开 Workload Identity）。
    C. 每 rank 写本地 41 GB overlay —— 9.3 GB/rank 塞得下，
       但 torch_dist 目录会碎在 64 个 pod 上，加载时若发生 resharding 会读不到。

    默认走 A。`--out` 指向格式化后挂载的路径。

用法（torchrun 内）：
    python import_hy3_ckpt.py --out /mnt/nvme/hy3-megatron --pp 2 --ep 16

单进程兜底（慢，需 600 GB 内存，仅供小规模调试）：
    python import_hy3_ckpt.py --out ./ckpt --single
"""
from __future__ import annotations

import argparse
import os
import time

HF_MODEL = "tencent/Hy3"
HF_BASE = "tencent/Hy3-Base"



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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="Megatron checkpoint 输出目录")
    p.add_argument("--base", action="store_true", help="转 Hy3-Base 而非 instruct 版")
    p.add_argument("--single", action="store_true", help="单进程 CPU 转换（需 ~600 GB 内存）")
    p.add_argument("--local", default=None, help="本地 HF 权重目录（避免回源下载）")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=2)
    p.add_argument("--ep", type=int, default=16)
    a = p.parse_args()

    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "False")

    from megatron.bridge import AutoBridge

    _ensure_hy3_bridge()

    hf_path = a.local or (HF_BASE if a.base else HF_MODEL)
    rank = int(os.environ.get("RANK", "0"))
    t0 = time.time()

    if a.single:
        if rank == 0:
            print(f"[import] 单进程转换 {hf_path} → {a.out}（预计数十分钟，吃满内存）", flush=True)
        AutoBridge.import_ckpt(hf_path, a.out, trust_remote_code=True)
    else:
        # 分布式路径：先初始化并行状态，让 to_megatron_model 只构建本 rank 的分片
        import torch
        import torch.distributed as dist
        from megatron.core import parallel_state

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=a.tp,
            pipeline_model_parallel_size=a.pp,
            expert_model_parallel_size=a.ep,
            expert_tensor_parallel_size=1,
        )
        if rank == 0:
            ws = dist.get_world_size()
            print(f"[import] 分布式转换 world={ws} TP{a.tp}/PP{a.pp}/EP{a.ep}  "
                  f"{hf_path} → {a.out}  每 rank 约 {597.6/ws:.1f} GB", flush=True)
        AutoBridge.import_ckpt(hf_path, a.out, trust_remote_code=True)
        dist.barrier()

    if rank == 0:
        print(f"[import] 完成，耗时 {time.time()-t0:.0f}s", flush=True)
        # 落一份体检信息，便于事后核对是不是转全了
        tot = 0
        for root, _, files in os.walk(a.out):
            tot += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        print(f"[import] 输出体积 {tot/1e9:.1f} GB（HF 侧 597.6 GB，bf16 应大致相当）", flush=True)


if __name__ == "__main__":
    main()
