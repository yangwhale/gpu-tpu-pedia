#!/bin/bash
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# =============================================================================
# PyTorch DDP (Distributed Data Parallel) 测试示例
# 使用 torchrun 启动分布式训练
# =============================================================================

echo "=============================================="
echo "PyTorch DDP Test Starting..."
echo "=============================================="
echo "Node Rank: $JOB_COMPLETION_INDEX"
echo "Total Nodes: $NNODES"
echo "GPUs per Node: $GPUS_PER_NODE"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "=============================================="

# 创建简单的 DDP 测试脚本
cat > /tmp/ddp_test.py << 'EOF'
#!/usr/bin/env python3
"""
Simple PyTorch DDP Test Script
验证分布式训练环境是否正确配置
"""

import os
import torch
import torch.distributed as dist

def main():
    # 清除 torchrun 自动设置的环境变量以消除 GIB shim 警告
    # 这些变量是 torchrun 在子进程中设置的，需要在 NCCL 初始化前清除
    for var in ["TORCH_NCCL_USE_COMM_NONBLOCKING", "TORCH_NCCL_ASYNC_ERROR_HANDLING"]:
        if var in os.environ:
            del os.environ[var]
    
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # 设置当前 GPU
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    
    print(f"[Rank {rank}/{world_size}] Local Rank: {local_rank}, "
          f"Device: cuda:{device}, "
          f"GPU Name: {torch.cuda.get_device_name(device)}")
    
    # 创建测试张量
    tensor = torch.ones(1024, 1024, device=device) * rank
    
    # 执行 all_reduce 操作
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    # 验证结果
    expected_sum = sum(range(world_size))
    actual_sum = tensor[0, 0].item()
    
    if rank == 0:
        print(f"\n[Test Result]")
        print(f"  Expected sum: {expected_sum}")
        print(f"  Actual sum: {actual_sum}")
        if abs(expected_sum - actual_sum) < 1e-5:
            print("  Status: PASSED ✓")
        else:
            print("  Status: FAILED ✗")
    
    # 同步所有进程
    dist.barrier()
    
    if rank == 0:
        print(f"\nAll {world_size} GPUs successfully communicated!")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
EOF

# 初始化 NCCL 环境（GIB 自动管理）
source /usr/local/gib/scripts/set_nccl_env.sh

# 使用 torchrun 启动分布式测试
# 注意：GIB shim 警告的环境变量由 torchrun 在子进程中设置，
# 需要在 Python 脚本中清除（见 ddp_test.py 中的处理）
torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$JOB_COMPLETION_INDEX \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  /tmp/ddp_test.py

echo ""
echo "=============================================="
echo "PyTorch DDP Test Completed!"
echo "=============================================="
