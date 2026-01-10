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
# NCCL 性能测试示例脚本
# 在所有节点上运行 NCCL all_reduce 性能测试
# =============================================================================

echo "=============================================="
echo "NCCL Performance Test Starting..."
echo "=============================================="

# 只在主节点（rank 0）上执行 MPI 启动
if [ "$JOB_COMPLETION_INDEX" -ne "0" ]; then
  echo "This is worker node $JOB_COMPLETION_INDEX, waiting for master to initiate test..."
  # 工作节点等待，MPI 会通过 SSH 启动进程
  exit 0
fi

echo "This is master node, initiating NCCL test across all nodes..."

# -----------------------------------------------------------------------------
# 1. 基本连通性测试
# -----------------------------------------------------------------------------
echo ""
echo "Step 1: Testing basic connectivity..."
mpirun --allow-run-as-root \
  -np $((8*$NNODES)) \
  -hostfile /etc/job-worker-services.txt \
  --mca orte_keep_fqdn_hostnames 1 \
  --mca plm_rsh_agent "ssh -q -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 2222" \
  hostname

if [ $? -ne 0 ]; then
  echo "ERROR: Connectivity test failed!"
  exit 1
fi
echo "Connectivity test passed!"

# -----------------------------------------------------------------------------
# 2. NCCL All-Reduce 性能测试
# -----------------------------------------------------------------------------
echo ""
echo "Step 2: Running NCCL all_reduce performance test..."
echo "Test parameters:"
echo "  - Nodes: $NNODES"
echo "  - GPUs per node: 8"
echo "  - Total GPUs: $((8*$NNODES))"
echo "  - Message sizes: 2M to 16G"
echo ""

mpirun --allow-run-as-root \
  -hostfile /etc/job-worker-services.txt \
  -wdir /third_party/nccl-tests \
  -mca plm_rsh_no_tree_spawn 1 \
  --mca orte_keep_fqdn_hostnames 1 \
  --map-by slot \
  --mca plm_rsh_agent "ssh -q -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 2222" \
  bash -c "source /tmp/export_init_env.sh && ./build/all_reduce_perf -b 2M -e 16G -f 2 -n 1 -g 1 -w 10"

echo ""
echo "=============================================="
echo "NCCL Performance Test Completed!"
echo "=============================================="
