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
# Unified Testbed 启动脚本
# 用于初始化分布式训练环境和节点间通信
# =============================================================================

echo "=============================================="
echo "Unified Testbed Launcher Starting..."
echo "=============================================="

# -----------------------------------------------------------------------------
# 1. 库路径配置
# -----------------------------------------------------------------------------
export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH:$LD_LIBRARY_PATH"
ldconfig $NCCL_PLUGIN_PATH
echo "Added $NCCL_PLUGIN_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""

# -----------------------------------------------------------------------------
# 1.1 通用依赖安装（仅主节点执行一次）
# -----------------------------------------------------------------------------
if [[ "$JOB_COMPLETION_INDEX" -eq "0" ]]; then
  echo "Installing common dependencies..."
  
  # 修复 NGC 镜像中的 triton ldconfig 路径问题
  TRITON_FILE="/usr/local/lib/python3.12/dist-packages/triton/backends/nvidia/driver.py"
  if [ -f "$TRITON_FILE" ]; then
    sed -i 's|libs = subprocess.check_output(\["ldconfig"|libs = subprocess.check_output(["/sbin/ldconfig"|g' $TRITON_FILE
    echo "  Fixed triton ldconfig path"
  fi
  
  # 安装常用工具（HuggingFace CLI, ModelScope 等）
  pip install --upgrade huggingface_hub[cli] modelscope -q 2>/dev/null
  echo "  Installed: huggingface_hub[cli], modelscope"
  
  # 刷新 shell 命令缓存，确保新安装的命令可用
  hash -r
fi

# 确保 pip 安装的命令在 PATH 中（所有节点都需要）
export PATH="/usr/local/bin:$PATH"
hash -r 2>/dev/null || true

echo "Launching Torch distributed on node rank $JOB_COMPLETION_INDEX out of $NNODES nodes"

# -----------------------------------------------------------------------------
# 2. SSH 服务配置
# -----------------------------------------------------------------------------

# 如果设置了 SSH 公钥环境变量，则添加到 authorized_keys
if [[ -n "${SSH_PUBLIC_KEY}" ]]; then
  echo "${SSH_PUBLIC_KEY}" >> /root/.ssh/authorized_keys
  echo "Added SSH public key from environment variable"
fi

# 创建 SSH 守护进程运行目录并启动服务
mkdir -p /run/sshd
/usr/sbin/sshd -p 2222
echo "Pod has started SSH daemon on port 2222"

# -----------------------------------------------------------------------------
# 3. 节点发现和拓扑感知（仅主节点执行）
# -----------------------------------------------------------------------------

if [ "$JOB_COMPLETION_INDEX" -eq "0" ]; then
  echo "Delaying 10 sec to allow SSH service to start on all nodes..."
  sleep 10

  # -------------------------------------------------------------------------
  # 3.1 节点发现和连通性测试
  # -------------------------------------------------------------------------
  
  echo "List of worker services:"
  for JOB_INDEX in $(seq 0 $((NNODES-1))); do
    # 构造 Kubernetes 服务的 FQDN
    WORKER="$HOSTNAME_PREFIX"0-"$JOB_INDEX.$DOMAIN_NAME"

    echo "  Ping $WORKER"
    echo -n "  Pong "; ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 2222 $WORKER hostname
    ssh_exit_code=$?
    
    # 重试机制：如果连接失败，每 2 秒重试一次
    while [ $ssh_exit_code -ne 0 ]; do
      echo "  (pong failed, retrying in 2 seconds)"
      sleep 2
      echo -n "  Pong "; ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 2222 $WORKER hostname 
      ssh_exit_code=$?
    done

    # -------------------------------------------------------------------------
    # 3.2 物理拓扑发现
    # -------------------------------------------------------------------------
    
    echo "  Querying $WORKER for VM physical location"
    LOCATION=$(ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 2222 $WORKER curl "-s" "-H" "\"Metadata-Flavor: Google\"" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/physical_host" )

    ssh_exit_code=$?
    while [ $ssh_exit_code -ne 0 ]; do
      echo "  (query failed, retrying in 2 seconds)"
      sleep 2
      LOCATION=$(ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 2222 $WORKER curl "-s" "-H" "\"Metadata-Flavor: Google\"" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/physical_host" )
      ssh_exit_code=$?
    done

    echo "  Got $LOCATION"
    echo "$LOCATION $WORKER" >> "/tmp/${HOSTNAME_PREFIX}worker-locations.txt"

    # 配置 SSH 客户端
    echo "Host $WORKER" >> /root/.ssh/config
    echo "  Port 2222" >> /root/.ssh/config
    echo "  StrictHostKeyChecking no" >> /root/.ssh/config
  done

  # -------------------------------------------------------------------------
  # 3.3 拓扑优化排序
  # -------------------------------------------------------------------------
  
  echo "Sorting VM list by physical location:"
  sort "/tmp/${HOSTNAME_PREFIX}worker-locations.txt" > /tmp/job-worker-rank-order.txt
  cat /tmp/job-worker-rank-order.txt | sed 's/^/  /'
  
  # 生成主机名列表文件
  cat /tmp/job-worker-rank-order.txt | awk '{print $2}' > /tmp/job-worker-hostnames-order.txt
  
  # 生成 MPI hostfile（每个节点 8 个 GPU slots）
  for WORKER in $(cat /tmp/job-worker-rank-order.txt | awk '{print $2}'); do
    echo "$WORKER slots=8" >> /etc/job-worker-services.txt
  done
  
  echo "Generated /etc/job-worker-services.txt:"
  cat /etc/job-worker-services.txt | sed 's/^/  /'
  # 将 hostfile 分发到所有节点
  echo ""
  echo "Distributing hostfile to all workers..."
  for WORKER in $(cat /tmp/job-worker-hostnames-order.txt); do
    scp /etc/job-worker-services.txt $WORKER:/etc/job-worker-services.txt
  done
  
  # 创建同步标志，让其他节点知道发现完成
  touch /tmp/node_discovery_complete
  for WORKER in $(cat /tmp/job-worker-hostnames-order.txt); do
    ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 2222 $WORKER touch /tmp/node_discovery_complete
  done
  echo "Node discovery and hostfile distribution complete!"
else
  # 工作节点：等待主节点完成节点发现
  echo "Waiting for master node to complete node discovery..."
  while [[ ! -f /tmp/node_discovery_complete ]]; do
    sleep 2
  done
  echo "Node discovery complete, proceeding..."
fi

# -----------------------------------------------------------------------------
# 4. 执行任务脚本（如果存在）
# -----------------------------------------------------------------------------

# 任务脚本通过 --set-file task_script=path/to/script.sh 挂载到固定路径
TASK_SCRIPT_PATH="/workload/task/task-script.sh"

if [[ -f "${TASK_SCRIPT_PATH}" ]]; then
  echo "=============================================="
  echo "Executing task script: ${TASK_SCRIPT_PATH}"
  echo "=============================================="
  bash "${TASK_SCRIPT_PATH}"
  SCRIPT_EXIT_CODE=$?
  echo "Task script exit code: ${SCRIPT_EXIT_CODE}"
elif [[ -f /workload/scripts/training-script.sh ]]; then
  # 向后兼容旧版配置
  echo "=============================================="
  echo "Executing legacy training script..."
  echo "=============================================="
  bash /workload/scripts/training-script.sh
else
  echo "=============================================="
  echo "No task script specified."
  echo "To run a task, deploy with:"
  echo "  --set-file task_script=path/to/your-script.sh"
  echo "=============================================="
fi

# -----------------------------------------------------------------------------
# 5. 调试模式处理
# -----------------------------------------------------------------------------

if [[ "${SLEEP_INFINITY}" == "true" ]]; then
  echo "=============================================="
  echo "SLEEP_INFINITY is enabled, keeping container running..."
  echo "You can exec into this pod for debugging."
  echo "=============================================="
  sleep infinity
fi

echo "=============================================="
echo "Workload completed"
echo "Pod on $(hostname --fqdn) is exiting"
echo "=============================================="
