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
#


export LD_LIBRARY_PATH="$NCCL_PLUGIN_PATH"
ldconfig $LD_LIBRARY_PATH
echo "Added $LD_LIBRARY_PATH to ldconfig:"
ldconfig -p | grep libcuda | sed 's/^/  /'
echo ""

echo "Launching Torch distributed on the node rank $JOB_COMPLETION_INDEX out of $NNODES nodes"
# 如果设置了SSH公钥环境变量，则添加到authorized_keys
if [[ -n "${SSH_PUBLIC_KEY}" ]]; then
  echo "${SSH_PUBLIC_KEY}" >> /root/.ssh/authorized_keys
  echo "Added SSH public key from environment variable"
fi
#service ssh restart

# -----------------------------------------------------------------------------
# 3. SSH 服务配置
# 用于节点间通信，MPI 需要通过 SSH 在不同节点间启动进程
# -----------------------------------------------------------------------------
mkdir /run/sshd  # 创建 SSH 守护进程运行目录
/usr/sbin/sshd -p 2222  # 在端口 2222 启动 SSH 服务（避免与主机 SSH 冲突）
echo "Pod has started SSH daemon"

if [ "$JOB_COMPLETION_INDEX" -eq "0" ]; then
  echo "Delaying 10 sec to allow SSH service to start"
  sleep 10

  # ---------------------------------------------------------------------------
  # 1 节点发现和连通性测试
  # ---------------------------------------------------------------------------
  
  echo "List of worker services:"
  # 遍历所有工作节点，建立 SSH 连接
  for JOB_INDEX in $(seq 0 $((NNODES-1))); do
    # 构造 Kubernetes 服务的 FQDN
    # 使用十进制格式 "0-$JOB_INDEX"，其中0是ReplicatedJob索引，JOB_INDEX是Pod索引
    WORKER="$HOSTNAME_PREFIX"0-"$JOB_INDEX.$DOMAIN_NAME"

    echo "  Ping $WORKER"
    # 测试 SSH 连接，获取远程主机名
    echo -n "  Pong "; ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 2222 $WORKER hostname
    ssh_exit_code=$?  # $? 是 bash 的特殊变量，保存上一个命令的退出状态码
                      # 0 表示成功，非 0 表示失败
    
    # 重试机制：如果连接失败（退出码非 0），每 2 秒重试一次
    while [ $ssh_exit_code -ne 0 ]; do
      echo "  (pong failed, retrying in 2 seconds)"
      sleep 2
      echo -n "  Pong "; ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 2222 $WORKER hostname 
      ssh_exit_code=$?
    done

    # ---------------------------------------------------------------------------
    # 2 物理拓扑发现
    # 查询每个节点的物理位置，用于优化通信拓扑
    # ---------------------------------------------------------------------------
    
    echo "  Querying $WORKER for VM physical location"
    # 通过 Google Cloud 元数据 API 获取物理主机信息
    LOCATION=$(ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 2222 $WORKER curl "-s" "-H" "\"Metadata-Flavor: Google\"" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/physical_host" )

    ssh_exit_code=$?  # 再次获取上一个 SSH 命令的退出状态码
    # 重试机制：确保能够获取到位置信息（如果 SSH 或 curl 命令失败）
    while [ $ssh_exit_code -ne 0 ]; do
      echo "  (query failed, retrying in 2 seconds)"
      sleep 2
      LOCATION=$(ssh -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 2222 $WORKER curl "-s" "-H" "\"Metadata-Flavor: Google\"" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/physical_host" )
      ssh_exit_code=$?
    done

    echo "  Got $LOCATION"
    # 记录物理位置和主机名的映射关系
    echo "$LOCATION $WORKER" >> "/tmp/${HOSTNAME_PREFIX}worker-locations.txt"

    # 配置 SSH 客户端，简化后续连接
    echo "Host $WORKER" >> /root/.ssh/config
    echo "  Port 2222" >> /root/.ssh/config
    echo "  StrictHostKeyChecking no" >> /root/.ssh/config
  done

  # ---------------------------------------------------------------------------
  # 3 拓扑优化排序
  # 根据物理位置对节点进行排序，优化网络通信效率
  # ---------------------------------------------------------------------------
  
  echo "Sorting VM list by physical location:"
  # 对节点按物理位置进行排序，优化网络通信拓扑
  #
  # 输入文件格式: "物理位置 主机名"，例如：
  # rack-1-host-3 worker-0.job.default.svc.cluster.local
  # rack-1-host-5 worker-1.job.default.svc.cluster.local
  # rack-2-host-1 worker-2.job.default.svc.cluster.local
  #
  # sort 命令会按第一列（物理位置）进行字典序排序
  # 这样相同机架或相近位置的节点会被排在一起，减少跨机架通信
  sort "/tmp/${HOSTNAME_PREFIX}worker-locations.txt" > /tmp/job-worker-rank-order.txt

  cat /tmp/job-worker-rank-order.txt | sed 's/^/  /'
  
  # 生成只包含主机名的文件
  cat /tmp/job-worker-rank-order.txt | awk '{print $2}' > /tmp/job-worker-hostnames-order.txt
  
  # 生成 MPI hostfile，每个节点分配 8 个 slots（对应 8 个 GPU）
  # awk '{print $2}' 提取排序后的主机名（第二列）
  for WORKER in $(cat /tmp/job-worker-rank-order.txt | awk '{print $2}'); do
    echo "$WORKER slots=8" >> /etc/job-worker-services.txt
  done
fi

# -----------------------------------------------------------------------------
# 调用 train.sh 脚本 example
# bash /workload/scripts/train.sh

# 根据环境变量决定是否保持容器运行
if [[ "${SLEEP_INFINITY}" == "true" ]]; then
  echo "SLEEP_INFINITY is enabled, keeping container running..."
  sleep infinity
fi

echo "Training completed"
echo "Pod on $(hostname --fqdn) is exiting"