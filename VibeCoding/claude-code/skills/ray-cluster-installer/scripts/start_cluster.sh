#!/bin/bash
# Ray 集群一键启动脚本
# 使用 gcloud compute ssh 通过内网连接

set -e

# 配置
HEAD_IP="10.8.0.79"
HEAD_INSTANCE="chrisya-b200-spot-mig-ase1-g4wg"
HEAD_ZONE="asia-southeast1-b"

WORKER_IP="10.8.0.80"
RAY_PORT=6379
DASHBOARD_PORT=8265
NUM_GPUS=8

# 远程命令前缀（加载 PATH）
REMOTE_PREFIX="export PATH=\$HOME/.local/bin:\$PATH &&"

echo "=============================================="
echo "  Ray 集群一键启动"
echo "=============================================="
echo "Head Node:   $HEAD_IP ($HEAD_INSTANCE)"
echo "Worker Node: $WORKER_IP (本机)"
echo "=============================================="

# 1. 先停止所有节点的 Ray
echo ""
echo "[1/4] 停止现有 Ray 进程..."
gcloud compute ssh $HEAD_INSTANCE --zone=$HEAD_ZONE --internal-ip \
    --command="$REMOTE_PREFIX ray stop --force 2>/dev/null || true" \
    -- -o LogLevel=ERROR -o ConnectTimeout=10 &
ray stop --force 2>/dev/null || true
wait
sleep 2

# 2. 启动 Head Node (远程)
echo ""
echo "[2/4] 启动 Head Node ($HEAD_IP)..."
gcloud compute ssh $HEAD_INSTANCE --zone=$HEAD_ZONE --internal-ip \
    --command="$REMOTE_PREFIX ray start --head \
        --port=$RAY_PORT \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=$DASHBOARD_PORT \
        --node-ip-address=$HEAD_IP \
        --num-gpus=$NUM_GPUS" \
    -- -o LogLevel=ERROR -o ConnectTimeout=10

# 等待 Head 完全启动
echo "等待 Head Node 就绪..."
sleep 3

# 3. 启动 Worker Node (本机)
echo ""
echo "[3/4] 启动 Worker Node ($WORKER_IP)..."
ray start \
    --address="$HEAD_IP:$RAY_PORT" \
    --node-ip-address=$WORKER_IP \
    --num-gpus=$NUM_GPUS

# 4. 验证集群
echo ""
echo "[4/4] 验证集群状态..."
sleep 2
python3 -c "
import ray
# 使用 auto 连接本地 ray 进程（已加入集群）
ray.init(address='auto')
nodes = ray.nodes()
alive = sum(1 for n in nodes if n['Alive'])
gpus = sum(n.get('Resources', {}).get('GPU', 0) for n in nodes if n['Alive'])
print(f'  节点数: {alive}')
print(f'  总 GPU: {int(gpus)}')
ray.shutdown()
"

echo ""
echo "=============================================="
echo "  ✓ Ray 集群启动成功！"
echo "=============================================="
echo "Dashboard: http://$HEAD_IP:$DASHBOARD_PORT"
echo "连接地址: ray://$HEAD_IP:$RAY_PORT"
echo "=============================================="
