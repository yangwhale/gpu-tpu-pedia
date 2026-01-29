#!/bin/bash
# Ray Head Node Startup Script
# Usage: HEAD_IP=10.8.0.79 ./start_head.sh

set -e

# Configuration (override with environment variables)
HEAD_IP="${HEAD_IP:-$(hostname -I | awk '{print $1}')}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)}"
NUM_CPUS="${NUM_CPUS:-$(nproc)}"

echo "=============================================="
echo "  Ray Head Node Startup"
echo "=============================================="
echo "Head IP:        $HEAD_IP"
echo "Ray Port:       $RAY_PORT"
echo "Dashboard Port: $DASHBOARD_PORT"
echo "GPUs:           $NUM_GPUS"
echo "CPUs:           $NUM_CPUS"
echo "=============================================="

# Stop existing Ray processes
echo "Stopping existing Ray processes..."
ray stop --force 2>/dev/null || true
sleep 2

# Start Head Node
echo "Starting Ray Head Node..."
ray start --head \
    --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$DASHBOARD_PORT \
    --node-ip-address=$HEAD_IP \
    --num-cpus=$NUM_CPUS \
    --num-gpus=$NUM_GPUS

echo ""
echo "=============================================="
echo "  Ray Head Node Started Successfully!"
echo "=============================================="
echo "Dashboard:       http://$HEAD_IP:$DASHBOARD_PORT"
echo "Cluster Address: ray://$HEAD_IP:$RAY_PORT"
echo ""
echo "Worker nodes can join with:"
echo "  ray start --address='$HEAD_IP:$RAY_PORT'"
echo "=============================================="
