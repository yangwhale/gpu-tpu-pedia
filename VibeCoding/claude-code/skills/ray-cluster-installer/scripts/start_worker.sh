#!/bin/bash
# Ray Worker Node Startup Script
# Usage: HEAD_IP=10.8.0.79 ./start_worker.sh

set -e

# Configuration (override with environment variables)
HEAD_IP="${HEAD_IP:?Error: HEAD_IP environment variable is required}"
RAY_PORT="${RAY_PORT:-6379}"
WORKER_IP="${WORKER_IP:-$(hostname -I | awk '{print $1}')}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)}"
NUM_CPUS="${NUM_CPUS:-$(nproc)}"

echo "=============================================="
echo "  Ray Worker Node Startup"
echo "=============================================="
echo "Head IP:    $HEAD_IP:$RAY_PORT"
echo "Worker IP:  $WORKER_IP"
echo "GPUs:       $NUM_GPUS"
echo "CPUs:       $NUM_CPUS"
echo "=============================================="

# Stop existing Ray processes
echo "Stopping existing Ray processes..."
ray stop --force 2>/dev/null || true
sleep 2

# Start Worker Node
echo "Connecting to Ray Head Node..."
ray start \
    --address="$HEAD_IP:$RAY_PORT" \
    --node-ip-address=$WORKER_IP \
    --num-cpus=$NUM_CPUS \
    --num-gpus=$NUM_GPUS

echo ""
echo "=============================================="
echo "  Ray Worker Node Connected!"
echo "=============================================="
echo "Connected to Head: $HEAD_IP:$RAY_PORT"
echo "Worker IP:         $WORKER_IP"
echo "=============================================="
