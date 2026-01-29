#!/bin/bash
# Ray 集群一键停止脚本

HEAD_INSTANCE="chrisya-b200-spot-mig-ase1-g4wg"
HEAD_ZONE="asia-southeast1-b"

echo "停止 Ray 集群..."

# 并行停止所有节点
gcloud compute ssh $HEAD_INSTANCE --zone=$HEAD_ZONE --command="ray stop --force 2>/dev/null || true" -- -o LogLevel=ERROR &
ray stop --force 2>/dev/null || true
wait

echo "✓ Ray 集群已停止"
