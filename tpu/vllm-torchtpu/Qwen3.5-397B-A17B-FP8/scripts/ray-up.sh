#!/usr/bin/env bash
# 路线 B：在 4 个 worker 上拉起 Ray 集群，把 32 个 device 组成单一 mesh。
#
# 上游只有裸 VM + docker 的多机脚本（scripts/multihost/run_cluster.sh），
# 没有 GKE 先例。但多机在代码里是一等公民不是脚本层 hack：
#   envs.py            TPU_MULTIHOST_BACKEND = env_with_choices(..., ["ray"])
#   tpu_platform.py    按 multihost_backend 分支走不同的 device 绑定
#   distributed/utils  get_kv_ips/ports 在 ray 模式下收集全部节点
#
# 在每个 pod 上跑: bash ray-up.sh <head|worker> <head_dns>
set -uo pipefail

ROLE="${1:?head 或 worker}"
HEAD="${2:-ttpu16-s-0-0.ttpu16}"

# GKE 注入了 TPU_WORKER_ID，但没注入 TPU_NODE_ID —— 而 distributed/utils.py 的
# get_node_id() 读的正是后者，默认 0。不补这一行，4 个节点会全都自认为 node 0，
# get_kv_ips() 收集到的 IP 表会塌成一个，KV transfer 全指向同一台。
export TPU_NODE_ID="${TPU_WORKER_ID:-0}"
export TPU_MULTIHOST_BACKEND=ray
export PYTHONUNBUFFERED=1
export VLLM_CACHE_ROOT=/work/vllmcache
export HF_HUB_OFFLINE=1

ray stop --force >/dev/null 2>&1 || true
sleep 2

if [ "$ROLE" = "head" ]; then
  ray start --head --port=6379 --dashboard-host=0.0.0.0 \
    --node-ip-address="$(hostname -i)" 2>&1 | tail -8
else
  # worker 要等 head 的 6379 起来，否则 ray start 直接失败退出
  for i in $(seq 1 60); do
    (echo > /dev/tcp/"${HEAD%%:*}"/6379) >/dev/null 2>&1 && break
    sleep 5
  done
  ray start --address="$HEAD:6379" --node-ip-address="$(hostname -i)" 2>&1 | tail -8
fi

sleep 5
echo "=== ray 集群状态（node=$TPU_NODE_ID role=$ROLE）==="
ray status 2>&1 | head -25
