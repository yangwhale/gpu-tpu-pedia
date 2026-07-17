#!/bin/bash
# DSV3 256 GPU — STRICT native Bridge recipe (full_iteration graph)
# Runs on each yw-{a,b,c,d}-N pod. Computes global node_rank from hostname.
set -eux

HOST=$(hostname)                 # e.g. yw-a-5
GROUP=$(echo "$HOST" | sed -E 's/^yw-([abcd])-[0-9]+$/\1/')
ORD=$(echo "$HOST" | sed -E 's/^yw-[abcd]-([0-9]+)$/\1/')
case "$GROUP" in
  a) OFFSET=0 ;;
  b) OFFSET=16 ;;
  c) OFFSET=32 ;;
  d) OFFSET=48 ;;
  *) echo "bad group $GROUP"; exit 1 ;;
esac
NODE_RANK=$((OFFSET + ORD))
MASTER_ADDR="yw-a-0.yw"

# ===== Base env (Bridge PERF_ENV_VARS, utils/executors.py) =====
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export NVTE_NORM_FWD_USE_CUDNN=1          # deepseek fp8_mx gb300 -> del_cudnn_ln=False (kept)
export NVTE_NORM_BWD_USE_CUDNN=1
export TORCH_NCCL_HIGH_PRIORITY=1
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
export NCCL_GRAPH_REGISTER=0              # =1 hangs on GB300 GIB; keep 0

# ===== full_iteration graph specific (perf_plugins.py:302-306) =====
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,graph_capture_record_stream_reuse:True"
export TORCH_NCCL_AVOID_RECORD_STREAMS=0

# ===== deepseek model-specific =====
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

# ===== cutedsl fused grouped MLP + a2a overlap =====
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8

# ===== hybridep NVL domain (EP=32, GB300 NVL72) =====
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32   # == ep_size
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128

# ===== CUDA connections (hybridep + sm100+ -> 32) =====
export CUDA_DEVICE_MAX_CONNECTIONS=32

# ===== LayerNorm SM margin (hybridep -> 20) =====
export NVTE_FWD_LAYERNORM_SM_MARGIN=20
export NVTE_BWD_LAYERNORM_SM_MARGIN=20

# ===== NCCL GIB (GKE GB300) =====
export NCCL_CONF_FILE=/usr/local/gib/configs/nccl.a4xmax.conf
export NCCL_IB_SPLIT_DATA_ON_QPS=1
export NCCL_CTA_POLICY=1
export NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME=eth0
export NCCL_SOCKET_IFNAME=eth0

cd /opt/Megatron-Bridge
pip install git+https://github.com/NVIDIA/dllogger#egg=dllogger >/dev/null 2>&1 || true

# Inner per-local-rank worker: numactl + NATIVE recipe (NO cuda_graph override)
cat > /tmp/worker.sh <<'WORKER'
#!/bin/bash
cd /opt/Megatron-Bridge
numactl --cpunodebind=$((LOCAL_RANK/2)) --membind=$((LOCAL_RANK/2)) \
python scripts/performance/run_script.py \
  -m deepseek -mr deepseek_v3 --task pretrain \
  -g gb300 -c fp8_mx -ng 256 \
  --data mock --max_steps 30 \
  --log_dir /tmp/nemo-results \
  -cv v2 \
  logger.log_throughput=true
WORKER
chmod +x /tmp/worker.sh

echo "=== DSV3 256GPU native full_iteration | host=$HOST rank=$NODE_RANK master=$MASTER_ADDR ==="
torchrun --nproc-per-node=4 --nnodes=64 --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=29600 --rdzv_conf timeout=1800 \
  --no-python bash /tmp/worker.sh 2>&1
