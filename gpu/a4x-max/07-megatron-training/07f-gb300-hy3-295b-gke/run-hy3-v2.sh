#!/bin/bash
# Hy3 (混元3, 295B-A21B) 64 GPU — GB300 NVL72 单域 16 节点
# 每个 yw-a-N pod 上跑。从 hostname 算 node_rank 0-15。
#
# 与 07e (DSV3) 的差异：
#   1. 单域 16 节点（非 4 域 64 节点）→ node_rank 无 group offset
#   2. EP=32 → NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=32（改 EP 必须同步改这里！）
#   3. Bridge 无 hy3 perf recipe → 走自研 pretrain 脚本 hy3_provider.py（非 run_script.py）
set -eux

HOST=$(hostname)                 # e.g. yw-a-5
NODE_RANK=$(echo "$HOST" | sed -E 's/^yw-a-([0-9]+)$/\1/')
MASTER_ADDR="yw-a-0.yw"
NNODES=16
EP_SIZE=32                       # 改这个必须同步改下面的 NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN

# ===== SSH 启动必加：继承容器完整 ENV（login shell 会丢 PATH/LD_LIBRARY_PATH/CUDA_HOME）=====
if [ -r /proc/1/environ ]; then
  while IFS= read -r -d '' __e; do export "$__e" 2>/dev/null || true; done < /proc/1/environ
fi
export PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

# ===== Base env (Bridge PERF_ENV_VARS, utils/executors.py) =====
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=False
export NCCL_NVLS_ENABLE=0
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export TORCH_NCCL_HIGH_PRIORITY=1
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_XET=1
export NCCL_GRAPH_REGISTER=0              # =1 在 GB300 GIB 下 rendezvous 挂死

# ===== full_iteration graph 专属（漏了会 StreamCaptureUnjoined 崩）=====
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,graph_capture_record_stream_reuse:True"
export TORCH_NCCL_AVOID_RECORD_STREAMS=0

# ===== MoE 数值（DSV3 血统，沿用）=====
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

# ===== cutedsl fused grouped MLP + a2a overlap =====
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export CUDNNFE_CLUSTER_OVERLAP_MARGIN=8

# ===== hybridep NVL domain（EP=32, GB300 NVL72 单域）=====
export NVLINK_DOMAIN_SIZE=72
export USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EP_SIZE   # 必须 == EP，否则 all-to-all 挂死
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

# Inner per-local-rank worker
cat > /tmp/worker.sh <<'WORKER'
#!/bin/bash
cd /opt/Megatron-Bridge
# 精度：首跑 BF16 对齐腾讯官方口径（官方三套 SFT 栈全 BF16，无 FP8 训练路径）。
# MoE 上 FP8 相对 BF16 实测只有 -5%~+5%（见 README 三节），FP8_MX 作为对照另跑。
numactl --cpunodebind=$((LOCAL_RANK/2)) --membind=$((LOCAL_RANK/2)) \
python /tmp/hy3_pretrain.py \
  --num-gpus 64 \
  --tp 1 --pp 2 --vpp 8 --ep 32 \
  --cuda-graph transformer_engine \
  --recompute-modules \
  --recompute-granularity none \
  --mbs 1 --gbs 2048 \
  --seq-length 4096 \
  --precision bf16 \
  --mtp-layers 0 \
  --max-steps 30
WORKER
chmod +x /tmp/worker.sh

echo "=== Hy3 295B 64GPU | host=$HOST rank=$NODE_RANK/$NNODES master=$MASTER_ADDR EP=$EP_SIZE ==="
torchrun --nproc-per-node=4 --nnodes=$NNODES --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=29600 --rdzv_conf timeout=1800 \
  --no-python bash /tmp/worker.sh 2>&1
