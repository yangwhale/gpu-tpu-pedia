#!/bin/bash
# FP8_MX vs BF16 训练效果对齐验证
#
# 同一 seed、同一数据、同一并行配置，各跑 N 步，逐步对比 lm loss。
# 判据：FP8 的 loss 曲线应与 BF16 逐步吻合，相对偏差在数值噪声内（经验阈值 <1%），
#       且无 NaN / 无 skipped iteration。偏差持续放大说明 FP8 量化损害了训练。
#
# 注意：loss 只在**最后一个 pipeline stage** 打印。PP=2 / 16 pod 时在 yw-a-15。
set -u
CTX=gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test
PODS=$(seq 0 15 | sed 's/^/yw-a-/')
DIR=~/gpu-tpu-pedia/gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke
LOSS_POD=yw-a-15          # 最后一个 PP stage
STEPS=${1:-20}
# 用 EP16（BF16 最优）保证两边配置完全一致，只有精度不同
COMMON="--num-gpus 64 --tp 1 --pp 2 --vpp 8 --ep 16 --mbs 1 --gbs 2048 --seq-length 4096 \
--mtp-layers 0 --cuda-graph full_iteration --cutedsl --a2a-overlap \
--recompute-granularity none --recompute-modules --max-steps $STEPS"

cleanup() {
  for a in 1 2 3; do
    echo "$PODS" | xargs -P 16 -I {} timeout 60 kubectl --context $CTX exec {} -- bash -c \
      'pkill -9 -f torchrun 2>/dev/null; sleep 2; nvidia-smi --query-compute-apps=pid --format=csv,noheader|sort -u|xargs -r kill -9 2>/dev/null; true' 2>/dev/null
    sleep 10
    MX=$(timeout 60 kubectl --context $CTX exec yw-a-0 -- bash -c 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|sort -rn|head -1' 2>/dev/null | tr -dc '0-9')
    echo "   cleanup#$a 残留 ${MX:-?}MiB"
    [ "${MX:-9999}" -lt 2000 ] 2>/dev/null && return 0
    if [ "$a" -ge 2 ]; then
      timeout 120 kubectl --context $CTX delete pod -l job=yw --grace-period=0 --force >/dev/null 2>&1
      for w in $(seq 1 40); do sleep 15
        R=$(timeout 60 kubectl --context $CTX get pods -l job=yw --no-headers 2>/dev/null|grep -c Running)
        [ "${R:-0}" -ge 16 ] && break; done
      sleep 10
      local B; B=$(base64 -w0 $DIR/hy3_pretrain.py)
      echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- bash -c \
        "echo $B | base64 -d > /tmp/hy3_pretrain.py" 2>/dev/null
    fi
  done
}

run_one() {
  local PREC=$1 TAG=$2
  echo "===== $TAG (precision=$PREC, seed 固定) $(TZ=Asia/Hong_Kong date +%H:%M:%S) ====="
  cleanup
  cat > /tmp/la.sh <<LAEOF
#!/bin/bash
if [ -r /proc/1/environ ]; then while IFS= read -r -d '' e; do export "\$e" 2>/dev/null||true; done < /proc/1/environ; fi
export PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:\${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=False NCCL_NVLS_ENABLE=0
export NVTE_NORM_FWD_USE_CUDNN=1 NVTE_NORM_BWD_USE_CUDNN=1 TORCH_NCCL_HIGH_PRIORITY=1
export HF_HUB_DISABLE_XET=1 NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,graph_capture_record_stream_reuse:True"
export TORCH_NCCL_AVOID_RECORD_STREAMS=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 CUDNNFE_CLUSTER_OVERLAP_MARGIN=8
export NVLINK_DOMAIN_SIZE=72 USE_MNNVL=1 NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=16
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128 CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=20 NVTE_BWD_LAYERNORM_SM_MARGIN=20
export NCCL_CONF_FILE=/usr/local/gib/configs/nccl.a4xmax.conf
export NCCL_IB_SPLIT_DATA_ON_QPS=1 NCCL_CTA_POLICY=1 NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
H=\$(hostname); R=\${H##*-}
cd /opt/Megatron-Bridge
cat > /tmp/wl.sh <<'W'
#!/bin/bash
cd /opt/Megatron-Bridge
numactl --cpunodebind=\$((LOCAL_RANK/2)) --membind=\$((LOCAL_RANK/2)) python /tmp/hy3_pretrain.py $COMMON --precision $PREC
W
chmod +x /tmp/wl.sh
torchrun --nproc-per-node=4 --nnodes=16 --node_rank=\$R --master_addr=yw-a-0.yw \\
  --master_port=29600 --rdzv_conf timeout=1800 --no-python bash /tmp/wl.sh 2>&1
LAEOF
  B=$(base64 -w0 /tmp/la.sh)
  echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- bash -c \
    "echo $B | base64 -d > /tmp/la.sh; chmod +x /tmp/la.sh; rm -f /tmp/la.log" 2>/dev/null
  echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- bash -c \
    'setsid nohup /tmp/la.sh > /tmp/la.log 2>&1 < /dev/null & sleep 1' 2>/dev/null

  local T0=$(date +%s)
  while :; do
    sleep 45
    local EL=$(( $(date +%s) - T0 ))
    local N=$(timeout 60 kubectl --context $CTX exec $LOSS_POD -- bash -c 'grep -c "lm loss" /tmp/la.log; true' 2>/dev/null | head -1 | tr -dc '0-9')
    echo "   ${EL}s loss 行数=${N:-0}/$STEPS"
    [ "${N:-0}" -ge "$STEPS" ] && break
    [ "$EL" -gt 2000 ] && break
  done
  timeout 90 kubectl --context $CTX exec $LOSS_POD -- bash -c \
    'grep -oE "iteration +[0-9]+/.*lm loss: [0-9.E+-]+" /tmp/la.log' 2>/dev/null > $DIR/loss_$TAG.txt
  echo "   -> 采集 $(wc -l < $DIR/loss_$TAG.txt) 条 loss"
}

run_one bf16    bf16
run_one fp8_mx  fp8
cleanup
echo "对齐实验完成 $(TZ=Asia/Hong_Kong date +%H:%M:%S)"
