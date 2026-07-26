#!/bin/bash
# 在 16 节点 64 卡上启动 Hy3-295B SFT。
#
# 与预训练启动脚本的差异（详见 SFT.md §4）：
#   - 不设 CUDA graph 相关变量：SFT 只跑 ~200 步，capture 的固定开销换不回来
#   - 不设 FP8：SFT 走 BF16，追求权重的精细调整而非吞吐
#   - 数据和 checkpoint 都在本地 RAID，不走网络
#
# 用法: ./run_sft.sh [epochs] [lr]
set -u
CTX=gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test
PODS=$(seq 0 15 | sed 's/^/yw-a-/')
DIR=~/gpu-tpu-pedia/gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke
EPOCHS=${1:-10}
LR=${2:-1e-5}

ARGS="--pretrained /raid/hy3-megatron --data /raid/sft_mixed \
--num-gpus 64 --tp 1 --pp 2 --ep 16 --seq-length 512 \
--gbs 32 --mbs 1 --epochs $EPOCHS --lr $LR --precision bf16 \
--train-samples 22144 \
--save /raid/hy3-sft --export-hf /raid/hy3-sft-hf"

# ---- 启动前彻底清理 ----
# 血泪教训：残留的 torchrun / hy3_sft 会占住 rendezvous 端口 29700，
# 新一轮在 yw-a-0 上直接 EADDRINUSE 退出，其余 15 节点随后报
# "ncclRemoteError: remote process exited"——看起来像别的节点挂了，
# 实际是 master 根本没起来。必须清干净并**验证**，不能只 pkill 一把了事。
# 注意 pkill 模式里的中括号：不加的话 `pkill -f hy3_sft.py` 会匹配到
# **执行这条命令的 bash 自己**（它的 cmdline 里就含这个字符串），
# 于是 shell 先把自己杀了，后面的命令一条都不执行 —— 表现为"清理了但没清掉"。
cleanup() {
  for a in 1 2 3; do
    echo "$PODS" | xargs -P 16 -I {} timeout 40 kubectl --context $CTX exec {} -- bash -c \
      'pkill -9 -f "[s]ft_launch"; pkill -9 -f "[h]y3_sft"; pkill -9 -f "[t]orchrun"; true' 2>/dev/null
    sleep 5
    N=$(echo "$PODS" | xargs -P 16 -I {} timeout 30 kubectl --context $CTX exec {} -- bash -c \
      'ps aux | grep -c "[h]y3_sft"' 2>/dev/null | awk '{s+=$1} END{print s+0}')
    echo "  清理第 $a 轮：残留 $N 个进程"
    [ "$N" = "0" ] && break
  done
  # 端口仍被占则连僵尸 CUDA context 一起清
  echo "$PODS" | xargs -P 16 -I {} timeout 40 kubectl --context $CTX exec {} -- bash -c \
    'nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9 2>/dev/null; true' 2>/dev/null
  sleep 3
}
cleanup

# 分发最新脚本
for f in hy3_sft.py; do
  B=$(base64 -w0 "$DIR/$f")
  echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- \
    bash -c "echo $B | base64 -d > /raid/$f" 2>/dev/null
done

cat > /tmp/sft_launch.sh <<LEOF
#!/bin/bash
# 继承 pod 的初始环境（GIB / NCCL 配置由 init 容器写入 PID 1 的 environ）
if [ -r /proc/1/environ ]; then
  while IFS= read -r -d '' e; do export "\$e" 2>/dev/null || true; done < /proc/1/environ
fi
export PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:\${LD_LIBRARY_PATH:-}"
export TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 HF_HOME=/raid/hf
export TOKENIZERS_PARALLELISM=False HF_HUB_DISABLE_XET=1
export PYTHONPATH=/raid/pylib:\${PYTHONPATH:-}        # HYV3Bridge 自愈路径

# NCCL / GIB —— 与预训练一致（README §6）
export NCCL_CONF_FILE=/usr/local/gib/configs/nccl.a4xmax.conf
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_SPLIT_DATA_ON_QPS=1 NCCL_CTA_POLICY=1 NCCL_DEBUG=WARN
export NCCL_NVLS_ENABLE=0 TORCH_NCCL_HIGH_PRIORITY=1
export NVLINK_DOMAIN_SIZE=72 USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=16
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128
export CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_NORM_FWD_USE_CUDNN=1 NVTE_NORM_BWD_USE_CUDNN=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=20 NVTE_BWD_LAYERNORM_SM_MARGIN=20
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

H=\$(hostname); R=\${H##*-}
cd /opt/Megatron-Bridge
cat > /tmp/sft_worker.sh <<'W'
#!/bin/bash
cd /opt/Megatron-Bridge
# NUMA 绑核：GB300 有 2 个 CPU NUMA（0-71 / 72-143），GPU0/1→node0，GPU2/3→node1
numactl --cpunodebind=\$((LOCAL_RANK/2)) --membind=\$((LOCAL_RANK/2)) \\
  python /raid/hy3_sft.py $ARGS
W
chmod +x /tmp/sft_worker.sh
torchrun --nproc-per-node=4 --nnodes=16 --node_rank=\$R \\
  --master_addr=yw-a-0.yw --master_port=29700 --rdzv_conf timeout=1800 \\
  --no-python bash /tmp/sft_worker.sh 2>&1
LEOF

B=$(base64 -w0 /tmp/sft_launch.sh)
echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- bash -c \
  "echo $B | base64 -d > /raid/sft_launch.sh; chmod +x /raid/sft_launch.sh; rm -f /raid/sft.log" 2>/dev/null
echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- bash -c \
  'setsid nohup /raid/sft_launch.sh > /raid/sft.log 2>&1 < /dev/null & sleep 1' 2>/dev/null
echo "SFT 已在 16 节点启动  epochs=$EPOCHS lr=$LR  $(TZ=Asia/Hong_Kong date +%H:%M:%S)"
echo "看日志: kubectl --context $CTX exec yw-a-0 -- tail -f /raid/sft.log"
echo "看 loss: kubectl --context $CTX exec yw-a-15 -- grep 'lm loss' /raid/sft.log   # 最后一个 PP stage"
