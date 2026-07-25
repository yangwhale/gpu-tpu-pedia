#!/bin/bash
# Hy3 64 GPU 消融实验自动化扫点框架
#
# 每个配置：清僵尸显存 -> 分发 -> 16 pod 启动 -> 等稳态 -> 采集指标 -> 记 CSV
# 结果写 results.csv，可随时 tail 查看。
#
# 用法: ./sweep.sh [起始序号]     # 支持中断后从第 N 个继续
set -u
CTX=gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test
PODS=$(seq 0 15 | sed 's/^/yw-a-/')
OUT=~/gpu-tpu-pedia/gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/results.csv
STEPS=8            # capture + ~5 稳态步，够取中位数
MAXWAIT=1500       # 单个配置最长等待秒数

# ---- 实验矩阵：名称|额外参数 ----
# 基准 V4 = --cuda-graph full_iteration --cutedsl --a2a-overlap --recompute-granularity none --recompute-modules
BASE="--num-gpus 64 --tp 1 --pp 2 --vpp 8 --ep 32 --mbs 1 --gbs 2048 --seq-length 4096 --mtp-layers 0"
V4="--cuda-graph full_iteration --cutedsl --a2a-overlap --recompute-granularity none --recompute-modules"

EXPS=(
# D 组补跑：40 层必须配 VPP=4（40/(PP2*VPP8=16)=2.5 无法整除，原 D 组因此 CRASH）
"D1_40layer_bf16|--cuda-graph full_iteration --cutedsl --a2a-overlap --recompute-granularity none --recompute-modules|--num-layers 40 --vpp 4"
"D2_40layer_bf16_mbs2|--cuda-graph full_iteration --cutedsl --a2a-overlap --recompute-granularity none --recompute-modules|--num-layers 40 --vpp 4 --mbs 2 --gbs 4096"
"D3_40layer_bf16_mbs4|--cuda-graph full_iteration --cutedsl --a2a-overlap --recompute-granularity none --recompute-modules|--num-layers 40 --vpp 4 --mbs 4 --gbs 8192"
"D4_40layer_fp8|--cuda-graph full_iteration --cutedsl --a2a-overlap --recompute-granularity none --recompute-modules|--num-layers 40 --vpp 4 --precision fp8_mx"
"D5_fp8_ep16_mbs2|--cuda-graph full_iteration --cutedsl --a2a-overlap --recompute-granularity none --recompute-modules|--precision fp8_mx --ep 16 --mbs 2 --gbs 4096"
"D6_fp8_mbs4|--cuda-graph full_iteration --cutedsl --a2a-overlap --recompute-granularity none --recompute-modules|--precision fp8_mx --mbs 4 --gbs 8192"
)

START=${1:-1}
[ -f "$OUT" ] || echo "idx,name,status,tflops_median,tflops_min,tflops_max,step_time_s,hbm_gb,tokens_per_s_per_gpu,mfu_pct,args" > "$OUT"

# grep -c 无匹配时退出码 1，会触发调用方 `|| echo 0` 产生 "0\n0"。
# 统一只取首行并剥非数字字符，调用方不再用 || 兜底。
kexec() { timeout 90 kubectl --context $CTX exec "$1" -- bash -c "$2" 2>/dev/null; }
knum()  { local v; v=$(timeout 90 kubectl --context $CTX exec "$1" -- bash -c "$2" 2>/dev/null | head -1 | tr -dc '0-9'); echo "${v:-0}"; }

cleanup() {
  # 按 PID 清僵尸 CUDA context（不用 pattern，避免自匹配），并**验证显存真的归零**。
  # 不验证会把上一轮残留误判成新配置 OOM（第一次跑 A1 就这么翻车）。
  for attempt in 1 2 3 4; do
    echo "$PODS" | xargs -P 16 -I {} timeout 60 kubectl --context $CTX exec {} -- bash -c \
      'pkill -9 -f torchrun 2>/dev/null; sleep 2; for r in 1 2; do nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u | xargs -r kill -9 2>/dev/null; sleep 2; done; true' 2>/dev/null
    sleep 10
    # 全 16 pod 里的最大占用
    MX=0
    for pp in yw-a-0 yw-a-5 yw-a-10 yw-a-15; do
      v=$(knum "$pp" 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1')
      [ "${v:-0}" -gt "$MX" ] 2>/dev/null && MX=$v
    done
    echo "   cleanup#$attempt 残留显存峰值 ${MX}MiB"
    [ "$MX" -lt 2000 ] && return 0
    # 第 2 次仍清不掉 -> 进程卡在 D 状态，kill -9 无效，驱动不回收 CUDA context。
    # 唯一解法：重建 pod（容器 PID namespace 销毁时内核才回收）。
    if [ "$attempt" -ge 2 ]; then
      echo "   kill 无效，重建 pod 释放 CUDA context ..."
      timeout 120 kubectl --context $CTX delete pod -l job=yw --grace-period=0 --force >/dev/null 2>&1
      for w in $(seq 1 40); do
        sleep 15
        R=$(timeout 60 kubectl --context $CTX get pods -l job=yw --no-headers 2>/dev/null | grep -c Running)
        echo "   pod 重建中 ${R}/16"
        [ "${R:-0}" -ge 16 ] && break
      done
      sleep 10
      redistribute
    fi
  done
  echo "   ⚠️ cleanup 未能清空（${MX}MiB），继续但结果可能失真"
}

redistribute() {
  # pod 重建后 /tmp 清空，需要重新分发训练脚本
  local B; B=$(base64 -w0 ~/gpu-tpu-pedia/gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/hy3_pretrain.py)
  echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- bash -c \
    "echo $B | base64 -d > /tmp/hy3_pretrain.py" 2>/dev/null
  echo "   已重新分发 hy3_pretrain.py"
}

for i in "${!EXPS[@]}"; do
  idx=$((i+1)); [ "$idx" -lt "$START" ] && continue
  IFS='|' read -r NAME FLAGS EXTRA <<< "${EXPS[$i]}"
  EXTRA=${EXTRA:-}
  # EXTRA 里的并行度覆盖要放最后（argparse 后者生效）
  ARGS="$BASE $FLAGS $EXTRA --max-steps $STEPS"
  echo "=========== [$idx/${#EXPS[@]}] $NAME  $(TZ=Asia/Hong_Kong date +%H:%M:%S) ==========="
  cleanup

  # EP 可能被 EXTRA 覆盖 -> 提取真实 EP 设 env
  EP=$(echo "$ARGS" | grep -oE '\-\-ep [0-9]+' | tail -1 | awk '{print $2}')
  cat > /tmp/sw.sh <<SWEOF
#!/bin/bash
if [ -r /proc/1/environ ]; then while IFS= read -r -d '' e; do export "\$e" 2>/dev/null||true; done < /proc/1/environ; fi
export PATH=/opt/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:\$PATH
export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:\${LD_LIBRARY_PATH:-}"
# ⚠️ 必须 OFFLINE=0：qwen3 骨架 recipe 要从 HF 取 Qwen config。
# pod 重建会清空容器内 HF cache，此时 OFFLINE=1 直接 LocalEntryNotFoundError。
export TRANSFORMERS_OFFLINE=0 HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=False NCCL_NVLS_ENABLE=0
export NVTE_NORM_FWD_USE_CUDNN=1 NVTE_NORM_BWD_USE_CUDNN=1 TORCH_NCCL_HIGH_PRIORITY=1
export HF_HUB_DISABLE_XET=1 NCCL_GRAPH_REGISTER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,graph_capture_record_stream_reuse:True"
export TORCH_NCCL_AVOID_RECORD_STREAMS=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 CUDNNFE_CLUSTER_OVERLAP_MARGIN=8
export NVLINK_DOMAIN_SIZE=72 USE_MNNVL=1
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EP
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128 CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=20 NVTE_BWD_LAYERNORM_SM_MARGIN=20
export NCCL_CONF_FILE=/usr/local/gib/configs/nccl.a4xmax.conf
export NCCL_IB_SPLIT_DATA_ON_QPS=1 NCCL_CTA_POLICY=1 NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
H=\$(hostname); R=\${H##*-}
cd /opt/Megatron-Bridge
cat > /tmp/w.sh <<'W'
#!/bin/bash
cd /opt/Megatron-Bridge
numactl --cpunodebind=\$((LOCAL_RANK/2)) --membind=\$((LOCAL_RANK/2)) python /tmp/hy3_pretrain.py $ARGS
W
chmod +x /tmp/w.sh
torchrun --nproc-per-node=4 --nnodes=16 --node_rank=\$R --master_addr=yw-a-0.yw \\
  --master_port=29600 --rdzv_conf timeout=1800 --no-python bash /tmp/w.sh 2>&1
SWEOF
  B=$(base64 -w0 /tmp/sw.sh)
  echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- bash -c \
    "echo $B | base64 -d > /tmp/sw.sh; chmod +x /tmp/sw.sh; rm -f /tmp/sw.log" 2>/dev/null
  echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- bash -c \
    'setsid nohup /tmp/sw.sh > /tmp/sw.log 2>&1 < /dev/null & sleep 1' 2>/dev/null

  # ---- 轮询等待 ----
  T0=$(date +%s); STATUS=RUNNING; HBM=0
  while :; do
    sleep 45
    EL=$(( $(date +%s) - T0 ))
    N=$(knum yw-a-0 'grep -c "Step Time" /tmp/sw.log; true')
    OOM=$(knum yw-a-0 'grep -ci "out of memory" /tmp/sw.log; true')
    PROC=$(knum yw-a-0 'pgrep -c -f hy3_pretrain; true')
    M=$(knum yw-a-0 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1')
    [ "${M:-0}" -gt "${HBM:-0}" ] 2>/dev/null && HBM=$M
    echo "   ${EL}s 步数=$N oom=$OOM proc=$PROC hbm=${HBM}MiB"
    if [ "$OOM" -gt 0 ]; then STATUS=OOM; break; fi
    if [ "$N" -ge "$STEPS" ]; then STATUS=OK; break; fi
    if [ "$PROC" -le 1 ] && [ "$EL" -gt 200 ]; then STATUS=CRASH; break; fi
    if [ "$EL" -gt "$MAXWAIT" ]; then STATUS=$([ "$N" -gt 3 ] && echo OK || echo HANG); break; fi
  done

  # ---- 采集 ----
  if [ "$STATUS" = OK ]; then
    STATS=$(kexec yw-a-0 'grep -oE "GPU utilization: [0-9.]+" /tmp/sw.log | awk "{print \$3}" | tail -5 | sort -n | awk "{a[NR]=\$1} END{printf \"%s %s %s\", a[int((NR+1)/2)], a[1], a[NR]}"')
    ST=$(kexec yw-a-0 'grep -oE "Step Time : [0-9.]+" /tmp/sw.log | awk "{print \$4}" | tail -5 | sort -n | awk "{a[NR]=\$1} END{print a[int((NR+1)/2)]}"')
    MED=$(echo "$STATS"|awk '{print $1}'); MIN=$(echo "$STATS"|awk '{print $2}'); MAX=$(echo "$STATS"|awk '{print $3}')
  else MED=""; MIN=""; MAX=""; ST=""; fi

  GBS=$(echo "$ARGS"|grep -oE '\-\-gbs [0-9]+'|tail -1|awk '{print $2}')
  SL=$(echo "$ARGS"|grep -oE '\-\-seq-length [0-9]+'|tail -1|awk '{print $2}')
  PREC=$(echo "$ARGS"|grep -oE '\-\-precision [a-z0-9_]+'|tail -1|awk '{print $2}'); PREC=${PREC:-bf16}
  PEAK=$([ "$PREC" = bf16 ] && echo 2700 || echo 5400)
  TOK=$(awk -v g="$GBS" -v s="$SL" -v t="${ST:-0}" 'BEGIN{if(t>0)printf "%.0f", g*s/t/64; else print ""}')
  MFU=$(awk -v m="${MED:-0}" -v p="$PEAK" 'BEGIN{if(m>0)printf "%.1f", m/p*100; else print ""}')
  HG=$(awk -v h="${HBM:-0}" 'BEGIN{printf "%.0f", h/1024}')
  echo "$idx,$NAME,$STATUS,$MED,$MIN,$MAX,$ST,$HG,$TOK,$MFU,\"$FLAGS $EXTRA\"" >> "$OUT"
  echo ">>> [$idx] $NAME = $STATUS  tflops=$MED  step=${ST}s  hbm=${HG}GB  tok/s/gpu=$TOK  mfu=${MFU}%"
done
cleanup
echo "全部完成 $(TZ=Asia/Hong_Kong date +%H:%M:%S)"
