#!/bin/bash
# Hy3 256 GPU 跨 4 域消融扫点（E 组）
#
# 与 64 卡版 sweep.sh 的差异：
#   1. 64 pod 跨 4 个 StatefulSet（yw-a/b/c/d），node_rank = 域序号*16 + pod序号
#   2. **跨域 MNNVL**：NCCL_MNNVL_ENABLE=0 + NCCL_CUMEM_ENABLE=0，USE_MNNVL=1 保持
#   3. 每个实验产出**启动时间线**（timeline.py --stamp 打戳，--parse 拆阶段）
#
# 用法: ./sweep256.sh [起始序号]
set -u
CTX=gke_tencent-gcp-taiji-poc_us-central1_gb300-gke-test
PODS=$(for g in a b c d; do seq 0 15 | sed "s/^/yw-$g-/"; done)
DIR=~/gpu-tpu-pedia/gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke
OUT=$DIR/results256.csv
STEPS=8
MAXWAIT=2400          # 256 卡启动更慢，放宽

BASE="--num-gpus 256 --seq-length 4096 --mtp-layers 0"
PERF="--cuda-graph full_iteration --cutedsl --a2a-overlap --recompute-granularity none --recompute-modules"

# 名称|性能开关|并行与批次覆盖|MNNVL模式(normal/nofix)
EXPS=(
"E13_mnnvl_negative|$PERF|--tp 1 --pp 4 --vpp 2 --ep 32 --mbs 2 --gbs 8192 --precision fp8_mx|nofix"
"E1_qwen3_style|$PERF|--tp 1 --pp 4 --vpp 2 --ep 32 --mbs 2 --gbs 8192 --precision fp8_mx|normal"
"E2_our64_scaled|$PERF|--tp 1 --pp 2 --vpp 8 --ep 32 --mbs 2 --gbs 16384 --precision fp8_mx|normal"
"E3_gbs_not_scaled|$PERF|--tp 1 --pp 2 --vpp 8 --ep 32 --mbs 2 --gbs 4096 --precision fp8_mx|normal"
"E4_tp2_mbs4|$PERF|--tp 2 --pp 2 --vpp 8 --ep 32 --mbs 4 --gbs 16384 --precision fp8_mx|normal"
"E5_tp2_pp4_mbs4|$PERF|--tp 2 --pp 4 --vpp 2 --ep 32 --mbs 4 --gbs 8192 --precision fp8_mx|normal"
"E5b_tp2_mbs2_ctrl|$PERF|--tp 2 --pp 2 --vpp 8 --ep 32 --mbs 2 --gbs 8192 --precision fp8_mx|normal"
"E7_recompute_mbs4|--cuda-graph full_iteration --cutedsl --a2a-overlap --recompute-granularity selective --recompute-modules moe_act|--tp 1 --pp 4 --vpp 2 --ep 32 --mbs 4 --gbs 16384 --precision fp8_mx|normal"
"E8_ep16|$PERF|--tp 1 --pp 4 --vpp 2 --ep 16 --mbs 2 --gbs 8192 --precision fp8_mx|normal"
"E9_ep64|$PERF|--tp 1 --pp 4 --vpp 2 --ep 64 --mbs 2 --gbs 8192 --precision fp8_mx|normal"
"E10_vpp4|$PERF|--tp 1 --pp 4 --vpp 4 --ep 32 --mbs 2 --gbs 8192 --precision fp8_mx|normal"
"E11_bf16|$PERF|--tp 1 --pp 4 --vpp 2 --ep 32 --mbs 2 --gbs 8192 --precision bf16|normal"
"E12_mtp1|$PERF|--tp 1 --pp 4 --vpp 2 --ep 32 --mbs 2 --gbs 8192 --precision fp8_mx --mtp-layers 1|normal"
)

START=${1:-1}
[ -f "$OUT" ] || echo "idx,name,status,tflops_median,tflops_min,tflops_max,step_time_s,hbm_gb,tokens_per_s_per_gpu,mfu_pct,startup_s,capture_s,args" > "$OUT"

knum() { local v; v=$(timeout 90 kubectl --context $CTX exec "$1" -- bash -c "$2" 2>/dev/null | head -1 | tr -dc '0-9'); echo "${v:-0}"; }
kexec() { timeout 90 kubectl --context $CTX exec "$1" -- bash -c "$2" 2>/dev/null; }

redistribute() {
  local B1 B2; B1=$(base64 -w0 $DIR/hy3_pretrain.py); B2=$(base64 -w0 $DIR/timeline.py)
  echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- bash -c \
    "echo $B1 | base64 -d > /tmp/hy3_pretrain.py; echo $B2 | base64 -d > /tmp/timeline.py" 2>/dev/null
}

cleanup() {
  for a in 1 2 3; do
    echo "$PODS" | xargs -P 16 -I {} timeout 60 kubectl --context $CTX exec {} -- bash -c \
      'pkill -9 -f torchrun 2>/dev/null; sleep 2; for r in 1 2; do nvidia-smi --query-compute-apps=pid --format=csv,noheader|sort -u|xargs -r kill -9 2>/dev/null; sleep 2; done; true' 2>/dev/null
    sleep 10
    MX=0
    for pp in yw-a-0 yw-b-0 yw-c-0 yw-d-0 yw-a-15 yw-d-15; do
      v=$(knum "$pp" 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|sort -rn|head -1')
      [ "${v:-0}" -gt "$MX" ] 2>/dev/null && MX=$v
    done
    echo "   cleanup#$a 残留 ${MX}MiB"
    [ "$MX" -lt 2000 ] && return 0
    if [ "$a" -ge 2 ]; then
      echo "   kill 无效 -> 重建 pod"
      timeout 150 kubectl --context $CTX delete pod -l job=yw --grace-period=0 --force >/dev/null 2>&1
      for w in $(seq 1 50); do sleep 15
        R=$(timeout 60 kubectl --context $CTX get pods -l job=yw --no-headers 2>/dev/null|grep -c Running)
        echo "   重建 ${R:-0}/64"; [ "${R:-0}" -ge 64 ] && break; done
      sleep 15; redistribute
    fi
  done
}

for i in "${!EXPS[@]}"; do
  idx=$((i+1)); [ "$idx" -lt "$START" ] && continue
  IFS='|' read -r NAME PERFF PAR MODE <<< "${EXPS[$i]}"
  ARGS="$BASE $PERFF $PAR --max-steps $STEPS"
  EP=$(echo "$ARGS" | grep -oE '\-\-ep [0-9]+' | tail -1 | awk '{print $2}')
  TP=$(echo "$ARGS" | grep -oE '\-\-tp [0-9]+' | tail -1 | awk '{print $2}')
  echo "=========== [$idx/${#EXPS[@]}] $NAME  $(TZ=Asia/Hong_Kong date +%H:%M:%S) ==========="
  cleanup

  # 跨域 MNNVL：normal = 正确设置；nofix = 故意不设（E13 负例）
  if [ "$MODE" = normal ]; then
    MNNVL_LINES='export NCCL_MNNVL_ENABLE=0
export NCCL_CUMEM_ENABLE=0'
  else
    MNNVL_LINES='# E13 负例：故意不设 NCCL_MNNVL_ENABLE / NCCL_CUMEM_ENABLE，吃 GIB 默认（=2）'
  fi
  # TP>1 必须开 sequence parallel（否则激活不被切分，白付通信）
  SPFLAG=""; [ "${TP:-1}" -gt 1 ] && SPFLAG="--sequence-parallel"

  cat > /tmp/sw256.sh <<SWEOF
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
export NVLINK_DOMAIN_SIZE=72 USE_MNNVL=1
$MNNVL_LINES
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EP
export NUM_OF_TOKENS_PER_CHUNK_COMBINE_API=128 CUDA_DEVICE_MAX_CONNECTIONS=32
export NVTE_FWD_LAYERNORM_SM_MARGIN=20 NVTE_BWD_LAYERNORM_SM_MARGIN=20
export NCCL_CONF_FILE=/usr/local/gib/configs/nccl.a4xmax.conf
export NCCL_IB_SPLIT_DATA_ON_QPS=1 NCCL_CTA_POLICY=1 NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME=eth0 NCCL_SOCKET_IFNAME=eth0
# node_rank = 域序号*16 + pod序号   (yw-{a,b,c,d}-N)
H=\$(hostname); G=\$(echo "\$H"|sed -E 's/^yw-([abcd])-[0-9]+$/\1/'); O=\${H##*-}
case "\$G" in a) F=0;; b) F=16;; c) F=32;; d) F=48;; esac
R=\$((F + O))
echo "### MNNVL 实际生效值: NCCL_MNNVL_ENABLE=\${NCCL_MNNVL_ENABLE:-<unset>} NCCL_CUMEM_ENABLE=\${NCCL_CUMEM_ENABLE:-<unset>} USE_MNNVL=\$USE_MNNVL"
cd /opt/Megatron-Bridge
cat > /tmp/w256.sh <<'W'
#!/bin/bash
cd /opt/Megatron-Bridge
numactl --cpunodebind=\$((LOCAL_RANK/2)) --membind=\$((LOCAL_RANK/2)) python /tmp/hy3_pretrain.py $ARGS $SPFLAG
W
chmod +x /tmp/w256.sh
torchrun --nproc-per-node=4 --nnodes=64 --node_rank=\$R --master_addr=yw-a-0.yw \\
  --master_port=29600 --rdzv_conf timeout=1800 --no-python bash /tmp/w256.sh 2>&1 \\
  | python3 -u /tmp/timeline.py --stamp
SWEOF
  B=$(base64 -w0 /tmp/sw256.sh)
  echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- bash -c \
    "echo $B | base64 -d > /tmp/sw256.sh; chmod +x /tmp/sw256.sh; rm -f /tmp/sw256.log" 2>/dev/null
  echo "$PODS" | xargs -P 16 -I {} timeout 90 kubectl --context $CTX exec {} -- bash -c \
    'setsid nohup /tmp/sw256.sh > /tmp/sw256.log 2>&1 < /dev/null & sleep 1' 2>/dev/null

  T0=$(date +%s); STATUS=RUNNING; HBM=0
  while :; do
    sleep 50
    EL=$(( $(date +%s) - T0 ))
    N=$(knum yw-a-0 'grep -c "Step Time" /tmp/sw256.log; true')
    OOM=$(knum yw-a-0 'grep -ci "out of memory" /tmp/sw256.log; true')
    ERR=$(knum yw-a-0 'grep -ci "CUDA error\|ncclUnhandled\|ncclInternal\|Invalid access of peer" /tmp/sw256.log; true')
    PROC=$(knum yw-a-0 'pgrep -c -f hy3_pretrain; true')
    M=$(knum yw-a-0 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1')
    [ "${M:-0}" -gt "${HBM:-0}" ] 2>/dev/null && HBM=$M
    echo "   ${EL}s 步数=$N oom=$OOM err=$ERR proc=$PROC hbm=${HBM}MiB"
    [ "$OOM" -gt 0 ] && { STATUS=OOM; break; }
    [ "$ERR" -gt 0 ] && { STATUS=NCCL_ERR; break; }
    [ "$N" -ge "$STEPS" ] && { STATUS=OK; break; }
    [ "$PROC" -le 1 ] && [ "$EL" -gt 300 ] && { STATUS=CRASH; break; }
    [ "$EL" -gt "$MAXWAIT" ] && { STATUS=$([ "$N" -gt 3 ] && echo OK || echo HANG); break; }
  done

  # ---- 采集吞吐 ----
  MED=""; MIN=""; MAX=""; ST=""
  if [ "$STATUS" = OK ]; then
    STATS=$(kexec yw-a-0 'grep -oE "GPU utilization: [0-9.]+" /tmp/sw256.log|awk "{print \$3}"|tail -5|sort -n|awk "{a[NR]=\$1} END{printf \"%s %s %s\", a[int((NR+1)/2)], a[1], a[NR]}"')
    ST=$(kexec yw-a-0 'grep -oE "Step Time : [0-9.]+" /tmp/sw256.log|awk "{print \$4}"|tail -5|sort -n|awk "{a[NR]=\$1} END{print a[int((NR+1)/2)]}"' | tr -dc '0-9.')
    MED=$(echo "$STATS"|awk '{print $1}'); MIN=$(echo "$STATS"|awk '{print $2}'); MAX=$(echo "$STATS"|awk '{print $3}')
  fi
  # ---- 采集启动时间线 ----
  kexec yw-a-0 'cat /tmp/sw256.log' > "$DIR/tl_${NAME}.log" 2>/dev/null
  TLOUT=$(python3 "$DIR/timeline.py" --parse "$DIR/tl_${NAME}.log" 2>/dev/null)
  echo "$TLOUT" > "$DIR/timeline_${NAME}.md"
  SU=$(echo "$TLOUT" | grep -oE "启动总耗时.*：[0-9.]+" | grep -oE "[0-9.]+$")
  CAP=$(echo "$TLOUT" | awk -F'|' '/CUDA graph capture 结束/{gsub(/[^0-9.]/,"",$4);print $4}')
  echo "   MNNVL 实际值: $(kexec yw-a-0 'grep -m1 "### MNNVL" /tmp/sw256.log' | sed 's/.*### //')"

  GBS=$(echo "$ARGS"|grep -oE '\-\-gbs [0-9]+'|tail -1|awk '{print $2}')
  PREC=$(echo "$ARGS"|grep -oE '\-\-precision [a-z0-9_]+'|tail -1|awk '{print $2}'); PREC=${PREC:-bf16}
  PEAK=$([ "$PREC" = bf16 ] && echo 2700 || echo 5400)
  TOK=$(awk -v g="$GBS" -v t="${ST:-0}" 'BEGIN{if(t>0)printf "%.0f", g*4096/t/256; else print ""}')
  MFU=$(awk -v m="${MED:-0}" -v p="$PEAK" 'BEGIN{if(m>0)printf "%.1f", m/p*100; else print ""}')
  HG=$(awk -v h="${HBM:-0}" 'BEGIN{printf "%.0f", h/1024}')
  echo "$idx,$NAME,$STATUS,$MED,$MIN,$MAX,$ST,$HG,$TOK,$MFU,${SU:-},${CAP:-},\"$PERFF $PAR $SPFLAG\"" >> "$OUT"
  echo ">>> [$idx] $NAME = $STATUS tflops=$MED step=${ST}s hbm=${HG}GB tok=$TOK mfu=${MFU}% 启动=${SU}s capture=${CAP}s"
done
cleanup
echo "256 卡扫点完成 $(TZ=Asia/Hong_Kong date +%H:%M:%S)"
