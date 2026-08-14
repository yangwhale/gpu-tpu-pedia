#!/bin/bash
# AOT（提前编译）—— 不占一张 TPU，就能把 64 芯片的训练步编出来。
#
# 用途有三个，按价值排：
#   1. 抢卡之前先知道会不会 OOM。编译器报的错跟真机一字不差（实测逐位吻合）。
#   2. 拿到每个 device 的显存分解（参数 / 临时 / 代码），比事后看峰值有用得多。
#   3. 顺带 dump 出 HLO，喂给各种图分析工具。
#
# ⚠️ 它**不省编译时间**。64 芯片上真机编译只要 46 秒，占启动的 1/5，
#    详见 AOT-COMPILE.md。别拿「省十几分钟编译」当理由。
#
# 前置：先 prep.sh 把代码推到 GCS。
#
# 用法（两种跑法，结果完全一致）：
#
#   A. 本机 docker（推荐，任意一台带 docker 的机器）
#      RUNNER=docker GCS_STAGE=gs://your-bucket/hy3 \
#      IMAGE=us-docker.pkg.dev/PROJECT/gcr.io/your-maxtext-latest:runner \
#      bash aot.sh probe1
#
#   B. GKE Job（机器不在手边时）
#      GCS_STAGE=... IMAGE=... bash aot.sh probe1
#      需要一个 CPU 节点池（默认 NODEPOOL=cpu-np），且**该节点池的服务账号
#      要有 GCS_STAGE 那个桶的读权限** —— 跨项目的桶会直接 403。
#
# 常用环境变量：
#   CPUS=32          给多少核（实测并行度封顶 ~12-14，给再多吃不下）
#   TOPO_NAME=...    目标拓扑，见下方警告
#   PDBS=12          per_device_batch_size，扫 batch 时改这个
#   SAVE_TO=gs://... 存编译产物并上传（Step 3 用；不设就只体检不存）
#   DUMP_GCS=gs://.. 把 HLO dump 打包上传
#
# ⚠️ 拓扑名按 **device 数**不是芯片数：v7 是 2 device/chip，
#    所以 64 芯片 = tpu7x-128 = 4x4x4。写成 tpu7x-64 会编成 32 芯片的图。
set -euo pipefail
RUN=${1:?用法: aot.sh <run-name>}; shift || true
GCS_STAGE=${GCS_STAGE:?}; IMAGE=${IMAGE:?}
CPUS=${CPUS:-32}; TOPO_NAME=${TOPO_NAME:-tpu7x-128}; SLICES=${SLICES:-1}
MODEL=${MODEL:-hunyuan3-295b}; NAME=hy3-aot-$RUN
RUNNER=${RUNNER:-k8s}; NODEPOOL=${NODEPOOL:-cpu-np}
DUMP_GCS=${DUMP_GCS:-}          # 设了就把 HLO dump 打包传上去
SAVE_TO=${SAVE_TO:-}            # 设了就存编译产物并上传到这个 GCS 前缀

# 跟 run.sh 的 v7 分支**完全一致**的 16 个 flag。
# 少一个都可能编不过 —— 实测：只留调度器那几个而漏掉 sparse core collective
# aggregator，会直接报 "Latency hiding layer scheduler requires sparse core
# collective aggregator to be enabled"。这些 flag 之间有依赖，不要自行精简。
FLAGS='--xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=65472
--xla_enable_async_all_gather=true
--xla_tpu_enable_sparse_core_reduce_scatter_v2=true
--xla_tpu_enable_sparse_core_collective_offload_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
--xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
--xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true
--xla_tpu_enable_sparse_core_collective_aggregator=true
--xla_tpu_use_tc_device_shape_on_sc=True --xla_sc_disable_megacore_partitioning=True
--xla_tpu_enable_latency_hiding_layer_scheduler=true
--xla_tpu_scheduler_percent_shared_memory_limit=150
--xla_tpu_enable_layer_scheduler_for_dependent_collectives=true
--xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false'
FLAGS=$(echo $FLAGS)

# MoE tile —— 生产必带，性能差 26%。
# ⚠️ 不要用 QUICKSTART 早期写的 tkcfg.py monkeypatch，那条路已经是空操作
#    （它打的类只在 use_tokamax_gmm=True 时才被调用，而那个开关在 v7 上死锁）。
#    正确入口就是下面这 18 个配置参数。TILE_MLP 必须 = base_moe_mlp_dim。
TILE=""; for m in wi wo; do for p in fwd dlhs drhs; do
  TILE="$TILE ${m}_tile_${p}_batch_seq=${TILE_BS:-512}"
  TILE="$TILE ${m}_tile_${p}_embed_dim=${TILE_EMB:-2048}"
  TILE="$TILE ${m}_tile_${p}_mlp_dim=${TILE_MLP:-1536}"
done; done

# 跟 run.sh 共用的模型/并行配置。改这里的话记得两边一起改，
# 否则 AOT 体检的是一个你并不会真跑的配置。
COMMON="override_model_config=True ici_fsdp_parallelism=-1 ici_tensor_parallelism=1 \
megablox=True sparse_matmul=True scan_layers=True \
sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 sa_block_q_dq=2048 sa_block_kv_dq=2048 \
remat_policy=custom decoder_layer_input=offload attention=flash \
allow_split_physical_axes=True tokenizer_type=tiktoken \
tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken"
EXTRA="per_device_batch_size=${PDBS:-12} max_target_length=4096 use_custom_sort_vjp=True \
sa_use_fused_bwd_kernel=True use_tokamax_splash=True out_proj=remat \
opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 use_iota_embed=True$TILE"

gcloud storage ls "$GCS_STAGE/hy3-maxtext.tgz" >/dev/null 2>&1 || {
  echo "找不到 $GCS_STAGE/hy3-maxtext.tgz —— 先跑 'GCS_STAGE=$GCS_STAGE bash prep.sh'"; exit 1; }

PKL=""; [ -n "$SAVE_TO" ] && PKL="compiled_trainstep_file=/out/$NAME.pkl"

# 计量脚本：镜像里**没有 /usr/bin/time**，用 python 的 rusage 自己包一层。
# 顺带给出平均并行度（cpu_time/wall），这个数比 wall 本身更能指导选机型。
METER=$(base64 -w0 <<'PYEOF'
import resource, subprocess, sys, time, threading, os
t0 = time.time(); peak = [0]
def poll():
    while True:
        try:
            for d in os.listdir('/proc'):
                if d.isdigit():
                    try:
                        for l in open('/proc/%s/status' % d):
                            if l.startswith('VmHWM'):
                                peak[0] = max(peak[0], int(l.split()[1])); break
                    except Exception: pass
        except Exception: pass
        time.sleep(5)
threading.Thread(target=poll, daemon=True).start()
rc = subprocess.call(sys.argv[1:])
w = time.time() - t0
u = resource.getrusage(resource.RUSAGE_CHILDREN)
cpu = u.ru_utime + u.ru_stime
print("\n@@@METER rc=%d wall=%.1fs cpu_time=%.1fs avg_cores=%.2f peak_rss=%.1fGiB"
      % (rc, w, cpu, cpu / max(w, 1e-9), peak[0] / 1048576.0))
PYEOF
)

# 容器里跑的那段，两种 runner 共用同一份，保证结果可比。
PAYLOAD=$(cat <<PEOF | base64 -w0
set -e
echo "=== \$(nproc) vCPU 可见 / 限额 $CPUS 核 ==="
echo $METER | base64 -d > /tmp/meter.py
[ -f /w/hy3-maxtext.tgz ] && cp /w/hy3-maxtext.tgz /tmp/p.tgz || gcloud storage cp $GCS_STAGE/hy3-maxtext.tgz /tmp/p.tgz
cd /deps && rm -rf src/maxtext && tar xzf /tmp/p.tgz
mkdir -p /tmp/xla_dump /out
# JAX_PLATFORMS=cpu：本机没有 TPU，靠 topology 描述编译。
# 日志里那两条 "could not determine TPU accelerator type / worker hostnames"
# 是正常噪音，不影响编译结果。
export JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=1
export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
python3 /tmp/meter.py python3 -m src.maxtext.trainers.pre_train.train_compile \
  src/maxtext/configs/base.yml \
  model_name=$MODEL run_name=$NAME base_output_directory=/tmp/out \
  compile_topology=$TOPO_NAME compile_topology_num_slices=$SLICES \
  compile_xla_flags="$FLAGS" $PKL \
  dataset_type=synthetic enable_checkpointing=False steps=10 \
  dtype=bfloat16 weight_dtype=float32 $COMMON $EXTRA $* 2>&1 | tail -60
echo "=== HLO dump ==="; du -sh /tmp/xla_dump; ls /tmp/xla_dump | wc -l
PEOF
)

if [ "$RUNNER" = docker ]; then
  W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; mkdir -p "$W/out"
  # 在 host 上拉包：容器里不一定有可用的 ADC，host 有。
  gcloud storage cp "$GCS_STAGE/hy3-maxtext.tgz" "$W/hy3-maxtext.tgz" >/dev/null 2>&1
  docker run --rm --cpus="$CPUS" -v "$W":/w -v "$W/out":/out "$IMAGE" \
    bash -c "echo $PAYLOAD | base64 -d > /tmp/run.sh; bash /tmp/run.sh" 2>&1 | tee /tmp/$NAME.log
  if [ -n "$SAVE_TO" ] && [ -f "$W/out/$NAME.pkl" ]; then
    # save_compiled() 是裸 open()，不认 gs://，所以只能先落本地再自己传。
    gcloud storage cp "$W/out/$NAME.pkl" "$SAVE_TO/$NAME.pkl"
    echo "编译产物 -> $SAVE_TO/$NAME.pkl  ($(du -h "$W/out/$NAME.pkl" | cut -f1))"
  fi
  echo; echo "日志留在 /tmp/$NAME.log"
else
  kubectl delete job "$NAME" --ignore-not-found=true --wait=true >/dev/null 2>&1
  # 注意别拼出空的分号段：args 是单条 bash -c，`...; ; ` 会直接语法错误
  # （2026-08-15 复跑文档时踩到，pod 起来就 "syntax error near unexpected token"）。
  POST="echo $PAYLOAD | base64 -d > /tmp/run.sh; bash /tmp/run.sh"
  [ -n "$SAVE_TO" ]  && POST="$POST; gcloud storage cp /out/$NAME.pkl $SAVE_TO/$NAME.pkl"
  [ -n "$DUMP_GCS" ] && POST="$POST; tar czf /tmp/d.tgz -C /tmp xla_dump; gcloud storage cp /tmp/d.tgz $DUMP_GCS/$NAME-xla_dump.tgz"
  cat <<YAML | kubectl apply -f - >/dev/null
apiVersion: batch/v1
kind: Job
metadata: {name: $NAME}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    spec:
      restartPolicy: Never
      nodeSelector: {cloud.google.com/gke-nodepool: $NODEPOOL}
      containers:
      - name: aot
        image: $IMAGE
        resources:
          requests: {cpu: "$CPUS", memory: 40Gi}
          limits:   {cpu: "$CPUS", memory: 60Gi}
        command: ["bash","-c"]
        args: ["$POST"]
YAML
  echo "[$NAME] 已提交（$TOPO_NAME × $SLICES slice，$CPUS 核）。"
  echo "  kubectl logs -l job-name=$NAME --tail=60 | grep -E 'Memory analysis|@@@METER|OOM|Error'"
fi

echo
echo "读数提示："
echo "  * 判 OOM **只看 temp_size 这一项**，跟 94.74 GB/device 比。"
echo "    不要把 argument+temp 加起来 —— argument 有相当部分被别名/donate 掉了，"
echo "    加起来会得到一个偏保守、能把可行配置误判成 OOM 的数。"
echo "  * 'Memory analysis' 是**每个 device** 的账，v7 是 2 device/chip"
echo "  * @@@METER 的 avg_cores 是平均并行度 —— 它封顶在 ~14，给再多核也吃不下"
echo "  * 最后一行必须是 'Finished train_compile.py successfully!'；"
echo "    编译失败也会留下 HLO dump，看到 dump 不等于成功"
