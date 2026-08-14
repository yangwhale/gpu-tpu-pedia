#!/bin/bash
# AOT（提前编译）—— 不占一张 TPU，就能把 64 芯片的训练步编出来。
#
# 用途有三个，按价值排：
#   1. 抢卡之前先知道会不会 OOM。编译器报的错跟真机一字不差。
#   2. 拿到每个 device 的显存分解（参数 / 临时 / 代码），比事后看峰值有用得多。
#   3. 顺带 dump 出 HLO，喂给各种图分析工具。
#
# 前置：跟 run.sh 一样，先 prep.sh 把代码推到 GCS。
#
# 用法：
#   GCS_STAGE=gs://your-bucket/hy3 \
#   IMAGE=us-docker.pkg.dev/PROJECT/gcr.io/your-maxtext-latest:runner \
#   bash aot.sh <run-name> [CPUS=32] [TOPO_NAME=tpu7x-128] [额外参数...]
#
# ⚠️ 拓扑名按 **device 数**不是芯片数：v7 是 2 device/chip，
#    所以 64 芯片 = tpu7x-128 = 4x4x4。写成 tpu7x-64 会编成 32 芯片的图。
set -euo pipefail
RUN=${1:?用法: aot.sh <run-name>}; shift || true
GCS_STAGE=${GCS_STAGE:?}; IMAGE=${IMAGE:?}
CPUS=${CPUS:-32}; TOPO_NAME=${TOPO_NAME:-tpu7x-128}; SLICES=${SLICES:-1}
MODEL=${MODEL:-hunyuan3-295b}; NAME=hy3-aot-$RUN
NODEPOOL=${NODEPOOL:-cpu-np}
DUMP_GCS=${DUMP_GCS:-}          # 设了就把 HLO dump 打包传上去

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
opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 use_iota_embed=True"

gcloud storage ls "$GCS_STAGE/hy3-maxtext.tgz" >/dev/null 2>&1 || {
  echo "找不到 $GCS_STAGE/hy3-maxtext.tgz —— 先跑 'GCS_STAGE=$GCS_STAGE bash prep.sh'"; exit 1; }

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

kubectl delete job "$NAME" --ignore-not-found=true --wait=true >/dev/null 2>&1
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
        args:
        - |
          set -e
          echo "=== \$(nproc) vCPU 可见 / 限额 $CPUS 核 ==="
          echo $METER | base64 -d > /tmp/meter.py
          gcloud storage cp $GCS_STAGE/hy3-maxtext.tgz /tmp/p.tgz
          cd /deps && rm -rf src/maxtext && tar xzf /tmp/p.tgz
          mkdir -p /tmp/xla_dump
          # JAX_PLATFORMS=cpu：本机没有 TPU，靠 topology 描述编译。
          # 日志里那两条 "could not determine TPU accelerator type / worker hostnames"
          # 是正常噪音，不影响编译结果。
          export JAX_PLATFORMS=cpu TF_CPP_MIN_LOG_LEVEL=1
          export XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"
          python3 /tmp/meter.py python3 -m src.maxtext.trainers.pre_train.train_compile \\
            src/maxtext/configs/base.yml \\
            model_name=$MODEL run_name=$NAME base_output_directory=/tmp/out \\
            compile_topology=$TOPO_NAME compile_topology_num_slices=$SLICES \\
            compile_xla_flags="$FLAGS" \\
            dataset_type=synthetic enable_checkpointing=False steps=10 \\
            dtype=bfloat16 weight_dtype=float32 $COMMON $EXTRA $* 2>&1 | tail -60
          echo "=== HLO dump ==="; du -sh /tmp/xla_dump; ls /tmp/xla_dump | wc -l
          if [ -n "$DUMP_GCS" ]; then
            tar czf /tmp/dump.tgz -C /tmp xla_dump
            gcloud storage cp /tmp/dump.tgz $DUMP_GCS/$NAME-xla_dump.tgz
            echo "dump -> $DUMP_GCS/$NAME-xla_dump.tgz"
          fi
YAML
echo "[$NAME] 已提交（$TOPO_NAME × $SLICES slice，$CPUS 核）。"
echo
echo "看结果："
echo "  kubectl logs -l job-name=$NAME --tail=60 | grep -E 'Memory analysis|@@@METER|OOM|Error'"
echo
echo "读数提示："
echo "  * 'Memory analysis' 那行是**每个 device** 的账，v7 是 2 device/chip"
echo "  * 单 device HBM ≈ argument + temp + generated_code（output 与 argument 别名，不叠加）"
echo "  * @@@METER 的 avg_cores 是平均并行度 —— 它封顶在 ~14，给再多核也吃不下"
