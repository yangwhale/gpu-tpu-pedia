#!/bin/bash
# 在 v5p 或 v7 上跑 hunyuan3 —— 一份代码，两套 XLA flag。
#
# 两个平台的差别只有三处：nodeSelector、每步的 flag 集、以及读数时的 device/chip 换算。
# 模型代码、补丁、配置完全共用（这是 2026-07-28 合并的结果，见 README §5.5）。
#
# 前置：先跑一次 prep.sh 把代码推到 GCS（只有改代码才要重跑）：
#   GCS_STAGE=gs://your-bucket/hy3 bash prep.sh
#
# 代码来自 yangwhale/maxtext 的 hunyuan3 分支，**整棵 src/maxtext 覆盖容器里那棵**。
# 不是只注入改动文件——那样测的是「我的改动 + 容器里的旧基座」，不是分支本身。
#
# 镜像两个平台共用同一个（新版 MaxText，src/maxtext 布局 + nnx）。
# 旧镜像驱动不了 Ironwood；新镜像 v5p / v7 都能驱动，已实测。
#
# 用法：
#   PLATFORM=v5p|v7 GCS_STAGE=gs://your-bucket/hy3 \
#   IMAGE=us-docker.pkg.dev/PROJECT/gcr.io/your-maxtext-latest:runner \
#   bash run.sh <run-name> [额外参数...]
set -euo pipefail
RUN=${1:?用法: run.sh <run-name>}; shift || true
PLATFORM=${PLATFORM:?需要 PLATFORM=v5p 或 v7}
GCS_STAGE=${GCS_STAGE:?}; IMAGE=${IMAGE:?}
MODEL=${MODEL:-hunyuan3-295b}; STEPS=${STEPS:-10}; NAME=hy3-$RUN

case "$PLATFORM" in
v5p)
  # 缩规模跑：NODES/TOPO 可用环境变量覆盖，两者必须自洽（NODES = 芯片数 ÷ 4）
  NODES=${NODES:-64}; ACCEL=tpu-v5p-slice; TOPO=${TOPO:-4x8x8}
  # 25 个 flag。SparseCore 那组集合通信卸载在 v5p 上是主要收益来源（§4.4）。
  # ⚠️ 这里的 dvfs_p_state=3 在 **v7 上等于默认值、等于没写**（2026-08-11 实测 R1≡R3）。
  #    v7 分支已改成 =7（+8.6%）。v5p 上的最优档未测，保持原样。
  # 注意：官方 DSV3 v5p 配方里还有 --2a886c8_chip_config_name=megachip_tccontrol，
  # 新版 libtpu 已经摘掉这个 flag，带上会直接 "Unknown command line flag" 退出。
  FLAGS='--xla_tpu_dvfs_p_state=3 --xla_tpu_scoped_vmem_limit_kib=65472
  --xla_tpu_enable_sparse_core_reduce_scatter_v2=true
  --xla_tpu_enable_sparse_core_collective_offload_all_gather=true
  --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true
  --xla_tpu_enable_all_gather_offload_tracing=true
  --xla_tpu_use_tc_device_shape_on_sc=True --xla_sc_disable_megacore_partitioning=True
  --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false
  --xla_enable_async_all_gather=true --xla_tpu_prefer_async_allgather_to_allreduce=true
  --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true
  --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true
  --xla_tpu_enable_concurrent_sparse_core_offloading=true
  --xla_tpu_aggressive_opt_barrier_removal=true
  --xla_tpu_enable_offloading_gather_to_sparsecore=true
  --xla_tpu_sparse_core_all_gather_latency_multiplier=1
  --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3
  --xla_tpu_enable_sparse_core_collective_aggregator=true
  --xla_tpu_enable_latency_hiding_layer_scheduler=true
  --xla_tpu_scheduler_percent_shared_memory_limit=150
  --xla_tpu_pcie_bandwidth_multiplier=0.03
  --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true
  --xla_sc_enable_instruction_fusion=false --xla_sc_disjoint_spmem=false'
  # tile 参数：旧版 3 个，新版拆成 18 个（{wi,wo} × {fwd,dlhs,drhs} × 3 维）
  # TILE_MLP 可覆盖：v7 上实测 tile 必须**等于** base_moe_mlp_dim(1536)，
  # 1024 除不尽会断言失败、512 能整除但更慢。v5p 默认仍是 1024，待验。
  TILE=""; for m in wi wo; do for p in fwd dlhs drhs; do
    TILE="$TILE ${m}_tile_${p}_batch_seq=512 ${m}_tile_${p}_embed_dim=1024 ${m}_tile_${p}_mlp_dim=${TILE_MLP:-1024}"
  done; done
  EXTRA="per_device_batch_size=8 max_target_length=8192 use_custom_sort_vjp=True
  sa_use_fused_bwd_kernel=False out_proj=remat$TILE"
  ;;
v7)
  NODES=${NODES:-16}; ACCEL=tpu7x; TOPO=${TOPO:-4x4x4}
  # 只带这 16 个（基线 3 + SparseCore 卸载 9 + 调度器 4）。
  # 补到 26 个也能跑（c2），但收益 ±0，所以保持精简。
  # **不要**照抄官方那套一次全开 —— w1 那轮死锁，元凶是同时开的
  # use_tokamax_gmm（§6.7），不是 flag 数本身。
  # 调度器组是唯一值钱的一组（+6.6%）；SparseCore 卸载组在 v7 上收益是 0。
  FLAGS='--xla_tpu_dvfs_p_state=7 --xla_tpu_scoped_vmem_limit_kib=65472 --xla_enable_async_all_gather=true
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
  # MoE tile —— 2026-08-15 补上。此前 v7 上两条能拿 tile 收益的路一条都没走：
  # 没开 use_tokamax_gmm（怕 §6.7 的死锁），也没传下面这 18 个配置参数，
  # 于是跑在默认 tile 上白丢 26%。
  # QUICKSTART 早期教的 tkcfg.py monkeypatch 对这条路**无效**：它打的
  # PallasMosaicTpuRaggedDot 只在 use_tokamax_gmm=True 时才被调到，而那个开关
  # 在 v7 上死锁、一直关着，生产走的是默认 megablox。加计数器实测「被调用 0 次」，
  # 但它照常打印 "[tkcfg] patched"，所以看起来是生效的。
  # 默认 megablox 路径吃的就是下面这 18 个配置参数（见 TUNING-v7 §3.4.2 路径对照表）。
  # tile_n 必须 = base_moe_mlp_dim(1536)：1024 除不尽会断言失败，512 能整除但更慢。
  # 实测 64 芯片：不带 525.4 TFLOP/s/chip → 带上 662.2（+26%）。
  TILE=""; for m in wi wo; do for p in fwd dlhs drhs; do
    TILE="$TILE ${m}_tile_${p}_batch_seq=${TILE_BS:-512}"
    TILE="$TILE ${m}_tile_${p}_embed_dim=${TILE_EMB:-2048}"
    TILE="$TILE ${m}_tile_${p}_mlp_dim=${TILE_MLP:-1536}"
  done; done
  # pdbs 默认 12（最优配方）。13 是 AOT 扫出来的上限、再快 0.66%，14 装不下。
  EXTRA="per_device_batch_size=${PDBS:-12} max_target_length=4096 use_custom_sort_vjp=True
  sa_use_fused_bwd_kernel=True use_tokamax_splash=True out_proj=remat
  opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 use_iota_embed=True$TILE"
  ;;
*) echo "PLATFORM 只能是 v5p 或 v7"; exit 1;;
esac
FLAGS=$(echo $FLAGS); EXTRA=$(echo $EXTRA)

# 两个平台共用的部分。踩过的雷：
#   use_tokamax_gmm=True   -> v7 上死锁（§6.7）；也是 use_gmm_v2 的强制前置，所以 gmm_v2 一并不可用
#   shard_exp_on_fsdp=True -> OOM，192 个专家换不来
#   shard_optimizer_over_data=True -> 优化器状态 sharding 退化成全复制，device_put 直接拒
COMMON="override_model_config=True ici_fsdp_parallelism=-1 ici_tensor_parallelism=1 \
megablox=True sparse_matmul=True scan_layers=True \
sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 sa_block_q_dq=2048 sa_block_kv_dq=2048 \
remat_policy=custom decoder_layer_input=offload attention=flash \
allow_split_physical_axes=True tokenizer_type=tiktoken \
tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken"

# DRYRUN=1：只把展开后的模型/并行参数打出来，不提交任何东西。
# 用途是跟 aot.sh 对账 —— AOT 体检的必须是你真会跑的那个配置，
# 两边任何一处漂移都会让体检结果失去意义。见 AOT-COMPILE.md「两个脚本对账」。
if [ -n "${DRYRUN:-}" ]; then
  for t in $COMMON $EXTRA "$@"; do echo "$t"; done | sort
  exit 0
fi

gcloud storage ls "$GCS_STAGE/hy3-maxtext.tgz" >/dev/null 2>&1 || {
  echo "找不到 $GCS_STAGE/hy3-maxtext.tgz —— 先跑 'GCS_STAGE=$GCS_STAGE bash prep.sh'"; exit 1; }

# NODEPOOL 用于「0 节点的 autoscaling 池」（DWS flex-start）：
# 默认的 exclusive-topology 注解要求 leader pod 先落地，follower 才能抄它的
# gke-nodepool 选择器。池子是空的时候 leader 永远落不了地，webhook 就会拒绝
# 创建 follower —— 4 个 pod 只出 1 个，autoscaler 也只看得见 1 个 pending。
# 直接把节点池写死在 nodeSelector 里，既保证同池独占，又不需要 leader 先行。
#
# NO_EXCLUSIVE_TOPOLOGY=1 用于「托管/排队制集群」（Kueue 等）：那里没有可以写死的
# 池名，池子要靠 autoscaler 现扩。带着注解时 leader 落不了地 → follower 建不出来
# → 16 个 pod 只出 1 个 → autoscaler 只看见 1 个 pending，永远不给你扩 16 台。
# 症状是 parallelism=16 而 status.active=1，且没有任何报错。去掉注解后 16 个 pod
# 一次全建出来，autoscaler 才看得到真实需求。TPU 的 gke-tpu-topology 选择器本身
# 就把范围限死在同拓扑的池里，实测不会分裂到多个池。
if [ -n "${NODEPOOL:-}" ]; then
  ANNO=""; POOLSEL="
              cloud.google.com/gke-nodepool: $NODEPOOL"
elif [ -n "${NO_EXCLUSIVE_TOPOLOGY:-}" ]; then
  ANNO=""; POOLSEL=""
else
  ANNO="alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool"; POOLSEL=""
fi

NS=${NAMESPACE:-priority-dev}
QUEUE=${QUEUE:-multislice-queue}
PRIO=${PRIORITY_CLASS:-medium}

kubectl delete jobset "$NAME" -n "$NS" --ignore-not-found=true --wait=false >/dev/null 2>&1
cat <<YAML | kubectl apply -f - >/dev/null
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: $NAME
  namespace: $NS
  labels:
    kueue.x-k8s.io/queue-name: $QUEUE
  annotations: {$ANNO}
spec:
  ttlSecondsAfterFinished: 7200
  failurePolicy: {maxRestarts: 0}
  replicatedJobs:
  - name: slice-job
    replicas: 1
    template:
      spec:
        parallelism: $NODES
        completions: $NODES
        backoffLimit: 0
        template:
          metadata:
            labels:
              declared-duration-minutes: "120"
          spec:
            priorityClassName: $PRIO
            restartPolicy: Never
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: $ACCEL
              cloud.google.com/gke-tpu-topology: $TOPO$POOLSEL
            hostNetwork: true
            dnsPolicy: ClusterFirstWithHostNet
            tolerations: [{operator: "Exists"}]
            containers:
            - name: jax-tpu
              image: $IMAGE
              ports: [{containerPort: 8471}, {containerPort: 8080}]
              securityContext: {privileged: true}
              command: ["bash","-c"]
              args:
              - |
                set -e
                gcloud storage cp $GCS_STAGE/hy3-maxtext.tgz /tmp/p.tgz
                cd /deps && rm -rf src/maxtext && tar xzf /tmp/p.tgz
                export JAX_PLATFORMS=tpu,cpu TPU_STDERR_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0
                export LIBTPU_INIT_ARGS='$FLAGS'
                python3 -m src.maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \\
                  model_name=$MODEL run_name=$NAME base_output_directory=/tmp/hy3out \\
                  dataset_type=synthetic enable_checkpointing=False steps=$STEPS \\
                  dtype=bfloat16 weight_dtype=float32 $COMMON $EXTRA $*
              resources: {limits: {google.com/tpu: 4}}
              volumeMounts: [{mountPath: /dev/shm, name: dshm}]
            volumes: [{name: dshm, emptyDir: {medium: Memory}}]
YAML
echo "[$NAME] 已提交（$PLATFORM，$NODES 台）。"
echo
echo "读结果前必看："
echo "  * **先确认 $NODES/$NODES Running 再看日志**。TPU 切片全有全无，人不齐时"
echo "    活着的 pod 会报 GetSliceInfo 失败 —— 那是症状不是病因。"
echo "  * 判错看**最早**那条，不是日志尾。配置非法会先起 TPU 再退，"
echo "    真正的报错是 'MAXTEXT CONFIG ERROR' / pydantic 的 'Value error'。"
echo "  * step 0 含编译，step 1/2 是 JAX 异步派发的假读数，稳态取 step >= 3。"
if [ "$PLATFORM" = v5p ]; then
echo "  * v5p 是 MegaCore，1 device = 1 chip，日志里的 TFLOP/s/device 不用换算。"
echo "  * MFU = TFLOP/s/device / 459"
echo "  * 预期：step ≈ 63.2 s，TFLOP/s/device ≈ 160.9，MFU ≈ 35.1%"
else
echo "  * v7 是 2 device/chip，per-chip = 日志值 × 2；MFU = per-chip / 2307"
echo "  * v7 编译约 46 s（80 层 / 64 芯片实测），真正慢的是建切片：TPU init 约 70 s"
echo "    —— '编译要 10-17 分钟' 是旧说法，已被 TUNING-v7 与 2026-08-15 的 AOT 对照实验两次推翻"
echo "  * 预期（本脚本默认 = pdbs 12 + 18 个 tile 参数 + dvfs=7）："
echo "      step ≈ 20.61 s，TFLOP/s/device ≈ 331.1，即 662.2 per-chip，MFU ≈ 28.70%"
echo "      PDBS=13 再快 0.66%（666.6 / 28.89%）；14 装不下（AOT 实测差 1.79 GB）"
echo "  * 明显低于这个数先查 tile：漏掉那 18 个参数会掉到 525 per-chip（-26%）"
echo "  * 对照：pdbs 8 无 tile 是 445 per-chip / 19.3%（旧基线）"
fi
