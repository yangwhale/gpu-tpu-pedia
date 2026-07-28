#!/bin/bash
# 在 64 芯片 v7 (Ironwood) 上跑 hunyuan3-295b —— 本仓库 §6.5 那个 c1 配置的复现脚本。
# 当前成绩：step 20.43 s，445.12 TFLOP/s/chip，MFU 19.29%，整机 205,314 tok/s。
# 注意这**还没调到位**，目标是 DSV3 在同硬件的 612.7 / 26.6%（§6.1）。
#
# 前置：
#   * 一个 4x4x4 的 tpu7x 节点池（16 台 tpu7x-standard-4t = 64 芯片）。
#     v7 不会自动建 placement policy，必须先手工建带拓扑的 workload policy：
#       gcloud compute resource-policies create workload-policy tpu7x-64chip \
#         --region=us-central1 --type=HIGH_THROUGHPUT --accelerator-topology=4x4x4
#       gcloud container node-pools create np-v7x-64 --cluster=... --region=us-central1 \
#         --node-locations=us-central1-c --machine-type=tpu7x-standard-4t \
#         --tpu-topology=4x4x4 --num-nodes=16 --spot --placement-policy=tpu7x-64chip \
#         --disk-type=hyperdisk-balanced --disk-size=200
#   * 镜像是**新版** MaxText（src/maxtext 布局、nnx）。旧版 libtpu 驱动不了 Ironwood。
#   * 已按下面打过补丁
#
# 补丁（一次性，新版布局用 port.py 而不是 .patch）：
#   cp -r $MAXTEXT_SRC /tmp/mt-v7 && cd /tmp/mt-v7
#   cp hunyuan3.py       src/maxtext/models/
#   cp hunyuan3-295b.yml src/maxtext/configs/models/
#   python3 port.py            # 改 6 个文件，见脚本内注释
#
# 用法：
#   MAXTEXT_SRC=/tmp/mt-v7 GCS_STAGE=gs://your-bucket/hy3 \
#   IMAGE=us-docker.pkg.dev/PROJECT/gcr.io/your-maxtext-latest:runner \
#   bash run-v7-64.sh <run-name> [额外参数...]
set -euo pipefail
RUN=${1:?用法: run-v7-64.sh <run-name>}; shift || true
MAXTEXT_SRC=${MAXTEXT_SRC:?}; GCS_STAGE=${GCS_STAGE:?}; IMAGE=${IMAGE:?}
STEPS=${STEPS:-12}; NAME=hy3-$RUN

# 只带这些 flag。**不要**照抄官方 30 个一次全开——实测会死锁（§6.5 的 w1）。
# 调度器组是唯一值钱的一组（+6.6%），SparseCore 卸载组在 v7 上是 0。
FLAGS='--xla_tpu_scoped_vmem_limit_kib=65472 --xla_enable_async_all_gather=true
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

# c1 配置。踩过的雷：
#   use_tokamax_gmm=True        -> 死锁 stalled chips（§6.7），千万别开
#   shard_exp_on_fsdp=True      -> OOM 109 G（§6.6），192 专家换不来
#   per_device_batch_size=12    -> OOM
ARGS="override_model_config=True ici_fsdp_parallelism=-1 ici_tensor_parallelism=1 \
per_device_batch_size=8 max_target_length=4096 \
megablox=True sparse_matmul=True scan_layers=True use_custom_sort_vjp=True \
sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 sa_block_q_dkv=2048 \
sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 sa_block_q_dq=2048 sa_block_kv_dq=2048 \
sa_use_fused_bwd_kernel=True use_tokamax_splash=True \
opt_type=adamw mu_dtype=bfloat16 grad_dtype=bfloat16 use_iota_embed=True \
allow_split_physical_axes=True \
remat_policy=custom decoder_layer_input=offload out_proj=remat \
attention=flash tokenizer_type=tiktoken \
tokenizer_path=src/maxtext/assets/tokenizer_llama3.tiktoken"

cd "$MAXTEXT_SRC"
tar czf /tmp/hy3inject-v7.tgz \
  src/maxtext/common/common_types.py src/maxtext/configs/types.py \
  src/maxtext/layers/decoders.py src/maxtext/layers/nnx_decoders.py src/maxtext/layers/moe.py \
  src/maxtext/utils/maxtext_utils.py src/maxtext/models/hunyuan3.py \
  src/maxtext/configs/models/hunyuan3-295b.yml
gsutil -q cp /tmp/hy3inject-v7.tgz "$GCS_STAGE/hy3inject-v7.tgz"

kubectl delete jobset "$NAME" --ignore-not-found=true --wait=false >/dev/null 2>&1
cat <<YAML | kubectl apply -f - >/dev/null
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: $NAME
  annotations: {alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool}
spec:
  ttlSecondsAfterFinished: 7200
  failurePolicy: {maxRestarts: 0}
  replicatedJobs:
  - name: slice-job
    replicas: 1
    template:
      spec:
        parallelism: 16      # 16 台 × 4 芯片 = 64 芯片 = 128 JAX device（v7 是 2 dev/chip）
        completions: 16
        backoffLimit: 0
        template:
          spec:
            restartPolicy: Never
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x
              cloud.google.com/gke-tpu-topology: 4x4x4
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
                gsutil -q cp $GCS_STAGE/hy3inject-v7.tgz /tmp/p.tgz
                cd /deps && tar xzf /tmp/p.tgz
                export JAX_PLATFORMS=tpu,cpu TPU_STDERR_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0
                export LIBTPU_INIT_ARGS='$FLAGS'
                python3 -m src.maxtext.trainers.pre_train.train src/maxtext/configs/base.yml \\
                  model_name=hunyuan3-295b run_name=$NAME \\
                  base_output_directory=/tmp/hy3out \\
                  dataset_type=synthetic enable_checkpointing=False steps=$STEPS \\
                  dtype=bfloat16 weight_dtype=float32 $ARGS $*
              resources: {limits: {google.com/tpu: 4}}
              volumeMounts: [{mountPath: /dev/shm, name: dshm}]
            volumes: [{name: dshm, emptyDir: {medium: Memory}}]
YAML
echo "[$NAME] 已提交。跟踪：kubectl logs -f job/$NAME-slice-job-0 -c jax-tpu"
echo
echo "读结果时注意（§3.6）："
echo "  * v7 是 2 device/chip，per-chip TFLOP/s = 日志值 × 2"
echo "  * MFU = per-chip / 2307"
echo "  * 稳态取 step >= 3；v7 编译要 10-17 分钟，比 v5p 慢很多"
echo "  * 预期：step ≈ 20.4 s，日志 TFLOP/s/device ≈ 222.6，即 445 per-chip，MFU ≈ 19.3%"
