#!/bin/bash
# 在 256 芯片 v5p 上跑 hunyuan3-295b —— 本仓库 §四 那个 36.72% MFU 的完整复现脚本。
#
# 前置：
#   * GKE 集群里有一个 4x8x8 的 v5p 节点池（64 台 ct5p-hightpu-4t = 256 芯片）
#   * 集群装了 JobSet controller
#   * 镜像里是 MaxText commit 3eb77db3c（旧版 src/MaxText 布局）
#   * 已按下面「补丁」一节把 hunyuan3 打进一个本地 MaxText 检出
#
# 补丁（一次性）：
#   git -C $MAXTEXT_SRC apply /path/to/register-hunyuan3.patch
#   cp hunyuan3.py            $MAXTEXT_SRC/src/MaxText/layers/
#   cp hunyuan3-295b.yml      $MAXTEXT_SRC/src/MaxText/configs/models/
#   cp hunyuan3-smoke.yml     $MAXTEXT_SRC/src/MaxText/configs/models/
#   python3 verify_hunyuan3.py --root $MAXTEXT_SRC   # 8 项应全过
#
# 用法：
#   MAXTEXT_SRC=/path/to/maxtext GCS_STAGE=gs://your-bucket/hy3 \
#   IMAGE=us-docker.pkg.dev/PROJECT/gcr.io/your-maxtext:tag \
#   bash run-v5p-256.sh <run-name> [额外的 MaxText 参数...]
set -euo pipefail

RUN=${1:?用法: run-v5p-256.sh <run-name> [extra args]}; shift || true
MAXTEXT_SRC=${MAXTEXT_SRC:?需要设置 MAXTEXT_SRC}
GCS_STAGE=${GCS_STAGE:?需要设置 GCS_STAGE，例如 gs://my-bucket/hy3}
IMAGE=${IMAGE:?需要设置 IMAGE}
STEPS=${STEPS:-15}
NAME=hy3-$RUN

# ---- 官方 DeepSeek3 v5p 配方的 30 个 XLA flag 里，MaxText 3eb77db3c 所带
# ---- libtpu（libtpu_lts_20250721）认识的 23 个，加 3 个 SparseCore 运行模式。
# ---- 少一个都可能报依赖缺失；多一个不认识的会让进程直接退出。
LIBTPU_FLAGS='--xla_tpu_dvfs_p_state=3 --xla_tpu_scoped_vmem_limit_kib=65472
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
--2a886c8_chip_config_name=megachip_tccontrol
--xla_sc_enable_instruction_fusion=false --xla_sc_disjoint_spmem=false'
LIBTPU_FLAGS=$(echo $LIBTPU_FLAGS)   # 压成一行

# ---- §4.4 消融出来的最优参数集 ----
# 关键几项及其实测代价（去掉会损失多少）：
#   ici_fsdp_parallelism=-1 且不开 EP  改成 EP=64 会 OOM 326 G（§4.2）
#   use_custom_sort_vjp=True           关掉 −6.15 pp（§4.4）
#   per_device_batch_size=8            降到 4 少 5.2 pp
#   max_target_length=8192             降到 4096 少 9.2 pp
#   out_proj=remat                     官方 DSV3 用 offload，Hy3 上要改回来
BEST_ARGS="ici_fsdp_parallelism=-1 ici_tensor_parallelism=1 \
per_device_batch_size=8 max_target_length=8192 \
megablox=True sparse_matmul=True scan_layers=True use_custom_sort_vjp=True \
sa_block_q=2048 sa_block_kv=2048 sa_block_kv_compute=2048 \
sa_block_q_dkv=2048 sa_block_kv_dkv=2048 sa_block_kv_dkv_compute=2048 \
sa_block_q_dq=2048 sa_block_kv_dq=2048 sa_use_fused_bwd_kernel=False \
remat_policy=custom decoder_layer_input=offload out_proj=remat \
tile_batch_seq=512 tile_embed_dim=1024 tile_mlp_dim=1024 attention=flash"

# 1) 把改过的文件打包上传（64 个 pod 用 kubectl cp 不现实，走 GCS 中转）
cd "$MAXTEXT_SRC"
tar czf /tmp/hy3inject.tgz \
  src/MaxText/common_types.py src/MaxText/pyconfig.py src/MaxText/maxtext_utils.py \
  src/MaxText/layers/decoders.py src/MaxText/layers/moe.py src/MaxText/layers/hunyuan3.py \
  src/MaxText/configs/base.yml \
  src/MaxText/configs/models/hunyuan3-295b.yml src/MaxText/configs/models/hunyuan3-smoke.yml
gsutil -q cp /tmp/hy3inject.tgz "$GCS_STAGE/hy3inject.tgz"

# 2) 提交 JobSet
kubectl delete jobset "$NAME" --ignore-not-found=true --wait=false >/dev/null 2>&1
cat <<YAML | kubectl apply -f - >/dev/null
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: $NAME
  annotations:
    alpha.jobset.sigs.k8s.io/exclusive-topology: cloud.google.com/gke-nodepool
spec:
  ttlSecondsAfterFinished: 7200
  failurePolicy: {maxRestarts: 0}
  replicatedJobs:
  - name: slice-job
    replicas: 1
    template:
      spec:
        parallelism: 64      # 64 台 × 4 芯片 = 256 芯片；多机 TPU 池全有全无
        completions: 64
        backoffLimit: 0
        template:
          spec:
            restartPolicy: Never
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu-v5p-slice
              cloud.google.com/gke-tpu-topology: 4x8x8
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
                gsutil -q cp $GCS_STAGE/hy3inject.tgz /tmp/p.tgz
                cd /deps && tar xzf /tmp/p.tgz
                export JAX_PLATFORMS=tpu,cpu TPU_STDERR_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0
                export LIBTPU_INIT_ARGS='$LIBTPU_FLAGS'
                python3 -m src.MaxText.train src/MaxText/configs/base.yml \\
                  model_name=hunyuan3-295b run_name=$NAME \\
                  base_output_directory=/tmp/hy3out \\
                  dataset_type=synthetic enable_checkpointing=False steps=$STEPS \\
                  dtype=bfloat16 weight_dtype=float32 \\
                  tokenizer_type=tiktoken \\
                  tokenizer_path=src/MaxText/assets/tokenizer_llama3.tiktoken \\
                  $BEST_ARGS $*
              resources: {limits: {google.com/tpu: 4}}
              volumeMounts: [{mountPath: /dev/shm, name: dshm}]
            volumes: [{name: dshm, emptyDir: {medium: Memory}}]
YAML
echo "[$NAME] 已提交。跟踪：kubectl logs -f job/$NAME-slice-job-0 -c jax-tpu"
echo
echo "读结果时注意（§3.7）："
echo "  * step 0 含编译，step 1/2 是 JAX 异步派发的假读数，稳态取 step >= 3"
echo "  * v5p 是 MegaCore，1 device = 1 chip，日志里的 TFLOP/s/device 不用乘 2"
echo "  * MFU = TFLOP/s/device / 459"
echo "  * 预期：step ≈ 59.6 s，TFLOP/s/device ≈ 168.6，MFU ≈ 36.7%，整机 ≈ 281,500 tok/s"
