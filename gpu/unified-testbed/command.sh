# =============================================================================
# A4 Unified Testbed - 快速命令参考
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 连接到 GKE 集群
# -----------------------------------------------------------------------------
export PROJECT=gpu-launchpad-playground
export REGION=asia-southeast1
export ZONE=asia-southeast1-b
export CLUSTER_NAME=chrisya-gke-a4
export GCS_BUCKET=chrisya-gpu-pg-ase1
export ARTIFACT_REGISTRY=asia-docker.pkg.dev/gpu-launchpad-playground/chrisya-gpu-pgp-repo

gcloud config set project ${PROJECT}
gcloud config set compute/zone ${ZONE}
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION

# -----------------------------------------------------------------------------
# 2. 设置路径
# -----------------------------------------------------------------------------
export REPO_ROOT=$(git rev-parse --show-toplevel)
export RECIPE_ROOT=$REPO_ROOT/gpu/unified-testbed
export WORKLOAD_IMAGE=$ARTIFACT_REGISTRY/unified-testbed-pytorch:25.06-py3

# -----------------------------------------------------------------------------
# 3. 构建 Docker 镜像（可选，如果需要自定义镜像）
# -----------------------------------------------------------------------------
cd $RECIPE_ROOT/docker
gcloud builds submit --region=${REGION} \
    --config cloudbuild.yml \
    --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY \
    --timeout "2h" \
    --machine-type=e2-highcpu-32 \
    --quiet \
    --async

# -----------------------------------------------------------------------------
# 4. 部署工作负载
# -----------------------------------------------------------------------------

# 4.1 基础模式（交互式调试，保持容器运行）
export WORKLOAD_NAME=$USER-testbed
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/gke-runtime/launchers/torchrun-startup.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    $WORKLOAD_NAME \
    $RECIPE_ROOT/gke-runtime/jobset

# 4.2 NCCL 测试模式
export WORKLOAD_NAME=$USER-nccl-test
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/examples/nccl-test.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "workload.envs[1].value"="false" \
    $WORKLOAD_NAME \
    $RECIPE_ROOT/gke-runtime/jobset

# 4.3 DDP 测试模式
export WORKLOAD_NAME=$USER-ddp-test
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/examples/torchrun-ddp-test.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "workload.envs[1].value"="false" \
    $WORKLOAD_NAME \
    $RECIPE_ROOT/gke-runtime/jobset

# 4.4 Pai-Megatron Qwen3 训练模式（需要 NFS）
export WORKLOAD_NAME=$USER-qwen3-training
export FILESTORE_IP=<FILESTORE_IP>
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/examples/pai-megatron-qwen3.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "volumes.nfs.enabled"=true \
    --set "volumes.nfs.ip"=$FILESTORE_IP \
    --set "workload.envs[1].value"="false" \
    $WORKLOAD_NAME \
    $RECIPE_ROOT/gke-runtime/jobset

# 4.5 Pai-Megatron Qwen3-Next 训练模式（需要 NFS）
export WORKLOAD_NAME=$USER-qwen3-next-training
export FILESTORE_IP=<FILESTORE_IP>
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/examples/pai-megatron-qwen3-next.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "volumes.nfs.enabled"=true \
    --set "volumes.nfs.ip"=$FILESTORE_IP \
    --set "workload.envs[1].value"="false" \
    $WORKLOAD_NAME \
    $RECIPE_ROOT/gke-runtime/jobset

# 4.6 Pai-Megatron Qwen3-Next + Lustre 高性能存储
export WORKLOAD_NAME=$USER-qwen3-next-lustre
export LUSTRE_IP=172.27.48.5
export LUSTRE_FS=lustrefs
export LUSTRE_INSTANCE=my-lustre
export LUSTRE_LOCATION=us-central1-a
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$RECIPE_ROOT/examples/pai-megatron-qwen3-next.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "volumes.lustre.enabled"=true \
    --set "volumes.lustre.ip"=$LUSTRE_IP \
    --set "volumes.lustre.filesystem"=$LUSTRE_FS \
    --set "volumes.lustre.instanceName"=$LUSTRE_INSTANCE \
    --set "volumes.lustre.projectId"=$PROJECT \
    --set "volumes.lustre.location"=$LUSTRE_LOCATION \
    --set "workload.envs[1].value"="false" \
    $WORKLOAD_NAME \
    $RECIPE_ROOT/gke-runtime/jobset

# -----------------------------------------------------------------------------
# 5. 查看工作负载状态
# -----------------------------------------------------------------------------

# 查看 Pod 状态
kubectl get pods | grep $USER

# 查看 JobSet 状态
kubectl get jobset | grep $USER

# 查看详细信息
kubectl describe jobset $WORKLOAD_NAME

# 查看事件
kubectl get events --sort-by=.metadata.creationTimestamp | tail -20

# -----------------------------------------------------------------------------
# 6. 查看日志
# -----------------------------------------------------------------------------

# 获取 Pod 名称
kubectl get pods | grep $WORKLOAD_NAME

# 查看主节点日志（替换 xxxxx 为实际 Pod 后缀）
kubectl logs -f $WORKLOAD_NAME-workload-0-0-xxxxx

# 进入容器调试
kubectl exec -it $WORKLOAD_NAME-workload-0-0-xxxxx -- bash

# 在容器内检查 GPU 状态
nvidia-smi

# 在容器内运行 NCCL 测试
/third_party/nccl-tests/build/all_reduce_perf -b 1M -e 1G -f 2

# -----------------------------------------------------------------------------
# 7. 卸载工作负载
# -----------------------------------------------------------------------------
helm uninstall $WORKLOAD_NAME

# 批量卸载所有测试工作负载
helm uninstall $USER-testbed
helm uninstall $USER-nccl-test
helm uninstall $USER-ddp-test
helm uninstall $USER-qwen3-training
helm uninstall $USER-qwen3-next-training
helm uninstall $USER-qwen3-next-lustre

# -----------------------------------------------------------------------------
# 8. 常用调试命令
# -----------------------------------------------------------------------------

# 检查 NCCL 环境变量
kubectl exec -it <POD_NAME> -- env | grep NCCL

# 检查 GIB 配置
kubectl exec -it <POD_NAME> -- cat /usr/local/gib/scripts/set_nccl_env.sh

# 检查 NFS 挂载
kubectl exec -it <POD_NAME> -- df -h | grep mnt

# 检查 Lustre 挂载
kubectl exec -it <POD_NAME> -- df -h | grep lustre

# 查看分布式训练环境变量
kubectl exec -it <POD_NAME> -- env | grep -E "(MASTER|NNODES|RANK|WORLD)"
