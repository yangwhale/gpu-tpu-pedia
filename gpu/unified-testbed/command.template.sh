# =============================================================================
# A4 Unified Testbed - 快速命令参考
# =============================================================================
# 使用说明:
#   1. 复制此文件为 command.sh: cp command.template.sh command.sh
#   2. 修改下方 <YOUR_...> 占位符为您的实际值
#   3. command.sh 已加入 .gitignore，不会被提交到版本控制
# =============================================================================

# -----------------------------------------------------------------------------
# 1. 连接到 GKE 集群
# -----------------------------------------------------------------------------
export PROJECT=<YOUR_PROJECT_ID>                    # Google Cloud 项目 ID
export REGION=<YOUR_REGION>                         # 区域，如 asia-southeast1
export ZONE=<YOUR_ZONE>                             # 可用区，如 asia-southeast1-b
export CLUSTER_NAME=<YOUR_CLUSTER_NAME>             # GKE 集群名称
export GCS_BUCKET=<YOUR_GCS_BUCKET>                 # GCS 存储桶名称
# Artifact Registry 格式：
#   多区域: asia-docker.pkg.dev, us-docker.pkg.dev, europe-docker.pkg.dev
#   单区域: us-central1-docker.pkg.dev, asia-southeast1-docker.pkg.dev
export ARTIFACT_REGISTRY=<YOUR_AR_LOCATION>-docker.pkg.dev/<YOUR_PROJECT_ID>/<YOUR_REPO>

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

# 部署结构说明:
# - 启动脚本已内置在 chart 中，无需额外指定
# - task_script: 可选，任务脚本（挂载到 /workload/task/task-script.sh）
#
# 基础模式: 不设置 task_script，容器保持运行，可 SSH 进入调试
# 测试模式: 设置 task_script 和 sleepInfinity=false，运行测试后退出

# 4.1 基础模式（交互式调试，保持容器运行）
# 只初始化分布式环境，不运行任何任务，容器保持运行供调试使用
export WORKLOAD_NAME=$USER-testbed
helm install $WORKLOAD_NAME $RECIPE_ROOT/gke-runtime/jobset \
    -f $RECIPE_ROOT/gke-runtime/values.yaml

# 4.2 NCCL 测试模式（自动运行 NCCL 测试后退出）
export WORKLOAD_NAME=$USER-nccl-test
helm install $WORKLOAD_NAME $RECIPE_ROOT/gke-runtime/jobset \
    -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file task_script=$RECIPE_ROOT/examples/nccl-test.sh \
    --set workload.sleepInfinity=false

# 4.3 DDP 测试模式（自动运行 PyTorch DDP 测试后退出）
export WORKLOAD_NAME=$USER-ddp-test
helm install $WORKLOAD_NAME $RECIPE_ROOT/gke-runtime/jobset \
    -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file task_script=$RECIPE_ROOT/examples/torchrun-ddp-test.sh \
    --set workload.sleepInfinity=false

# 4.4 Pai-Megatron Qwen3 训练模式（使用 values.yaml 中的存储配置，默认 Lustre）
export WORKLOAD_NAME=$USER-qwen3-training
helm install $WORKLOAD_NAME $RECIPE_ROOT/gke-runtime/jobset \
    -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file task_script=$RECIPE_ROOT/examples/pai-megatron-qwen3.sh \
    --set workload.sleepInfinity=false

# 4.5 Pai-Megatron Qwen3-Next 训练模式（使用 values.yaml 中的存储配置，默认 Lustre）
export WORKLOAD_NAME=$USER-qwen3-next-training
helm install $WORKLOAD_NAME $RECIPE_ROOT/gke-runtime/jobset \
    -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file task_script=$RECIPE_ROOT/examples/pai-megatron-qwen3-next.sh \
    --set workload.sleepInfinity=false

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
