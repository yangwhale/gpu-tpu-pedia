#!/usr/bin/env bash
# =============================================================================
# 07-mtc-setup.sh —— §7.1：打开 Multi-Tier Checkpointing（addon → HNS 桶 → IAM → CheckpointConfiguration → 验证）
#
# 出处（公开文档 https://docs.cloud.google.com/kubernetes-engine/docs/how-to/machine-learning/training/multi-tier-checkpointing ）：
#   - addon：clusters update --update-addons=HighScaleCheckpointing=ENABLED,GcsFuseCsiDriver=ENABLED；另需 --workload-pool
#     集群是 zonal，所以 --location=$ZONE（location = 集群所在 zone，highScaleCheckpointingConfig/gcsFuseCsiDriverConfig
#     均 enabled、workloadPool=<project>.svc.id.goog）
#   - IAM：principal://…/subject/ns/gke-managed-checkpointing/sa/gke-checkpointing-multitier-node → roles/storage.objectUser
#   - 验证 kubectl get pod -n gke-managed-checkpointing（每个加速器节点一个 multitier-driver）；卸载 = 删 CR
#   - 未开 addon 时 apply 报 no matches for kind "CheckpointConfiguration"
#   - HNS 桶：https://cloud.google.com/storage/docs/create-hns-bucket（--uniform-bucket-level-access --enable-hierarchical-namespace）
#   - DaemonSet 标签 k8s-app=high-scale-checkpointing、highscalecheckpointing.gke.io/config-name=<CR 名>
#
# 用法：
#   ./07-mtc-setup.sh              # dry-run：检查现状，打印将执行的命令
#   ./07-mtc-setup.sh --execute    # 缺什么补什么
#   ./07-mtc-setup.sh --verify     # 只验证
#   ./07-mtc-setup.sh --delete --execute   # 删 CR（driver 被撤、ramdisk 释放）；不删桶、不关 addon
# 幂等：每步先查；addon 已开、桶已存在、IAM 已绑、CR 已存在都跳过（CR 用 kubectl apply，改了 env 重跑即更新）。
#       集群升级前后要删了重建 CR（driver DaemonSet 不自动升级：公开文档「Cluster upgrades」）。
# 前置：container.clusters.update、storage.buckets.create / setIamPolicy；kubectl 集群管理员；envsubst（gettext）；
#       multitier driver ≥0.8.4 → 有效下限 GKE 1.37.0-gke.3165000（当时只有它和 3503000 带；1.32.4-gke.1415000 只是 addon 门槛）
#       （实测集群实际 v0.9.0-gke.2）。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
rb_env_check PROJECT REGION ZONE CLUSTER MTC_BUCKET MTC_CONFIG_NAME MTC_INMEM_SIZE

EXEC=0; MODE=setup
for a in "$@"; do case "$a" in --execute) EXEC=1;; --verify) MODE=verify;; --delete) MODE=delete;; esac; done
run() { echo "+ $*"; if [[ $EXEC == 1 ]]; then "$@"; else echo "  (dry-run，未执行)"; fi; }
CR_YAML="$(dirname "$0")/07-checkpoint-configuration.yaml"

verify() {
  echo "== 验证"
  local tpu drv ready
  # 期望数按 DaemonSet 实际 nodeSelector 数：实测集群的 DS 额外带 cloud.google.com/gke-networking-dra-driver=true（控制器加的 [推断]），
  # 当时两千二百多个 TPU 节点里只有两千多个有这个标签、也正好两千多个 driver pod
  local sel; sel=$(kubectl get ds -n gke-managed-checkpointing -l "highscalecheckpointing.gke.io/config-name=$MTC_CONFIG_NAME" -o json \
    | jq -r '[.items[0].spec.template.spec.nodeSelector // {} | to_entries[] | "\(.key)=\(.value)"] | join(",")')
  tpu=$(kubectl get nodes -l "${sel:-cloud.google.com/gke-tpu-accelerator=tpu7x}" --no-headers 2>/dev/null | wc -l)
  echo "  DS nodeSelector：${sel:-<DS 不存在>}；TPU 节点总数 $(kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x --no-headers 2>/dev/null | wc -l)"
  drv=$(kubectl get pods -n gke-managed-checkpointing -l "highscalecheckpointing.gke.io/config-name=$MTC_CONFIG_NAME" --no-headers 2>/dev/null | wc -l)
  ready=$(kubectl get pods -n gke-managed-checkpointing -l "highscalecheckpointing.gke.io/config-name=$MTC_CONFIG_NAME" -o json \
    | jq '[.items[] | select(.status.phase=="Running" and ([.status.containerStatuses[]?.ready] | all))] | length')
  echo "  CR：$(kubectl get checkpointconfigurations.checkpointing.gke.io "$MTC_CONFIG_NAME" -o name 2>&1)"
  echo "  匹配 nodeSelector 的节点 $tpu；driver pod $drv；全部容器 Ready 的 $ready（期望三者相等；实测集群每 pod 6/6 容器）"
  kubectl get pods -n gke-managed-checkpointing --no-headers 2>/dev/null | awk '$3!="Running"{print "  未 Running：",$0}' | head -20
  [[ $tpu -gt 0 && $tpu == "$ready" ]] && echo "  OK" || { echo "  未达标：kubectl describe pod <上面的 pod> -n gke-managed-checkpointing（403 = 桶 IAM，见公开文档 Troubleshoot）"; return 1; }
}

case $MODE in
  verify) verify; exit $?;;
  delete)
    run kubectl delete checkpointconfigurations.checkpointing.gke.io "$MTC_CONFIG_NAME" --ignore-not-found
    echo "验证：kubectl get pods -n gke-managed-checkpointing -l highscalecheckpointing.gke.io/config-name=$MTC_CONFIG_NAME  → 期望 No resources"
    exit 0;;
esac

# ---- 1. addon + Workload Identity
cj=$(gcloud container clusters describe "$CLUSTER" --location="$ZONE" --project="$PROJECT" --format=json)
hsc=$(jq -r '.addonsConfig.highScaleCheckpointingConfig.enabled // false' <<<"$cj")
fuse=$(jq -r '.addonsConfig.gcsFuseCsiDriverConfig.enabled // false' <<<"$cj")
pool=$(jq -r '.workloadIdentityConfig.workloadPool // ""' <<<"$cj")
echo "现状 HighScaleCheckpointing=$hsc GcsFuse=$fuse workloadPool=${pool:-<无>}"
[[ -n "${pool:-}" ]] || run gcloud container clusters update "$CLUSTER" --location="$ZONE" --project="$PROJECT" --workload-pool="$PROJECT.svc.id.goog"
[[ "$hsc" == true && "$fuse" == true ]] || run gcloud container clusters update "$CLUSTER" --location="$ZONE" --project="$PROJECT" \
  --update-addons=HighScaleCheckpointing=ENABLED,GcsFuseCsiDriver=ENABLED
# 回滚：--update-addons=HighScaleCheckpointing=DISABLED（先删 CR）[需验证：公开文档只写了 ENABLED 的形式]

# ---- 2. HNS 桶（同 region）
if gcloud storage buckets describe "gs://$MTC_BUCKET" --project="$PROJECT" --format=json >/tmp/mtc-bucket.json 2>/dev/null; then
  hns=$(jq -r '(.hierarchical_namespace // .hierarchicalNamespace // {}) | .enabled // false' /tmp/mtc-bucket.json)
  [[ "$hns" == true ]] && echo "桶 gs://$MTC_BUCKET 已存在（HNS）" \
    || { echo "桶 gs://$MTC_BUCKET 已存在但没开 HNS——HNS 不能事后开，换名重建" >&2; exit 1; }
else
  run gcloud storage buckets create "gs://$MTC_BUCKET" --project="$PROJECT" --location="$REGION" \
    --uniform-bucket-level-access --enable-hierarchical-namespace
fi

# ---- 3. IAM（重复 add 是幂等的）
PNUM=$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')
run gcloud storage buckets add-iam-policy-binding "gs://$MTC_BUCKET" \
  --member="principal://iam.googleapis.com/projects/$PNUM/locations/global/workloadIdentityPools/$PROJECT.svc.id.goog/subject/ns/gke-managed-checkpointing/sa/gke-checkpointing-multitier-node" \
  --role=roles/storage.objectUser
# 回滚：同参数的 remove-iam-policy-binding

# ---- 4. CheckpointConfiguration
echo "+ envsubst < $CR_YAML | kubectl apply -f -"
envsubst '${MTC_CONFIG_NAME} ${MTC_BUCKET} ${MTC_INMEM_SIZE}' < "$CR_YAML" | grep -v '^#' | sed 's/^/    /'
if [[ $EXEC == 1 ]]; then
  envsubst '${MTC_CONFIG_NAME} ${MTC_BUCKET} ${MTC_INMEM_SIZE}' < "$CR_YAML" | kubectl apply -f -
  echo "等 driver DaemonSet 铺满（每节点拉镜像+挂 gcsfuse，约数分钟）…"; sleep 120; verify
fi
