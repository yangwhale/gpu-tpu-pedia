#!/usr/bin/env bash
# =============================================================================
# 02-create-cluster.sh —— §2.2：建 zonal GKE 集群（含系统池 default-pool），然后取凭据、验证
#
# 目标状态 = 实测集群（逐项对照见下方注释）。flag 名按公开 gcloud 参考核对：
#   https://cloud.google.com/sdk/gcloud/reference/container/clusters/create
#   （--addons 取值含 NodeLocalDNS / GcsFuseCsiDriver / HighScaleCheckpointing / GcePersistentDiskCsiDriver /
#     GcpFilestoreCsiDriver；--monitoring 取值含 JOBSET，且这些组件要求开 Managed Prometheus；DCGM 是 NVIDIA GPU 专用，TPU 集群不开；
#     --cluster-dns 取值 clouddns|default|kubedns）
# 集群级 flag 组合的已验证来源：
#   - 系统池规格：参考集群蓝图（n2-standard-32、min 18，CoreDNS 反亲和要求 ≥15 台）/ 实测集群 n2-standard-16 × 6
# [需验证] 整条命令没有在任何集群上原样跑过；先 --dry-run 看一眼再执行。
#
# 幂等：集群已存在就跳过 create，直接做验证。
# 回滚：gcloud container clusters delete "$CLUSTER" --location="$ZONE" --project="$PROJECT"
#       （删集群会连带删所有节点池；reservation 本身不受影响，sub-block 会被释放回 reservation）
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
RB_CREATING=1 rb_env_check PROJECT REGION ZONE CLUSTER GKE_VERSION   # RB_CREATING=1：顺带判集群名长度上限
DRY=0; [[ "${1:-}" == --dry-run ]] && DRY=1
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }

case "$CLUSTER_DNS" in
  clouddns) DNS_FLAGS=(--cluster-dns=clouddns --cluster-dns-scope=cluster) ;;   # 实测：CLOUD_DNS / CLUSTER_SCOPE
  coredns)  DNS_FLAGS=() ;;                                                     # kube-dns 起步，§2.6 再切自管 CoreDNS
  *) echo "CLUSTER_DNS 只能是 coredns|clouddns" >&2; exit 1 ;;
esac

if gcloud container clusters describe "$CLUSTER" --location="$ZONE" --project="$PROJECT" >/dev/null 2>&1; then
  echo "集群 $CLUSTER 已存在，跳过创建"
else
  run gcloud container clusters create "$CLUSTER" \
    --project="$PROJECT" --location="$ZONE" \
    --cluster-version="$GKE_VERSION" --release-channel="$RELEASE_CHANNEL" \
    --network="$NET_MAIN" --subnetwork="$SUBNET_MAIN" \
    --enable-ip-alias --cluster-ipv4-cidr="$POD_CIDR" --services-ipv4-cidr="$SERVICES_CIDR" \
    --default-max-pods-per-node=110 \
    --enable-dataplane-v2 --enable-multi-networking \
    --enable-private-nodes --master-ipv4-cidr="$MASTER_CIDR" \
    --enable-dns-access --no-enable-master-authorized-networks \
    `# ⚠ 照实测集群/参考样例：控制面 IP 端点不限来源（仍需 IAM）。使用方安全要求时改 --enable-master-authorized-networks --master-authorized-networks=<网段>（02 §2.2 表注）` \
    --enable-slice-controller \
    --addons=NodeLocalDNS,GcsFuseCsiDriver,HighScaleCheckpointing,GcePersistentDiskCsiDriver,GcpFilestoreCsiDriver \
    "${DNS_FLAGS[@]}" \
    --workload-pool="${PROJECT}.svc.id.goog" \
    --enable-managed-prometheus \
    --monitoring=SYSTEM,STORAGE,HPA,POD,DAEMONSET,DEPLOYMENT,STATEFULSET,CADVISOR,KUBELET,JOBSET \
    --logging=SYSTEM,WORKLOAD \
    --machine-type="$SYSTEM_MACHINE_TYPE" --num-nodes="$SYSTEM_POOL_NODES" \
    --disk-type=pd-balanced --disk-size=100 --image-type=COS_CONTAINERD --enable-gvnic \
    --workload-metadata=GKE_METADATA \
    --enable-shielded-nodes
fi
# ↑ --workload-metadata=GKE_METADATA 显式写上（与 02-create-tpu-nodepools.sh 对齐）：系统池要跑 06c 的 gcsfuse 冒烟 / 构建 pod，
#   gcsfuse CSI 要求节点池开 Workload Identity。实测集群 workloadPool 已开，但 default-pool 与少数单网卡池
#   workloadMetadataConfig 为空，在上面挂 gcsfuse 报「Workload Identity Federation is not enabled on node」
#   （实测），所以这里显式写上，不依赖默认值。

[[ $DRY == 1 ]] && exit 0
echo "== 取凭据（DNS endpoint，不依赖 IP 白名单）"
gcloud container clusters get-credentials "$CLUSTER" --location="$ZONE" --project="$PROJECT" --dns-endpoint

echo "== 验证：与目标状态对照"
gcloud container clusters describe "$CLUSTER" --location="$ZONE" --project="$PROJECT" --format="yaml(
  currentMasterVersion,releaseChannel.channel,location,
  addonsConfig.sliceControllerConfig,addonsConfig.highScaleCheckpointingConfig,addonsConfig.gcsFuseCsiDriverConfig,addonsConfig.dnsCacheConfig,
  networkConfig.datapathProvider,networkConfig.enableMultiNetworking,networkConfig.dnsConfig,
  clusterIpv4Cidr,privateClusterConfig.enablePrivateNodes,privateClusterConfig.masterIpv4CidrBlock,
  controlPlaneEndpointsConfig.dnsEndpointConfig.allowExternalTraffic,workloadIdentityConfig.workloadPool,
  monitoringConfig.managedPrometheusConfig)"
kubectl get crd slices.accelerator.gke.io
kubectl get nodes -l cloud.google.com/gke-nodepool="$SYSTEM_POOL" --no-headers | awk '{print $2}' | sort | uniq -c
wim=$(gcloud container node-pools describe "$SYSTEM_POOL" --cluster="$CLUSTER" --location="$ZONE" --project="$PROJECT" \
        --format='value(config.workloadMetadataConfig.mode)')
[[ "$wim" == GKE_METADATA ]] && echo "OK   $SYSTEM_POOL workloadMetadata=GKE_METADATA" \
  || echo "FAIL $SYSTEM_POOL workloadMetadata='$wim'（06c 的 gcsfuse 会挂不上）→ gcloud container node-pools update $SYSTEM_POOL --cluster=$CLUSTER --location=$ZONE --workload-metadata=GKE_METADATA"
