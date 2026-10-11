#!/usr/bin/env bash
# =============================================================================
# 02-create-networks.sh —— §2.1：主 VPC + 2 个 DCN VPC（MTU 8896）、子网、防火墙、Cloud Router、Cloud NAT
#
# 出处：
#   - 网段 / 私有 Google 访问：参考集群蓝图 （subnet_private_access: true）
#   - 命名：实测集群节点池配置快照（<cluster>-main-mtu9k-<zone> 等）
#   - 防火墙源地址段：参考集群蓝图注释里项目自动化规则的范围（10/8,172.16/12,192.168/16,100.64/10，tcp+udp+icmp）
#     原样例的 firewall-rules create 没写 --source-ranges（= 0.0.0.0/0），这里收窄成内网段。
#
# 幂等：每个资源先 describe，已存在就跳过（不会改已存在资源的参数——MTU/网段建后不可改，发现不一致要人工处理）。
# 回滚：./02-create-networks.sh --delete --dry-run 看清单，确认后 --delete --yes（按 NAT → Router → 防火墙 → 子网 → VPC 逆序删）。
#   有集群还在用这些网络时脚本直接拒绝——NAT / Router / 防火墙在集群活着时是删得掉的，只有子网和 VPC 会被 GCE 拒。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
RB_CREATING=1 rb_env_check PROJECT REGION ZONE CLUSTER   # RB_CREATING=1：顺带判集群名长度上限

MODE=create; DRY=0
YES=0
for a in "$@"; do case "$a" in --delete) MODE=delete;; --dry-run) DRY=1;; --yes) YES=1;; esac; done
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }
exists() { "$@" >/dev/null 2>&1; }

FW_SRC="10.0.0.0/8,172.16.0.0/12,192.168.0.0/16,100.64.0.0/10"
NETS=("$NET_MAIN:$SUBNET_MAIN:$SUBNET_MAIN_RANGE" "$NET_DCN1:$SUBNET_DCN1:$SUBNET_DCN1_RANGE" "$NET_DCN2:$SUBNET_DCN2:$SUBNET_DCN2_RANGE")

if [[ $MODE == create ]]; then
  for spec in "${NETS[@]}"; do
    IFS=: read -r net sub range <<<"$spec"
    exists gcloud compute networks describe "$net" --project="$PROJECT" \
      || run gcloud compute networks create "$net" --mtu="$MTU" --bgp-routing-mode=regional \
           --subnet-mode=custom --project="$PROJECT"
    exists gcloud compute networks subnets describe "$sub" --region="$REGION" --project="$PROJECT" \
      || run gcloud compute networks subnets create "$sub" --network="$net" --range="$range" \
           --region="$REGION" --enable-private-ip-google-access --project="$PROJECT"
    exists gcloud compute firewall-rules describe "${net}-allow-internal" --project="$PROJECT" \
      || run gcloud compute firewall-rules create "${net}-allow-internal" --network="$net" \
           --direction=INGRESS --allow=tcp,udp,icmp --source-ranges="$FW_SRC" --project="$PROJECT"
  done
  # 只给主 VPC（eth0，默认路由所在）建 Router/NAT；样例在附加 VPC 上也建了一套，DCN 流量在 VPC 内，用不到 NAT。
  exists gcloud compute routers describe "$ROUTER" --region="$REGION" --project="$PROJECT" \
    || run gcloud compute routers create "$ROUTER" --network="$NET_MAIN" --region="$REGION" --project="$PROJECT"
  exists gcloud compute routers nats describe "$NAT" --router="$ROUTER" --region="$REGION" --project="$PROJECT" \
    || run gcloud compute routers nats create "$NAT" --router="$ROUTER" --region="$REGION" --project="$PROJECT" \
         --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges --enable-logging \
         --enable-dynamic-port-allocation --min-ports-per-vm=32 --max-ports-per-vm=1024
  echo "== 验证"
  gcloud compute networks list --project="$PROJECT" --filter="name~^${CLUSTER}-" --format="table(name,mtu,x_gcloud_subnet_mode)"
  gcloud compute networks subnets list --project="$PROJECT" --regions="$REGION" --filter="name~^${CLUSTER}-" \
    --format="table(name,network.basename(),ipCidrRange,privateIpGoogleAccess)"
  gcloud compute routers nats describe "$NAT" --router="$ROUTER" --region="$REGION" --project="$PROJECT" \
    --format="yaml(natIpAllocateOption,enableDynamicPortAllocation,minPortsPerVm,maxPortsPerVm,sourceSubnetworkIpRangesToNat)"
else
  # 闸门：NAT / Router / 防火墙规则在集群还活着时也删得掉（只有子网和 VPC 会被拒），
  # 删了 = 2,304 台节点断出网 + 两张 DCN VPC 的内网放行规则消失 → 多片任务全部挂住。所以先确认没有集群在用这些网络。
  users=""
  for spec in "${NETS[@]}"; do
    IFS=: read -r net _ _ <<<"$spec"
    users+=$(gcloud container clusters list --project="$PROJECT" --filter="network=$net" --format="value(name)" 2>/dev/null || echo "?查询失败")
  done
  if [[ -n "$users" ]]; then echo "拒绝删除：这些网络还有集群在用（或查询失败）：$users" >&2; exit 2; fi
  if [[ $YES != 1 && $DRY != 1 ]]; then echo "删除要显式加 --yes（先用 --delete --dry-run 看清单）" >&2; exit 2; fi
  run gcloud compute routers nats delete "$NAT" --router="$ROUTER" --region="$REGION" --project="$PROJECT" --quiet || true
  run gcloud compute routers delete "$ROUTER" --region="$REGION" --project="$PROJECT" --quiet || true
  for spec in "${NETS[@]}"; do
    IFS=: read -r net sub _ <<<"$spec"
    run gcloud compute firewall-rules delete "${net}-allow-internal" --project="$PROJECT" --quiet || true
    run gcloud compute networks subnets delete "$sub" --region="$REGION" --project="$PROJECT" --quiet || true
    run gcloud compute networks delete "$net" --project="$PROJECT" --quiet || true
  done
fi
