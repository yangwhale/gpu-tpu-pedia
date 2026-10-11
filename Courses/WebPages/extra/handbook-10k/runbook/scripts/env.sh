# shellcheck shell=bash
# =============================================================================
# env.sh —— 全书唯一的变量块（模板）。复制一份填好，每个终端 `source ~/runbook-env.sh`，不要在章节里另起一套。
#
# 用法：
#   cp runbook/scripts/env.sh ~/runbook-env.sh && vi ~/runbook-env.sh   # 填 <…> 占位
#   source ~/runbook-env.sh && rb_env_check                        # 缺值就报错退出
#
# 命名规则：后缀带 zone；
# 术语（同手册 00 §6）：「实测集群」= 本书脚本实际跑过的整 pod；「参考集群」= Google 另一个整 pod，用 Cluster Toolkit 蓝图建。
# 网段取自参考集群蓝图（实测集群子网网段未采集，Pod CIDR 两边一致 10.224.0.0/12）。
# 本文件只 export，不做任何修改云资源的动作；重复 source 是幂等的。
# =============================================================================

# ---- 项目与位置 ---------------------------------------------------------------
export PROJECT="${PROJECT:-<使用方 GKE 项目 ID>}"      # 写项目 ID，不写项目编号
export REGION="${REGION:-<region>}"                   # 例：us-central1
export ZONE="${ZONE:-<zone>}"                         # 例：us-central1-b。集群是 zonal
export CLUSTER="${CLUSTER:-<cluster>}"                # 上限 = 63 − len(ZONE) − 33：us-central1-b 17 字符（经验教训第 13 条；CTK 的 24 不适用本书脚本），rb_env_check 会算

# ---- 容量 -------------------------------------------------------------------
export RESERVATION="${RESERVATION:-}"                 # 名字到手前留空；rb_env_check 会拦住
export BLOCK="${BLOCK:-${RESERVATION}-block-0001}"    # 整 pod = 1 个 block
export CUBES_EXPECTED="${CUBES_EXPECTED:-144}"        # sub-block 数
export CUBES_MAX_JOB="${CUBES_MAX_JOB:-128}"          # 最大规划任务 128 cube = 8,192 chip（00 §4 账本；SLO 线见下面的 CUBES_SLO）
# 01/08 章用：
export CUBES_SLO="${CUBES_SLO:-130}"        # SLO 线，多数资料 130/144，待书面确认（09 D02）
export SPARE_CUBES="${SPARE_CUBES:-}"                 # 每日留空 S（配额 = min(CUBES_MAX_JOB, 健康 cube − S)×64 里的 S，同 03-addons.sh quota；不是规划空出的 16，见 09 D04），与使用方书面约定；空 = 未约定，08-daily-check 按 0 算并告警，03-addons.sh quota 不给建议值。
#   故意不放进 rb_env_check 默认必填：02-create-tpu-nodepools.sh 用默认列表，D04 定下来之前会被拦死；要它的脚本自己列

# ---- 版本 -------------------------------------------------------------------
export GKE_VERSION="${GKE_VERSION:-<BOM 里的 GKE 版本>}"   # 按 BOM 填；本书实测走 RAPID 通道
export RELEASE_CHANNEL="${RELEASE_CHANNEL:-rapid}"
export NODE_VERSION="${NODE_VERSION:-${GKE_VERSION}}"

# ---- 网络（建池前冻结，建完不能改：经验教训第 17 条）-------------------------------------
export NET_MAIN="${NET_MAIN:-${CLUSTER}-main-mtu9k-${ZONE}}"
export SUBNET_MAIN="${SUBNET_MAIN:-${CLUSTER}-privatesubnet-1-${ZONE}}"
export NET_DCN1="${NET_DCN1:-${CLUSTER}-dcn-privatenet-2-${ZONE}}"
export SUBNET_DCN1="${SUBNET_DCN1:-${CLUSTER}-privatesubnet-2-${ZONE}}"
export NET_DCN2="${NET_DCN2:-${CLUSTER}-dcn-privatenet-3-${ZONE}}"
export SUBNET_DCN2="${SUBNET_DCN2:-${CLUSTER}-privatesubnet-3-${ZONE}}"
export MTU="${MTU:-8896}"
export SUBNET_MAIN_RANGE="${SUBNET_MAIN_RANGE:-10.11.0.0/18}"   # 参考集群蓝图 primary_subnet_ip
export SUBNET_DCN1_RANGE="${SUBNET_DCN1_RANGE:-10.12.0.0/18}"   # 参考集群蓝图 tpu_subnet_1_ip
export SUBNET_DCN2_RANGE="${SUBNET_DCN2_RANGE:-10.13.0.0/18}"   # 参考集群蓝图 tpu_subnet_2_ip
export POD_CIDR="${POD_CIDR:-10.224.0.0/12}"                    # 两个集群一致；/14 在 1,011 VM 处耗尽
export SERVICES_CIDR="${SERVICES_CIDR:-10.0.32.0/20}"           # 参考集群蓝图 services_ip_range（实测集群用的是自动分配 34.118.224.0/20）
export MASTER_CIDR="${MASTER_CIDR:-172.16.0.32/28}"             # 两个集群一致
export ROUTER="${ROUTER:-${CLUSTER}-router-${REGION}}"
export NAT="${NAT:-${CLUSTER}-nat-${REGION}}"

# ---- 节点池 -----------------------------------------------------------------
export MACHINE_TYPE="${MACHINE_TYPE:-tpu7x-standard-4t}"
export TPU_TOPOLOGY="${TPU_TOPOLOGY:-4x4x4}"                    # PROVISION_ONLY 在 tpu7x 上只接受 4x4x4
export NODES_PER_POOL="${NODES_PER_POOL:-16}"
export POLICY_NAME="${POLICY_NAME:-${CLUSTER}-ss-policy}"       # 例：<集群名>-ss-policy
export TPU_DISK_TYPE="${TPU_DISK_TYPE:-hyperdisk-balanced}"
export TPU_DISK_SIZE_GB="${TPU_DISK_SIZE_GB:-100}"
export TPU_MAX_PODS="${TPU_MAX_PODS:-32}"                       # 每节点 /26；110 会让 Pod CIDR 提前耗尽
export POOL_PREFIX="${POOL_PREFIX:-tpu7x-pool-sb}"            # 池名必须带 sub-block 号
export SYSTEM_POOL="${SYSTEM_POOL:-default-pool}"   # ⚠ 别改名：02-coredns.yaml 与 03-slice-controller.yaml 把 nodeSelector 写死成 default-pool；改了这里不改那两处，slice-controller Pending → fail-closed webhook 挡住全集群建 Job
export SYSTEM_MACHINE_TYPE="${SYSTEM_MACHINE_TYPE:-n2-standard-32}"  # 参考集群；例：n2-standard-16 × 6
export SYSTEM_POOL_NODES="${SYSTEM_POOL_NODES:-18}"             # 自管 CoreDNS 15 副本强制反亲和 → 至少 15 台（参考集群 min 18）
# DNS 路线（§2.6，上线前定，改要滚全部节点）：
#   coredns  = kube-dns 起步 + 自管 CoreDNS 15 副本 + NodeLocal DNSCache（参考集群蓝图；要求系统池 ≥15 台）
#   clouddns = Cloud DNS（cluster scope）+ NodeLocal DNSCache（系统池可小到 6 × n2-standard-16）
export CLUSTER_DNS="${CLUSTER_DNS:-coredns}"

# ---- K8s 侧 -----------------------------------------------------------------
export KUEUE_VERSION="${KUEUE_VERSION:-0.18.10}"                # 钉 0.18.x ≥ 0.18.5（01 §1.6）
export JOBSET_VERSION="${JOBSET_VERSION:-0.12.0}"               # 参考集群的 jobset-values.yaml
export NS="${NS:-<使用方任务 namespace>}"                        # 使用方任务 namespace，必填，不给默认值（默认 default 会掩盖漏写 -n $NS 的命令）；实测集群是显式填的 NS=default
export LOCAL_QUEUE="${LOCAL_QUEUE:-multislice-queue}"
# 03/04 章新增（03-addons.sh、04-*.sh 用）：
export TPU_QUOTA="${TPU_QUOTA:-}"                               # ClusterQueue nominalQuota（芯片数）= min(CUBES_MAX_JOB, 健康 cube − SPARE_CUBES)×64，下限 0，每天重算；`03-addons.sh quota` 给建议值。
#   健康 cube 定义同 03-addons.sh quota：16 个节点全 Ready + partition-4x4x4-state=HEALTHY + allocatable google.com/tpu=4 + 没有 PartOfBrokenSlice 污点（被隔离的不算）
export INSPECT_NS="${INSPECT_NS:-rb-inspect}"                  # 04 §4.1 巡检/网卡检查专用 namespace；已写进 03-kueue-values.yaml 的 Kueue 排除列表，改名两处一起改
export TEST_IMAGE="${TEST_IMAGE:-mirror.gcr.io/library/python@sha256:4d1caded1f729ae443eb803f26ffde7b61e696aeaef62f099abb6dd6b14257c7}"  # 04 冒烟镜像：参考作业 JobSet 用的 python 镜像 digest。
#   使用方要换成自己集群能拉到的镜像 digest（例如使用方镜像仓库里的同一镜像）；mirror.gcr.io 在使用方网络下能不能拉 —— 未核
# 05/07 章用：
export CLUSTER_QUEUE="${CLUSTER_QUEUE:-cluster-queue}"          # 03 §3.5 的 CQ 名（实测集群）
export TPU_FLAVOR="${TPU_FLAVOR:-tpu7x-flavor-v2}"              # 03 §3.5 的 ResourceFlavor 名
export MTC_CONFIG_NAME="${MTC_CONFIG_NAME:-mtc-config}"         # CheckpointConfiguration 名，集群级（参考作业）
export MTC_BUCKET="${MTC_BUCKET:-${CLUSTER}-mtc-${REGION}}"     # MTC 备份桶，必须 HNS（实测集群用的是 <数据桶>）
export MTC_INMEM_SIZE="${MTC_INMEM_SIZE:-100Gi}"                # ≈ 本地检查点 ×2.2；实测集群 tmpfs-size-mib=102400
# 06c 章用：
export DATA_BUCKET="${DATA_BUCKET:-<数据桶名，不带 gs://>}"      # 实测集群 <数据桶>
export DATA_PV="${DATA_PV:-${NS}-data-pv}"                      # 实测集群 <数据卷>
export DATA_PVC="${DATA_PVC:-${NS}-data-pvc}"                   # 实测集群 <数据卷>（ns default）
#   DATA_PV / DATA_PVC 在 source 时就按 NS 定好了名字，之后再改 NS 不会跟着变。在 $INSPECT_NS 建巡检版时，三个变量必须一起覆盖：
#   NS=$INSPECT_NS DATA_PV=${DATA_PV}-inspect DATA_PVC=${DATA_PVC}-inspect envsubst < 06-data-pv-pvc.yaml（PV 是集群级对象，名字不能和任务版重复；见 06c §6.1）
export KSA="${KSA:-default}"                                    # 训练 pod 的 K8s SA；实测集群 ns/default/sa/default
export RUN_NAME="${RUN_NAME:-my-run}"                          # overlay/日志目录都按它命名
export JOBSET_NAME="${JOBSET_NAME:-my-run}"                    # 换成你的作业名；≤ 20 字符，且 namespace + 名字 + worker + 下标位数 ≤ 32 → ns=default、10+ 片时 ≤ 17
export ENV_TAR="${ENV_TAR:-pretrain/env/maxtext-<commit7>-py312.tar}"  # 桶内路径
export GITHUB_SECRET="${GITHUB_SECRET:-github-token}"          # 私有代码仓库 token 的 K8s Secret（key: token）

# ---- 工作目录 ----------------------------------------------------------------
# 本文件就在 runbook/scripts/ 里时自动推 RUNBOOK_DIR；复制出去用时必须手填。
_rb_self="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd)"
if [[ ( -z "${RUNBOOK_DIR:-}" || "${RUNBOOK_DIR}" == \<* ) && -f "$_rb_self/../scripts/env.sh" ]]; then RUNBOOK_DIR="$(cd "$_rb_self/.." && pwd)"; fi
export RUNBOOK_DIR="${RUNBOOK_DIR:-<解压目录>/runbook}"      # 复制出去用时必填，例：$HOME/runbook
# 没填时 source 当场就提示（不退出，避免把交互 shell 关掉）；05/07/08 用 "$RUNBOOK_DIR"/scripts/… 的命令会失败
[[ -f "$RUNBOOK_DIR/scripts/env.sh" ]] || echo "env: RUNBOOK_DIR=$RUNBOOK_DIR 不是 runbook 目录，请在 ~/runbook-env.sh 里填（rb_env_check 会报同样的错）" >&2
export STATE_DIR="${STATE_DIR:-$HOME/runbook-state/${CLUSTER}}"     # 脚本写日志/清单的地方
mkdir -p "$STATE_DIR" 2>/dev/null || true

rb_env_check() {   # 用法：rb_env_check            检查默认 6 项
  local bad=0 v      #       rb_env_check A B ...  只检查列出的变量（例如建 VPC 时还没有 RESERVATION）
  local vars=("$@"); (( ${#vars[@]} )) || vars=(PROJECT REGION ZONE CLUSTER RESERVATION GKE_VERSION)
  for v in "${vars[@]}"; do
    if [[ -z "${!v}" || "${!v}" == \<* ]]; then echo "env: $v 未填（当前='${!v}'）" >&2; bad=1; fi
  done
  if [[ ! -f "$RUNBOOK_DIR/scripts/env.sh" ]]; then echo "env: RUNBOOK_DIR=$RUNBOOK_DIR 不是 runbook 目录（复制出去的 env 要手填）" >&2; bad=1; fi
  if [[ "$PROJECT" =~ ^[0-9]+$ ]]; then echo "env: PROJECT 是编号不是 ID" >&2; bad=1; fi
  if [[ "$ZONE" != "$REGION"-* ]]; then echo "env: ZONE=$ZONE 不在 REGION=$REGION 里" >&2; bad=1; fi
  # 集群名上限按最长派生名算：防火墙规则 ${CLUSTER}-dcn-privatenet-2-${ZONE}-allow-internal ≤ 63（经验教训第 13 条；us-central1-b → 17）
  local maxc=$(( 63 - ${#ZONE} - 18 - 15 ))   # 18 = "-dcn-privatenet-2-"，15 = "-allow-internal"
  # 只在「要新建」的脚本里判（调用时加前缀 `RB_CREATING=1 rb_env_check …`：02-create-networks.sh / 02-create-cluster.sh）。
  # 只在新建时拦：已建好的集群（这些运维脚本都列了 CLUSTER）不能因名字长度被拦。已有集群的名字改不了，运维脚本只提示不拦。
  if (( ${#CLUSTER} > maxc )); then
    if [[ "${RB_CREATING:-0}" == 1 ]]; then echo "env: CLUSTER 名 ${#CLUSTER} 字符 > $maxc（ZONE=$ZONE 下的上限），02 §2.1 建防火墙会失败" >&2; bad=1; fi
  fi
  return $bad
}
