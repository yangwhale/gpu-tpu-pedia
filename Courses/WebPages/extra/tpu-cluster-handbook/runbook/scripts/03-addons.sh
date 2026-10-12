#!/usr/bin/env bash
# =============================================================================
# 03-addons.sh —— 第 03 章全部改状态的动作：JobSet、Kueue、kueue-slice-controller、Kueue 对象，以及它们的验证/回滚/拆除
#
# 用法：
#   ./03-addons.sh preflight                    # 只读：工具、kubectl 上下文、托管 Slice CRD、default-pool
#   ./03-addons.sh jobset          [--dry-run]  # §3.1 helm upgrade --install jobset
#   ./03-addons.sh kueue           [--dry-run]  # §3.2 helm upgrade --install kueue（改 03-kueue-values.yaml 后也跑它）
#   ./03-addons.sh slice-controller [--dry-run] # §3.3 kubectl apply 03-slice-controller.yaml
#   ./03-addons.sh quota [--assume-isolated=<ids>] [--assume-restored=<ids>]   # §3.5 只读：按当前 HEALTHY cube 数算建议的 TPU_QUOTA（全书配额唯一口径）
#   ./03-addons.sh queues          [--dry-run]  # §3.4/§3.5 PriorityClass → Topology/AC/RF → CQ/LQ
#   ./03-addons.sh verify                        # §3.7 全部只读检查
#   ./03-addons.sh rollback-kueue <REV>  | rollback-jobset <REV>     # helm rollback 到指定 revision（先看 helm history）
#   ./03-addons.sh remove-slice-controller [--dry-run]               # 先删 webhook 再删其余
#   ./03-addons.sh teardown        [--dry-run]  # 整章逆序拆除（Kueue 对象 → slice-controller → kueue → jobset）
#
# 出处：
#   - Helm 安装形态（release 名、chart 仓库、namespace、create_namespace、wait、timeout 1200s）：
#     参考集群蓝图里 jobset / kueue 两段
#   - `helm upgrade` 不会应用 JobSet chart 的 crds/ 目录，升级要手工 apply CRD；Kueue 的 CRD 在 templates/ 里由 Helm 管
#   - 装前确认 chart 存在：`helm show chart ...`
#   - Kueue 改配置后 rollout restart
#   - Kueue 对象删除顺序（clusterqueue → resourceflavor → admissioncheck → topology，必须在 Kueue controller 还活着时删）
#   - slice-controller 的 webhook fail-closed → 先删 MutatingWebhookConfiguration
#   - 验证命令：见手册 03 章
#
# 幂等：helm upgrade --install / kubectl apply --server-side 重跑只会收敛到同一状态；queues 在 Topology 已存在时
#   apply 同样内容是 no-op，但**改了 levels 会被 API 拒绝（field is immutable）——这是设计如此，见 §3.4 铁律**。
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

CMD="${1:-}"; shift || true
DRY=0; ARGS=()
for a in "$@"; do case "$a" in --dry-run) DRY=1;; *) ARGS+=("$a");; esac; done
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }
need() { command -v "$1" >/dev/null || { echo "缺工具：$1" >&2; exit 1; }; }

JOBSET_CHART=oci://registry.k8s.io/jobset/charts/jobset
KUEUE_CHART=oci://registry.k8s.io/kueue/charts/kueue
SC_NS=slice-controller-system
SC_WEBHOOK=slice-controller-mutating-webhook-configuration

check_context() {
  local ctx; ctx="$(kubectl config current-context)"
  [[ "$ctx" == *"$CLUSTER"* ]] || { echo "kubectl 上下文 '$ctx' 不含 CLUSTER=$CLUSTER —— 停（清单会干净地 apply 到别的集群）" >&2; exit 1; }
}

case "$CMD" in
preflight)
  rb_env_check PROJECT REGION ZONE CLUSTER
  need kubectl; need helm; need envsubst; need jq
  helm version --short                                   # OCI chart 需要 Helm ≥ 3.8
  check_context
  kubectl get crd slices.accelerator.gke.io              # 托管 GKE Slice Controller 已开（02 出口）
  echo "default-pool Ready 节点数（slice-controller 2 副本 + 反亲和，至少 2）："
  kubectl get nodes -l cloud.google.com/gke-nodepool="$SYSTEM_POOL" --no-headers | grep -cw Ready
  helm show chart "$JOBSET_CHART" --version "$JOBSET_VERSION" | grep -E '^(name|version|appVersion)'
  helm show chart "$KUEUE_CHART"  --version "$KUEUE_VERSION"  | grep -E '^(name|version|appVersion)'
  ;;

jobset)
  rb_env_check PROJECT REGION ZONE CLUSTER; check_context
  tag="$(awk '/^ *tag:/{print $2; exit}' "$HERE/03-jobset-values.yaml")"
  [[ "$tag" == "v$JOBSET_VERSION" ]] || { echo "03-jobset-values.yaml image.tag=$tag 与 JOBSET_VERSION=$JOBSET_VERSION 不一致（见手册 03 章）" >&2; exit 1; }
  if helm status jobset -n jobset-system >/dev/null 2>&1; then
    # 升级路径：helm upgrade 不碰 crds/，先手工 server-side apply 新版 CRD（见手册）
    run helm pull "$JOBSET_CHART" --version "$JOBSET_VERSION" --untar --untardir "$STATE_DIR/jobset-chart-$JOBSET_VERSION"
    run kubectl apply --server-side --force-conflicts -f "$STATE_DIR/jobset-chart-$JOBSET_VERSION/jobset/crds/"
  fi
  run helm upgrade --install jobset "$JOBSET_CHART" --version "$JOBSET_VERSION" \
      --namespace jobset-system --create-namespace \
      -f "$HERE/03-jobset-values.yaml" --wait --timeout 20m
  [[ $DRY == 1 ]] || { kubectl get deploy,pods -n jobset-system
    kubectl get deploy -n jobset-system -o jsonpath='{.items[*].spec.template.spec.containers[*].image}{"\n"}'; }
  ;;

kueue)
  rb_env_check PROJECT REGION ZONE CLUSTER; check_context
  first=1; helm status kueue -n kueue-system >/dev/null 2>&1 && first=0
  run helm upgrade --install kueue "$KUEUE_CHART" --version "$KUEUE_VERSION" \
      --namespace kueue-system --create-namespace \
      -f "$HERE/03-kueue-values.yaml" --wait --timeout 20m
  # 已有 release 时 ConfigMap 变了 controller 不一定自动重读——显式重启
  [[ $first == 1 ]] || run kubectl rollout restart deployment -n kueue-system kueue-controller-manager
  run kubectl rollout status deployment -n kueue-system kueue-controller-manager --timeout=10m
  [[ $DRY == 1 ]] || helm history kueue -n kueue-system --max 5
  ;;

slice-controller)
  rb_env_check PROJECT REGION ZONE CLUSTER; check_context
  echo "注意：apply 到 2 个 pod Ready 之间，全集群 Job/JobSet/LWS 创建会被 fail-closed webhook 拒绝（一般 < 1 min）[推断]"
  run kubectl apply --server-side -f "$HERE/03-slice-controller.yaml"
  run kubectl rollout status deployment -n "$SC_NS" slice-controller-controller-manager --timeout=10m
  ;;

quota)
  # 建议值 = min(CUBES_MAX_JOB, 健康 cube − SPARE_CUBES) × 64；SPARE_CUBES = 每日留空 S（09 D04）
  # 一个 cube 算健康 = 16 个节点全 Ready、分区标签 HEALTHY（只算 HEALTHY；节点标签本来也只有 HEALTHY / UNHEALTHY，
  # 05 §5.3 实测）、allocatable google.com/tpu=4、且没有被 05-isolate-cube.sh 打上 PartOfBrokenSlice taint
  # （隔离的 cube 标签照样是 HEALTHY，不排除就会把已扣的 64 芯又算回来）。
  # 全书配额口径以这里为准：05-isolate-cube.sh 调本子命令重算，08-daily-check.sh / env.sh 注释 / queues 模板注释都应与此一致。
  # 可选（给 05-isolate-cube.sh 用，只影响计算、不改集群）：
  #   --assume-isolated=<id,id>  把这些 cube 当作已隔离（dry-run 时 taint 还没打）
  #   --assume-restored=<id,id>  把这些 cube 当作已去 taint（--undo 时）
  # 这两个参数是后加的，未在集群上跑过 [未实测]；验证：在任一集群上对比 `quota` 与 `quota --assume-isolated=<一个健康 cube>`，
  # 健康数应差 1。
  iso='[]'; res='[]'
  for a in ${ARGS[@]+"${ARGS[@]}"}; do case "$a" in
    --assume-isolated=*) iso="$(jq -cn --arg s "${a#*=}" '$s | split(",") | map(select(. != ""))')";;
    --assume-restored=*) res="$(jq -cn --arg s "${a#*=}" '$s | split(",") | map(select(. != ""))')";;
    *) echo "quota：未知参数 $a" >&2; exit 1;; esac; done
  healthy="$(kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x -o json | jq -r --argjson iso "$iso" --argjson res "$res" '
    [.items[] | select(.metadata.labels["cloud.google.com/gke-tpu-partition-4x4x4-state"]=="HEALTHY")
     | select(any(.status.conditions[]; .type=="Ready" and .status=="True"))
     | select(.status.allocatable["google.com/tpu"]=="4")
     | .metadata.labels["cloud.google.com/gke-tpu-partition-4x4x4-id"] as $id
     | select(any($iso[]; . == $id) | not)
     | select((any(.spec.taints[]?; .key=="PartOfBrokenSlice") and (any($res[]; . == $id) | not)) | not)
     | $id]
    | group_by(.) | map(select(length==16)) | length')"
  echo "健康 cube（16 节点全 Ready + HEALTHY + TPU=4 + 未隔离）= $healthy；SPARE_CUBES=${SPARE_CUBES:-<未填>}"
  if [[ -n "${SPARE_CUBES:-}" ]]; then
    q=$(( (healthy - SPARE_CUBES) * 64 )); (( q < 0 )) && q=0
    (( q > ${CUBES_MAX_JOB:-128} * 64 )) && q=$(( ${CUBES_MAX_JOB:-128} * 64 ))   # 上限 = 最大规划任务 CUBES_MAX_JOB（128 cube，00 §4）
    echo "建议 TPU_QUOTA=$q    # = min(128, 健康 − S) × 64，负数取 0"
  else
    echo "⚠ SPARE_CUBES 未填（env.sh），不给建议 TPU_QUOTA；先与使用方书面约定每日留空 S 再填（09 D04）" >&2
  fi
  ;;

queues)
  rb_env_check PROJECT REGION ZONE CLUSTER RESERVATION TPU_QUOTA; check_context
  [[ "$TPU_QUOTA" =~ ^[0-9]+$ ]] && (( TPU_QUOTA % 64 == 0 )) || { echo "TPU_QUOTA=$TPU_QUOTA 必须是 64 的整数倍（按芯片数）" >&2; exit 1; }
  mkdir -p "$STATE_DIR/03"
  envsubst '${RESERVATION}' < "$HERE/03-kueue-topology-flavor.yaml.tmpl" > "$STATE_DIR/03/topology-flavor.yaml"
  envsubst '${TPU_QUOTA} ${NS} ${LOCAL_QUEUE}' < "$HERE/03-kueue-queues.yaml.tmpl" > "$STATE_DIR/03/queues.yaml"
  if grep -n '\${' "$STATE_DIR"/03/*.yaml | grep -v ':[[:space:]]*#'; then echo "渲染后仍有未替换变量" >&2; exit 1; fi
  echo "渲染结果在 $STATE_DIR/03/"
  kubectl get namespace "$NS" >/dev/null 2>&1 || run kubectl create namespace "$NS"
  run kubectl apply --server-side -f "$HERE/03-kueue-priority.yaml"
  run kubectl apply --server-side -f "$STATE_DIR/03/topology-flavor.yaml"
  # nominalQuota 也会被 05-isolate-cube.sh / 05 §5.5 用 kubectl patch 改（另一个 field manager），不加 --force-conflicts 会报冲突；
  # 加了就以本次 TPU_QUOTA 为准——所以 TPU_QUOTA 必须是当天 `quota` 子命令的输出（它已经扣掉隔离的 cube），别手填旧值。
  run kubectl apply --server-side --force-conflicts -f "$STATE_DIR/03/queues.yaml"
  ;;

verify)
  check_context
  echo "== JobSet（期望 3/3，镜像 v$JOBSET_VERSION）"; kubectl get deploy -n jobset-system
  kubectl get deploy -n jobset-system -o jsonpath='{.items[*].spec.template.spec.containers[*].image}{"\n"}'
  echo "== Kueue（期望 3/3，镜像 v$KUEUE_VERSION）"; kubectl get deploy -n kueue-system
  kubectl get deploy -n kueue-system -o jsonpath='{.items[*].spec.template.spec.containers[*].image}{"\n"}'
  echo "== Kueue 生效配置（期望 waitForPodsReady 为非注释行，timeout 30m / recoveryTimeout 10m）"
  kubectl get cm kueue-manager-config -n kueue-system -o jsonpath='{.data.controller_manager_config\.yaml}' \
    | grep -E '^\s*(waitForPodsReady|timeout|recoveryTimeout|manageJobsWithoutQueueName):' \
    || echo "  !! 一行都没有：waitForPodsReady / manageJobsWithoutQueueName 全是注释（= 关闭）→ 经验教训第 34 条"
  # ↑ 在 set -euo pipefail 下 grep 零命中 = 整个 verify 静默 exit 1，恰好在它要抓的那种情况下后面的检查全没跑，所以要兜住
  echo "== Kueue 容器参数（期望含 TopologyAwareScheduling=true,FailureRecoveryPolicy=true）"
  kubectl get deploy -n kueue-system kueue-controller-manager -o jsonpath='{.spec.template.spec.containers[0].args}{"\n"}'
  echo "== slice-controller（期望 2/2 Running 在两个不同节点；PDB ALLOWED DISRUPTIONS 1；webhook 3 个）"
  kubectl get pods -n "$SC_NS" -o wide
  kubectl get pdb -n "$SC_NS"
  kubectl get mutatingwebhookconfiguration "$SC_WEBHOOK" -o jsonpath='{range .webhooks[*]}{.name}{" "}{.failurePolicy}{"\n"}{end}'
  kubectl get deploy -n "$SC_NS" slice-controller-controller-manager -o jsonpath='{.spec.template.spec.containers[0].image}{"\n"}'
  echo "== Kueue 对象"
  kubectl get topology
  kubectl get admissioncheck -o custom-columns=NAME:.metadata.name,CONTROLLER:.spec.controllerName,ACTIVE:'.status.conditions[?(@.type=="Active")].status'
  kubectl get resourceflavor
  kubectl get clusterqueue -o custom-columns=NAME:.metadata.name,ACTIVE:'.status.conditions[?(@.type=="Active")].status',REASON:'.status.conditions[?(@.type=="Active")].reason',QUOTA:'.spec.resourceGroups[0].flavors[0].resources[0].nominalQuota'
  kubectl get localqueue -A
  kubectl get priorityclass,workloadpriorityclass | grep -vE 'system-(cluster|node)-critical'
  ;;

rollback-kueue|rollback-jobset)
  rel="${CMD#rollback-}"; ns="$rel-system"; rev="${ARGS[0]:-}"
  [[ -n "$rev" ]] || { helm history "$rel" -n "$ns"; echo "用法：$0 $CMD <REVISION>" >&2; exit 1; }
  run helm rollback "$rel" "$rev" -n "$ns" --wait --timeout 20m
  [[ "$rel" == kueue ]] && run kubectl rollout restart deployment -n kueue-system kueue-controller-manager
  echo "提醒：线上手工 patch 不在 Helm 历史里，rollback 会把它们一起抹掉/复活（参考集群）"
  ;;

remove-slice-controller|teardown)
  check_context
  echo "== 前置检查：不能还有 Kueue 建的 Slice / 已准入的 Workload（否则 Slice finalizer 没人摘 → 经验教训第 40 条）"
  kubectl get workloads -A --no-headers 2>/dev/null | head || true
  # Kueue 建的 Slice 带 accelerator.gke.io/owner-workload-name 注解（GA 指南 Troubleshooting Scheduler）；手动 Slice 没有
  kubectl get slices -o jsonpath='{range .items[*]}{.metadata.name}{"\towner="}{.metadata.annotations.accelerator\.gke\.io/owner-workload-name}{"\n"}{end}' 2>/dev/null | head || true
  if [[ $DRY == 0 ]] && kubectl get workloads -A --no-headers 2>/dev/null | grep -q .; then
    echo "还有 Workload：先删对应 JobSet/Job，等 kubectl get slices 里 Kueue 建的切片消失，再重跑" >&2; exit 1; fi
  if [[ "$CMD" == teardown ]]; then
    # Kueue controller 还活着时删 Kueue 对象，否则 resource-in-use finalizer 永远摘不掉
    run kubectl delete localqueue --all -A --ignore-not-found --timeout=90s
    for k in clusterqueue resourceflavor admissioncheck topology; do
      run kubectl delete "$k" --all --ignore-not-found --timeout=90s
    done
    run kubectl delete -f "$HERE/03-kueue-priority.yaml" --ignore-not-found
  fi
  # fail-closed webhook 先走，再删 Deployment；反过来会有一段时间全集群建不了 Job
  run kubectl delete mutatingwebhookconfiguration "$SC_WEBHOOK" --ignore-not-found
  run kubectl delete -f "$HERE/03-slice-controller.yaml" --ignore-not-found --wait=true
  if [[ "$CMD" == teardown ]]; then
    run helm uninstall kueue -n kueue-system --wait --timeout 15m
    run helm uninstall jobset -n jobset-system --wait --timeout 15m
    echo "JobSet CRD 在 chart 的 crds/ 里，helm uninstall 不删；整集群清理时再手工删："
    echo "  kubectl get crd -o name | grep jobset.x-k8s.io | xargs -r kubectl delete"
    echo "== 验证"
    [[ $DRY == 1 ]] || { kubectl get ns kueue-system jobset-system "$SC_NS" 2>&1 | sed 's/^/  /'
      kubectl get crd | grep -E 'kueue.x-k8s.io|jobset.x-k8s.io' || echo "  Kueue CRD 已无"; }
  fi
  ;;

*) sed -n '2,20p' "$0"; exit 1;;
esac
