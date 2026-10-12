#!/usr/bin/env bash
# =============================================================================
# 04-host-inspect.sh —— §4.1 逐台主机巡检：每台 host 一个手动 2x2x1 Slice + 一个钉住该 host 的单 pod Job
#
# 用法：
#   ./04-host-inspect.sh plan    [--pool POOL]                 # 只读：列出要巡检的节点和它们的 2x2x1 partition id → $STATE_DIR/04/inspect-plan.tsv
#   ./04-host-inspect.sh apply   [--wave N] [--limit K] [--dry-run]   # 为 plan 里的前 K 台（默认全部）建 Slice + Job，打标签 wave=N
#   ./04-host-inspect.sh status                                # 只读：Slice 状态 / Job 完成情况汇总 / 各波从 apply 起的耗时
#   ./04-host-inspect.sh results                               # 只读：收集每台的 INSPECT_RESULT 行 → $STATE_DIR/04/inspect-results.txt
#   ./04-host-inspect.sh cleanup [--dry-run]                   # 删巡检 Job 与巡检 Slice（必须做）
#
# 做法照抄参考集群上的一次全量主机巡检：
#   - Slice：accelerator.gke.io/v1beta1，cluster 级对象，topology 2x2x1，partitionIds = 该节点标签
#     cloud.google.com/gke-tpu-partition-2x2x1-id，注解 slice.gke.io/retry-on-failure: "true"
#   - Job：单 pod，nodeSelector kubernetes.io/hostname + gke-tpu-topology: 2x2x1 + gke-tpu-accelerator: tpu7x
#     + gke-accelerator-count: "4"，hostNetwork，google.com/tpu 4（Slice ACTIVE 后节点才带 gke-tpu-topology=2x2x1，pod 在此之前 Pending）
#   - 与那次巡检的差别：namespace 用 $INSPECT_NS（Kueue 排除列表里，见 03-kueue-values.yaml 文件头）；
#     默认测试内容是 JAX device 数（期望 8），不是那次巡检的 <H2D 基准镜像>（那个镜像在 Google 自有仓库，
#     使用方能否拉取未知 [需验证]）。要跑 H2D/D2H 带宽，设 INSPECT_IMAGE / INSPECT_CMD 覆盖。
#
# 幂等：apply 用 kubectl apply，已存在的 Slice/Job 不重复建；Job 已完成的不会重跑（要重跑先 cleanup 再 apply）。
# 回滚/清理：cleanup。它只删带 app=rb-inspect 标签的对象，不会碰任务切片。
# 前提：03 出口满足；只在维护窗口跑（巡检切片占着的 cube 编不进多 cube 切片 → 经验教训第 33 条）。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

CMD="${1:-}"; shift || true
DRY=0; WAVE=1; LIMIT=0; POOL=""
while (( $# )); do case "$1" in
  --dry-run) DRY=1; shift;; --wave) WAVE="$2"; shift 2;; --limit) LIMIT="$2"; shift 2;; --pool) POOL="$2"; shift 2;;
  *) echo "未知参数 $1" >&2; exit 1;; esac; done
OUT="$STATE_DIR/04"; mkdir -p "$OUT"; PLAN="$OUT/inspect-plan.tsv"
LABEL="app=rb-inspect"
INSPECT_IMAGE="${INSPECT_IMAGE:-$TEST_IMAGE}"
INSPECT_CMD="${INSPECT_CMD:-pip install -q -U --pre jax jaxlib libtpu requests -f https://storage.googleapis.com/jax-releases/libtpu_releases.html && python -c 'import jax,os; n=jax.local_device_count(); print(\"INSPECT_RESULT\", os.environ[\"NODE_NAME\"], \"devices=%d\" % n, \"OK\" if n==8 else \"MISMATCH\")'}"

case "$CMD" in
plan)
  sel="cloud.google.com/gke-tpu-accelerator=tpu7x"; [[ -n "$POOL" ]] && sel="$sel,cloud.google.com/gke-nodepool=$POOL"
  kubectl get nodes -l "$sel" -o json | jq -r '.items[] | [.metadata.name,
      .metadata.labels["cloud.google.com/gke-nodepool"],
      (.metadata.labels["cloud.google.com/gke-tpu-partition-2x2x1-id"] // "MISSING"),
      (.metadata.labels["cloud.google.com/gke-tpu-slice"] // "-"),
      (if ([.status.conditions[]? | select(.type=="Ready" and .status=="True")] | length) == 1 then "Ready" else "NotReady" end)] | @tsv' > "$PLAN"
  echo "节点 $(wc -l < "$PLAN") 台 → $PLAN"
  echo "缺 2x2x1-id 的（不能巡检，回 02 §2.7 TAS 闸门）：$(awk -F'\t' '$3=="MISSING"' "$PLAN" | wc -l)"
  echo "已在某个切片里的（apply 会跳过）：$(awk -F'\t' '$4!="-"' "$PLAN" | wc -l)"
  echo "NotReady 的（apply 会跳过：Job 起不来，只会占着 2x2x1 Slice；直接按 02 §2.5 / 08 §8.5 报修）：$(awk -F'\t' '$5=="NotReady"' "$PLAN" | wc -l)"
  ;;

apply)
  [[ -s "$PLAN" ]] || { echo "先跑 plan" >&2; exit 1; }
  kubectl get namespace "$INSPECT_NS" >/dev/null 2>&1 || { echo "+ kubectl create namespace $INSPECT_NS"; [[ $DRY == 1 ]] || kubectl create namespace "$INSPECT_NS"; }
  # 记 apply 时刻，status 据此报这一波已用多少秒（先跑 1 池 16 台量出耗时，再排全量；144 池的耗时不能凭空估）
  [[ $DRY == 1 ]] || date +%s > "$OUT/inspect-wave$WAVE.apply.epoch"
  n=0
  while IFS=$'\t' read -r node pool pid inslice ready; do
    [[ "$pid" == MISSING || "$inslice" != "-" || "${ready:-Ready}" == NotReady ]] && continue
    (( LIMIT > 0 && n >= LIMIT )) && break
    sfx="${node#gke-tpu-}"; name="insp-${sfx}"; name="${name:0:59}"     # GCE slice 名上限 59 字符（GA 指南）
    manifest=$(cat <<EOF
apiVersion: accelerator.gke.io/v1beta1
kind: Slice
metadata:
  name: ${name}
  labels: {app: rb-inspect, wave: "${WAVE}"}
  annotations: {slice.gke.io/retry-on-failure: "true"}
spec:
  type: tpu7x
  topology: 2x2x1
  partitionIds: ["${pid}"]
---
apiVersion: batch/v1
kind: Job
metadata:
  name: ${name}
  namespace: ${INSPECT_NS}
  labels: {app: rb-inspect, wave: "${WAVE}"}
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 86400
  template:
    metadata:
      labels: {app: rb-inspect, node: "${node}"}
    spec:
      restartPolicy: Never
      hostNetwork: true
      dnsPolicy: ClusterFirstWithHostNet
      nodeSelector:
        kubernetes.io/hostname: "${node}"
        cloud.google.com/gke-tpu-topology: 2x2x1
        cloud.google.com/gke-tpu-accelerator: tpu7x
        cloud.google.com/gke-accelerator-count: "4"
      tolerations:
      - {key: google.com/tpu, operator: Exists, effect: NoSchedule}
      containers:
      - name: inspect
        image: ${INSPECT_IMAGE}
        securityContext: {privileged: true}
        env:
        - {name: NODE_NAME, value: "${node}"}
        - {name: NODEPOOL_NAME, value: "${pool}"}
        - {name: PARTITION_ID, value: "${pid}"}
        command: ["bash", "-c", $(printf '%s' "$INSPECT_CMD" | jq -Rs .)]
        resources:
          requests: {google.com/tpu: "4"}
          limits:   {google.com/tpu: "4"}
EOF
)
    if [[ $DRY == 1 ]]; then echo "---"; echo "$manifest"; else echo "$manifest" | kubectl apply -f - >/dev/null && echo "applied $name ($node)"; fi
    n=$((n+1))
  done < "$PLAN"
  echo "共 $n 台，wave=$WAVE"
  ;;

status)
  echo "== 巡检 Slice 状态分布（期望全部 ACTIVE；过渡态 >10–15 min = 经验教训第 39 条）"
  kubectl get slices -l "$LABEL" --no-headers 2>/dev/null | awk '{print $4}' | sort | uniq -c
  echo "== 巡检 Job（COMPLETIONS 1/1 = 完成）"
  kubectl get jobs -n "$INSPECT_NS" -l "$LABEL" --no-headers 2>/dev/null | awk '{print $2}' | sort | uniq -c
  echo "== 失败的 Job"
  kubectl get jobs -n "$INSPECT_NS" -l "$LABEL" -o json | jq -r '.items[] | select(.status.failed>0) | .metadata.name'
  echo "== 各波耗时（从 apply 起；全部 1/1 时记下来，填 04 §4.1「实测耗时」）[后加，未实测]"
  for f in "$OUT"/inspect-wave*.apply.epoch; do
    [[ -e "$f" ]] || { echo "  （没有 apply 记录）"; break; }
    w="${f##*/inspect-wave}"; w="${w%.apply.epoch}"
    done_n=$(kubectl get jobs -n "$INSPECT_NS" -l "$LABEL,wave=$w" -o json | jq '[.items[]|select(.status.succeeded>=1)]|length')
    all_n=$(kubectl get jobs -n "$INSPECT_NS" -l "$LABEL,wave=$w" -o json | jq '.items|length')
    last=$(kubectl get jobs -n "$INSPECT_NS" -l "$LABEL,wave=$w" -o json | jq -r '[.items[].status.completionTime // empty]|max // "-"')
    echo "  wave=$w：apply 后 $(( $(date +%s) - $(cat "$f") )) s，完成 $done_n/$all_n，最后一个完成于 $last（减 apply 时刻 $(date -u -d @"$(cat "$f")" +%FT%TZ) 即本波耗时）"
  done
  ;;

results)
  : > "$OUT/inspect-results.txt"
  for p in $(kubectl get pods -n "$INSPECT_NS" -l "$LABEL" -o name); do
    kubectl logs -n "$INSPECT_NS" "$p" 2>/dev/null | grep -E '^(INSPECT_RESULT|NODE_BENCH_RESULT)' >> "$OUT/inspect-results.txt" || echo "NO_RESULT ${p#pod/}" >> "$OUT/inspect-results.txt"
  done
  echo "结果 → $OUT/inspect-results.txt"; awk '{print $NF}' "$OUT/inspect-results.txt" | sort | uniq -c
  ;;

cleanup)
  run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }
  run kubectl delete jobs -n "$INSPECT_NS" -l "$LABEL" --ignore-not-found --wait=true
  run kubectl delete slices -l "$LABEL" --ignore-not-found --wait=false
  echo "Slice 删除走 DEACTIVATING → 消失，约 30 s/个；不要手摘 finalizer。确认：kubectl get slices -l $LABEL 为空"
  ;;

*) sed -n '2,12p' "$0"; exit 1;;
esac
