#!/usr/bin/env bash
# =============================================================================
# 04-teardown-check.sh —— §4.9 每级之后「拆干净了没有」，只读
#
# 用法：./04-teardown-check.sh [--keep REGEX]     # REGEX = 有意保留的 Slice 名（例如 '^my-manual-'），不算残留
# 退出码：0 = 干净；1 = 有残留/重叠（逐条打印原因）
#
# 检查项与出处：
#   1. 残留 Slice（除 --keep 外）——残留会悄悄吃容量、伪装成 GCE_STOCKOUT；Kueue 建的带
#      accelerator.gke.io/owner-workload-name 注解
#   2. 同一 partition id 出现在两个 Slice 里 = 事故 → 经验教训第 39 条
#   3. 过渡态（ACTIVATING / DEACTIVATING / INCOMPLETE）Slice——超过 10–15 min = bug → 经验教训第 39 条
#   4. 残留 Workload / 巡检对象（app=rb-inspect）——只算没 Finished 的、且名字不匹配 --keep 的 Workload
#      （已 Finished 的 Workload 不算残留，否则集群上只要跑过作业就永远 rc=1）。
#      Kueue 的 Workload 名是 jobset-<JobSet名>-<hash5>，--keep 的正则要同时能匹配 Slice 名和 Workload 名，
#      例：--keep '^(default-jobset-|jobset-)?(<你的作业名前缀>)'。
#   5. 节点上 cloud.google.com/gke-tpu-slice 标签指向已不存在的 Slice（只报告、不判失败）。
#      实测：2x2x1 手动片删除后（23 s 到 NotFound）gke-tpu-slice / gke-tpu-topology 两个标签立即自动摘掉；
#      多 cube 片删后何时摘 [需验证]。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
KEEP='^$'
[[ "${1:-}" == --keep ]] && KEEP="$2"
bad=0
slices_json="$(kubectl get slices -o json)"

echo "== 1. Slice（有意保留的正则：$KEEP）"
left="$(jq -r --arg k "$KEEP" '.items[] | select(.metadata.name | test($k) | not) |
  "\(.metadata.name)\t\(.spec.topology)\towner=\(.metadata.annotations["accelerator.gke.io/owner-workload-name"] // "-")"' <<<"$slices_json")"
if [[ -n "$left" ]]; then echo "$left"; echo "  ↑ 残留 Slice"; bad=1; else echo "  无残留"; fi

echo "== 2. partition id 重叠"
dup="$(jq -r '[.items[] | .metadata.name as $n | .spec.partitionIds[]? | {p: ., n: $n}] | group_by(.p)[] | select(length>1) |
  "\(.[0].p)\t\(map(.n) | join(","))"' <<<"$slices_json")"
if [[ -n "$dup" ]]; then echo "$dup"; echo "  ↑ 重叠 → 经验教训第 39 条"; bad=1; else echo "  无重叠"; fi

echo "== 3. 过渡态 Slice（看 AGE，>10–15 min 升级处理 → 经验教训第 39 条）"
kubectl get slices --no-headers 2>/dev/null | awk '$4 ~ /ACTIVATING|DEACTIVATING|INCOMPLETE/' | sed 's/^/  /' || true

echo "== 4. 残留 Workload / 巡检对象"
wl="$(kubectl get workloads -A -o json 2>/dev/null | jq -r --arg k "$KEEP" '.items[]
  | select(([.status.conditions[]? | select(.type=="Finished" and .status=="True")] | length) == 0)
  | select(.metadata.name | test($k) | not)
  | "\(.metadata.namespace)\t\(.metadata.name)"' || true)"
[[ -n "$wl" ]] && { echo "$wl" | sed 's/^/  /'; echo "  ↑ 未结束的 Workload（Finished 的已排除）"; bad=1; } || echo "  无未结束的 Workload"
insp="$(kubectl get jobs,ds -n "$INSPECT_NS" --no-headers 2>/dev/null; kubectl get slices -l app=rb-inspect --no-headers 2>/dev/null)" || true
[[ -n "$insp" ]] && { echo "$insp" | sed 's/^/  /'; echo "  ↑ 跑 04-host-inspect.sh cleanup / 04-nic-check.sh cleanup"; bad=1; } || echo "  无巡检对象"

echo "== 5. 节点上的 gke-tpu-slice 标签是否都指向现存 Slice（只报告）[需验证]"
existing="$(jq -r '.items[].metadata.name' <<<"$slices_json" | sort -u)"
kubectl get nodes -l cloud.google.com/gke-tpu-slice -o jsonpath='{range .items[*]}{.metadata.labels.cloud\.google\.com/gke-tpu-slice}{"\n"}{end}' \
  | sort | uniq -c | { n=0; while read -r cnt s; do grep -qx "$s" <<<"$existing" || { echo "  $cnt 个节点挂着已不存在的切片名 $s"; n=$((n+1)); }; done
      (( n == 0 )) && echo "  OK：无悬挂标签"; true; }

exit $bad
