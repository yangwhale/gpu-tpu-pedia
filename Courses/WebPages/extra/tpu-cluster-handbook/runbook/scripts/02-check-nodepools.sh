#!/usr/bin/env bash
# =============================================================================
# 02-check-nodepools.sh —— §2.7：逐池健康检查（只读，可随时重跑）
#
# 每个 TPU 池一行：GKE 池状态 / 节点数 / Ready 数 / distinct 4x4x4 partition id 数 / partition state /
#                   allocatable google.com/tpu / 网卡数（节点注解 networking.gke.io/nic-info）/ DRA 标签
# 判据（一行全满足 = OK）：
#   status=RUNNING、nodes=ready=16、ids=1、state 恰好 = HEALTHY:16（16 个节点全 HEALTHY）、tpu=4、nics=3、dra=true
#   - 其他条件都满足、只是 state 里出现 DEGRADED → 判 DEGRADED，单独计数，不算进 OK（处置见 02 §2.7「DEGRADED 池处置」）。
#   - 其余 → CHECK。
#   要点：state 必须整串匹配，只锚开头会把「HEALTHY:15,UNHEALTHY:1」和 DEGRADED 都算成 OK，
#     而 02 §2.7 的「OK ≥ 130」闸门用的就是这个数。本判据未在集群上跑过 [需验证]。
#   注意：实测集群现场 GKE 标签 partition-4x4x4-state 只见过 HEALTHY / UNHEALTHY，GCE 侧 DEGRADED 的 sub-block
#     在这里显示为 UNHEALTHY → 判 CHECK（09 D01）；别的 pod 上见过标签值 DEGRADED（BAD_ICI），
#     所以留了 DEGRADED 分支。
#     要区分「GCE DEGRADED」和「真坏」，对照 01-precheck.sh 第 4 段存的 sub-block 明细（healthInfo）。
# 另外两个判据：
#   - TAS 闸门两个数（带 id 的节点 = 16×N，distinct id = N）
#   - 每个 4x4x4 id 下正好 16 个 Ready 节点
# 依赖：kubectl、jq、gcloud；已 get-credentials。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
rb_env_check PROJECT ZONE CLUSTER

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
gcloud container node-pools list --cluster="$CLUSTER" --location="$ZONE" --project="$PROJECT" \
  --filter="config.machineType=$MACHINE_TYPE" --format="value(name,status)" | sort > "$TMP/pools"
kubectl get nodes -l cloud.google.com/gke-tpu-accelerator=tpu7x -o json > "$TMP/nodes.json"

jq -r '.items[] | [
  .metadata.labels["cloud.google.com/gke-nodepool"],
  ((.status.conditions[] | select(.type=="Ready") | .status) // "Unknown"),
  (.metadata.labels["cloud.google.com/gke-tpu-partition-4x4x4-id"] // "-"),
  (.metadata.labels["cloud.google.com/gke-tpu-partition-4x4x4-state"] // "-"),
  (.status.allocatable["google.com/tpu"] // "0"),
  ((.metadata.annotations["networking.gke.io/nic-info"] // "[]") | fromjson | length),
  (.metadata.labels["cloud.google.com/gke-networking-dra-driver"] // "-")
] | @tsv' "$TMP/nodes.json" | sort > "$TMP/nodes.tsv"

printf "%-24s %-12s %5s %5s %3s %-18s %-5s %-5s %-5s %s\n" POOL STATUS NODES READY IDS STATE TPU NICS DRA VERDICT
ok=0; deg=0; bad=0
while read -r pool status; do
  awk -F'\t' -v p="$pool" '$1==p' "$TMP/nodes.tsv" > "$TMP/p"
  nodes=$(wc -l < "$TMP/p"); ready=$(awk -F'\t' '$2=="True"' "$TMP/p" | wc -l)
  ids=$(awk -F'\t' '$3!="-"{print $3}' "$TMP/p" | sort -u | wc -l)
  state=$(cut -f4 "$TMP/p" | sort | uniq -c | awk '{printf "%s:%s,", $2, $1}' | sed 's/,$//')
  tpu=$(cut -f5 "$TMP/p" | sort -u | tr '\n' ',' | sed 's/,$//')
  nics=$(cut -f6 "$TMP/p" | sort -u | tr '\n' ',' | sed 's/,$//')
  dra=$(cut -f7 "$TMP/p" | sort -u | tr '\n' ',' | sed 's/,$//')
  verdict=CHECK
  if [[ "$status" == RUNNING && $nodes == "$NODES_PER_POOL" && $ready == "$NODES_PER_POOL" && $ids == 1 \
        && "$tpu" == 4 && "$nics" == 3 && "$dra" == true ]]; then
    if [[ "$state" =~ ^HEALTHY:${NODES_PER_POOL}$ ]]; then verdict=OK
    elif [[ "$state" == *DEGRADED* ]]; then verdict=DEGRADED
    fi
  fi
  case $verdict in OK) ok=$((ok+1));; DEGRADED) deg=$((deg+1));; *) bad=$((bad+1));; esac
  printf "%-24s %-12s %5s %5s %3s %-18s %-5s %-5s %-5s %s\n" "$pool" "$status" "$nodes" "$ready" "$ids" "${state:--}" "${tpu:--}" "${nics:--}" "${dra:--}" "$verdict"
done < "$TMP/pools"

echo "== 汇总"
echo "池 OK=$ok  DEGRADED=$deg  需处理=$bad  （出口条件：OK ≥ 130，只数全 HEALTHY 的池，见 00 §4 账本；"
echo "  DEGRADED 按 02 §2.7「DEGRADED 池处置」；需处理的按 §2.5 对模式；不达标的退路见 02 §2.7 检查点）"
echo "TAS 闸门：带 4x4x4 id 的节点 = $(awk -F'\t' '$3!="-"' "$TMP/nodes.tsv" | wc -l)（期望 16×N）；distinct id = $(awk -F'\t' '$3!="-"{print $3}' "$TMP/nodes.tsv" | sort -u | wc -l)（期望 N）"
echo "每个 id 下的 Ready 节点数分布（期望全是 16）："
awk -F'\t' '$3!="-" && $2=="True"{c[$3]++} END{for(i in c) print c[i]}' "$TMP/nodes.tsv" | sort | uniq -c | awk '{print "  "$2" 个节点的 id："$1" 个"}'
echo "缺 id 标签的 TPU 节点：$(awk -F'\t' '$3=="-"' "$TMP/nodes.tsv" | wc -l)（非 0 → 经验教训第 50 条）"
