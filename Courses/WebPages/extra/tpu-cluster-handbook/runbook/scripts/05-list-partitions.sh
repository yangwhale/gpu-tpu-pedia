#!/usr/bin/env bash
# =============================================================================
# 05-list-partitions.sh —— §5.3 / §5.7：按 4x4x4 cube 汇总节点标签，列出 partition id、是否可用、被谁占
#
# 出处：
#   - partition id = 节点标签 cloud.google.com/gke-tpu-partition-4x4x4-id；state 标签 -4x4x4-state；
#     只选 HEALTHY 且池里 16 个节点全 Ready 的 cube（已去掉 DEGRADED：公开文档有 DEGRADED 值（仅 4x4x4 级），
#     本集群实测没见到；遇到 DEGRADED 按 01 §1.9 口径不算可用，05 §5.3；
#     与 03-addons.sh quota 的健康口径一致）
#   - 节点注解 node.gke.io/created-by-mig = 该 cube 的 MIG 全路径
#   - Slice CR 的 spec.partitionIds 即「已被占」；过期 Slice 会永久占着 id
#   - 节点上有 gke-tpu-slice 标签 = 有活着的切片（含 sub-slice），不能再进 super-slice
#   - 隔离 taint 键 PartOfBrokenSlice
#
# 用法：
#   ./05-list-partitions.sh                      # 全部 cube 一行一个（TSV）
#   ./05-list-partitions.sh --free               # 只列可用于新 Slice 的 cube
#   ./05-list-partitions.sh --emit my-slice 8x8x16 # 从可用 cube 里取 A×B×C/64 个，打印一份 Slice YAML（不 apply）
# 只读：不改任何对象，可随时重跑。依赖：kubectl、jq。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

MODE=all; NAME=""; TOPO=""
case "${1:-}" in
  --free) MODE=free;;
  --emit) MODE=emit; NAME="${2:?slice 名}"; TOPO="${3:?topology，如 4x8x8}";;
esac

L=cloud.google.com/gke-tpu-partition-4x4x4-id
nodes_json=$(kubectl get nodes -l "$L" -o json)
used=$(kubectl get slices.accelerator.gke.io -o json | jq -c '[.items[] | {n:.metadata.name, p:.spec.partitionIds[]}]')

table=$(jq -r --argjson used "$used" '
  [.items[] | {
     id:    .metadata.labels["cloud.google.com/gke-tpu-partition-4x4x4-id"],
     state: .metadata.labels["cloud.google.com/gke-tpu-partition-4x4x4-state"],
     pool:  .metadata.labels["cloud.google.com/gke-nodepool"],
     mig:   ((.metadata.annotations["node.gke.io/created-by-mig"] // "") | split("/") | last),
     ready: ([.status.conditions[] | select(.type=="Ready" and .status=="True")] | length),
     broken: ([(.spec.taints // [])[] | select(.key=="PartOfBrokenSlice")] | length),
     slicelbl: (.metadata.labels["cloud.google.com/gke-tpu-slice"] // "")
  } | select(.id != null)]
  | group_by(.id)[]
  | {id: .[0].id, pool: .[0].pool, mig: .[0].mig,
     nodes: length, ready: (map(.ready) | add),
     states: (map(.state) | unique | join(",")),
     broken: (map(.broken) | add),
     slicelbl: (map(.slicelbl) | unique | map(select(. != "")) | join(",")) }
  | . as $c
  | $c + {usedby: ([$used[] | select(.p == $c.id) | .n] | unique | join(","))}
  | [.id, .pool, .mig, "\(.ready)/\(.nodes)", .states, (if .broken>0 then "TAINTED" else "-" end),
     (if .usedby=="" then "-" else .usedby end), (if .slicelbl=="" then "-" else .slicelbl end)]
  | @tsv' <<<"$nodes_json")

is_free() { awk -F'\t' '$4=="16/16" && $5=="HEALTHY" && $6=="-" && $7=="-" && $8=="-"'; }
# 第 8 列非空 = cube 上还有 sub-slice（如 2x2x1 巡检切片）活着，编不进 super-slice

case $MODE in
  all)  printf 'PARTITION_ID\tNODEPOOL\tMIG\tREADY\tSTATE\tTAINT\tSLICE_CR\tNODE_SLICE_LABEL\n'; echo "$table" | sort -k2;;
  free) echo "$table" | is_free | sort -k2
        echo "free cubes: $(echo "$table" | is_free | grep -c . || true) / total $(echo "$table" | grep -c . || true)" >&2;;
  emit)
    IFS=x read -r a b c <<<"$TOPO"; n=$(( a*b*c/64 ))
    mapfile -t ids < <(echo "$table" | is_free | sort -k2 | head -n "$n" | cut -f1)
    (( ${#ids[@]} == n )) || { echo "只有 ${#ids[@]} 个可用 cube，需要 $n" >&2; exit 1; }
    sed -e "s/__NAME__/$NAME/" -e "s/__TOPOLOGY__/$TOPO/" \
        -e "/__PARTITION_IDS__/{s/.*//;}" "$(dirname "$0")/05-slice.yaml" | sed '/^$/d'
    for i in "${ids[@]}"; do echo "  - $i"; done;;
esac
