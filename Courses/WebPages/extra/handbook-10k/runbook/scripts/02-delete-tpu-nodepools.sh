#!/usr/bin/env bash
# =============================================================================
# 02-delete-tpu-nodepools.sh —— §2.4 的回滚：删 TPU 池（默认 dry-run，要 --yes 才真删）
#
#   ./02-delete-tpu-nodepools.sh --only <节点池>,<节点池>    # 删指定池
#   ./02-delete-tpu-nodepools.sh --all --yes                                    # 删全部 $POOL_PREFIX* 池（拆集群前）
#
# ⚠ 容量事故期间不要删健康的池——删了 sub-block 可能被别人/别的池占走，回不来。
# ⚠ 池上还有 ACTIVE Slice 时先按 05 §5.5 删 Slice，再删池（否则 finalizer / multi-MIG 残留）。
# 删除命令：gcloud container node-pools delete（公开 gcloud 参考）；10 个并发删除各十几秒、零 409。
# 幂等：已不存在的池跳过。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
rb_env_check PROJECT ZONE CLUSTER
YES=0; ALL=0; ONLY=""; PARALLEL=10
while (( $# )); do case "$1" in --yes) YES=1;; --all) ALL=1;; --only) ONLY=$2; shift;; --parallel) PARALLEL=$2; shift;; esac; shift; done
if (( ALL )); then
  POOLS=$(gcloud container node-pools list --cluster="$CLUSTER" --location="$ZONE" --project="$PROJECT" \
            --filter="name~^${POOL_PREFIX}" --format="value(name)")
else POOLS=$(tr ',' '\n' <<<"$ONLY"); fi
[[ -n "$POOLS" ]] || { echo "没有要删的池（用 --only 或 --all）"; exit 0; }
echo "将删除 $(wc -w <<<"$POOLS") 个池："; echo "$POOLS" | head -20
# 从「打印一下」改成硬闸——目标池里还有节点属于某个 Slice（带 gke-tpu-slice 标签）就拒绝，
# 否则删池会留下 finalizer / multi-MIG 残留，也违反 05 章「先删 Slice 再删池」的红线。
busy=""
for p in $POOLS; do
  n=$(kubectl get nodes -l "cloud.google.com/gke-nodepool=$p,cloud.google.com/gke-tpu-slice" -o name 2>/dev/null | wc -l) \
    || n="?"
  [[ "$n" == 0 ]] || busy+="$p($n) "
done
if [[ -n "$busy" ]]; then
  echo "拒绝：这些池还有节点在 Slice 里（或 kubectl 查询失败）：$busy" >&2
  echo "先按 05 §5.5 删 Slice（05-delete-slice-chain.sh），确认 kubectl get slices 为空再删池" >&2; exit 2
fi
echo "目标池里没有属于 Slice 的节点。集群里现有 Slice："; kubectl get slices -o name 2>/dev/null | head || true
(( YES )) || { echo "dry-run：加 --yes 才执行"; exit 0; }
echo "$POOLS" | xargs -r -P "$PARALLEL" -I{} gcloud container node-pools delete {} \
  --cluster="$CLUSTER" --location="$ZONE" --project="$PROJECT" --quiet
echo "== 验证：剩余池"; gcloud container node-pools list --cluster="$CLUSTER" --location="$ZONE" --project="$PROJECT" --format="value(name,status)"
