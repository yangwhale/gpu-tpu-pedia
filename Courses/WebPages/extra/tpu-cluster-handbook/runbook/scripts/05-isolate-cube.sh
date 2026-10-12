#!/usr/bin/env bash
# =============================================================================
# 05-isolate-cube.sh —— §5.7：把坏 cube 从调度里拿掉（taint），并按 03-addons.sh quota 口径重算 Kueue 配额；--undo 反向
#
# 出处：
#   - 配额口径：google.com/tpu 按芯片计，1 cube = 64；nominalQuota = min(128, 健康 cube − 备件 S) × 64（03 §3.5）。
#     现在不再做「± n×64」增量：直接调 03-addons.sh quota（--assume-isolated / --assume-restored）重算绝对值再 patch。
#     原因：健康 − S > 128 时配额本来就封顶 8192，隔离一个 cube 不该再扣。健康只算 HEALTHY，与 quota 子命令一致。
#     [需验证]：在测试集群对一个健康 cube 跑 dry-run，看打印的「重算」值是否等于
#     `03-addons.sh quota --assume-isolated=<该 id>` 的建议值，封顶时应打印「配额不变」。
#   - 用 JSON patch 的 test 操作先校验 flavor 名和旧值，避免并发改配额时写错位（RFC 6902 test；kubectl --type=json 支持）
#
# 用法：
#   ./05-isolate-cube.sh --reason "<工单号> host 坏" <partition-id> [...]            # dry-run
#   ./05-isolate-cube.sh --reason "..." --execute <partition-id> [...]
#   ./05-isolate-cube.sh --undo --execute <partition-id> [...]                    # Google 确认修好后
#   ./05-isolate-cube.sh --list                                                   # 看台账与当前 taint
# 幂等：台账 $STATE_DIR/isolated-cubes.tsv 记着已隔离的 id；已在台账里的 id 不会重复处理；配额每次按当前节点状态重算绝对值，重跑结果一样；taint 用 --overwrite。
# 回滚：--undo（去 taint + 按 quota 口径重算配额 + 从台账删行）。
# 前置：kubectl 对 nodes（patch）与 clusterqueues.kueue.x-k8s.io（patch）有写权限；jq；env 里填了 SPARE_CUBES（重算要用，没填就停）。
# 注意：NoSchedule 不赶走已在跑的 pod；在跑的任务靠 Slice FAILED → Kueue 驱逐（§5.6）或 waitForPodsReady.recoveryTimeout（07 §7.2）。
#       Kueue TAS 选 cube 时会避开 pod 不容忍的 taint，是用 taint 隔离的前提 [需验证：用一次 dry-run 提交确认]。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
rb_env_check PROJECT CLUSTER

EXEC=0; UNDO=0; REASON=""; LIST=0; IDS=()
while (($#)); do case "$1" in
  --execute) EXEC=1;; --undo) UNDO=1;; --reason) REASON="$2"; shift;; --list) LIST=1;;
  -*) echo "未知选项 $1" >&2; exit 64;; *) IDS+=("$1");; esac; shift; done
LEDGER="$STATE_DIR/isolated-cubes.tsv"; touch "$LEDGER"
# dry-run 时打印可直接复制的命令（%q 转义，否则 `in (...)` 的括号贴进 shell 会报语法错），
# 并把同一条命令加 --dry-run=server 交给 API server 校验一遍（不落库；能提前抓到 RBAC 不够、JSON patch test 不过）
run() { echo "+ $(printf '%q ' "$@")"
  if [[ $EXEC == 1 ]]; then "$@"
  elif out=$("$@" --dry-run=server -o name 2>&1); then echo "  (dry-run，未执行；server 端试跑通过：$(wc -l <<<"$out") 个对象)"
  else echo "  (dry-run，未执行；!! server 端试跑失败：$out)"; echo "  「The request is invalid」= JSON patch 的 test 没过（flavor 名/旧配额被别人改了），重跑本脚本取新值" ; DRYFAIL=1; fi; }
DRYFAIL=0
L=cloud.google.com/gke-tpu-partition-4x4x4-id

if [[ $LIST == 1 ]]; then
  echo "== 台账 $LEDGER（id / 时间 / 原因）"; cat "$LEDGER"
  echo "== 当前带 PartOfBrokenSlice taint 的 cube（id: 节点数）"
  kubectl get nodes -l "$L" -o json | jq -r '.items[] | select([(.spec.taints//[])[]|.key]|index("PartOfBrokenSlice")) | .metadata.labels["'"$L"'"]' | sort | uniq -c
  exit 0
fi
(( ${#IDS[@]} )) || { echo "至少给一个 partition id（05-list-partitions.sh 第 1 列）" >&2; exit 64; }
[[ $UNDO == 1 || -n "$REASON" ]] || { echo "隔离必须写 --reason" >&2; exit 64; }

# 只处理「状态会变」的 id：隔离时跳过已在台账的，--undo 时跳过不在台账的
todo=(); for id in "${IDS[@]}"; do
  if grep -q "^$id"$'\t' "$LEDGER"; then (( UNDO )) && todo+=("$id") || echo "跳过 $id：已在台账"
  else (( UNDO )) && echo "跳过 $id：不在台账" || todo+=("$id"); fi
done
(( ${#todo[@]} )) || { echo "没有要改的 id"; exit 0; }
sel="$L in ($(IFS=,; echo "${todo[*]}"))"
n=${#todo[@]}
[[ -n "${SPARE_CUBES:-}" && "${SPARE_CUBES}" != \<* ]] || { echo "SPARE_CUBES 没填：配额要按 03-addons.sh quota 口径重算，先填 ~/runbook-env.sh（09 D04）" >&2; exit 1; }

# ---- 1. taint / 去 taint
for id in "${todo[@]}"; do
  c=$(kubectl get nodes -l "$L=$id" -o name | wc -l); echo "cube $id：$c 个节点"; (( c > 0 )) || { echo "  找不到节点，停" >&2; exit 1; }
done
if (( UNDO )); then run kubectl taint nodes -l "$sel" PartOfBrokenSlice=true:NoSchedule-
else run kubectl taint nodes -l "$sel" PartOfBrokenSlice=true:NoSchedule --overwrite; fi

# ---- 2. 配额：按 03-addons.sh quota 口径重算绝对值（不做 ± n×64 增量）
cq=$(kubectl get clusterqueue "$CLUSTER_QUEUE" -o json)
read -r gi fl ri cur < <(jq -r --arg f "$TPU_FLAVOR" '
  .spec.resourceGroups | to_entries[] | .key as $g | .value.flavors | to_entries[] | select(.value.name==$f) | .key as $fl
  | .value.resources | to_entries[] | select(.value.name=="google.com/tpu") | "\($g) \($fl) \(.key) \(.value.nominalQuota)"' <<<"$cq") || true
# ↑ `|| true`：flavor 名不对时 jq 无输出、read 返回 1，set -e 会让脚本一声不吭退出（实测），走不到下面的报错
[[ -n "${cur:-}" ]] || { echo "CQ $CLUSTER_QUEUE 里找不到 flavor $TPU_FLAVOR 的 google.com/tpu（env: CLUSTER_QUEUE / TPU_FLAVOR）" >&2; exit 1; }
ids_csv="$(IFS=,; echo "${todo[*]}")"
if (( UNDO )); then assume="--assume-restored=$ids_csv"; else assume="--assume-isolated=$ids_csv"; fi
qout="$(bash "$(dirname "$0")/03-addons.sh" quota "$assume")" || { echo "03-addons.sh quota 失败（看上面的报错），配额未改" >&2; exit 1; }
echo "$qout" | sed 's/^/  [03-addons.sh quota '"$assume"'] /'
new="$(sed -n 's/^建议 TPU_QUOTA=\([0-9]*\).*/\1/p' <<<"$qout")"
[[ -n "$new" ]] || { echo "03-addons.sh quota 没给出建议值（SPARE_CUBES？），停" >&2; exit 1; }
cap=$(( ${CUBES_MAX_JOB:-128} * 64 ))
echo "配额 $CLUSTER_QUEUE/$TPU_FLAVOR：现值 $cur → 重算 $new（= min(${CUBES_MAX_JOB:-128}, 健康 − S=$SPARE_CUBES) × 64，$( ((UNDO)) && echo "把 $n 个 cube 算回健康" || echo "已排除这 $n 个 cube")）"
(( new == cap )) && echo "  注：健康 − S ≥ ${CUBES_MAX_JOB:-128}，配额封顶 $cap（单作业上限 CUBES_MAX_JOB），这批 cube 进出不改变配额"
# 以重算值为准：现值若被人手改过，也会被拉回 quota 口径（JSON patch 的 test 仍防并发改）
if [[ "$cur" == "$new" ]]; then
  echo "  配额不变，跳过 patch"
else
  patch=$(jq -nc --arg p "/spec/resourceGroups/$gi/flavors/$fl" --arg r "/spec/resourceGroups/$gi/flavors/$fl/resources/$ri/nominalQuota" \
    --arg f "$TPU_FLAVOR" --arg old "$cur" --arg new "$new" \
    '[{op:"test",path:($p+"/name"),value:$f},{op:"test",path:$r,value:$old},{op:"replace",path:$r,value:$new}]')
  run kubectl patch clusterqueue "$CLUSTER_QUEUE" --type=json -p "$patch"
fi

# ---- 3. 台账
if [[ $EXEC == 1 ]]; then
  for id in "${todo[@]}"; do
    if (( UNDO )); then sed -i "/^$id\t/d" "$LEDGER"; else printf '%s\t%s\t%s\n' "$id" "$(date -u +%FT%TZ)" "$REASON" >> "$LEDGER"; fi
  done
fi

# ---- 4. 验证
if [[ $EXEC != 1 ]]; then
  echo "== dry-run 结束：上面两条命令没执行（server 端试跑 $([[ $DRYFAIL == 0 ]] && echo 全部通过 || echo 有失败)）。现值：taint 节点 $(kubectl get nodes -l "$sel" -o json | jq '[.items[]|select([(.spec.taints//[])[]|.key]|index("PartOfBrokenSlice"))]|length')，nominalQuota $cur；加 --execute 才生效"
  exit $DRYFAIL
fi
echo "== 验证"
echo "  taint 节点数（隔离后期望 16×n=$((16*n))，--undo 后期望 0）：$(kubectl get nodes -l "$sel" -o json | jq '[.items[]|select([(.spec.taints//[])[]|.key]|index("PartOfBrokenSlice"))]|length')"
echo "  nominalQuota 现值：$(kubectl get clusterqueue "$CLUSTER_QUEUE" -o jsonpath="{.spec.resourceGroups[$gi].flavors[$fl].resources[$ri].nominalQuota}")（期望 $new）"
echo "  规则：nominalQuota = min(${CUBES_MAX_JOB:-128}, 健康 cube − 备件 S=$SPARE_CUBES) × 64（03 §3.5）；健康 cube 只算 HEALTHY + 16/16 Ready + TPU=4 + 未隔离，复核跑 03-addons.sh quota"
