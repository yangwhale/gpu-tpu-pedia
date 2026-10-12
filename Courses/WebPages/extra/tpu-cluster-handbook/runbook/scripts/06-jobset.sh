#!/usr/bin/env bash
# =============================================================================
# 06-jobset.sh —— 06c §6.4–§6.6：渲染 / 提交 / 盯盘 / 停 / 恢复 / 删除 / 验收 JIT 基线作业 JobSet
#
# 出处：
#   - JobSet 模板（RECIPE=ref 时用你自备的 06-jobset-ref.yaml，下载包不含）
#   - Workload 名 = jobset-<JobSet 名>-<5 位 hash>，ownerReferences 指向 JobSet；
#     Slice 名 = <ns>-jobset-<JobSet 名>-<hash>-worker-<i>
#   - 停任务：Workload .spec.active=false → 驱逐且不再排队（https://kueue.sigs.k8s.io/docs/concepts/workload/ 「Active」）；
#     直接 patch JobSet spec.suspend 在 Kueue 管理下的行为 [需验证]（Kueue 自己也在写 suspend）
#
# 用法：[阶梯变量] ./06-jobset.sh <子命令>
#   阶梯变量（06c §6.5 的四级阶梯用；全不设 = JIT 基线作业原样 10+ × 4x8x8）：
#     SLICES=N            片数 → JobSet replicas；overlay 的 num_slices / dcn_data_parallelism 由 `overlay` 子命令同步写
#     TOPO=4x8x8           每片形状 → slice-topology 注解；completions = 芯片数 ÷ 4；ICI_FSDP_PARALLELISM = 芯片数 × 2（= 每片 device 数）
#     STEPS=50             训练步数（验收按 STEPS−1 核对）
#     TRAIN_SCRIPT=pretrain_moe_large.sh   启动脚本。**1 cube 冒烟别换成 pretrain_moe_small.sh**：那个脚本写的是
#                          LOAD_PARAMETERS_PATH=${LOAD_PARAMETERS_PATH:-<小号模型预训练 ckpt>}，JobSet 里的 "" 挡不住（:- 把空串当未设），
#                          会静默加载 /models/pretrain/moe-small/… 的预训练权重（读源码发现）；用下面的 MODEL_NAME
#     MODEL_NAME=          不设 = 脚本默认 moe-large；1 cube 冒烟用 moe-small（大号模型脚本认 MODEL_NAME env，随机初始化、flag 与大号模型完全相同）
#     EXTRA_LIBTPU_ARGS=   追加到 LIBTPU_INIT_ARGS 末尾（启动脚本把自己的默认 flag 放在前面，用户值在后 → 同名 flag 以你的为准，
#                          出处：启动脚本的 flag 拼接顺序）；做单 flag A/B 用
#     POD_PRIORITY=high    pod 模板 priorityClassName，默认 high（03-kueue-priority.yaml 建的 800 档）；04-gen-jobset.sh --pod-priority 同义。
#                          生产训练不能不写：不写 = 优先级 0，低于验收任务的 low=100，验收任务反过来抢生产（04 §4.2「优先级」）
#                          ⚠ **共享集群上一律 POD_PRIORITY=（清空）**：默认 high 会静默抢占其他团队的低优先级任务
#                          （共享集群上可能已有同名 high，值和你不同）。
#                          起 render / preflight 在 POD_PRIORITY 非空时查集群里有没有**别人的**更低优先级 TPU pod，有就大声警告（不拦）
#     AVOID_MANUAL=1       apply 前过一遍 05-avoid-manual-slices.py（避开手工 Slice 占着的 cube，经验教训第 33 条）
#     RECIPE=dsv3（默认）= 上游公开 MaxText 的 DeepSeek-V3 671B 配方（06b）：
#                          模板换成 06a-jobset-dsv3.yaml、参数读 06a-dsv3-recipe.env；不要 env tar / PVC / overlay / Secret，
#                          只要 PROJECT RUN_NAME JOBSET_NAME；可选 OUTPUT_DIR（默认 pod 本地 /tmp/maxtext-out）。
#                          SLICES / TOPO / STEPS / EXTRA_LIBTPU_ARGS / POD_PRIORITY / AVOID_MANUAL 照常生效；MODEL_NAME / TRAIN_SCRIPT 不适用
#                          RECIPE=ref = 你自备的配方：模板 06-jobset-ref.yaml、overlay 06-config-overlay-ref.yml 放在本脚本同目录
#                          （下载包里没有这两个文件，要自己准备）；MODEL_NAME / TRAIN_SCRIPT 只对它生效
#     EXCLUDE_POOLS=       后加（经验教训第 48 条，未实测）：render 时给每个 replicatedJob 追加
#                          cloud.google.com/gke-nodepool NotIn [池名…] 的 required 亲和（逗号或空格分隔）；
#                          EXCLUDE_POOLS=auto = 现场跑一遍下面的 notin-pools 取名单。降级没同步成 label，
#                          TAS 看不见坏 cube，只能提交前手工排除；名单当天重查，别复用昨天的
#     XLA_FLAGS_ENV=       只对 dsv3：覆盖容器的 XLA_FLAGS（默认读 recipe.env）；单片量「关 pass 的代价」时设空。
#                          多片（SLICES>1）却没有 collective-pipeliner-forward → preflight FAIL（AOT：多片不带 = OOM）
#   render      打印渲染后的 YAML（不改任何东西）
#   notin-pools 列出提交前该排除的节点池（只读；后加，未实测）：sub-block 不是 HEALTHY 16/16 的池，
#               加上有节点带维护标签 / 维护 taint / NotReady 的池（06c §6.10 第 9 条）。池名到 stdout，原因到 stderr
#   overlay-print  只打印 overlay（不上传）
#   overlay     按 SLICES 生成本 run 的 config_overlay.yml 并上传（已存在且内容不同 → 拒绝，换 RUN_NAME）
#   preflight   检查 PVC / Secret / overlay / env tar / LocalQueue 都在（只读）
#   apply       preflight 通过后 kubectl apply
#   watch       JobSet / Workload / Slice / Pod 概况（只读，可反复跑）
#   logs        跟 worker-0-0 的日志
#   stop        Workload active=false（释放切片，不删对象）      resume  active=true（重新排队）
#   suspend     patch JobSet spec.suspend=true [需验证]           unsuspend
#   verify      验收：step 数、loss、生效 flag、存档
#   steps       从 worker-0-0 日志算稳态步时 p50 / 均值 / 慢步占比 / TFLOP/s / HBM（跑着也能用），06c §6.6 判读用
#   delete      删 JobSet（Workload / Slice / Pod 随之回收）并确认 Slice 已释放
# 幂等：apply 同名同内容 → unchanged；同名改了内容会被拒（JobSet 大部分 spec 不可变）→ 先 delete 或换 JOBSET_NAME/RUN_NAME（如 -v2）。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
RECIPE="${RECIPE:-dsv3}"
case "$RECIPE" in
  ref)
    rb_env_check PROJECT DATA_BUCKET RUN_NAME JOBSET_NAME ENV_TAR
    VARS='$JOBSET_NAME $NS $LOCAL_QUEUE $DATA_PVC $DATA_BUCKET $RUN_NAME $ENV_TAR $GITHUB_SECRET' ;;
  dsv3)   # 06b：公开配置配方
    rb_env_check PROJECT RUN_NAME JOBSET_NAME
    source "$(dirname "$0")/06a-dsv3-recipe.env"
    RECIPE_XLA_FLAGS_ENV="${XLA_FLAGS_ENV-$RECIPE_XLA_FLAGS_ENV}"
    [[ "$RUN_NAME" == my-run || "$JOBSET_NAME" == my-run ]] && { echo "RECIPE=dsv3 但 RUN_NAME/JOBSET_NAME 还是 env.sh 的默认 my-run：先 export（06b §3）" >&2; exit 2; }
    export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/maxtext-out}" RECIPE_IMAGE RECIPE_MODEL_ARGS RECIPE_XLA_FLAGS_ENV
    export RECIPE_LIBTPU_ARGS="$RECIPE_XLA_FLAGS $RECIPE_RUNTIME_FLAGS"
    VARS='$JOBSET_NAME $NS $LOCAL_QUEUE $RUN_NAME $OUTPUT_DIR $RECIPE_IMAGE $RECIPE_MODEL_ARGS $RECIPE_LIBTPU_ARGS $RECIPE_XLA_FLAGS_ENV' ;;
  *) echo "RECIPE=$RECIPE 不认识（ref | dsv3）" >&2; exit 2 ;;
esac
SLICES="${SLICES:-8}"; TOPO="${TOPO:-4x8x8}"; STEPS="${STEPS:-50}"   # 默认 8 只是演示值，不是本书实测的片数
TRAIN_SCRIPT="${TRAIN_SCRIPT:-pretrain_moe_large.sh}"; EXTRA_LIBTPU_ARGS="${EXTRA_LIBTPU_ARGS:-}"
EXCLUDE_POOLS="${EXCLUDE_POOLS:-}"
POD_PRIORITY="${POD_PRIORITY-high}"   # 默认 high：不写优先级 = 0，比验收任务的 low=100 还低，会被验收任务抢占 → 全部重切；显式 POD_PRIORITY= 才关掉
AVOID_MANUAL="${AVOID_MANUAL:-1}"; MODEL_NAME="${MODEL_NAME:-}"
EXTRA_ESC=$(printf '%s' "$EXTRA_LIBTPU_ARGS" | sed -e 's/[\\/&]/\\&/g')   # flag 里带 / 或 & 时 sed 替换不再报错
IFS=x read -r _a _b _c <<<"$TOPO"; CHIPS=$(( _a * _b * _c )); PODS=$(( CHIPS / 4 )); FSDP=$(( CHIPS * 2 ))
(( CHIPS % 64 == 0 )) || { echo "TOPO=$TOPO 不是整 cube（Kueue 路径只收 4x4x4 的整数倍，04 §4.2）" >&2; exit 2; }
shared_cluster_check() {   # POD_PRIORITY 非空时，找「不是本 JobSet」的、已准入且优先级更低的 Kueue Workload；只警告不拦
  [[ -n "$POD_PRIORITY" && -z "${SCC_DONE:-}" ]] || return 0
  SCC_DONE=1                                   # apply = preflight + render，只报一次
  command -v kubectl >/dev/null 2>&1 || return 0
  local val lower
  val=$(kubectl --request-timeout=10s get priorityclass "$POD_PRIORITY" -o jsonpath='{.value}' 2>/dev/null) || return 0
  [[ "$val" =~ ^-?[0-9]+$ ]] || return 0
  # 查 Workload 不查 pod：Kueue 抢占的单位是 Workload；大集群上 get pods -A -o json 要 100 s 以上（实测），Workload 1–2 s
  lower=$(kubectl --request-timeout=20s get workloads.kueue.x-k8s.io -A --no-headers \
      -o custom-columns='NS:.metadata.namespace,OWNER:.metadata.ownerReferences[0].name,PRIO:.spec.priority,CQ:.status.admission.clusterQueue' 2>/dev/null \
    | awk -v js="$JOBSET_NAME" -v v="$val" '$4 != "<none>" && $2 != js { p = ($3 == "<none>") ? 0 : $3; if (p + 0 < v + 0) printf "%s/%s（priority %s，ClusterQueue %s）\n", $1, $2, p, $4 }') || return 0
  [[ -n "$lower" ]] || return 0
  {
    echo "  ██ WARN 共享集群？POD_PRIORITY=$POD_PRIORITY（值 $val）高于下面这些已准入、不属于本 JobSet 的 Workload："
    head -20 <<<"$lower" | sed 's/^/  ██      /'
    (( $(wc -l <<<"$lower") > 20 )) && echo "  ██      …共 $(wc -l <<<"$lower") 个"
    echo "  ██ 同一 ClusterQueue 下 Kueue 会为本任务抢占它们（共享集群上可能已有同名 high，值和你不同）。"
    echo "  ██ 独占集群、而且这些都是你自己的任务 → 可以忽略；共享集群 → 用 POD_PRIORITY= 重来，并 render 确认 YAML 里没有 priorityClassName（06c §6.4）"
  } >&2
}
notin_pools() {   # 经验教训第 48 条 [需验证]：照 05-avoid-manual-slices.py 的节点判据，另加 sub-block healthInfo
  rb_env_check RESERVATION BLOCK ZONE PROJECT POOL_PREFIX >&2 || return 1
  {
    # ① GCE 侧：sub-block 不是 HEALTHY 16/16 → 对应池（池名规则同 02-create-tpu-nodepools.sh：前缀 + 三位 sub-block 号）
    gcloud compute reservations sub-blocks list "$RESERVATION" --block-name="$BLOCK" --zone="$ZONE" --project="$PROJECT" \
        --format="csv[no-heading](name,healthInfo.healthStatus,healthInfo.healthyHostCount)" \
      | while IFS=, read -r sb h n; do
          [[ "$h" == HEALTHY && "$n" == 16 ]] && continue
          pool="${POOL_PREFIX}$(printf '%03d' "$((10#${sb##*-subblock-}))")"
          echo "  notin $pool  sub-block ${h:-?} ${n:-?}/16" >&2; echo "$pool"
        done
    # ② GKE 侧：有节点带维护标签 / 维护 taint / NotReady / unreachable 的池（维护标签只提前几分钟出现，挡不住未公告维护）
    kubectl get nodes -o json | jq -r '.items[]
      | select(.metadata.labels["cloud.google.com/gke-nodepool"] != null)
      | select((.metadata.labels["cloud.google.com/scheduled-maintenance-time"] != null)
               or any(.spec.taints[]?; .key=="cloud.google.com/maintenance-window-started" or .key=="node.kubernetes.io/unreachable" or .key=="node.kubernetes.io/not-ready")
               or any(.status.conditions[]?; .type=="Ready" and .status!="True"))
      | "\(.metadata.labels["cloud.google.com/gke-nodepool"])\t\(.metadata.name)"' \
      | while IFS=$'\t' read -r pool node; do echo "  notin $pool  节点 $node 维护中 / NotReady" >&2; echo "$pool"; done
  } | grep . | sort -u
  # 局限：sub-block HEALTHY 16/16 但主机在计划外维护（某个 sub-block 上出过的 FAILURE_INFRA）这两路都看不见，要靠把降级同步成 label + taint 的自动化才能根治（经验教训第 48 条，需自建）
}
exclude_filter() {   # EXCLUDE_POOLS 非空时追加 gke-nodepool NotIn；写法同 05-avoid-manual-slices.py 的 inject()（每条 term 都加）
  local pools="$EXCLUDE_POOLS"
  [[ "$pools" == auto ]] && pools=$(notin_pools | paste -sd, -)
  pools=$(tr ' ' ',' <<<"$pools" | tr -s ',' | sed 's/^,//; s/,$//')
  [[ -n "$pools" ]] || { cat; return; }
  python3 -c '
import sys, yaml
names = sorted(set(sys.argv[1].split(",")))
expr = {"key": "cloud.google.com/gke-nodepool", "operator": "NotIn", "values": names}
docs = list(yaml.safe_load_all(sys.stdin)); n = 0
for d in docs:
    if not d or d.get("kind") != "JobSet": continue
    for rj in d["spec"]["replicatedJobs"]:
        ps = rj["template"]["spec"]["template"]["spec"]
        terms = (ps.setdefault("affinity", {}).setdefault("nodeAffinity", {})
                 .setdefault("requiredDuringSchedulingIgnoredDuringExecution", {}).setdefault("nodeSelectorTerms", []))
        if not terms: terms.append({"matchExpressions": [dict(expr)]})
        for t in terms:
            if dict(expr) not in t.setdefault("matchExpressions", []): t["matchExpressions"].append(dict(expr))
        n += 1
joined = " ".join(names)
print(f"# 已给 {n} 个 replicatedJob 注入 gke-nodepool NotIn（{len(names)} 个池）：{joined}", file=sys.stderr)
yaml.safe_dump_all(docs, sys.stdout, sort_keys=False, width=10000, allow_unicode=True)
' "$pools"
}
render() { render_raw | exclude_filter; }
render_raw() {
  shared_cluster_check
  if [[ "$RECIPE" == dsv3 ]]; then render_dsv3; return; fi
  envsubst "$VARS" < "$(dirname "$0")/06-jobset-ref.yaml" | sed -E \
    -e "s/^(    replicas: )[0-9N]+( |$)/\1${SLICES}\2/" \
    -e "s/^(        (completions|parallelism): )64( |$)/\1${PODS}\3/" \
    -e "s/(cloud.google.com\/gke-tpu-slice-topology: )4x8x8/\1${TOPO}/" \
    -e "s/\{name: STEPS, value: \"50\"\}/{name: STEPS, value: \"${STEPS}\"}/" \
    -e "s/\{name: ICI_FSDP_PARALLELISM, value: \"512\"\}/{name: ICI_FSDP_PARALLELISM, value: \"${FSDP}\"}/" \
    -e "s/scripts\/pretrain_moe_large\.sh/scripts\/${TRAIN_SCRIPT}/g" \
    -e "s/(\{name: LIBTPU_INIT_ARGS, value: \"[^\"]*)\"\}/\1${EXTRA_ESC:+ ${EXTRA_ESC}}\"}/" \
    ${MODEL_NAME:+-e "s/^( +)- \{name: STEPS, value: (\"[0-9]+\")\}$/\1- {name: STEPS, value: \2}\n\1- {name: MODEL_NAME, value: \"${MODEL_NAME}\"}/"} \
  | if [[ -n "$POD_PRIORITY" ]]; then sed -E "s/^(            )restartPolicy: Never$/\1restartPolicy: Never\n\1priorityClassName: ${POD_PRIORITY}/"; else cat; fi
}
render_dsv3() {   # 06b：ici_fsdp_parallelism=-1 自动吃满每片 device，所以换 TOPO 不用改 FSDP 度
  envsubst "$VARS" < "$(dirname "$0")/06a-jobset-dsv3.yaml" | sed -E \
    -e "s/^(    replicas: )[0-9N]+( |$)/\1${SLICES}\2/" \
    -e "s/^(        (completions|parallelism): )64( |$)/\1${PODS}\3/" \
    -e "s/(cloud.google.com\/gke-tpu-slice-topology: )4x8x8/\1${TOPO}/" \
    -e "s/dcn_data_parallelism=N steps=50/dcn_data_parallelism=${SLICES} steps=${STEPS}/" \
    -e "s/(\{name: LIBTPU_INIT_ARGS, value: \"[^\"]*)\"\}/\1${EXTRA_ESC:+ ${EXTRA_ESC}}\"}/" \
  | if [[ -n "$POD_PRIORITY" ]]; then sed -E "s/^(            )restartPolicy: Never$/\1restartPolicy: Never\n\1priorityClassName: ${POD_PRIORITY}/"; else cat; fi
}
overlay_text() {  # JIT 基线作业 overlay 去掉注释，片数换成 SLICES
  grep -v '^#' "$(dirname "$0")/06-config-overlay-ref.yml" \
    | sed -E "s/^(dcn_data_parallelism|num_slices): [0-9N]+$/\1: ${SLICES}/"
}
OVL="gs://$DATA_BUCKET/configs/submit/$RUN_NAME/config_overlay.yml"
wl()     { kubectl -n "$NS" get workloads -o json | jq -r --arg n "$JOBSET_NAME" \
             '.items[] | select(any(.metadata.ownerReferences[]?; .kind=="JobSet" and .name==$n)) | .metadata.name'; }
w00()    { kubectl -n "$NS" get pods -l jobset.sigs.k8s.io/jobset-name="$JOBSET_NAME",jobset.sigs.k8s.io/job-index=0,batch.kubernetes.io/job-completion-index=0 \
             -o jsonpath='{.items[0].metadata.name}' 2>/dev/null; }
ok()  { echo "  OK   $*"; }
bad() { echo "  FAIL $*"; BAD=1; }

preflight() {
  BAD=0
  if [[ "$RECIPE" == dsv3 ]]; then preflight_common; return $BAD; fi
  [[ "$(kubectl -n "$NS" get pvc "$DATA_PVC" -o jsonpath='{.status.phase}' 2>/dev/null)" == Bound ]] \
    && ok "PVC $DATA_PVC Bound" || bad "PVC $DATA_PVC 不是 Bound → 06c §6.1"
  kubectl -n "$NS" get secret "$GITHUB_SECRET" >/dev/null 2>&1 && ok "Secret $GITHUB_SECRET" || bad "缺 Secret $GITHUB_SECRET → 06-build-env.sh 提示的命令"
  kubectl -n "$NS" get localqueue "$LOCAL_QUEUE" >/dev/null 2>&1 && ok "LocalQueue $LOCAL_QUEUE" || bad "缺 LocalQueue → 03 §3.5"
  gcloud storage ls "gs://$DATA_BUCKET/$ENV_TAR" >/dev/null 2>&1 && ok "env tar" || bad "gs://$DATA_BUCKET/$ENV_TAR 不存在 → 06-build-env.sh"
  if o=$(gcloud storage cat "$OVL" 2>/dev/null); then
    n=$(sed -n 's/^num_slices: *//p' <<<"$o"); d=$(sed -n 's/^dcn_data_parallelism: *//p' <<<"$o")
    [[ "$n" == "$SLICES" && "$d" == "$SLICES" ]] && ok "overlay（num_slices = dcn_data_parallelism = $SLICES）" \
      || bad "overlay 里 num_slices=$n dcn_data_parallelism=$d，和 SLICES=$SLICES 对不上 → 换 RUN_NAME 后 $0 overlay"
  else bad "overlay 不存在 → SLICES=$SLICES $0 overlay"; fi
  preflight_common
  return $BAD
}
preflight_common() {
  if [[ "$RECIPE" == dsv3 ]]; then
    kubectl -n "$NS" get localqueue "$LOCAL_QUEUE" >/dev/null 2>&1 && ok "LocalQueue $LOCAL_QUEUE" || bad "缺 LocalQueue → 03 §3.5"
    if (( SLICES > 1 )) && [[ "$RECIPE_XLA_FLAGS_ENV" != *collective-pipeliner-forward* ]]; then
      bad "SLICES=$SLICES 但 XLA_FLAGS 里没有关 collective-pipeliner-forward：AOT 实测多片 temp 95.87 GiB > 94.74，编译必 OOM（06b §5）"
    else ok "XLA_FLAGS='${RECIPE_XLA_FLAGS_ENV}'"; fi
    [[ "$RECIPE_MODEL_ARGS" == *enable_checkpointing=True* && "$OUTPUT_DIR" != gs://* ]] \
      && bad "开了检查点但 OUTPUT_DIR=$OUTPUT_DIR 在 pod 本地：pod 一删检查点就没了 → OUTPUT_DIR=gs://…"
    [[ "$OUTPUT_DIR" == gs://* ]] && { gcloud storage ls "${OUTPUT_DIR%/}/" >/dev/null 2>&1 && ok "OUTPUT_DIR $OUTPUT_DIR 可列" \
      || echo "  WARN OUTPUT_DIR $OUTPUT_DIR 列不出来（本机身份；pod 用的是 ns/$NS/sa/$KSA，要 01 §1.5 ③ 授权）"; }
    TRAIN_SCRIPT="06b DeepSeek-V3 配方（${RECIPE_IMAGE##*@}）"
  fi
  [[ -n "$POD_PRIORITY" ]] || echo "  WARN POD_PRIORITY 为空：本任务优先级 = 0，任何带 low 的验收任务都能抢占它"
  if [[ -n "$POD_PRIORITY" ]]; then
    kubectl get priorityclass "$POD_PRIORITY" >/dev/null 2>&1 && ok "PriorityClass $POD_PRIORITY" \
      || bad "PriorityClass $POD_PRIORITY 不存在（pod 会被拒；现有：$(kubectl get priorityclass -o name 2>/dev/null | tr '\n' ' ')）"
  fi
  shared_cluster_check 2>&1
  free=$(kubectl get nodes -l cloud.google.com/gke-tpu-partition-4x4x4-state=HEALTHY -o name | wc -l)
  [[ "$RECIPE" == dsv3 ]] && FSDP="$(( FSDP / 2 )) × fsdp_transpose 2（ici_fsdp_parallelism=-1 自动）"
  echo "  INFO 本 run：${SLICES} 片 × ${TOPO}（${CHIPS} 芯/片，${PODS} pod/片，ICI_FSDP=${FSDP}），脚本 ${TRAIN_SCRIPT}，STEPS=${STEPS}${EXTRA_LIBTPU_ARGS:+，追加 flag：$EXTRA_LIBTPU_ARGS}"
  echo "  INFO HEALTHY 标签的 TPU 节点 $free 台（需 $(( SLICES * PODS )) = ${SLICES} 片 × ${PODS}；DEGRADED 是否可用看 slice-controller 的 --default-slice-health-node-affinity）"
}

case "${1:-}" in
  render)    render ;;
  notin-pools) notin_pools ;;
  overlay-print) overlay_text ;;   # 只打印本 run 的 overlay（不上传；不能写数据桶时自己放 ConfigMap 用）
  preflight) preflight ;;
  overlay)
    [[ "$RECIPE" == dsv3 ]] && { echo "  OK   RECIPE=dsv3 不用 overlay（参数全在 06a-dsv3-recipe.env，渲染进命令行）"; exit 0; }
    new=$(overlay_text)
    if old=$(gcloud storage cat "$OVL" 2>/dev/null); then
      [[ "$old" == "$new" ]] && { echo "  OK   $OVL 已存在且相同"; exit 0; }
      echo "  FAIL $OVL 已存在且内容不同——别改跑过的 run 的 overlay，换 RUN_NAME（06c §6.3）"; diff <(echo "$old") <(echo "$new") | head; exit 1
    fi
    printf '%s\n' "$new" | gcloud storage cp - "$OVL" && echo "  OK   已上传 $OVL（num_slices = dcn_data_parallelism = $SLICES）" ;;
  apply)
    preflight || exit 1
    if [[ "$AVOID_MANUAL" == 1 ]]; then render | "$(dirname "$0")/05-avoid-manual-slices.py" - - | kubectl apply -f -
    else render | kubectl apply -f -; fi ;;
  watch)
    kubectl -n "$NS" get jobset "$JOBSET_NAME" -o wide
    kubectl -n "$NS" get workloads | awk -v n="jobset-$JOBSET_NAME-" 'NR==1 || index($1,n)==1'
    kubectl get slices.accelerator.gke.io | awk -v n="jobset-$JOBSET_NAME-" 'NR==1 || index($1,n)'
    echo "ACTIVE slices: $(kubectl get slices.accelerator.gke.io --no-headers | grep "jobset-$JOBSET_NAME-" | grep -c ' ACTIVE ' || true) / $SLICES"
    kubectl -n "$NS" get pods -l jobset.sigs.k8s.io/jobset-name="$JOBSET_NAME" --no-headers \
      | awk '{print $3}' | sort | uniq -c | sed 's/^/  pods: /'                      # 期望 SLICES×PODS 个 Running，完成后同数 Completed
    ;;
  logs)      kubectl -n "$NS" logs -f "$(w00)" -c jax-tpu ;;
  stop)      kubectl -n "$NS" patch workload "$(wl)" --type=merge -p '{"spec":{"active":false}}' ;;
  resume)    kubectl -n "$NS" patch workload "$(wl)" --type=merge -p '{"spec":{"active":true}}' ;;
  suspend)   kubectl -n "$NS" patch jobset "$JOBSET_NAME" --type=merge -p '{"spec":{"suspend":true}}' ;;
  unsuspend) kubectl -n "$NS" patch jobset "$JOBSET_NAME" --type=merge -p '{"spec":{"suspend":false}}' ;;
  verify)
    out="$STATE_DIR/run-$RUN_NAME"; mkdir -p "$out"
    kubectl -n "$NS" get jobset "$JOBSET_NAME" -o jsonpath='{range .status.conditions[*]}{.type}={.status} {.reason}{"\n"}{end}'
    p=$(w00); kubectl -n "$NS" logs "$p" -c jax-tpu > "$out/worker-0-0.log"
    last=$(grep -oE 'completed step: [0-9]+' "$out/worker-0-0.log" | tail -1 | awk '{print $3}')
    want=$(( STEPS - 1 ))
    [[ "$last" == "$want" ]] && echo "  OK   最后一步 $last（STEPS-1）" || echo "  FAIL 最后一步 '$last'，期望 $want（经验教训第 44 条：Completed ≠ 成功）"
    grep -E 'completed step: (0|3|'"$want"')[,]' "$out/worker-0-0.log" | sed -E 's/.*(completed step: [0-9]+, seconds: [0-9.]+).*(loss: [0-9.]+).*/  \1  \2/'
    grep -m1 -E 'jax.devices\(\) returned|Num_devices:' "$out/worker-0-0.log" | sed 's/^/  /'   # 期望 SLICES × 芯片数 × 2（JIT 基线作业与 dsv3 10+ 片都是数千）；JIT 基线作业打 jax.devices()，上游 MaxText（dsv3）打 Num_devices
    echo "  期望 device 数 $(( SLICES * CHIPS * 2 ))"
    grep -m1 'TPU runtime versions' "$out/worker-0-0.log" | sed 's/^/  /'
    [[ "$RECIPE" == dsv3 ]] && echo "  期望 jax=0.12.0.dev20260923 libtpu=0.0.49（06b §2）" || echo "  期望 jax=0.11.0 jaxlib=0.11.0 libtpu=0.0.48" 
    grep -m1 'LIBTPU_INIT_ARGS' "$out/worker-0-0.log" > "$out/effective-libtpu-flags.txt" || true   # 生效 flag 存档
    grep -m1 '^XLA_FLAGS:' "$out/worker-0-0.log" >> "$out/effective-libtpu-flags.txt" || true        # RECIPE=dsv3：多片必须有 disable_hlo_passes
    if [[ "$RECIPE" == ref ]]; then gcloud storage ls "gs://$DATA_BUCKET/pretrain/moe-large/$RUN_NAME/" || true
    elif [[ "$OUTPUT_DIR" == gs://* ]]; then gcloud storage ls "${OUTPUT_DIR%/}/$RUN_NAME/" || true; fi                        # 期望有 train-*.log（worker-0-0 同步）
    echo "  存档：$out/"
    ;;
  steps)
    p=$(w00); f=$(mktemp); kubectl -n "$NS" logs "$p" -c jax-tpu > "$f"
    grep -m1 'Total memory size' "$f" | sed -E 's/.*(Total memory size: [0-9.]+ GB).*/  HBM（编译报告）\1 \/ 94.74/'
    grep -m1 'Total estimated memory size' "$f" | sed -E 's/.*(Total estimated memory size: [0-9.]+ GB).*(estimated temp size: [0-9.]+ GB).*/  HBM（编译报告，上游 MaxText 写法）\1，\2 \/ 94.74（编译期初筛看 temp，OOM 以实机峰值为准）/'   # RECIPE=dsv3
    python3 - "$f" <<'EOF'
import re, statistics as st, sys
rows = [(int(m[1]), float(m[2]), float(m[3]), float(m[4])) for m in
        re.finditer(r"completed step: (\d+), seconds: ([\d.]+), TFLOP/s/device: ([\d.]+).*?loss: ([\d.]+)", open(sys.argv[1]).read())]
if not rows: sys.exit("  还没有 completed step 行（编译期：JIT 基线作业从 jax.devices 到 step 0 四十多分钟；dsv3 单片基准约 6 min，其中有 4–5 min 一行日志都没有——都别急着杀）")
print(f"  最后一步 {rows[-1][0]}；step0 {rows[0][1]:.0f} s；loss {rows[0][3]:.3f} → {rows[-1][3]:.3f}")
ss = [r for r in rows if r[0] >= 3]                     # 口径：step>=3，剔除 <0.3×中位 或 >2.1×中位 的假象步
# （阈值取中位的倍数，适配任意步时）
if len(ss) < 5: sys.exit(f"  稳态步只有 {len(ss)} 个，再等等")
med0 = st.median(r[1] for r in ss)
ok = [r for r in ss if 0.3 * med0 <= r[1] <= 2.1 * med0]
t = [r[1] for r in ok]; p50 = st.median(t); mean = st.mean(t)
slow = [x for x in t if x > 1.2 * p50]                 # 慢步口径全书统一 >1.2×稳态 p50（00 术语表 g-slow-step）
print(f"  稳态 {len(ok)} 步（剔除假象 {len(ss)-len(ok)}）：p50 {p50:.2f} s，均值 {mean:.2f} s（+{(mean/p50-1)*100:.1f}%），"
      f"慢步(>1.2×p50) {len(slow)} 个 = {len(slow)/len(t):.0%}")
print(f"  TFLOP/s/device p50 {st.median(r[2] for r in ok):.1f}  → MFU {st.median(r[2] for r in ok)/1153.5:.1%}（÷1153.5，BF16 峰值口径）")
print(("  判读：均值比 p50 高 >23%（验收线 1.23，06c §6.6）→ 按 06c §6.6 ④ 分诊" if mean / p50 > 1.23 else
       "  判读：有慢步（均值 ÷ p50 在 1.05–1.23）：没过优化目标但在验收线内，报告里写慢步占比") if mean / p50 > 1.05 else "  判读：单峰")
EOF
    rm -f "$f" ;;
  delete)
    kubectl -n "$NS" delete jobset "$JOBSET_NAME" --ignore-not-found --wait=true
    sleep 30
    left=$(kubectl get slices.accelerator.gke.io --no-headers 2>/dev/null | grep -c "jobset-$JOBSET_NAME-" || true)
    [[ "$left" == 0 ]] && echo "  OK   Slice 已全部释放" || echo "  WARN 还有 $left 个 Slice（拆除要几分钟；>15 min 见 05 §5.6 / 08 告警 #4、#15）"
    ;;
  *) awk 'NR==2; NR>2 && /^# =+$/ {print; exit} NR>2' "$0"; exit 2 ;;   # 打印整个文件头（原来写死 2–40 行，头长了就截掉 verify/steps/delete）
esac
