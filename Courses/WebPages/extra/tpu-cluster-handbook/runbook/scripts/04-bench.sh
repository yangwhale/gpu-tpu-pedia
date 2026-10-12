#!/usr/bin/env bash
# =============================================================================
# 04-bench.sh —— 04 章「组件验收」C1–C3 / C6 与激活延迟 A 的驱动：生成 → 提交 → 等完 → 收日志 → 判定 → 删除
#
# 用法：
#   ./04-bench.sh configmap                                    # 把 04-bench/*.py 建成 ConfigMap（$BENCH_CM，默认 rb-bench）
#   ./04-bench.sh gen <形状> <c1|c3|c1c3|ici1d> [--p7] [--hlo] [04-gen-jobset.sh 的其余参数] > js.yaml
#   ./04-bench.sh run <形状> <c1|c3|c1c3|ici1d> [同上]           # gen → 05-avoid-manual-slices.py → apply → 等 → collect → judge → delete
#   ./04-bench.sh collect <jobset名> [输出目录]                # 收全部 pod 日志 + Workload/Slice 快照（要在删 JobSet 之前）
#   ./04-bench.sh timeline <输出目录>                          # A：提交 → QuotaReserved → Admitted(≈ACTIVE+1–2 s) → pod 起 → 完成（秒）
#   ./04-bench.sh judge <输出目录> [--shape S]                 # = 04-bench/bench_judge.py，FAIL 时 exit 1
#   ./04-bench.sh gcsfuse --pool P [--hosts 16] [--readers 32] --file <桶内相对路径> [--name gcs16r32] [--ns NS] [--pvc PVC] > job.yaml   # C6
#       默认 --ns $INSPECT_NS、--pvc ${INSPECT_DATA_PVC:-${DATA_PVC}-inspect}（= 06-data-pv-pvc.yaml 头注释里 $INSPECT_NS 变体的名字）：gcsfuse Job 不申请 TPU、不带队列名，放 $NS 会被 Kueue 挂住
#   ./04-bench.sh cleanup                                      # 删 app=rb-bench 的 JobSet / Job（$NS 和 $INSPECT_NS）和 ConfigMap
#
# 模式：c1 = 单芯片算力 + HBM + host↔device（每 host 各测各的，~2–3 min）；c3 = 片内 ICI 按物理轴（process 0 打印，~2 min）；
#       c1c3 = 两个都跑（~5 min）；ici1d = 一维 mesh 版 ICI（c2_ici_bench.py，只做对照，验收用 c3）；c1ici1d = c1 + ici1d。
#       注意编号：04 正文的卡片 C2 = HBM 带宽，含在 c1 里测；和这里的 ici1d 无关。传 c2 / c1c2 会直接报错，
#       免得照「C1 + C2 + C3」去跑 c2 结果跑成一维 ICI。脚本文件名 c2_ici_bench.py 与卡片编号无关。
#   --p7  ：LIBTPU_INIT_ARGS 加 --xla_tpu_dvfs_p_state=7（功耗档 A/B；阈值分档，见 04-bench/thresholds.csv）
#   --hlo ：C3_HLO=1，打印编译后 HLO 里的集合通信指令计数（核对「测的到底是不是那个集合通信」）
#
# 来源：本书前两轮实测的驱动脚本，按本书变量改写：
#   - JobSet = 04-gen-jobset.sh 原样生成（DEVICE_CHECK 先过，再跑基准）+ 挂 ConfigMap 到 /bench + 标签 app=rb-bench；
#   - 轮询 pod 一律按标签筛（集群 6,000+ pod 时不加标签的 get pods 要 10 s，时间线会因此失真）；
#   - 时间从对象的 lastTransitionTime 读，不靠轮询时刻；
#   - JAX process 0 不一定是 worker-0-0（实测见过 0-2 / 0-3 / 0-7 / 0-29 / 0-51），所以 collect 收全部 pod；
#   - 被抢占（Workload 出现 Preempted）立刻删掉，别让它自动重排回去抢 cube（实测踩过），本轮不算数、重跑。
# 不改集群状态的子命令：gen / timeline / judge / gcsfuse（只打印）。改状态的：configmap / run / cleanup。
# =============================================================================
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
CM="${BENCH_CM:-rb-bench}"
TIMEOUT="${BENCH_TIMEOUT:-1800}"      # 从 apply 起的总超时（秒）；排队 + 激活 + 跑，c1c3 在 8x16x16 上约 10 min
QTIMEOUT="${BENCH_QUEUE_TIMEOUT:-600}" # 迟迟不 Admitted 的上限（激活判超时门槛别低于 5 min，见 04 §4.7 形状阶梯）
ts() { date -u +%FT%TZ; }

gen() {
  local shape="$1" mode="$2"; shift 2
  local p7=0 hlo=0 rest=() name="" slices=1
  while (( $# )); do case "$1" in
    --p7) p7=1; shift;; --hlo) hlo=1; shift;;
    --name) name="$2"; shift 2;;
    --slices) slices="$2"; rest+=("$1" "$2"); shift 2;;
    *) rest+=("$1"); shift;; esac; done
  [[ "$mode" =~ ^(c1c2|c2)$ ]] && { echo "模式 $mode 无效：一维 mesh ICI 对照用 ici1d / c1ici1d；卡片 C2（HBM）含在 c1 里" >&2; exit 1; }
  [[ "$mode" =~ ^(c1|c3|c1c3|ici1d|c1ici1d)$ ]] || { echo "模式只能是 c1 / c3 / c1c3 / ici1d / c1ici1d" >&2; exit 1; }
  # 多片时默认名带 m<片数>（否则 1 片和 2 片同名，并跑会撞名）
  name="${name:-b${mode}-${shape//x/}$( ((slices>1)) && echo "m$slices" || true)$( ((p7)) && echo p7 || true)}"
  local patch; patch="$(cat <<'PY'
import sys, yaml
mode, cm, p7, hlo = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4] == "1"
head = []; body = []
for l in sys.stdin:
    (head if l.startswith("#") and not body else body).append(l)
d = yaml.safe_load("".join(body))
d["metadata"].setdefault("labels", {})["app"] = "rb-bench"
spec = d["spec"]["replicatedJobs"][0]["template"]["spec"]["template"]["spec"]
c = spec["containers"][0]
run = {"c1": ["python /bench/c1_chip_bench.py"], "ici1d": ["python /bench/c2_ici_bench.py"], "c3": ["python /bench/c3_ici_axes.py"],
       "c1c3": ["python /bench/c1_chip_bench.py", "python /bench/c3_ici_axes.py"],
       "c1ici1d": ["python /bench/c1_chip_bench.py", "python /bench/c2_ici_bench.py"]}[mode]
c["command"][2] = c["command"][2].rstrip("\n") + "\n" + "\n".join(run) + "\n"
env = c.setdefault("env", [])
env.append({"name": "PYTHONUNBUFFERED", "value": "1"})
if hlo:
    env.append({"name": "C3_HLO", "value": "1"})
if p7:
    for e in env:
        if e["name"] == "LIBTPU_INIT_ARGS":
            e["value"] = (e["value"] + " --xla_tpu_dvfs_p_state=7").strip(); break
    else:
        env.append({"name": "LIBTPU_INIT_ARGS", "value": "--xla_tpu_dvfs_p_state=7"})
c["volumeMounts"] = c.get("volumeMounts", []) + [{"name": "bench", "mountPath": "/bench", "readOnly": True}]
spec["volumes"] = spec.get("volumes", []) + [{"name": "bench", "configMap": {"name": cm}}]
sys.stdout.write("".join(head) + f"# 04-bench.sh：模式 {mode}{'，P7' if p7 else ''}{'，C3_HLO=1' if hlo else ''}；ConfigMap {cm} 挂 /bench\n")
yaml.safe_dump(d, sys.stdout, sort_keys=False, width=10000, allow_unicode=True)
PY
)"
  # YAML 走 stdin，所以补丁代码用 -c 传（heredoc 会占掉 stdin）
  bash "$HERE/04-gen-jobset.sh" "$shape" --name "$name" ${rest[@]+"${rest[@]}"} | python3 -c "$patch" "$mode" "$CM" "$p7" "$hlo"
}

collect() {
  local js="$1" out="${2:-$STATE_DIR/04/bench/$1}"; mkdir -p "$out"
  local sel="jobset.sigs.k8s.io/jobset-name=$js"
  kubectl -n "$NS" get workloads -o json | jq --arg n "jobset-$js-" '[.items[]|select(.metadata.name|startswith($n))|{name:.metadata.name, created:.metadata.creationTimestamp, conds:.status.conditions}]' > "$out/workload.json"
  kubectl get slices -o json | jq --arg n "-$js-" '[.items[]|select(.metadata.name|contains($n))|{name:.metadata.name, partitions:.spec.partitionIds, conds:.status.conditions}]' > "$out/slices.json"
  kubectl -n "$NS" get pods -l "$sel" -o json | jq -r '.items[]|[.metadata.name, .spec.nodeName, .status.startTime]|@tsv' > "$out/pods.txt"
  cut -f1 "$out/pods.txt" | xargs -P 16 -I{} sh -c "kubectl -n '$NS' logs {} -c main > '$out/pod-{}.log' 2>&1"
  echo "# collect：$(wc -l < "$out/pods.txt") 个 pod 日志 → $out" >&2
}

timeline() {
  python3 - "$1" <<'PY'
import json, os, sys
from datetime import datetime, timezone
d = sys.argv[1]
T = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
t0 = float(open(os.path.join(d, "apply.epoch")).read()) if os.path.exists(os.path.join(d, "apply.epoch")) else None
w = (json.load(open(os.path.join(d, "workload.json"))) or [{}])[0]
if t0 is None and w.get("created"): t0 = T(w["created"])
got = {c["type"]: T(c["lastTransitionTime"]) for c in w.get("conds") or [] if c["status"] == "True"}
act = [T(c["lastTransitionTime"]) for s in json.load(open(os.path.join(d, "slices.json"))) for c in s.get("conds") or []
       if c["type"] == "Ready" and c.get("reason") == "ACTIVE"]
# 按任意空白切（pods.txt 可能是空格分隔，只认 tab 会让 pod_first/pod_last 恒为「-」）
starts = sorted(T(l.split()[2]) for l in open(os.path.join(d, "pods.txt")) if len(l.split()) >= 3)
rel = lambda x: "-" if x is None else f"{x - t0:.0f}"
print("quota_reserved\tslice_active\tadmitted\tpod_first\tpod_last\tfinished\tpreempted")
print("\t".join([rel(got.get("QuotaReserved")), rel(max(act) if act else None), rel(got.get("Admitted")),
                 rel(starts[0] if starts else None), rel(starts[-1] if starts else None), rel(got.get("Finished")),
                 "YES（本轮不算，重跑）" if "Preempted" in got else "no"]))
if not act and got.get("Admitted"):
    print("# Slice 的 ACTIVE 条件已被 DEACTIVATING 覆盖：按 Admitted − 1 s 估（实测 5 个样本 Admitted 恒在 ACTIVE 后 1–2 s）")
PY
}

cmd="${1:-}"; shift || true
case "$cmd" in
configmap)
  kubectl -n "$NS" create configmap "$CM" --from-file="$HERE/04-bench/c1_chip_bench.py" --from-file="$HERE/04-bench/c2_ici_bench.py" \
    --from-file="$HERE/04-bench/c3_ici_axes.py" --dry-run=client -o yaml | kubectl label --local -f - app=rb-bench -o yaml | kubectl apply -f - ;;
gen) gen "$@" ;;
run)
  shape="$1"; mode="$2"
  y="$(gen "$@")"; js="$(awk '/^  name:/{print $2; exit}' <<<"$y")"
  out="$STATE_DIR/04/bench/$js"; mkdir -p "$out"; printf '%s\n' "$y" > "$out/gen.yaml"
  if kubectl get slices -o json | jq -e '[.items[]|select(.metadata.annotations["accelerator.gke.io/owner-workload-name"]==null)]|length>0' >/dev/null; then
    python3 "$HERE/05-avoid-manual-slices.py" "$out/gen.yaml" "$out/js.yaml"      # 有手工 Slice：避开它们占的 cube
  else cp "$out/gen.yaml" "$out/js.yaml"; fi
  kubectl -n "$NS" get configmap "$CM" >/dev/null 2>&1 || { echo "先跑 ./04-bench.sh configmap" >&2; exit 1; }
  date +%s > "$out/apply.epoch"; echo "$(ts) APPLY $js" | tee "$out/run.log"
  kubectl apply -f "$out/js.yaml" >> "$out/run.log"
  t0=$(cat "$out/apply.epoch"); adm=""
  while :; do
    t=$(( $(date +%s) - t0 ))
    conds="$(kubectl -n "$NS" get workloads -o json | jq -r --arg n "jobset-$js-" '[.items[]|select(.metadata.name|startswith($n))|.status.conditions[]?|select(.status=="True")|.type]|join(",")')"
    term="$(kubectl -n "$NS" get jobset "$js" -o jsonpath='{.status.terminalState}' 2>/dev/null || true)"
    echo "t=${t}s wl=[$conds] term=$term" >> "$out/run.log"
    [[ -z "$adm" && "$conds" == *Admitted* ]] && { adm=$t; echo "$(ts) ADMITTED t=${t}s" | tee -a "$out/run.log"; }
    [[ "$conds" == *Preempted* ]] && { echo "$(ts) PREEMPTED：本轮不算，删掉重跑（别让它自动重排）" | tee -a "$out/run.log"; break; }
    [[ -n "$term" ]] && break
    [[ -z "$adm" ]] && (( t > QTIMEOUT )) && { echo "$(ts) 超过 ${QTIMEOUT}s 未准入：kubectl describe workload 看 QuotaReserved 的 message（04「没准入时先分清是哪一种」）" | tee -a "$out/run.log"; break; }
    (( t > TIMEOUT )) && { echo "$(ts) 总超时 ${TIMEOUT}s" | tee -a "$out/run.log"; break; }
    sleep 5
  done
  collect "$js" "$out"
  echo "$(ts) END $js term=${term:-none}" | tee -a "$out/run.log"
  kubectl -n "$NS" delete jobset "$js" --wait=false >> "$out/run.log" 2>&1 || true
  echo "== A 激活时间线（秒，相对 apply）"; timeline "$out" | tee "$out/timeline.tsv"
  rc=0; python3 "$HERE/04-bench/bench_judge.py" "$out" | tee "$out/judge.txt" || rc=$?
  [[ "${term:-}" == Completed ]] || { echo "JobSet 未 Completed（${term:-none}）：本轮判 FAIL 或重跑" >&2; rc=1; }
  exit $rc ;;
collect) collect "$@" ;;
timeline) timeline "$1" ;;
judge) python3 "$HERE/04-bench/bench_judge.py" "$@" ;;
gcsfuse)
  # 默认放 $INSPECT_NS（03 排除出 Kueue 管理）：这个 Job 不申请 TPU、不带队列名，放受管的 $NS 会永远 suspended。
  # 前提是 $INSPECT_NS 里有指向数据桶的 PV/PVC，且 ns/$INSPECT_NS/sa/default 有桶读权限——命令见 04 C6「前提」[未实测]。
  POOL=""; HOSTS=16; READERS=32; FILE=""; NAME=""; GNS="${INSPECT_NS:-rb-inspect}"; GPVC="${INSPECT_DATA_PVC:-}"
  while (( $# )); do case "$1" in
    --pool) POOL="$2"; shift 2;; --hosts) HOSTS="$2"; shift 2;; --readers) READERS="$2"; shift 2;;
    --file) FILE="$2"; shift 2;; --name) NAME="$2"; shift 2;;
    --ns) GNS="$2"; shift 2;; --pvc) GPVC="$2"; shift 2;; *) echo "未知参数 $1" >&2; exit 1;; esac; done
  [[ -n "$POOL" && -n "$FILE" ]] || { echo "要 --pool（开了 Workload Identity 的 TPU 池）和 --file（桶内 ≥10 GB 的文件）" >&2; exit 1; }
  GPVC="${GPVC:-${DATA_PVC}-inspect}"   # 与 06-data-pv-pvc.yaml 的 $INSPECT_NS 变体同名
  [[ "$GNS" == "$NS" ]] && echo "# 注意：--ns 等于 \$NS（$NS），Kueue 会把这个无队列名的 Job 挂住；除非你确认 $NS 不受 Kueue 管" >&2
  JOB="${NAME:-gcs${HOSTS}r${READERS}}" N="$HOSTS" POOL="$POOL" FILE="$FILE" READERS="$READERS" NS="$GNS" DATA_PVC="$GPVC" \
    envsubst '$JOB $NS $N $POOL $FILE $READERS $DATA_PVC $TEST_IMAGE' < "$HERE/04-bench/gcsfuse-read.yaml.tmpl" ;;
cleanup)
  kubectl -n "$NS" delete jobset,job -l app=rb-bench --ignore-not-found
  kubectl -n "${INSPECT_NS:-rb-inspect}" delete job -l app=rb-bench --ignore-not-found   # C6 gcsfuse Job 默认在这里
  kubectl -n "$NS" delete configmap "$CM" --ignore-not-found ;;
*) sed -n '2,24p' "$0"; exit 1 ;;
esac
