#!/usr/bin/env bash
# =============================================================================
# 01-precheck.sh —— 01 章全部「建集群前」检查一次跑完，每项打印 PASS / FAIL / WARN / MANUAL
#
# 出处（逐项见下方每个 check 的注释）：
#   - 工具：gke-gcloud-auth-plugin https://cloud.google.com/kubernetes-engine/docs/how-to/cluster-access-for-kubectl ；
#     Helm ≥3.8.0 默认开 OCI（Helm 官方页 https://helm.sh/docs/topics/registries/ 没写版本号，
#     版本号取自第三方转述，如 https://github.com/celo-org/charts）[需验证]
#
# 用法：
#   source ~/runbook-env.sh && ./01-precheck.sh            # 全部检查
#   ./01-precheck.sh --skip-quota                      # 跳过配额（没有 quotas API 权限时）
# 只读：不改任何云资源，可随时重跑（交付当天务必再跑一次：配额和 sub-block 健康会变，见手册 01 章）。
# 退出码：有任何 FAIL → 1；只有 WARN/MANUAL → 0。
# 覆盖范围：第 0–6 段 = 01 §1.1–§1.4 + §1.5 ①②；第 7 段（新增，未实测）
#   = GKE 版本下限 + 数据桶 / MTC 桶存在性、数据桶授权、MTC 桶 HNS，只报 WARN / MANUAL，不报 FAIL。
#   不覆盖：MTC addon 和 multitier driver 实际版本（要集群建好才能查，见 07 §7.1 第 1–2 条）、ENV_TAR / GITHUB_SECRET（06c 章建）。
# 依赖：gcloud（含 beta 组件）、jq、python3。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

SKIP_QUOTA=0
[[ "${1:-}" == "--skip-quota" ]] && SKIP_QUOTA=1

NFAIL=0; NWARN=0
pass()   { printf 'PASS    %-34s %s\n' "$1" "$2"; }
fail()   { printf 'FAIL    %-34s %s\n        -> %s\n' "$1" "$2" "$3"; NFAIL=$((NFAIL+1)); }
warn()   { printf 'WARN    %-34s %s\n        -> %s\n' "$1" "$2" "$3"; NWARN=$((NWARN+1)); }
manual() { printf 'MANUAL  %-34s %s\n' "$1" "$2"; }

NODES=$(( CUBES_EXPECTED * NODES_PER_POOL ))          # 144 × 16 = 2,304 台 VM
CHIPS=$(( NODES * 4 ))                                 # 每台 tpu7x-standard-4t 4 chip → 9,216
MIN_HEALTHY="${CUBES_SLO:-130}"                  # 01 检查点：≥130 个 cube HEALTHY 16/16

echo "== 0. 变量"
if rb_env_check PROJECT REGION ZONE RESERVATION; then pass "env" "PROJECT=$PROJECT ZONE=$ZONE RESERVATION=$RESERVATION"
else fail "env" "变量没填全" "编辑 ~/runbook-env.sh 后重新 source；RESERVATION 为空说明名字还没到手，停在这里"; exit 1; fi

echo "== 1. 工具"
have() { command -v "$1" >/dev/null 2>&1; }
for t in gcloud kubectl jq python3 envsubst; do
  have "$t" && pass "tool:$t" "$(command -v "$t")" || fail "tool:$t" "找不到" "安装 $t 后重跑"
done
if gcloud compute reservations sub-blocks list --help >/dev/null 2>&1; then pass "tool:gcloud sub-blocks" "可用（GA 轨道）"
else fail "tool:gcloud sub-blocks" "没有 reservations sub-blocks 命令（SDK 太旧）" "gcloud components update（apt 装的 gcloud 用 apt install --only-upgrade google-cloud-cli）"; fi
# beta 组件只在 08 章 monitoring channels list、05 章清 multi-MIG 隐藏参数时用到：缺了只报 WARN
gcloud beta --help >/dev/null 2>&1 && pass "tool:gcloud beta" "可用（08 渠道列表 / 05 multi-MIG 用）" \
  || warn "tool:gcloud beta" "不可用" "gcloud components install beta（只有 08 / 05 两处要）"
have gke-gcloud-auth-plugin && pass "tool:gke-gcloud-auth-plugin" "ok" \
  || fail "tool:gke-gcloud-auth-plugin" "找不到（kubectl 连 GKE 要它）" "gcloud components install gke-gcloud-auth-plugin"
if have helm; then
  hv=$(helm version --template '{{.Version}}' 2>/dev/null | sed 's/^v//')
  if [[ "$(printf '%s\n%s\n' 3.8.0 "$hv" | sort -V | head -1)" != 3.8.0 ]]; then
    fail "tool:helm" "v$hv < 3.8.0" "升级 helm：03 章 Kueue/JobSet 走 oci:// chart"
  else pass "tool:helm" "v$hv（≥3.8.0 才默认支持 oci:）"; fi
else fail "tool:helm" "找不到" "安装 helm ≥3.8.0（03 章要用）"; fi

echo "== 2. 项目 / API / IAM"
st=$(gcloud projects describe "$PROJECT" --format='value(lifecycleState)' 2>/dev/null || true)
[[ "$st" == ACTIVE ]] && pass "project" "ACTIVE" || fail "project" "lifecycleState='$st'" "确认 PROJECT 是项目 ID 且你有权限"

enabled=$(gcloud services list --enabled --project "$PROJECT" --format='value(config.name)' 2>/dev/null || true)
for api in serviceusage.googleapis.com container.googleapis.com compute.googleapis.com; do
  grep -qx "$api" <<<"$enabled" && pass "api:$api" "enabled" \
    || fail "api:$api" "未开" "gcloud services enable $api --project $PROJECT（见手册；幂等，已开再跑无副作用）"
done

acct=$(gcloud config get-value account 2>/dev/null)
roles=$(gcloud projects get-iam-policy "$PROJECT" --flatten='bindings[].members' \
          --filter="bindings.members:${acct}" --format='value(bindings.role)' 2>/dev/null || true)
for r in roles/container.admin roles/compute.networkAdmin roles/iam.serviceAccountUser; do
  if grep -qx "$r" <<<"$roles" || grep -qx roles/owner <<<"$roles"; then pass "iam:$r" "$acct 直接持有"
  else warn "iam:$r" "$acct 没有直接绑定" "可能经由组授予（这里看不到组成员）；否则找项目管理员授予（见手册 01 章）"; fi
done

echo "== 3. reservation 本体（01 §1.1）"
rj=$(gcloud compute reservations describe "$RESERVATION" --zone "$ZONE" --project "$PROJECT" --format=json 2>/dev/null || true)
if [[ -z "$rj" ]]; then
  fail "reservation" "describe 失败" "名字/zone/项目不对，或 reservation 不在本项目；停止"
else
  chk() { # 名称 jq路径 期望 失败处置
    local got; got=$(jq -r "$2 // \"<空>\"" <<<"$rj")
    [[ "$got" == "$3" ]] && pass "$1" "$got" || fail "$1" "得到 '$got'，期望 '$3'" "$4"; }
  chk "res.status"            .status                                                  READY  "等 Google 侧建好（见手册）"
  chk "res.operationalMode"   .advancedDeploymentControl.reservationOperationalMode    ALL_CAPACITY "不是 All Capacity，找 Google 确认 reservation 类型"
  chk "res.deploymentType"    .deploymentType                                          DENSE  "找 Google 确认"
  chk "res.schedulingType"    .schedulingType                                          GROUPED "找 Google 确认（维护方式影响 08 §8.2）"
  chk "res.specificRequired"  .specificReservationRequired                             true   "找 Google 确认"
  chk "res.acceleratorCount"  '.aggregateReservation.reservedResources[0].accelerator.acceleratorCount' "$CHIPS" \
      "空或不足 = 芯片总数未确认，升级给 Google，不要假设 9216"
  chk "res.vmFamily"          .aggregateReservation.vmFamily                           VM_FAMILY_CLOUD_TPU_POD_SLICE_TPU7X "不是 tpu7x pod，停"
  del=$(jq -r '.deleteAtTime // empty' <<<"$rj")
  if [[ -z "$del" ]]; then warn "res.deleteAtTime" "无" "向 Google 确认到期时间"
  else
    left=$(( ( $(date -d "$del" +%s) - $(date +%s) ) / 86400 ))
    (( left > 30 )) && pass "res.deleteAtTime" "$del（还剩 $left 天）→ 写进运维日历" \
                    || warn "res.deleteAtTime" "$del（只剩 $left 天）" "到期 reservation 被删、GKE 立即删 TPU 节点，现在就找 Google 续期"
  fi
fi

echo "== 4. block / sub-block（01 §1.2）"
n=$(gcloud compute reservations blocks describe "$RESERVATION" --block-name "$BLOCK" --zone "$ZONE" --project "$PROJECT" \
      --format='value(resource.reservationSubBlockCount)' 2>/dev/null || true)
# 实测：blocks describe 的输出外面包了一层 resource:（jq keys = ["resource"]），
# 原写法 value(reservationSubBlockCount) 取到空。字段一律从 .resource 下取。
[[ "$n" == "$CUBES_EXPECTED" ]] && pass "block.subBlockCount" "$n" \
  || fail "block.subBlockCount" "得到 '$n'，期望 $CUBES_EXPECTED" "BLOCK 名不对（默认 \${RESERVATION}-block-0001）或 pod 没交满；或 gcloud 输出结构又变了——先 blocks describe ... --format=json | jq keys 看一眼"
sbj=$(gcloud compute reservations sub-blocks list "$RESERVATION" --block-name "$BLOCK" --zone "$ZONE" --project "$PROJECT" \
        --format=json 2>/dev/null || echo '[]')
jq -r '.[] | [(.name|split("-")|last), .healthInfo.healthStatus, (.healthInfo.healthyHostCount//0), (.inUseHostCount//0)] | @tsv' \
  <<<"$sbj" > "$STATE_DIR/subblocks-$(date +%Y%m%d-%H%M).tsv"
usable=$(jq '[.[] | select(.healthInfo.healthStatus=="HEALTHY" and (.healthInfo.healthyHostCount//0)==16)] | length' <<<"$sbj")
total=$(jq 'length' <<<"$sbj")
(( usable >= MIN_HEALTHY )) && pass "subblock.usable" "$usable / $total 个 HEALTHY 且 healthyHostCount=16（门槛 $MIN_HEALTHY）" \
  || fail "subblock.usable" "$usable / $total（门槛 $MIN_HEALTHY）" "把 DEGRADED 清单（$STATE_DIR/subblocks-*.tsv）交 Google；按操作口径重算计划（01 §1.2）"

echo "== 5. 维护（01 §1.3）"
bj=$(gcloud compute reservations blocks describe "$RESERVATION" --block-name "$BLOCK" --zone "$ZONE" --project "$PROJECT" --format=json 2>/dev/null || echo '{}')
# 同上：字段在 .resource 下。(.resource // .) 兼容两种输出。原写法在实测集群上给出假 PASS（实际 ongoing=3）。
if [[ "$(jq -r '(.resource // .) | has("reservationSubBlockCount")' <<<"$bj" 2>/dev/null)" != true ]]; then
  warn "maintenance.parse" "blocks describe 输出结构不认识（没找到 reservationSubBlockCount）" "先 --format=json | jq keys；下面的 pending/ongoing=0 不可信，改看 08-daily-check 的 sub-block 维护列"
fi
pend=$(jq -r '(.resource // .) | .reservationMaintenance.maintenancePendingCount // 0' <<<"$bj")
ong=$(jq -r '(.resource // .) | .reservationMaintenance.maintenanceOngoingCount // 0' <<<"$bj")
ugm=$(jq -c '(.resource // .) | .reservationMaintenance.upcomingGroupMaintenance // {}' <<<"$bj")
if (( pend == 0 && ong == 0 )); then pass "maintenance" "pending=0 ongoing=0"
else warn "maintenance" "pending=$pend ongoing=$ong $ugm" "交付前把可触发（canReschedule=true）的 pending 维护做完，流程见 08 §8.2"; fi

echo "== 6. 配额（01 §1.4；读 limit 不读 usage）"
if (( SKIP_QUOTA )); then manual "quota" "已跳过（--skip-quota）"; else
  reg=$(gcloud compute regions describe "$REGION" --project "$PROJECT" --format=json 2>/dev/null || echo '{}')
  rq() { jq -r --arg m "$1" '.quotas[]? | select(.metric==$m) | "\(.limit|floor) \(.usage|floor)"' <<<"$reg"; }
  for pair in "IN_USE_ADDRESSES:$((NODES+32))" "INSTANCES:$NODES" "INSTANCE_GROUPS:$((CUBES_EXPECTED+10))"; do
    m=${pair%%:*}; need=${pair##*:}; v=$(rq "$m")
    if [[ -z "$v" ]]; then manual "quota:$m" "regions describe 里没有这一项，去 Console 配额页人工看（需求 ≥$need）"
    else read -r lim use <<<"$v"
      (( lim >= need )) && pass "quota:$m" "limit $lim（usage $use，需求 ≥$need）" \
        || fail "quota:$m" "limit $lim < 需求 $need" "提配额申请"; fi
  done
  v=$(rq RESOURCE_POLICIES); [[ -n "$v" ]] && { read -r lim use <<<"$v"; manual "quota:RESOURCE_POLICIES" "limit $lim usage $use ← 每个 super-slice +1，留足余量（先例上万）"; }

  qi() { # quota-id  → 打印本 region 的 value（多维时打印全部维度，交人判断）
    gcloud quotas info describe "$1" --service=compute.googleapis.com --project="$PROJECT" --format=json --quiet </dev/null 2>/dev/null \
    | jq -r --arg r "$REGION" '.dimensionsInfos[]? | select(.dimensions.region==$r) | "\(.details.value) \(.dimensions|tostring)"' || true; }
  # 实测：quotas API 没开 / 没权限时 gcloud 非零退出，在 set -euo pipefail 下整个脚本静默退出，
  # 连「== 结果」都不打印——别把没输出当成「配额 PASS」。|| true 让它落到下面的 MANUAL。
  hdb_need=$(( NODES * TPU_DISK_SIZE_GB ))
  hv=$(qi HDB-TOTAL-GB-per-project-region | head -1 | awk '{print $1}' || true)
  if [[ -z "$hv" ]]; then manual "quota:HDB-TOTAL-GB" "读不到（需要 quotas API 权限）；需求 ≥$hdb_need GB"
  elif (( hv >= hdb_need )); then pass "quota:HDB-TOTAL-GB" "$hv GB ≥ $NODES × $TPU_DISK_SIZE_GB = $hdb_need"
  else fail "quota:HDB-TOTAL-GB" "$hv GB < $hdb_need" "提 QIR；交期 8–14 周；或请 Google 侧用容量调配手段解决"; fi
  tv=$(qi TPUS-PER-TPU-FAMILY-per-project-region | grep -i '7x' | head -1 | awk '{print $1}' || true)
  if [[ -z "$tv" ]]; then manual "quota:TPUS-PER-TPU-FAMILY" "没找到带 7X 的维度；人工看：gcloud quotas info describe TPUS-PER-TPU-FAMILY-per-project-region --service=compute.googleapis.com（需求 ≥$CHIPS）"
  elif (( tv >= CHIPS )); then pass "quota:TPUS-PER-TPU-FAMILY(7X)" "$tv ≥ $CHIPS"
  else fail "quota:TPUS-PER-TPU-FAMILY(7X)" "$tv < $CHIPS" "提配额（先例：不够就提配额申请）"; fi
  if prefs=$(gcloud quotas preferences list --project="$PROJECT" --format=json --quiet </dev/null 2>/dev/null); then
    recon=$(jq -r '.[]? | select(.reconciling==true) | "\(.quotaId) \(.quotaConfig.grantedValue) -> \(.quotaConfig.preferredValue)"' <<<"$prefs" || true)
    [[ -z "$recon" ]] && pass "quota:preferences" "没有 reconciling 中的申请" \
      || warn "quota:preferences" "有申请在途：$(tr '\n' ';' <<<"$recon")" "reconciling=true 表示还没生效，只信上面 describe 的数；另查 preferredValue 有没有被人改小"
  else manual "quota:preferences" "读不到（需要 quotas API 权限），人工看有没有在途申请"; fi   # 原写法读失败也报 PASS（实测发现）
  manual "quota:multi-MIG"  "Google 侧 override（先例约千），要书面确认"
  manual "quota:GCE Reads/min" "ReadsPerMinutePerProjectPerRegion 走配额提升申请，按 (MIG+MMIG)×12/min 起算"
  manual "quota:CPUS 聚合" "reservation 是否吃聚合 CPUS 未证实，要 Google 书面答复"
fi

echo "== 7. 版本 / 桶（01 §1.5 ③④、§1.6；后加，未实测：本段只静态写好、没在任何项目上跑过）"
# 版本下限出处：super-slicing ≥1.35.2-gke.1842000、sub-slicing ≥1.36.0-gke.3712000（00 §3 / 01 §1.6）；
# MTC driver ≥0.8.4 当时只有 1.37.0-gke.3165000 / 3503000 带（07 §7.1 第 2 条），回补 1.36 未核。
vge() { [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" == "$2" ]]; }   # vge A B：A ≥ B
if [[ -z "${GKE_VERSION:-}" || "$GKE_VERSION" == \<* ]]; then
  manual "gke.version" "GKE_VERSION 未填（BOM 还没定）；定了以后重跑本段（01 §1.6）"
else
  for pair in "1.35.2-gke.1842000:super-slicing" "1.36.0-gke.3712000:sub-slicing"; do
    floor=${pair%%:*}; what=${pair##*:}
    vge "$GKE_VERSION" "$floor" && pass "gke.version>=$what" "$GKE_VERSION ≥ $floor" \
      || warn "gke.version>=$what" "$GKE_VERSION < $floor" "换 BOM 里的 GKE 版本（01 §1.6）"
  done
  if vge "$GKE_VERSION" 1.37.0-gke.3165000; then
    pass "gke.version>=MTC" "$GKE_VERSION ≥ 1.37.0-gke.3165000（只说明不低于已知带 driver 0.9.0 的版本；建完按 07 §7.1 第 2 条查实际 driver）"
  else
    warn "gke.version>=MTC" "$GKE_VERSION < 1.37.0-gke.3165000" "要 Google 工程书面确认这个版本带不带 multitier driver ≥0.8.4（07 §7.1 第 2 条），否则换版本"
  fi
  sc=$(gcloud container get-server-config --zone "$ZONE" --project "$PROJECT" --format=json 2>/dev/null || echo '{}')
  ch=$(tr '[:lower:]' '[:upper:]' <<<"${RELEASE_CHANNEL:-}")
  if [[ "$sc" == '{}' ]]; then manual "gke.version.offered" "get-server-config 读不到；人工看 $ZONE 现在还提供不提供 $GKE_VERSION"
  elif jq -e --arg v "$GKE_VERSION" --arg c "$ch" '[.channels[]? | select(.channel==$c) | .validVersions[]?] | index($v)' <<<"$sc" >/dev/null; then
    pass "gke.version.offered" "$GKE_VERSION 在 $ch 频道的 validVersions 里"
  elif jq -e --arg v "$GKE_VERSION" '.validMasterVersions // [] | index($v)' <<<"$sc" >/dev/null; then
    warn "gke.version.offered" "$GKE_VERSION 在 validMasterVersions 里，但不在 $ch 频道" "核对 RELEASE_CHANNEL 与 BOM（01 §1.6）"
  else
    warn "gke.version.offered" "$ZONE 当前不提供 $GKE_VERSION" "版本可能已下架，和 Google 重新确认 BOM（01 §1.6）"
  fi
fi

if [[ -z "${DATA_BUCKET:-}" || "$DATA_BUCKET" == \<* ]]; then
  manual "bucket:data" "DATA_BUCKET 未填，先按 01 §1.5 ④ 建桶"
elif ! dbj=$(gcloud storage buckets describe "gs://$DATA_BUCKET" --project "$PROJECT" --format=json 2>/dev/null); then
  warn "bucket:data" "gs://$DATA_BUCKET 读不到（不存在或没权限）" "按 01 §1.5 ④ 建桶，或找桶主授权"
else
  pass "bucket:data" "gs://$DATA_BUCKET 存在（location $(jq -r '.location // "?"' <<<"$dbj")）"
  if [[ "$NS" == \<* ]]; then
    manual "bucket:data.iam" "NS 未填，没法拼 Workload Identity principal（01 §1.5 ③）"
  else
    sub="ns/$NS/sa/${KSA:-default}"
    if gcloud storage buckets get-iam-policy "gs://$DATA_BUCKET" --format=json 2>/dev/null \
         | jq -e --arg s "$sub" '[.bindings[]? | select(.role=="roles/storage.objectUser") | .members[]? | select(endswith("/subject/" + $s))] | length > 0' >/dev/null; then
      pass "bucket:data.iam" "roles/storage.objectUser 已授给 …/subject/$sub"
    else
      warn "bucket:data.iam" "没找到 roles/storage.objectUser → …/subject/$sub（或读不到 IAM）" "跑 01 §1.5 ③ 的 add-iam-policy-binding；经组授权的这里看不到"
    fi
  fi
fi

if ! mj=$(gcloud storage buckets describe "gs://$MTC_BUCKET" --project "$PROJECT" --format=json 2>/dev/null); then
  warn "bucket:mtc" "gs://$MTC_BUCKET 读不到（不存在或没权限）" "用 07-mtc-setup.sh 建 HNS 桶（07 §7.1 第 3 条）；不用 MTC 可忽略"
else
  hns=$(jq -r '(.hierarchical_namespace.enabled // .hierarchicalNamespace.enabled) | if . == null then "<无此字段>" else tostring end' <<<"$mj")
  case "$hns" in
    true)  pass "bucket:mtc.hns" "gs://$MTC_BUCKET 已开 HNS" ;;
    false) warn "bucket:mtc.hns" "gs://$MTC_BUCKET 没开 HNS" "HNS 只能建桶时开，换名重建（07 §7.1 第 3 条）" ;;
    *)     manual "bucket:mtc.hns" "describe 输出里没找到 HNS 字段（字段名未核），人工看 Console 或 --format=json 全文" ;;
  esac
fi

echo "== 结果：FAIL=$NFAIL WARN=$NWARN（sub-block 清单已存 $STATE_DIR）"
(( NFAIL == 0 ))
