#!/usr/bin/env bash
# =============================================================================
# 04-gen-jobset.sh —— 为 04「形状阶梯」每一级生成一个 Kueue 路径的冒烟 JobSet（只打印 YAML，不 apply）
#
# 用法：
#   ./04-gen-jobset.sh <形状> [--slices N] [--name NAME] [--multi-nic] [--priority WPC | --pod-priority PC]  > js.yaml
#   kubectl apply -f js.yaml
#   形状：2x2x1 2x2x2 2x2x4 2x4x4 4x4x4 4x4x8 4x8x8 8x8x8 8x8x16 8x16x16
#   例：./04-gen-jobset.sh 4x4x4                         # 4.3
#       ./04-gen-jobset.sh 4x8x8 --slices N --multi-nic # 4.8（10+ 片 over DCN）
#   ./04-gen-jobset.sh ladder                            # 打印阶梯表（形状 / 片数 / pod / chip / 期望 device）
#
# 每个 pod：装 JAX → 打印 jax.device_count() → 和期望值比，不等就 exit 1（Job 失败 = 这一级没过）。
#   期望 device = 芯片数 × 2 × 片数（v7x 每芯片 2 个 device，00 §0 口径）。
#
# 出处：
#   - 测试命令：GA 用户指南 multislice JobSet 示例（python:3.12 镜像；
#     `pip install -U --pre jax jaxlib libtpu requests -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`；
#     `python -c 'import jax; print("Total TPU devices:", jax.device_count())'`）
#   - 骨架字段（queue-name label、gke-tpu-slice-topology 注解、只写 gke-tpu-accelerator nodeSelector、hostNetwork、
#     ClusterFirstWithHostNet、privileged、TPU requests==limits 4、backoffLimit 0、Indexed）：实测跑通的参考作业 JobSet
#     与 04「JobSet 骨架」
#   - 镜像默认 $TEST_IMAGE = 参考作业用的 python 镜像 digest
#   - --multi-nic：nodeSelector gke-networking-dra-driver:"true" + LIBTPU_INIT_ARGS --megascale_grpc_interface_prefixes=eth1,eth2,lo
#   - 多片的 MEGASCALE_* / TPU_WORKER_HOSTNAMES 由平台注入到子 Job，JobSet 里不用写
#   - 名字 ≤20 字符
#   - --priority WPC：给 JobSet 加 label kueue.x-k8s.io/priority-class: WPC（Kueue 的 WorkloadPriorityClass，03-kueue-priority.yaml
#     建了 high/medium/low）。不写 = 没有优先级（按 0 算 [推断]）。在已有其他高优先级作业的集群上，不写优先级的验收任务
#     会被新来的任务抢占（Workload Evicted/Preempted「Preempted to accommodate a workload ... due to prioritization
#     in the ClusterQueue」→ Requeued）；在有生产训练的集群上跑验收，按 04「验收任务的优先级」一节选档。
#   - **--priority 指向不存在的 WorkloadPriorityClass 时静默卡死**（集群里没有任何 WPC 时实测如此）：JobSet 一直 suspend、
#     不生成 Workload、没有任何 Event，只有 kueue-controller-manager 日志里有
#     `Handling job with no workload ... WorkloadPriorityClass.kueue.x-k8s.io "low" not found`。
#     所以生成时会先 `kubectl get workloadpriorityclass` 查一下，查不到就报错退出。
#   - --pod-priority PC：改在 pod 模板写 priorityClassName: PC（普通 K8s PriorityClass）。Kueue 没有 WPC 标签时拿它当 Workload
#     优先级（实测：pod 写 high → Workload priority 取该 PriorityClass 的值、priorityClassRef.kind=PriorityClass）。只有 PriorityClass、没有 WPC 的集群用这个。
#     两个都不写 = 优先级 0，通常比集群里已有的所有档位都低，是「最让路」的选择。
#   - 集群里有手工 Slice 时，生成后再过一遍 05-avoid-manual-slices.py，否则 TAS 可能选中被手工片占着的 cube。
#
# 注意：每个 pod 都从公网 pip 安装（要 Cloud NAT 出网，经验教训第 16 条；大量 pod 同时现装既慢又不可复现，经验教训第 27 条——
#   实测集群从单片到 10+ 片同时装都没被限流）。
#   若有自建镜像，预装好 jax/libtpu 后把 TEST_IMAGE 换掉、把 pip 那行删掉即可。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

SHAPES="2x2x1 2x2x2 2x2x4 2x4x4 4x4x4 4x4x8 4x8x8 8x8x8 8x8x16 8x16x16"
chips_of() { local a b c; IFS=x read -r a b c <<<"$1"; echo $((a*b*c)); }

if [[ "${1:-}" == ladder ]]; then
  printf "%-8s %-7s %-6s %-7s %-7s %s\n" 级别 形状 片数 pod数 chip 期望device
  for row in "4.2 2x2x1 1" "4.3 4x4x4 1" "4.4 4x4x8 1" "4.5 4x8x8 1" "4.6 8x8x8 1" "4.7 8x16x16 1" "4.8 4x8x8 ${MS_SLICES:-8}"; do
    read -r lv sh n <<<"$row"; c=$(chips_of "$sh")
    printf "%-8s %-9s %-6s %-7s %-7s %s\n" "$lv" "$sh" "$n" $((c/4*n)) $((c*n)) $((c*2*n))
  done; exit 0
fi

SHAPE="${1:?用法：$0 <形状>|ladder [--slices N] [--name NAME] [--multi-nic] [--priority WPC | --pod-priority PC]}"; shift
[[ " $SHAPES " == *" $SHAPE "* ]] || { echo "不支持的形状 $SHAPE（可选：$SHAPES）" >&2; exit 1; }
SLICES=1; NAME=""; MULTINIC=0; PRIO=""; PODPRIO=""
while (( $# )); do case "$1" in
  --slices) SLICES="$2"; shift 2;; --name) NAME="$2"; shift 2;; --multi-nic) MULTINIC=1; shift;;
  --priority) PRIO="$2"; shift 2;;
  --pod-priority) PODPRIO="$2"; shift 2;;
  *) echo "未知参数 $1" >&2; exit 1;; esac; done

CHIPS=$(chips_of "$SHAPE"); HOSTS=$((CHIPS/4)); EXPECT=$((CHIPS*2*SLICES))
NAME="${NAME:-s${SHAPE//x/-}-n${SLICES}}"
(( ${#NAME} <= 20 )) || { echo "名字 $NAME 超过 20 字符" >&2; exit 1; }
# Slice 名 = {ns}-jobset-{name}-{hash5}-worker-{i}，上限 48 → ns + name + "worker" + 下标位数 ≤ 32（按 Slice 名上限 48 倒推，[推断]；命名预算见经验教训第 13 条）
_budget=$(( ${#NS} + ${#NAME} + 6 + ${#SLICES} ))
(( _budget <= 32 )) || { echo "名字预算超了：namespace(${#NS}) + JobSet 名(${#NAME}) + worker(6) + 下标(${#SLICES}) = $_budget > 32（经验教训第 13 条；Slice 名上限 48）→ 缩短 NAME" >&2; exit 1; }
(( CHIPS % 64 == 0 || SLICES == 1 )) || echo "# 警告：sub-cube 形状（<4x4x4）做多片没有先例" >&2

EXTRA_SEL=""; EXTRA_ENV=""; EXTRA_LBL=""
[[ -n "$PRIO" ]] && EXTRA_LBL="    kueue.x-k8s.io/priority-class: ${PRIO}"
if [[ -n "$PRIO" ]] && command -v kubectl >/dev/null && ! kubectl get workloadpriorityclass "$PRIO" >/dev/null 2>&1; then
  echo "WorkloadPriorityClass $PRIO 不存在（或无权读）：带这个标签的 JobSet 会永远 suspend 且没有 Event。" >&2
  echo "  现有 WPC：$(kubectl get workloadpriorityclass -o name 2>/dev/null | tr '\n' ' ')；只有 PriorityClass 的集群改用 --pod-priority" >&2
  exit 1
fi
EXTRA_PC=""; [[ -n "$PODPRIO" ]] && EXTRA_PC="            priorityClassName: ${PODPRIO}"
if (( MULTINIC )); then
  EXTRA_SEL='              cloud.google.com/gke-networking-dra-driver: "true"'
  EXTRA_ENV='              - {name: LIBTPU_INIT_ARGS, value: "--megascale_grpc_interface_prefixes=eth1,eth2,lo"}'
fi

cat <<EOF
# 生成：04-gen-jobset.sh $SHAPE --slices $SLICES   → ${SLICES} 片 × ${SHAPE}：${HOSTS} pod/片，共 $((HOSTS*SLICES)) pod、$((CHIPS*SLICES)) chip，期望 jax.device_count() = ${EXPECT}
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: ${NAME}
  namespace: ${NS}
  labels:
    kueue.x-k8s.io/queue-name: ${LOCAL_QUEUE}
${EXTRA_LBL}
spec:
  failurePolicy:
    maxRestarts: 0
  replicatedJobs:
  - name: worker
    replicas: ${SLICES}                  # Kueue 为每个 replica 建一个 Slice
    template:
      spec:
        backoffLimit: 0
        completionMode: Indexed
        parallelism: ${HOSTS}
        completions: ${HOSTS}
        template:
          metadata:
            annotations:
              cloud.google.com/gke-tpu-slice-topology: ${SHAPE}
          spec:
            restartPolicy: Never
${EXTRA_PC}
            hostNetwork: true
            dnsPolicy: ClusterFirstWithHostNet
            nodeSelector:
              cloud.google.com/gke-tpu-accelerator: tpu7x   # 不写 gke-tpu-topology，Kueue/webhook 会加
${EXTRA_SEL}
            tolerations:
            - {key: google.com/tpu, operator: Exists, effect: NoSchedule}
            containers:
            - name: main
              image: ${TEST_IMAGE}
              securityContext: {privileged: true}
              ports: [{containerPort: 8471}]
              env:
              - {name: EXPECT_DEVICES, value: "${EXPECT}"}
${EXTRA_ENV}
              command:
              - bash
              - -c
              - |
                set -ex
                pip install -U --pre jax jaxlib libtpu requests -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
                python -c 'import jax; print("Total TPU devices:", jax.device_count(), "local:", jax.local_device_count())'
                python -c 'import jax,os,sys; n=jax.device_count(); e=int(os.environ["EXPECT_DEVICES"]); print("DEVICE_CHECK", "OK" if n==e else "MISMATCH", n, e); sys.exit(0 if n==e else 1)'
              resources:
                requests: {google.com/tpu: "4"}
                limits:   {google.com/tpu: "4"}
EOF
