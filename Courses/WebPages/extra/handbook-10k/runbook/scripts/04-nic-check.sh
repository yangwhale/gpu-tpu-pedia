#!/usr/bin/env bash
# =============================================================================
# 04-nic-check.sh —— 「节点 / 网卡健康」：tcp-data-split 必须 off，dmesg 里不能有 gve 复位
#
# 用法：
#   ./04-nic-check.sh node <NODE>           # 单台：kubectl debug node（交互式，看完 exit；再跑 cleanup 删 debug pod）
#   ./04-nic-check.sh fleet [--dry-run]     # 全部 TPU 节点：在 $INSPECT_NS 起一个只读 DaemonSet，每个 pod 打一行 NIC_CHECK
#   ./04-nic-check.sh collect               # 汇总 DaemonSet 输出 → $STATE_DIR/04/nic-check.txt，列出不合格节点
#   ./04-nic-check.sh cleanup [--dry-run]   # 删 DaemonSet 和 kubectl debug 留下的 node-debugger-* pod
#
# 出处：
#   - 节点上执行的方式：蓝图明确「no IAP SSH to nodes ... Use `kubectl debug node/...` instead」
#   - kubectl debug node 的 pod 跑在宿主的 IPC/Network/PID namespace，根文件系统挂在 /host；要特权用 --profile=sysadmin；
#     用完要手工删 pod
#     → 因为共享宿主网络 namespace，容器里的 ethtool 看到的就是宿主的 eth0；dmesg 读的是同一个内核
#   - 检查项与期望：`ethtool -g eth0 | grep -i "tcp data split"` 期望 off（它是 ring 参数，在 -g 里，不在 -k 的 offload 列表里——
#     实测 netshoot 自带 ethtool 7.0：-k 里 0 行含 split，-g 输出 `TCP data split:		off`；按 -k 取值为空，
#     collect 会把每一台都判不合格）；`dmesg -T | grep -i "gve.*reset"` 期望无输出；
#     `NETDEV WATCHDOG` / `gve ... Performing reset` 为故障特征
#   - namespace 用 $INSPECT_NS：它在 Kueue 排除列表里（03-kueue-values.yaml）。在被 Kueue 管的 namespace 里，
#     manageJobsWithoutQueueName: true 会让 Kueue 接管不带队列名的 Pod
#
# NIC_IMAGE：需要带 ethtool 的镜像。默认 nicolaka/netshoot 只是候选（资料里没有指定镜像）[需验证]；上线前换成使用方镜像仓库里
#   按 digest 钉住的镜像。不要依赖宿主上有 ethtool——已有记录显示 tpu_network_optimizer 自带的 ethtool 6.1 在 6.12 内核上不工作。
#
# 只读：不改任何网卡设置。已知：运行时把 tcp-data-split 关掉会在有流量时重分配 ring，本身就会卡队列——
#   发现是 on 的节点，按 经验教训第 19 条 走「升级 GKE 版本 + 重建池」，不要现场 ethtool -K。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"

CMD="${1:-}"; shift || true
DRY=0; ARG=""
for a in "$@"; do case "$a" in --dry-run) DRY=1;; *) ARG="$a";; esac; done
run() { echo "+ $*"; [[ $DRY == 1 ]] || "$@"; }
NIC_IMAGE="${NIC_IMAGE:-nicolaka/netshoot}"
OUT="$STATE_DIR/04"; mkdir -p "$OUT"
CHECK='echo "NIC_CHECK node=$NODE_NAME tcp-data-split=$(ethtool -g eth0 2>/dev/null | awk -F: "/TCP data split/{gsub(/[ \t]/,\"\",\$2); print \$2}") gve_reset=$(dmesg -T 2>/dev/null | grep -ciE "gve.*reset|NETDEV WATCHDOG")"'

ensure_ns() { kubectl get namespace "$INSPECT_NS" >/dev/null 2>&1 || run kubectl create namespace "$INSPECT_NS"; }

case "$CMD" in
node)
  [[ -n "$ARG" ]] || { echo "用法：$0 node <NODE>" >&2; exit 1; }
  ensure_ns
  echo "进去后执行：  ethtool -g eth0 | grep -i 'tcp data split' ; dmesg -T | grep -iE 'gve.*reset|NETDEV WATCHDOG' ; exit"
  run kubectl debug "node/$ARG" -n "$INSPECT_NS" -it --profile=sysadmin --image="$NIC_IMAGE"
  echo "记得清理：$0 cleanup"
  ;;

fleet)
  ensure_ns
  manifest=$(cat <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: rb-nic-check
  namespace: ${INSPECT_NS}
  labels: {app: rb-nic-check}
spec:
  selector: {matchLabels: {app: rb-nic-check}}
  template:
    metadata:
      labels: {app: rb-nic-check}
    spec:
      hostNetwork: true
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu7x
      tolerations:
      - operator: Exists              # TPU 节点带 google.com/tpu taint；cordon 过的节点也要查
      containers:
      - name: check
        image: ${NIC_IMAGE}
        securityContext: {privileged: true}   # dmesg 需要
        env:
        - name: NODE_NAME
          valueFrom: {fieldRef: {fieldPath: spec.nodeName}}
        command: ["sh", "-c", $(printf '%s; sleep infinity' "$CHECK" | jq -Rs .)]
        resources:
          requests: {cpu: 10m, memory: 32Mi}
          limits:   {cpu: 200m, memory: 128Mi}
EOF
)
  if [[ $DRY == 1 ]]; then echo "$manifest"; else echo "$manifest" | kubectl apply -f -; fi
  echo "等 DESIRED=READY：kubectl get ds -n $INSPECT_NS rb-nic-check ；然后 $0 collect"
  ;;

collect)
  : > "$OUT/nic-check.txt"
  for p in $(kubectl get pods -n "$INSPECT_NS" -l app=rb-nic-check -o name); do
    kubectl logs -n "$INSPECT_NS" "$p" 2>/dev/null | grep '^NIC_CHECK' >> "$OUT/nic-check.txt" || echo "NIC_CHECK pod=${p#pod/} NO_OUTPUT" >> "$OUT/nic-check.txt"
  done
  echo "共 $(wc -l < "$OUT/nic-check.txt") 行 → $OUT/nic-check.txt"
  echo "== 不合格（tcp-data-split 不是 off，或 gve_reset>0，或无输出）："
  grep -vE 'tcp-data-split=off gve_reset=0$' "$OUT/nic-check.txt" || echo "  无"
  ;;

cleanup)
  run kubectl delete ds -n "$INSPECT_NS" rb-nic-check --ignore-not-found
  for p in $(kubectl get pods -n "$INSPECT_NS" -o name 2>/dev/null | grep '^pod/node-debugger-' || true); do
    run kubectl delete -n "$INSPECT_NS" "$p" --ignore-not-found
  done
  ;;

*) sed -n '2,10p' "$0"; exit 1;;
esac
