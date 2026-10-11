#!/usr/bin/env bash
# =============================================================================
# 06-build-env.sh —— 06c §6.2：构建并上传预构建 Python 环境 tar（渲染 06-envbuild-pod.yaml → 跑 → 验 → 清理）
#
# 出处：构建脚本见 06-envbuild-pod.yaml 头注释；
#   Secret 用法：kubectl create secret generic（https://kubernetes.io/docs/tasks/configmap-secret/managing-secret-using-kubectl）；
#   对象检查：gcloud storage ls（https://cloud.google.com/sdk/gcloud/reference/storage/ls）。
#
# 用法：
#   ./06-build-env.sh --dry-run        # 只打印渲染后的 Pod YAML
#   ./06-build-env.sh                  # 目标对象已存在就跳过（幂等）
#   ./06-build-env.sh --force          # 已存在也重建（会覆盖同名对象——换代码版本时请改 ENV_TAR 名字，别覆盖）
# MAXTEXT_REPO / MAXTEXT_BRANCH 必填（不填直接报错退出），MAXTEXT_COMMIT 填完整 commit；换代码版本时在调用前 export MAXTEXT_* / LIBTPU_VERSION / ENV_TAR。
# 前提：已 source env.sh；${GITHUB_SECRET} 已建在 $BUILD_NS（见下方提示）；ns/$BUILD_NS/sa/$KSA 对桶有 objectUser（06c §6.1 警告框）。
# namespace [需验证]：构建 pod 是非 TPU pod，放进 $NS 会被 Kueue 永远挂 SchedulingGated（06c §6.1 警告框），
#   所以默认放 $INSPECT_NS（Kueue 排除列表里那个）。BUILD_NS=$NS 可放回 $NS（只适合没装 Kueue pod 集成的集群）。
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/env.sh"
rb_env_check PROJECT DATA_BUCKET ENV_TAR INSPECT_NS
BUILD_NS="${BUILD_NS:-$INSPECT_NS}"

export MAXTEXT_REPO="${MAXTEXT_REPO:?填你的 MaxText 仓库 URL}"
export MAXTEXT_BRANCH="${MAXTEXT_BRANCH:?填你的 MaxText 分支}"
export MAXTEXT_COMMIT="${MAXTEXT_COMMIT:-<commit>}"
export LIBTPU_VERSION="${LIBTPU_VERSION:-0.0.48}"
export IMAGE="${IMAGE:-mirror.gcr.io/library/python@sha256:4d1caded1f729ae443eb803f26ffde7b61e696aeaef62f099abb6dd6b14257c7}"
export AR_PYPI="${AR_PYPI-${REGION}-python.pkg.dev/${PROJECT}/pypi-remote/simple/}"   # 设 AR_PYPI= 为空则走默认 PyPI
export ENVBUILD_POD="${ENVBUILD_POD:-envbuild-${MAXTEXT_COMMIT:0:7}}"
VARS='$ENVBUILD_POD $NS $SYSTEM_POOL $GITHUB_SECRET $MAXTEXT_REPO $MAXTEXT_BRANCH $MAXTEXT_COMMIT $LIBTPU_VERSION $AR_PYPI $PROJECT $DATA_BUCKET $ENV_TAR $IMAGE'
render() { NS="$BUILD_NS" envsubst "$VARS" < "$(dirname "$0")/06-envbuild-pod.yaml"; }   # 模板里 ${NS} 填 BUILD_NS

case "${1:-}" in --dry-run) render; exit 0;; esac
OBJ="gs://${DATA_BUCKET}/${ENV_TAR}"

if [[ "${1:-}" != --force ]] && gcloud storage ls -l "$OBJ" >/dev/null 2>&1; then
  echo "已存在，跳过：$(gcloud storage ls -l "$OBJ" | head -1)"; exit 0; fi

kubectl -n "$BUILD_NS" get secret "$GITHUB_SECRET" >/dev/null 2>&1 || {
  echo "缺 Secret $BUILD_NS/$GITHUB_SECRET。先建（token 别写进 shell 历史）：" >&2
  echo "  read -rs T && kubectl -n $BUILD_NS create secret generic $GITHUB_SECRET --from-literal=token=\"\$T\"; unset T" >&2
  exit 1; }

kubectl -n "$BUILD_NS" delete pod "$ENVBUILD_POD" --ignore-not-found --wait=true   # 上次残留（restartPolicy Never 的 pod 不能 apply 覆盖）
render | kubectl apply -f -
echo "等待构建（参考 JIT 基线作业从建构建 pod 到提交 JobSet 不到 10 min [推断：按 pod 文件名时间与 JobSet creationTimestamp]；超时 90 min）…"
kubectl -n "$BUILD_NS" wait pod "$ENVBUILD_POD" --for=jsonpath='{.status.phase}'=Succeeded --timeout=90m || {
  echo "构建失败或超时，最后 50 行：" >&2; kubectl -n "$BUILD_NS" logs "$ENVBUILD_POD" --tail=50 >&2 || true; exit 1; }

# ---- 验证 -------------------------------------------------------------------
kubectl -n "$BUILD_NS" logs "$ENVBUILD_POD" > "$STATE_DIR/envbuild-${ENVBUILD_POD}.log"
grep -q '^ENV_BUILD_OK' "$STATE_DIR/envbuild-${ENVBUILD_POD}.log" || { echo "日志里没有 ENV_BUILD_OK" >&2; exit 1; }
grep -E '^VERSIONS ' "$STATE_DIR/envbuild-${ENVBUILD_POD}.log" || true     # 期望 JIT 基线作业：jax 0.11.0 jaxlib 0.11.0 libtpu 0.0.48
gcloud storage ls -l "$OBJ"                                               # 期望约 10.3 GB
echo "日志（含 VERSIONS 行，存档用）：$STATE_DIR/envbuild-${ENVBUILD_POD}.log"

# ---- 清理 -------------------------------------------------------------------
kubectl -n "$BUILD_NS" delete pod "$ENVBUILD_POD" --wait=false
