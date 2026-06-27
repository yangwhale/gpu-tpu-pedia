#!/bin/bash
# build-megatron-image.sh — 在 GB200 节点本地 build + push Megatron 镜像 (native arm64)
#
# Why 节点本地:
#   - native arm64 build, 无 qemu emulation (5-10× 快)
#   - 节点自带 gcloud SA token, push Artifact Registry 不需额外 auth
#   - buildah 不需要 daemon, 不和 containerd 冲突
#   - 跑在节点上避免任何 master/cloudtop SSH 依赖 (ansible-friendly)
#
# 用法 (节点本地, ansible push):
#   bash build-megatron-image.sh             # build + push
#   bash build-megatron-image.sh --no-push   # build only
#
# 前置: 节点上有 Dockerfile.tencentos4-megatron-k8s134 + env.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../host/env.sh"

PUSH=true
for arg in "$@"; do [ "$arg" = "--no-push" ] && PUSH=false; done

DOCKERFILE="$SCRIPT_DIR/${MEGATRON_DOCKERFILE:-Dockerfile.ngc-megatron-k8s134}"
IMAGE="$MEGATRON_IMAGE"

[ -f "$DOCKERFILE" ] || { echo "ERROR: $DOCKERFILE not found (ansible 是否一并推送了 Dockerfile?)"; exit 1; }

echo "==========================================================="
echo "  Build Megatron image (native arm64 on $(hostname -s))"
echo "  base    = $MEGATRON_BASE_IMAGE"
echo "  TE      = $TRANSFORMER_ENGINE_VERSION"
echo "  MG-LM   = $MEGATRON_LM_TAG"
echo "  target  = $IMAGE"
echo "  push    = $PUSH"
echo "==========================================================="

# 1) 装 buildah + gcloud-cli (idempotent)
echo "[1/3] ensure buildah + gcloud-cli installed"
command -v buildah >/dev/null 2>&1 || sudo dnf install -y buildah 2>&1 | tail -3
command -v gcloud  >/dev/null 2>&1 || sudo dnf install -y google-cloud-cli 2>&1 | tail -3
buildah --version
gcloud --version | head -1

# 2) Artifact Registry auth (节点 SA 自带 token)
if $PUSH; then
  echo "[2/3] auth Artifact Registry via node SA"
  sudo gcloud auth configure-docker us-east1-docker.pkg.dev --quiet 2>&1 | tail -3
  sudo mkdir -p /etc/containers
  sudo cp -f /root/.docker/config.json /etc/containers/auth.json 2>/dev/null || \
    sudo gcloud auth print-access-token | sudo buildah login -u oauth2accesstoken \
      --password-stdin us-east1-docker.pkg.dev
fi

# 3) buildah bud + push
echo "[3/3] buildah bud (native arm64, full CUDA/TE/Megatron stack — 估计 30-60 min)"
sudo buildah bud \
  --platform linux/arm64 \
  --build-arg TE_VER=$TRANSFORMER_ENGINE_VERSION \
  --build-arg MEGATRON_TAG=$MEGATRON_LM_TAG \
  -t $IMAGE \
  -f "$DOCKERFILE" \
  "$SCRIPT_DIR"

echo '--- built images ---'
sudo buildah images | grep megatron

if $PUSH; then
  # build 耗时 > 1h 时 gcloud SA token (1h) 已过期, push 前重新 auth
  echo "[push] re-auth Artifact Registry (token 1h 过期防御)"
  sudo gcloud auth print-access-token | sudo buildah login -u oauth2accesstoken \
    --password-stdin us-east1-docker.pkg.dev 2>&1 | tail -2

  sudo buildah push "$IMAGE"
  echo
  echo "✅ pushed: $IMAGE"
  echo "Use in pod yaml:"
  echo "  sed -i 's|<MEGATRON_IMAGE>|$IMAGE|g' yamls/k8s134/k8s134-megatron-train-dranet.yaml"
fi
