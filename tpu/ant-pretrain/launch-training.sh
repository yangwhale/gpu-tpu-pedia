#!/bin/bash
# Run pretrain_al_model.sh on all pods whose names start with "tpuv7x-32-xl"

set -e

WORKDIR="/tmp/ramdisk/ant-pretrain"

# Get all pod names starting with tpuv7x-32-xl
PODS=$(kubectl get pods --no-headers -o custom-columns=":metadata.name" | grep "^tpuv7x-32-xl")

if [[ -z "$PODS" ]]; then
    echo "No pods found with prefix 'tpuv7x-32-xl'"
    exit 1
fi

POD_COUNT=$(echo "$PODS" | wc -l | tr -d ' ')
echo "Found $POD_COUNT pod(s) matching 'tpuv7x-32-xl'"

# Build the setup + launch command to execute on each pod
read -r -d '' POD_CMD << 'EOFCMD' || true
set -e
WORKDIR="/tmp/ramdisk/ant-pretrain"
VENV_DIR="${WORKDIR}/maxtext_venv"
LOG_DIR="${WORKDIR}/logs"

# Copy code from GCS-FUSE (read-only) to ramdisk (writable) if not already there
if [[ ! -d "$WORKDIR/src" ]]; then
    echo "=== Copying code from GCS-FUSE to ramdisk ==="
    mkdir -p "$WORKDIR"
    cp -r /ant-pretrain-code/* "$WORKDIR/"
    echo "=== Code copied ==="
fi

cd "$WORKDIR"

# Check if uv is available and maxtext can be imported
NEED_SETUP=0
if ! command -v uv &>/dev/null; then
    echo "uv not found, need setup"
    NEED_SETUP=1
elif ! python3 -c "import MaxText" &>/dev/null; then
    echo "maxtext not importable, need setup"
    NEED_SETUP=1
fi

if [[ $NEED_SETUP -eq 1 ]]; then
    echo "=== Installing dependencies ==="
    pip install uv
    uv venv --python 3.12 --seed --system-site-packages --clear "$VENV_DIR"
    source "${VENV_DIR}/bin/activate"
    # Install maxtext deps (.[tpu] will install PyPI JAX, we'll fix below)
    uv pip install -e .[tpu] --resolution=lowest
    # Pin JAX 0.9.1 + libtpu nightly (tested working combo for TPU v7)
    pip install jax==0.9.1 jaxlib==0.9.1 \
      -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ \
      -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    install_maxtext_github_deps
    pip uninstall -y optax
    pip install git+https://github.com/google-deepmind/optax
    echo "=== Dependencies installed ==="
else
    echo "=== MaxText already installed, using global Python ==="
fi

mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/pretrain_al_model_$(date +%Y%m%d_%H%M%S).log"

nohup bash scripts/pretrain_al_model.sh >"$LOG_FILE" 2>&1 < /dev/null &
TRAIN_PID=$!

if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "Failed to start training process"
    exit 1
fi

echo "Training started in background. PID=${TRAIN_PID}, log=${LOG_FILE}"
EOFCMD

# Execute the setup + launch command on all pods in parallel
PIDS=()
for POD in $PODS; do
    echo "Launching on pod: $POD"
    kubectl exec "$POD" -- bash -c "$POD_CMD" &
    PIDS+=($!)
done

echo "Waiting for all pod launch commands to finish..."
FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        FAILED=$((FAILED + 1))
    fi
done

if [[ $FAILED -gt 0 ]]; then
    echo "$FAILED pod(s) failed to launch training"
    exit 1
fi

echo "Training launched successfully on all pods"
