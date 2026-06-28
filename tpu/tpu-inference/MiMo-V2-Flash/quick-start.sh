#!/usr/bin/env bash
###############################################################################
#
#  MiMo-V2-Flash Inference on TPU v7x-8 with SGLang-JAX
#  =====================================================
#
#  INTRODUCTION
#  ------------
#
#  Today I want to walk you through running Xiaomi's MiMo-V2-Flash model
#  on a single TPU v7x-8 host using SGLang-JAX.
#
#  But first, let me give you some context on why SGLang-JAX matters for
#  the TPU inference ecosystem.
#
#  We already have vLLM TPU as a solid inference engine. vLLM provides
#  fully tested, production-ready support for mainstream model families
#  — Qwen (including Qwen3, Qwen3.5, Qwen3-Coder-480B), Gemma (3 and 4),
#  and Llama (3.1, 3.3). These models all pass unit tests, correctness
#  tests, and performance benchmarks on TPU.
#
#  SGLang-JAX complements vLLM in several important ways:
#
#    1. EXCLUSIVE MODEL COVERAGE — Some model families are only available
#       in SGLang-JAX on TPU today. MiMo (V2-Flash and V2.5-Pro) is not
#       listed in vLLM TPU's support matrix at all. SGLang-JAX also
#       provides first-class support for GLM-4 MoE, Grok-2, and Bailing
#       MoE — models frequently requested by enterprise customers.
#
#    2. OPTIMIZED MOE INFERENCE — SGLang-JAX ships two production-
#       validated MoE backends: "fused" (a Pallas kernel that combines
#       expert computation and all-to-all communication) and "epmoe"
#       (GMM-based expert-parallel dispatch). Our benchmarks on the
#       latest codebase show fused + dp=2 achieves 782 tok/s on v7x-8.
#       This level of MoE-specific tuning is critical for models with
#       hundreds of experts, like MiMo-V2-Flash (256 experts).
#
#    3. MTP (MULTI-TOKEN PREDICTION) — SGLang-JAX is the first inference
#       engine on TPU to support speculative decoding with MTP. Models
#       that natively train with MTP heads can leverage this for
#       significant latency improvements.
#
#    4. FAST RESPONSE TO CUSTOMER NEEDS — When a customer needs a
#       specific model or configuration, the SGLang-JAX team can turn
#       around support quickly. This agility is critical for enterprise
#       customers with unique requirements.
#
#  So think of it this way: vLLM is our workhorse for mainstream models
#  with proven, tested performance. SGLang-JAX extends our reach into
#  complex MoE architectures and cutting-edge models where production-
#  grade MoE kernel support is essential. Together, they give customers
#  full coverage on TPU.
#
#  Now, let's get hands-on. I'll show you how to deploy MiMo-V2-Flash
#  from zero on a single TPU v7x-8 node in GKE. The entire process takes
#  about 20 minutes, and you'll have a fully functional OpenAI-compatible
#  inference endpoint.
#
###############################################################################

set -euo pipefail

###############################################################################
# ENVIRONMENT VARIABLES
#
# Configure these for your GKE environment before running.
# No project IDs, cluster names, or node pool names are hardcoded
# in the commands below — everything comes from these variables.
###############################################################################

export GKE_PROJECT="${GKE_PROJECT:?Set GKE_PROJECT to your GCP project ID}"
export GKE_CLUSTER="${GKE_CLUSTER:?Set GKE_CLUSTER to your GKE cluster name}"
export GKE_REGION="${GKE_REGION:?Set GKE_REGION to your GKE cluster region}"
export GKE_CONTEXT="gke_${GKE_PROJECT}_${GKE_REGION}_${GKE_CLUSTER}"
export JOB_NAME="${JOB_NAME:-mimo-v2-flash-test}"
export TPU_IMAGE="${TPU_IMAGE:-us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1}"
export MODEL_ID="${MODEL_ID:-XiaomiMiMo/MiMo-V2-Flash}"

# kubectl wrapper — override KUBECTL if you need a custom wrapper
KUBECTL="${KUBECTL:-kubectl --context=${GKE_CONTEXT}}"

###############################################################################
# STEP 0 — Create a TPU v7x-8 Pod on GKE
#
# We start by submitting a Kubernetes Job that requests a TPU v7x-8 node.
# The node pool is configured with autoscaling (min=0, max=1), so when we
# submit this Job, the autoscaler will automatically provision a Spot TPU
# machine for us. This usually takes 3 to 5 minutes.
#
# A few things to note about the manifest:
#   - We use nodeSelector labels (tpu-accelerator and tpu-topology) to
#     target the correct node pool.
#   - The topology "2x2x1" means 4 TPU chips = 8 devices on a single host.
#   - We mount /dev/shm as a 500Gi memory-backed volume. This is critical
#     because the model weights are about 70GB — if we download them to
#     the boot disk (only 100GB), the node gets evicted for disk pressure.
#   - The container simply runs "sleep infinity" so we can exec into it
#     and run commands interactively.
###############################################################################

step0_create_pod() {
  echo "=== Step 0: Creating TPU v7x-8 Pod ==="

  cat > /tmp/tpu-e2e-pod.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: default
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      nodeSelector:
        cloud.google.com/gke-tpu-accelerator: tpu7x
        cloud.google.com/gke-tpu-topology: 2x2x1
      tolerations:
        - key: "google.com/tpu"
          operator: "Exists"
          effect: "NoSchedule"
      containers:
        - name: tpu-shell
          image: ${TPU_IMAGE}
          command: ["sleep", "infinity"]
          resources:
            requests:
              google.com/tpu: "4"
            limits:
              google.com/tpu: "4"
          volumeMounts:
            - name: dev-shm
              mountPath: /dev/shm
      volumes:
        - name: dev-shm
          emptyDir:
            medium: Memory
            sizeLimit: 500Gi
EOF

  $KUBECTL delete job ${JOB_NAME} 2>/dev/null || true
  $KUBECTL apply -f /tmp/tpu-e2e-pod.yaml
  echo "Waiting for Pod to become Ready (Spot TPU ~3-5 min)..."
  $KUBECTL wait --for=condition=Ready pod -l job-name=${JOB_NAME} --timeout=600s
  echo "Pod is Ready!"
}

###############################################################################
# STEP 1 — Verify the Pod and TPU devices
#
# Before we install anything, let's make sure the Pod is healthy and all
# 8 TPU devices are visible to JAX. The v7x-8 machine has 4 physical TPU
# chips, each exposing 2 logical devices, giving us 8 devices total with
# 768 GB of HBM memory.
###############################################################################

step1_verify() {
  echo "=== Step 1: Verifying Pod and TPU devices ==="
  local POD
  POD=$($KUBECTL get pods -l job-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}')
  echo "Pod name: $POD"

  $KUBECTL exec "$POD" -- bash -c '
    echo "Python version: $(python3 --version 2>&1)"
    echo "VFIO devices:   $(ls /dev/vfio/ 2>/dev/null | wc -l)"
    echo "JAX devices:    $(python3 -c "import jax; print(jax.device_count())" 2>/dev/null)"
  '
  echo "Expected: 8 JAX devices"
}

###############################################################################
# STEP 2 — Install SGLang-JAX from source
#
# An important note here: we install from source, not from PyPI. The PyPI
# package (version 0.0.2 at the time of recording) is missing critical
# MoE-related arguments like --ep-size and --moe-backend. These are
# essential for MiMo-V2-Flash which is a Mixture-of-Experts model with
# 256 experts.
#
# We use uv as our package manager — it's much faster than pip for
# resolving and installing dependencies. The full installation takes
# about 2 minutes.
###############################################################################

step2_install() {
  echo "=== Step 2: Installing SGLang-JAX from source ==="
  local POD
  POD=$($KUBECTL get pods -l job-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}')

  $KUBECTL exec "$POD" -- bash -c '
    set -e
    pip install uv 2>&1 | tail -1
    uv venv --python 3.12 /opt/sglang-env 2>&1 | tail -1

    if [ ! -d /opt/sglang-jax ]; then
      echo "Cloning sglang-jax repository..."
      cd /opt && git clone https://github.com/sgl-project/sglang-jax.git 2>&1 | tail -1
    fi

    source /opt/sglang-env/bin/activate
    cd /opt/sglang-jax
    echo "Installing sglang-jax (this takes ~2 min)..."
    uv pip install -e "python[all]" 2>&1 | tail -3

    echo ""
    echo "Verifying installation..."
    python -m sgl_jax.launch_server --help 2>&1 | grep -q "ep-size" \
      && echo "OK: sglang-jax installed with MoE support" \
      || echo "FAIL: ep-size argument not found"
  '
}

###############################################################################
# STEP 3 — Launch the SGLang-JAX inference server
#
# Now for the main event. We launch the SGLang-JAX server with parameters
# optimized for MiMo-V2-Flash on a single v7x-8 host.
#
# Let me walk through the key parameters:
#
#   --tp-size 8        Tensor Parallelism across all 8 TPU devices
#   --dp-size 2        Data Parallelism — the attention path uses TP=4
#                      (8/2), while MoE layers use full EP=8
#   --ep-size 8        Expert Parallelism — distributes 256 experts
#                      across 8 devices (32 experts per device)
#   --moe-backend fused  On the latest codebase, the fused Pallas
#                      kernel outperforms epmoe by ~23% on v7x-8.
#                      It fuses expert computation with all-to-all
#                      communication for higher throughput.
#   --swa-full-tokens-ratio 0.25  MiMo uses hybrid attention: 9 layers
#                      of full attention + 39 layers of sliding window.
#                      This ratio allocates 25% of KV cache to full
#                      attention layers and 75% to SWA layers.
#   --attention-backend fa  FlashAttention kernel for performance
#
# Two critical environment variables:
#   HF_HOME=/dev/shm/hf_cache  — Model weights (~70GB) download to
#     the memory-backed tmpfs instead of the 100GB boot disk. Without
#     this, the node gets evicted for ephemeral storage pressure.
#   JAX_COMPILATION_CACHE_DIR=/dev/shm/jit_cache  — Persist XLA
#     compilation cache so restarts are faster.
#
# The model will be automatically downloaded from HuggingFace on first
# launch. The full cold start (download + weight loading + FP8 dequant
# + XLA compilation) takes about 15 minutes.
###############################################################################

step3_launch() {
  echo "=== Step 3: Launching SGLang-JAX server ==="
  local POD
  POD=$($KUBECTL get pods -l job-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}')

  $KUBECTL exec "$POD" -- bash -c "cat > /tmp/start_sglang.sh << 'SCRIPT'
#!/bin/bash
set -e
source /opt/sglang-env/bin/activate

export HF_HOME=/dev/shm/hf_cache
export JAX_COMPILATION_CACHE_DIR=/dev/shm/jit_cache

echo \"[\$(date)] Starting SGLang-JAX with MiMo-V2-Flash...\"

python -u -m sgl_jax.launch_server \\
    --model-path ${MODEL_ID} \\
    --trust-remote-code \\
    --tp-size 8 --dp-size 2 --ep-size 8 \\
    --moe-backend fused \\
    --host 0.0.0.0 --port 30271 \\
    --page-size 256 --context-length 262144 \\
    --chunked-prefill-size 4096 \\
    --dtype bfloat16 --mem-fraction-static 0.95 \\
    --swa-full-tokens-ratio 0.25 --skip-server-warmup \\
    --max-running-requests 128 \\
    --attention-backend fa
SCRIPT
chmod +x /tmp/start_sglang.sh"

  # Launch in background
  $KUBECTL exec "$POD" -- bash -c '
    rm -f /tmp/sglang.log
    setsid bash /tmp/start_sglang.sh >> /tmp/sglang.log 2>&1 < /dev/null &
    echo "Server launched (PID=$!). Logs: /tmp/sglang.log"
    sleep 3
    tail -5 /tmp/sglang.log
  '
}

###############################################################################
# STEP 4 — Wait for cold start to complete
#
# The server needs about 15 minutes to become ready. Here's what happens
# during cold start:
#
#   1. Model download from HuggingFace    (~3 min, 157 safetensors files)
#   2. Weight loading + MoE expert layout  (~3 min, 282 MoE weight groups)
#   3. FP8 dequantization for attention    (~1 min)
#   4. XLA compilation + precompilation    (~8 min, multiple batch sizes)
#
# We poll the /health endpoint every 10 seconds until we get HTTP 200.
# You can also watch the detailed logs with:
#   kubectl exec <pod> -- tail -f /tmp/sglang.log
###############################################################################

step4_wait_ready() {
  echo "=== Step 4: Waiting for cold start (~15 min) ==="
  local POD
  POD=$($KUBECTL get pods -l job-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}')

  for i in $(seq 1 120); do
    CODE=$($KUBECTL exec "$POD" -- \
      curl -sf -o /dev/null -w "%{http_code}" http://localhost:30271/health 2>/dev/null \
      || echo "000")
    ELAPSED=$((i * 10))
    echo "T+${ELAPSED}s  HTTP ${CODE}"
    if [ "$CODE" = "200" ]; then
      echo ""
      echo "Server is READY!"
      return 0
    fi
    sleep 10
  done
  echo "TIMEOUT: server did not start within 20 minutes"
  return 1
}

###############################################################################
# STEP 5 — Smoke test: verify the model works
#
# Let's send a few requests to make sure everything is working correctly.
#
# First, a simple math question to verify basic generation. Then we send
# the same question 3 times to verify that the KV cache is functioning
# properly — this is especially important for MoE models with hybrid
# SWA attention, because cache accounting bugs can cause subsequent
# requests to produce garbled output.
#
# We actually discovered and patched 5 such bugs when deploying the
# larger MiMo-V2.5-Pro model. MiMo-V2-Flash has been clean so far,
# but it's always good to verify.
###############################################################################

step5_smoke_test() {
  echo "=== Step 5: Smoke test ==="
  local POD
  POD=$($KUBECTL get pods -l job-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}')

  echo "--- Math question ---"
  $KUBECTL exec "$POD" -- bash -c '
    curl -s http://localhost:30271/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"MiMo-V2-Flash\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+3? Answer with just the number.\"}],\"max_tokens\":50,\"temperature\":0}" \
      | python3 -c "import sys,json; r=json.load(sys.stdin); print(\"Answer:\", r[\"choices\"][0][\"message\"][\"content\"], \"| finish_reason:\", r[\"choices\"][0][\"finish_reason\"])"
  '

  echo ""
  echo "--- KV cache consistency check (3 identical requests) ---"
  for i in 1 2 3; do
    echo -n "  Request $i: "
    $KUBECTL exec "$POD" -- bash -c '
      curl -s http://localhost:30271/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"MiMo-V2-Flash\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer in one word.\"}],\"max_tokens\":20,\"temperature\":0}" \
        | python3 -c "import sys,json; r=json.load(sys.stdin); print(r[\"choices\"][0][\"message\"][\"content\"])"
    '
  done
  echo ""
  echo "Expected: all 3 responses should say 'Paris'"
}

###############################################################################
# STEP 6 — Throughput benchmark
#
# This runs a standard throughput benchmark using the exact same parameters
# as the SGLang-JAX official documentation: 256 random prompts with 16K
# input tokens and 1K output tokens each, at concurrency 64.
#
# We tested all four combinations of moe-backend × dp-size. Results:
#
#   moe-backend  dp  Output tok/s  Median ITL  Mean TPOT
#   fused        2   782           25.8 ms     53.2 ms    <-- best
#   fused        1   721           31.0 ms     59.4 ms
#   epmoe        2   636           32.6 ms     65.8 ms
#   epmoe        1   523           33.4 ms     77.2 ms
#
# Key finding: on the latest codebase (dev533), the fused Pallas kernel
# now outperforms epmoe by ~23% across both dp settings. This reverses
# the conclusion from the SGLang-JAX docs (commit b787fdef), which
# reported epmoe was 18-26% faster. The fused kernel has likely been
# optimized in subsequent commits.
#
# dp=2 consistently gives ~20% higher throughput than dp=1 across both
# backends, because the attention path runs at TP=4 with data
# parallelism while MoE layers still use the full EP=8.
###############################################################################

step6_benchmark() {
  echo "=== Step 6: Throughput benchmark ==="
  local POD
  POD=$($KUBECTL get pods -l job-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}')

  $KUBECTL exec "$POD" -- bash -c "
    source /opt/sglang-env/bin/activate
    export HF_HOME=/dev/shm/hf_cache
    python -m sgl_jax.bench_serving \
      --backend sgl-jax \
      --port 30271 \
      --dataset-name random \
      --num-prompts 256 \
      --random-input 16384 \
      --random-output 1024 \
      --max-concurrency 64 \
      --random-range-ratio 1 \
      --warmup-requests 0 \
      --tokenizer ${MODEL_ID}
  "
}

###############################################################################
# HELPER — View server logs
###############################################################################

show_log() {
  local POD
  POD=$($KUBECTL get pods -l job-name=${JOB_NAME} -o jsonpath='{.items[0].metadata.name}')
  $KUBECTL exec "$POD" -- tail -${1:-30} /tmp/sglang.log
}

###############################################################################
# HELPER — Clean up (delete Job, node pool auto-scales back to 0)
###############################################################################

cleanup() {
  echo "=== Cleanup: deleting Job ==="
  $KUBECTL delete job ${JOB_NAME}
  echo "Done. The node pool will auto-scale down to 0."
}

###############################################################################
# USAGE
###############################################################################

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  cat << 'USAGE'
MiMo-V2-Flash on TPU v7x-8 — Quick Start

Prerequisites:
  export GKE_PROJECT="your-gcp-project-id"
  export GKE_CLUSTER="your-gke-cluster-name"
  export GKE_REGION="us-central1"

Then source this script and run step by step:
  source quick-start.sh

  step0_create_pod     # Create GKE Pod (Spot TPU, ~3-5 min)
  step1_verify         # Verify TPU devices
  step2_install        # Install SGLang-JAX from source (~2 min)
  step3_launch         # Launch inference server (background)
  step4_wait_ready     # Wait for cold start (~15 min)
  step5_smoke_test     # Verify model output
  step6_benchmark      # Throughput benchmark (optional)
  show_log [N]         # View last N lines of server log
  cleanup              # Delete Job and release TPU
USAGE
fi
