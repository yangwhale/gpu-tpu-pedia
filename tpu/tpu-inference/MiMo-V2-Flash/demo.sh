#!/usr/bin/env bash
###############################################################################
#
#  MiMo-V2-Flash Inference on TPU v7x-8 — Live Demo
#  ==================================================
#
#  Deploy Xiaomi's MiMo-V2-Flash (256-expert MoE, hybrid attention)
#  on a single TPU v7x-8 host using SGLang-JAX.
#
#  Copy-paste each block sequentially. ~20 min end to end.
#
###############################################################################


###############################################################################
# Step 1 — Create TPU pod (run on local machine)
#
# Submit a K8s Job requesting Spot TPU v7x-8. Node pool auto-scales
# from 0. Takes ~3-5 min. /dev/shm mounted as 500Gi RAM disk for
# model weights (boot disk is only 100GB, model is 70GB).
###############################################################################

# --- copy from here ---

export GKE_PROJECT="cloud-tpu-multipod-dev"
export GKE_CLUSTER="chrisya-v7x-v3"
export GKE_REGION="us-central1"
export JOB_NAME="mimo-v2-flash-demo"

# If on cc-tw (needs gLinux ADC proxy):
KUBECTL="$HOME/CloseCrab/scripts/gke-kubectl.sh"
# If on gLinux or machine with direct gcloud auth:
# KUBECTL="kubectl --context=gke_${GKE_PROJECT}_${GKE_REGION}_${GKE_CLUSTER}"

$KUBECTL delete job $JOB_NAME 2>/dev/null

cat > /tmp/tpu-demo-pod.yaml << 'EOF'
apiVersion: batch/v1
kind: Job
metadata:
  name: mimo-v2-flash-demo
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
          image: us-docker.pkg.dev/cloud-tpu-images/jax-ai-image/tpu:jax0.8.1-rev1
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

$KUBECTL apply -f /tmp/tpu-demo-pod.yaml

# --- end copy ---


###############################################################################
# Check pod status (run on local machine, repeat until Running)
###############################################################################

# --- copy from here ---
$KUBECTL get pods -l job-name=mimo-v2-flash-demo
# --- end copy ---


###############################################################################
# Step 2 — Enter the pod (run on local machine)
###############################################################################

# --- copy from here ---
POD=$($KUBECTL get pods -l job-name=mimo-v2-flash-demo -o jsonpath='{.items[0].metadata.name}')
echo "Pod: $POD"
$KUBECTL exec -it $POD -- bash
# --- end copy ---


###############################################################################
# Step 3 — Install SGLang-JAX from source (run INSIDE pod)
#
# Install from source (not PyPI) because PyPI 0.0.2 is missing
# --ep-size and --moe-backend flags needed for MoE models.
# Uses uv for fast dependency resolution. ~2 min.
###############################################################################

# --- copy from here ---
pip install uv
uv venv --python 3.12 /opt/sglang-env
cd /opt && git clone https://github.com/sgl-project/sglang-jax.git
source /opt/sglang-env/bin/activate
cd /opt/sglang-jax
uv pip install -e "python[all]"
python -m sgl_jax.launch_server --help | grep -q "ep-size" && echo "OK" || echo "FAIL"
# --- end copy ---


###############################################################################
# Step 4 — Launch inference server (run INSIDE pod)
#
# Best config: fused MoE backend, dp=2, tp=8, ep=8.
# Cold start ~8-15 min (download 70GB + XLA compile).
# Poll /health until 200.
###############################################################################

# --- copy from here ---
source /opt/sglang-env/bin/activate
export HF_HOME=/dev/shm/hf_cache
export JAX_COMPILATION_CACHE_DIR=/dev/shm/jit_cache
rm -f /tmp/sglang.log /tmp/libtpu_lockfile

python -u -m sgl_jax.launch_server \
    --model-path XiaomiMiMo/MiMo-V2-Flash \
    --trust-remote-code \
    --tp-size 8 --dp-size 2 --ep-size 8 \
    --moe-backend fused \
    --host 0.0.0.0 --port 30271 \
    --page-size 256 --context-length 262144 \
    --chunked-prefill-size 4096 \
    --dtype bfloat16 --mem-fraction-static 0.95 \
    --swa-full-tokens-ratio 0.25 --skip-server-warmup \
    --max-running-requests 128 \
    --attention-backend fa \
    >> /tmp/sglang.log 2>&1 &

echo "Launched PID=$!. Tail log: tail -f /tmp/sglang.log"
# --- end copy ---

# Wait for server ready (run INSIDE pod, separate terminal or after bg):
# --- copy from here ---
while true; do
  CODE=$(curl -sf -o /dev/null -w "%{http_code}" http://localhost:30271/health 2>/dev/null || echo "000")
  echo "$(date +%H:%M:%S) health=$CODE"
  [ "$CODE" = "200" ] && echo "SERVER READY" && break
  sleep 10
done
# --- end copy ---


###############################################################################
# Step 5 — Smoke test (run INSIDE pod)
#
# Math question + KV cache consistency check (3 identical requests).
###############################################################################

# --- copy from here ---
curl -s http://localhost:30271/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"MiMo-V2-Flash","messages":[{"role":"user","content":"What is 2+3? Answer with just the number."}],"max_tokens":50,"temperature":0}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('Answer:', r['choices'][0]['message']['content'])"

for i in 1 2 3; do
  echo -n "#$i: "
  curl -s http://localhost:30271/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"MiMo-V2-Flash","messages":[{"role":"user","content":"What is the capital of France? One word."}],"max_tokens":20,"temperature":0}' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
done
# --- end copy ---


###############################################################################
# Step 6 — Throughput benchmark (run INSIDE pod)
#
# 256 prompts × 16K input × 1K output, concurrency 64. ~7 min.
# Expect: ~780 tok/s output throughput, ~53ms TPOT, ~26ms ITL.
###############################################################################

# --- copy from here ---
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
  --tokenizer XiaomiMiMo/MiMo-V2-Flash
# --- end copy ---


###############################################################################
# Cleanup (run on local machine)
###############################################################################

# --- copy from here ---
$KUBECTL delete job mimo-v2-flash-demo
# Node pool auto-scales back to 0.
# --- end copy ---
