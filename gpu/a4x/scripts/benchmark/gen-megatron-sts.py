#!/usr/bin/env python3
"""Generate k8s134-megatron-{model}-{N}n-{tp}t{pp}p-sts.yaml.

Self-driven StatefulSet — each pod runs torchrun --node_rank=$ORDINAL directly,
no sshd / mpirun barrier. master_addr = headless service DNS of pod-0.

Image us-east1-docker.pkg.dev/.../megatron-ngc:tev2.15-mgcore_r0.16.0-pt26.05-py3-v2
  - pre-installed /opt/Megatron-LM (core_r0.16.0)
  - NGC pytorch:26.05-py3 base, TE 2.15, CUDA 13.2
  - GIB v1.1.2 injected via init container (NCCL plugin)

hostPath /var/megatron-cache → node-local NVMe (root partition, ~140 GB free).
  Note: lssd nvme0n1 (3.2TB) not used — device numbering inconsistent across worker
        + TencentOS 4 minimal lacks mkfs.ext4; root nvme suffices for JIT cache.

Mock data + NullTokenizer (no real dataset / tokenizer file needed).

Usage:
  gen-megatron-sts.py <model> <num_gpus> <mbs> <gbs> <tp> <pp> <out-file>

  model: llama2-7b | llama2-13b | llama3-70b
  num_gpus: total GPUs (must be multiple of 4 — GB200 has 4 GPU/node)
  mbs: micro-batch size per GPU
  gbs: global batch size
  tp: tensor parallel
  pp: pipeline parallel
  out-file: yaml output path
"""
import sys

MODELS = {
    "llama2-7b": dict(
        layers=32, hidden=4096, ffn=11008, heads=32,
        gqa=False, rope_base=10000, vocab=32000,
    ),
    "llama2-13b": dict(
        layers=40, hidden=5120, ffn=13824, heads=40,
        gqa=False, rope_base=10000, vocab=32000,
    ),
    "llama3-70b": dict(
        layers=80, hidden=8192, ffn=28672, heads=64,
        gqa=True, query_groups=8, kv_channels=128,
        rope_base=500000, vocab=128256,
    ),
}

IMAGE = "us-east1-docker.pkg.dev/gpu-launchpad-playground/forrest-repo-us-east1/megatron-ngc:tev2.15-mgcore_r0.16.0-pt26.05-py3-v2"
GIB_IMAGE = "us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:v1.1.2"
SEQ_LEN = 4096
TRAIN_ITERS = 50          # 50 iter ≈ 30-60s steady-state for benchmark
LOG_INTERVAL = 1


def model_args(model, mbs, gbs, tp, pp, seq=SEQ_LEN):
    m = MODELS[model]
    a = [
        # --- Model architecture
        "--use-mcore-models",
        "--transformer-impl", "transformer_engine",
        f"--num-layers", str(m["layers"]),
        f"--hidden-size", str(m["hidden"]),
        f"--ffn-hidden-size", str(m["ffn"]),
        f"--num-attention-heads", str(m["heads"]),
        f"--seq-length", str(seq),
        f"--max-position-embeddings", str(seq),
        "--position-embedding-type", "rope",
        f"--rotary-base", str(m["rope_base"]),
        "--rotary-percent", "1.0",
        "--normalization", "RMSNorm",
        "--swiglu",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--attention-backend", "fused",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--init-method-std", "0.02",
        "--no-masked-softmax-fusion",
    ]
    if m.get("gqa"):
        a += [
            "--group-query-attention",
            "--num-query-groups", str(m["query_groups"]),
            "--kv-channels", str(m["kv_channels"]),
        ]
    # --- dtype
    a += ["--bf16"]
    # --- Data + tokenizer (mock)
    a += [
        "--tokenizer-type", "NullTokenizer",
        "--vocab-size", str(m["vocab"]),
        "--mock-data",
    ]
    # --- Training loop
    a += [
        "--micro-batch-size", str(mbs),
        "--global-batch-size", str(gbs),
        f"--train-iters", str(TRAIN_ITERS),
        "--lr", "0.00015",
        "--min-lr", "0.00001",
        "--lr-decay-style", "cosine",
        "--lr-warmup-iters", "2",
        "--clip-grad", "1.0",
        "--weight-decay", "0.1",
        "--adam-beta1", "0.9",
        "--adam-beta2", "0.95",
        f"--log-interval", str(LOG_INTERVAL),
        # NB: --log-throughput is megatron-core 0.16 buggy on multi-node + mock-data → silent skip iteration log entirely.
        # We don't enable it; instead compute TFLOPs/GPU manually from elapsed_ms via 6N formula in extract-megatron-stats.py.
        "--save-interval", "100000",
        "--eval-interval", "1000",  # megatron-core 0.16 quirk: even with --eval-iters 0, eval_interval default=None crashes get_train_valid_test_num_samples (int // None). Set to a large value.
        "--eval-iters", "0",
        "--no-load-optim",
        "--no-load-rng",
    ]
    # --- Parallel
    a += [
        "--tensor-model-parallel-size", str(tp),
        "--pipeline-model-parallel-size", str(pp),
    ]
    if tp > 1:
        a += ["--sequence-parallel"]
    # --- perf
    a += [
        "--cross-entropy-loss-fusion",
        "--manual-gc",
        "--use-distributed-optimizer",
        "--overlap-grad-reduce",
        "--overlap-param-gather",
    ]
    return a


def main():
    if len(sys.argv) != 8:
        print(__doc__)
        sys.exit(1)
    model = sys.argv[1]
    num_gpus = int(sys.argv[2])
    mbs = int(sys.argv[3])
    gbs = int(sys.argv[4])
    tp = int(sys.argv[5])
    pp = int(sys.argv[6])
    out = sys.argv[7]

    assert model in MODELS, f"model must be one of {list(MODELS)}"
    assert num_gpus % 4 == 0, "num_gpus must be multiple of 4 (GB200 4 GPU/node)"
    num_nodes = num_gpus // 4
    dp = num_gpus // (tp * pp)
    assert tp * pp * dp == num_gpus, "tp * pp must divide num_gpus"
    assert gbs % (mbs * dp) == 0, f"gbs={gbs} must be divisible by mbs*dp={mbs*dp}"

    # Sanitize model name for k8s resource names (no dots, lowercase only)
    model_label = model.replace(".", "-")
    prefix = f"meg-{model_label}-{num_gpus}g"

    cd_name = f"{prefix}-cd"
    channel_tpl = f"{cd_name}-channel"
    group = f"{prefix}-g1"
    rdma_rct = f"rdma-nics-{group}"

    margs = model_args(model, mbs, gbs, tp, pp)
    margs_str = " \\\n            ".join(margs)

    master_dns = f"{group}-0.{group}"

    # Build pod entry script
    pod_entry = f"""        - |
          set -eu
          echo "=== [{prefix}] pod $(hostname) starting at $(date) ==="

          # Wait for GIB libs to be installed by init container
          ls /usr/local/gib/lib64/libnccl-net.so > /dev/null 2>&1 || {{ echo "GIB libs missing"; exit 1; }}

          # NCCL env (gIB + MNNVL)
          export LD_LIBRARY_PATH=/usr/local/gib/lib64:/usr/local/nvidia/lib64:${{LD_LIBRARY_PATH:-}}
          source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null || true
          export NCCL_NET=gIB
          export NCCL_MNNVL_ENABLE=2
          export NCCL_CUMEM_ENABLE=1
          export NCCL_IB_GID_INDEX=3
          export NCCL_IB_QPS_PER_CONNECTION=4
          export NCCL_IB_TC=52
          export NCCL_IB_FIFO_TC=84
          export NCCL_IB_ADAPTIVE_ROUTING=1
          export NCCL_PXN_C2C=1
          export NCCL_NVLS_ENABLE=0
          export CUDA_DEVICE_MAX_CONNECTIONS=1
          # multi-node megatron rank-0 stdout buffered → iter log invisible until exit. force flush.
          export PYTHONUNBUFFERED=1

          ORDINAL=$(hostname | sed 's/.*-//')
          MASTER_ADDR="{master_dns}"
          MASTER_PORT=6000
          NNODES={num_nodes}

          echo "=== ORDINAL=$ORDINAL NNODES=$NNODES MASTER=$MASTER_ADDR ==="
          echo "IMEX: $(ls /dev/nvidia-caps-imex-channels/ 2>&1)"
          echo "GPUs: $(nvidia-smi -L 2>&1)"
          echo "Hostname: $(hostname); Pod IP: $(hostname -i)"

          mkdir -p /var/megatron-cache /tmp/torch-extensions
          export TORCH_EXTENSIONS_DIR=/tmp/torch-extensions

          # Fix DeepEP NCCL ABI mismatch:
          # GIB-injected libnccl.so.2.30.4 != pip nvidia-nccl-cu13 file content (filecmp strict).
          # megatron-core auto-imports deep_ep → hard AssertionError. We don't use deepep for llama.
          # Two-pronged: LD_PRELOAD pip libnccl (so DeepEP filecmp passes if it does try import), AND
          # remove deep_ep so megatron skips it cleanly.
          export LD_PRELOAD=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2:${{LD_PRELOAD:-}}
          rm -rf /opt/DeepEP /usr/local/lib/python3.12/dist-packages/deep_ep* 2>/dev/null || true

          cd /opt/Megatron-LM
          # Patch pip constraint (NGC base is fine but ensure)
          [ -f /etc/pip/constraint.txt ] && sed -i 's/setuptools==78\\.1\\.0/setuptools>=78.1.0/' /etc/pip/constraint.txt 2>/dev/null || true

          # Wait for master DNS resolvable (rank > 0 only)
          if [ "$ORDINAL" != "0" ]; then
            for i in $(seq 1 60); do
              if getent hosts $MASTER_ADDR > /dev/null 2>&1; then
                echo "master $MASTER_ADDR resolved"; break
              fi
              echo "waiting master DNS ($i/60)"; sleep 5
            done
          fi

          echo "=== Starting torchrun ==="
          exec torchrun \\
            --nproc_per_node=4 \\
            --nnodes=$NNODES \\
            --node_rank=$ORDINAL \\
            --master_addr=$MASTER_ADDR \\
            --master_port=$MASTER_PORT \\
            /opt/Megatron-LM/pretrain_gpt.py \\
            {margs_str}
"""

    yaml = f"""# Megatron-LM training — {model} ({num_gpus} GPU, {num_nodes} node, tp={tp} pp={pp} dp={dp})
# mbs={mbs} gbs={gbs} seq={SEQ_LEN} bf16 mock-data
# Auto-gen by scripts/k8s134/gen-megatron-sts.py
# Self-driven: each pod runs torchrun directly (no sshd/mpirun barrier)
---
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: {cd_name}
spec:
  numNodes: 0
  channel:
    resourceClaimTemplate:
      name: {channel_tpl}
---
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata:
  name: {rdma_rct}
spec:
  spec:
    devices:
      requests:
      - name: rdma-nics
        exactly:
          deviceClassName: rdma-devices
          count: 4
---
apiVersion: v1
kind: Service
metadata:
  name: {group}
spec:
  selector:
    app: {group}
  clusterIP: None
  publishNotReadyAddresses: true
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: {group}
spec:
  serviceName: {group}
  replicas: {num_nodes}
  podManagementPolicy: Parallel
  selector:
    matchLabels:
      app: {group}
  template:
    metadata:
      labels:
        app: {group}
    spec:
      tolerations:
      - operator: "Exists"
      nodeSelector:
        nvidia.com/gpu.family: blackwell
      affinity:
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - {{ key: app, operator: In, values: [{group}] }}
            topologyKey: nvidia.com/gpu.clique
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector:
              matchExpressions:
              - {{ key: app, operator: In, values: [{group}] }}
            topologyKey: kubernetes.io/hostname
      volumes:
      - name: shared-memory
        emptyDir:
          medium: "Memory"
          sizeLimit: 250Gi
      - name: gib
        emptyDir: {{}}
      - name: cache
        hostPath:
          path: /var/megatron-cache
          type: DirectoryOrCreate
      initContainers:
      - name: gib-installer
        image: {GIB_IMAGE}
        command: ["/bin/sh", "-c"]
        args:
        - |
          set -e
          /scripts/container_entry.sh install --install-nccl
          cp -a /usr/local/gib/. /target/gib/
          cp -a /usr/lib/aarch64-linux-gnu/libibverbs.so* /target/gib/lib64/
          cp -a /usr/lib/aarch64-linux-gnu/libmlx5.so* /target/gib/lib64/
          cp -a /usr/lib/aarch64-linux-gnu/librdmacm.so* /target/gib/lib64/
          cp -a /usr/lib/aarch64-linux-gnu/libibumad.so* /target/gib/lib64/ 2>/dev/null || true
          mkdir -p /target/gib/lib64/libibverbs
          cp -a /usr/lib/aarch64-linux-gnu/libibverbs/libmlx5-rdmav34.so /target/gib/lib64/libibverbs/ 2>/dev/null || true
          echo "GIB v1.1.2 + rdma-core installed"
        volumeMounts:
        - name: gib
          mountPath: /target/gib
      containers:
      - image: {IMAGE}
        name: pytorch
        resources:
          limits:
            nvidia.com/gpu: 4
          claims:
          - name: rdma-nics
          - name: compute-domain-channel
        securityContext:
          privileged: true
        volumeMounts:
        - name: shared-memory
          mountPath: /dev/shm
        - name: gib
          mountPath: /usr/local/gib
        - name: cache
          mountPath: /var/megatron-cache
        command: ["/bin/bash", "-c"]
        args:
{pod_entry}
      resourceClaims:
      - name: rdma-nics
        resourceClaimTemplateName: {rdma_rct}
      - name: compute-domain-channel
        resourceClaimTemplateName: {channel_tpl}
"""
    with open(out, "w") as f:
        f.write(yaml)
    print(f"Wrote {out}: {model} {num_gpus}GPU ({num_nodes}n) tp{tp}pp{pp}dp{dp} mbs={mbs} gbs={gbs} seq={SEQ_LEN}")


if __name__ == "__main__":
    main()
