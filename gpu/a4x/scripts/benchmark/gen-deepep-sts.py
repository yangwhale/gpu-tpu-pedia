#!/usr/bin/env python3
"""Generate k8s134-deepep-{N}node-{D}domain-sts.yaml — DeepEP test, master 自驱动 (无 kubectl exec).

Usage: gen-deepep-sts.py <N> <D> <out>
  N: 总 pod 数 (1, 2, 4, 18, 36)
  D: ComputeDomain 数 (1 = same clique, 2 = cross clique, only for N>=2)

每 yaml shape (跟 NCCL gen-nccl-multinode-sts.py 一致):
- StatefulSet × D (D=1 → 1 STS replicas=N, D=2 → 2 STS replicas=N/2)
- ComputeDomain × D (per-clique IMEX channel)
- ResourceClaimTemplate × D (4 RDMA NIC per pod)
- headless Service × D
- master = g1-0 (ordinal 0 of g1 group), 用 sshd:222 barrier 等其他 pod ready, 然后顺序跑 tests

测试 set (per N):
- N==1: intranode + low_latency + test_ep (3 tests, all WORLD_SIZE=1 内部 4 process)
- N>=2: test_internode + test_ep (2 tests, WORLD_SIZE=N nproc=4)

DeepEP env (from Phase 9 + dave run-deepep-2node.sh):
- LD_PRELOAD pip libnccl.so.2 (DeepEP filecmp check_nccl_so)
- LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:/usr/local/gib/lib64:...
  (nvshmem 3.7 优先 over NGC system 3.6.5, gib for NCCL_NET=gIB)
- EP_NCCL_ROOT_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nccl
- EP_JIT_CACHE_DIR=/tmp/deepep-jit-cache
- NCCL_NET=gIB + PXN_C2C=1 + IB_ADAPTIVE_ROUTING=1 + IB_QPS_PER_CONNECTION=4 + IB_TC=52 + IB_FIFO_TC=84
- NCCL_NVLS_ENABLE=0 + NCCL_MNNVL_ENABLE=2 (D=1 intra-clique)/0 (D=2 cross-clique)
- NCCL_IB_MERGE_NICS=0 (DeepEP issue #628: rail-aware GIN, 不要让 gIB 合并多 NIC)
- EP_BUFFER_DEBUG=1 + NCCL_DEBUG=INFO + NCCL_DEBUG_SUBSYS=INIT,NET,ENV (debug 期间必开)
- TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
- cpuset 16-139 via taskset (避 gve IRQ on CPU 0-15)
"""
import sys

args = sys.argv[1:]
if len(args) != 3:
    print(__doc__)
    sys.exit(1)

N = int(args[0])
D = int(args[1])
OUT = args[2]
assert D in (1, 2), "domains must be 1 or 2"
if N == 1:
    assert D == 1, "1 node only single domain"
else:
    assert N % D == 0, f"N={N} must be divisible by D={D}"

PER_DOM = N // D
PREFIX = f"deepep-{N}n"
GPUS_PER_NODE = 4
WORLD_SIZE = N
IMAGE = "us-east1-docker.pkg.dev/gpu-launchpad-playground/forrest-repo-us-east1/megatron-ngc:tev2.15-mgcore_r0.16.0-pt26.05-py3-v2"

# tests:
# legacy/test_internode.py 需要 num_ranks > 8 (3+ nodes), 2n 跑不了
if N == 1:
    TESTS = [
        ("intranode",   "cd /opt/DeepEP && taskset -c 16-139 python3 tests/legacy/test_intranode.py --num-processes 4 --allow-mnnvl"),
        ("low_latency", "cd /opt/DeepEP && taskset -c 16-139 python3 tests/legacy/test_low_latency.py --num-processes 4 --allow-mnnvl"),
        ("test_ep",     "cd /opt/DeepEP && taskset -c 16-139 python3 tests/elastic/test_ep.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 6 --num-experts 256 --num-sms 64"),
    ]
elif N == 2:
    TESTS = [
        ("test_ep",     "cd /opt/DeepEP && taskset -c 16-139 python3 tests/elastic/test_ep.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 6 --num-experts 256 --num-sms 64"),
    ]
else:
    TESTS = [
        ("test_internode", f"cd /opt/DeepEP && taskset -c 16-139 python3 tests/legacy/test_internode.py --num-processes 4"),
        ("test_ep",        f"cd /opt/DeepEP && taskset -c 16-139 python3 tests/elastic/test_ep.py --num-processes 4 --num-tokens 1024 --hidden 7168 --num-topk 6 --num-experts 256 --num-sms 64"),
    ]

# hostNetwork = True for N>=2: DRANET path raw ibverbs fail (Phase 9 memory), 必须 hostNetwork 用 host bond2-5 RDMA
HOST_NETWORK = (N >= 2)

# NCCL MNNVL: 2026-06-21 实测修正前述误判
# D=1 (同 clique): MNNVL=2 production 推荐 — combine SU 660 GB/s 跟 dave task1 §13.7 一致
#                  (NCCL 把 cross-host 透明转 NVLink-C2C, SO 显示 0 但 SU 真实)
# D=2 (跨 clique): MNNVL=0 必须 (跨 clique 没 NVLink fabric)
# dave task2 §10.5 用 MNNVL=0 是 RDMA-only baseline, 不是 production 路径
MNNVL = "2" if D == 1 else "0"

def gl(d):
    return f"{PREFIX}-g{d}"

def cd_name(d):
    return f"{PREFIX}-cd-g{d}"

def channel_tpl(d):
    return f"{cd_name(d)}-channel"

def pod_dns(d, i):
    return f"{gl(d)}-{i}.{gl(d)}"

all_hosts = []
for d in range(1, D+1):
    for i in range(PER_DOM):
        all_hosts.append(pod_dns(d, i))
ALL_HOSTS_STR = " ".join(all_hosts)
MASTER_DNS = pod_dns(1, 0)

# build test runner block — master only (ordinal 0 of g1)
test_runs = []
for test_name, cmd in TESTS:
    if N == 1:
        # single pod tests: 直接 local
        test_runs.append(f'''
            echo "=== [{PREFIX}] {test_name} ==="
            {cmd} 2>&1''')
    else:
        # multi-pod tests: torchrun across N pods
        test_runs.append(f'''
            echo "=== [{PREFIX}] {test_name} on {N} pods x {GPUS_PER_NODE} GPU ==="
            # launch test on all peers in parallel (1 process per pod, internal --num-processes 4)
            for peer in ${{ALL_HOSTS}}; do
              RANK_I=$(echo $ALL_HOSTS_ARRAY | tr ' ' '\\n' | grep -n "^${{peer}}$" | cut -d: -f1)
              RANK_I=$((RANK_I - 1))
              if [ "$peer" = "$SELF" ]; then continue; fi
              ssh -p 222 -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no $peer "
                export MASTER_ADDR={MASTER_DNS}
                export MASTER_PORT=8411
                export WORLD_SIZE={WORLD_SIZE}
                export RANK=$RANK_I
                export LOCAL_RANK=0
                cd /opt/DeepEP
                {cmd.replace('cd /opt/DeepEP && ', '')} > /tmp/{test_name}-rank-$RANK_I.log 2>&1 &
                echo \\$!
              " &
            done
            # master = rank 0
            export MASTER_ADDR={MASTER_DNS}
            export MASTER_PORT=8411
            export WORLD_SIZE={WORLD_SIZE}
            export RANK=0
            export LOCAL_RANK=0
            {cmd} 2>&1 | tee /tmp/{test_name}-rank-0.log
            wait
            echo "=== [{PREFIX}] {test_name} done, collecting peer logs ==="
            for peer in ${{ALL_HOSTS}}; do
              if [ "$peer" = "$SELF" ]; then continue; fi
              RANK_I=$(echo $ALL_HOSTS_ARRAY | tr ' ' '\\n' | grep -n "^${{peer}}$" | cut -d: -f1)
              RANK_I=$((RANK_I - 1))
              echo "--- rank $RANK_I @ $peer ---"
              ssh -p 222 -o ConnectTimeout=5 -o BatchMode=yes -o StrictHostKeyChecking=no $peer "cat /tmp/{test_name}-rank-$RANK_I.log 2>&1"
            done''')

ALL_TESTS = "\n".join(test_runs)

out = []
out.append(f"""# DeepEP Test — {N} node × {D} ComputeDomain (StatefulSet, master 自驱动 sshd)
# Auto-gen by gen-deepep-sts.py. Master = g1-0 ordinal 0.
# Tests: {', '.join(t for t, _ in TESTS)}
# Image: {IMAGE} (DeepEP 2.0.0 + nvshmem 3.7 + nccl 2.30.4 baked, no pip install)
""")

# ComputeDomain(s)
for d in range(1, D+1):
    out.append(f"""---
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata: {{ name: {cd_name(d)} }}
spec:
  numNodes: 0
  channel:
    resourceClaimTemplate: {{ name: {channel_tpl(d)} }}""")

# per-group RCT + Service + StatefulSet
for d in range(1, D+1):
    rct = f"rdma-nics-{gl(d)}"

    out.append(f"""---
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata: {{ name: {rct} }}
spec:
  spec:
    devices:
      requests:
      - {{ name: rdma-nics, exactly: {{ deviceClassName: rdma-devices, count: 4 }} }}""")

    out.append(f"""---
apiVersion: v1
kind: Service
metadata: {{ name: {gl(d)} }}
spec:
  selector: {{ app: {gl(d)} }}
  clusterIP: None
  publishNotReadyAddresses: true""")

    inter_anti = ""
    if d == 2:
        inter_anti = f"""
          - labelSelector: {{ matchExpressions: [{{ key: app, operator: In, values: [{gl(1)}] }}] }}
            topologyKey: nvidia.com/gpu.clique"""

    if N == 1:
        # N=1 single pod: master 自驱动跑 test (没 cross-pod 协调)
        master_entry = f"""        - |
          set -eu
          source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null || true
          mkdir -p /run/sshd
          chmod 700 /root/.ssh
          chmod 600 /root/.ssh/* 2>/dev/null || true
          chmod 644 /root/.ssh/*.pub 2>/dev/null || true
          /usr/sbin/sshd
          ORDINAL=$(echo ${{POD_NAME:-$(hostname)}} | sed 's/.*-//')
          SELF="${{POD_NAME:-$(hostname)}}.{gl(d)}"
          echo "=== [{PREFIX}-master] ${{SELF}} IMEX=$(ls /dev/nvidia-caps-imex-channels/ 2>&1) ==="
          set +e
{ALL_TESTS}
          echo "=== [{PREFIX}] DONE ==="
          sleep 7200"""
    else:
        # N>=2 dave-style: 所有 pod sleep infinity + sshd 起着供 kubectl exec connect
        # launch 通过外部 script `run-deepep.sh` 跑 (从 master node kubectl exec parallel)
        # 见 dave_doc_v2/scripts/run-deepep-2node.sh + run-deepep-128gpu.sh 验证模式
        master_entry = f"""        - |
          set -eu
          source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null || true
          mkdir -p /run/sshd
          chmod 700 /root/.ssh
          chmod 600 /root/.ssh/* 2>/dev/null || true
          chmod 644 /root/.ssh/*.pub 2>/dev/null || true
          /usr/sbin/sshd
          ORDINAL=$(echo ${{POD_NAME:-$(hostname)}} | sed 's/.*-//')
          SELF="${{POD_NAME:-$(hostname)}}.{gl(d)}"
          echo "=== [{PREFIX}] ${{SELF}} ready (ordinal=$ORDINAL, group={gl(d)}, dave-style) IMEX=$(ls /dev/nvidia-caps-imex-channels/ 2>&1) ==="
          echo "Launch test via: bash scripts/k8s134/run-deepep.sh {N} {D}"
          sleep infinity"""

    host_net_line = "      hostNetwork: true\n      dnsPolicy: ClusterFirstWithHostNet\n" if HOST_NETWORK else ""
    rdma_resource_claim = "" if HOST_NETWORK else f"""          - {{ name: rdma-nics }}
"""
    rdma_resource_claim_decl = "" if HOST_NETWORK else f"""      - {{ name: rdma-nics, resourceClaimTemplateName: {rct} }}
"""

    out.append(f"""---
apiVersion: apps/v1
kind: StatefulSet
metadata: {{ name: {gl(d)} }}
spec:
  serviceName: {gl(d)}
  replicas: {PER_DOM}
  podManagementPolicy: Parallel
  selector:
    matchLabels: {{ app: {gl(d)} }}
  template:
    metadata:
      labels: {{ app: {gl(d)}, app-wide: {PREFIX} }}
    spec:
      tolerations: [{{ operator: "Exists" }}]
      nodeSelector: {{ feature.node.kubernetes.io/pci-10de.present: "true" }}
      imagePullSecrets: [{{ name: ar-secret }}]
{host_net_line}      affinity:
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector: {{ matchExpressions: [{{ key: app, operator: In, values: [{gl(d)}] }}] }}
            topologyKey: nvidia.com/gpu.clique
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector: {{ matchExpressions: [{{ key: app-wide, operator: In, values: [{PREFIX}] }}] }}
            topologyKey: kubernetes.io/hostname{inter_anti}
      volumes:
      - {{ name: shared-memory, emptyDir: {{ medium: "Memory", sizeLimit: 250Gi }} }}
      - {{ name: gib, emptyDir: {{}} }}
      - {{ name: ssh-keys, emptyDir: {{}} }}
      initContainers:
      - name: gib-installer
        image: us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:v1.1.2
        command: ["/bin/sh", "-c"]
        args:
        - |
          set -e
          /scripts/container_entry.sh install --install-nccl
          cp -a /usr/local/gib/. /target/gib/
          # 删 GIB 自带 v58 libibverbs/libmlx5/librdmacm/libibumad (跟 NGC v59 ABI 冲突, memory: gib_libibverbs_abi_mismatch.md)
          mkdir -p /target/gib/lib64/_disabled
          cd /target/gib/lib64 && mv libibverbs.so* librdmacm.so* libmlx5.so* libibumad.so* _disabled/ 2>/dev/null || true
          # cp GIB pre-baked SSH keys (shared across all pods for cross-pod ssh barrier + mpirun)
          cp -a /root/.ssh/. /target/ssh/
        volumeMounts:
        - {{ name: gib, mountPath: /target/gib }}
        - {{ name: ssh-keys, mountPath: /target/ssh }}
      containers:
      - image: {IMAGE}
        name: deepep
        imagePullPolicy: IfNotPresent
        resources:
          limits: {{ nvidia.com/gpu: 4 }}
          claims:
{rdma_resource_claim}          - {{ name: compute-domain-channel }}
        securityContext: {{ privileged: true }}
        volumeMounts:
        - {{ name: shared-memory, mountPath: /dev/shm }}
        - {{ name: gib, mountPath: /usr/local/gib }}
        - {{ name: ssh-keys, mountPath: /root/.ssh }}
        env:
        - {{ name: LD_PRELOAD, value: "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2" }}
        - {{ name: LD_LIBRARY_PATH, value: "/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem/lib:/usr/local/gib/lib64:/usr/local/nvidia/lib64" }}
        - {{ name: EP_NCCL_ROOT_DIR, value: "/usr/local/lib/python3.12/dist-packages/nvidia/nccl" }}
        - {{ name: EP_JIT_CACHE_DIR, value: "/tmp/deepep-jit-cache" }}
        - {{ name: NCCL_NET, value: "gIB" }}
        - {{ name: NCCL_PXN_C2C, value: "1" }}
        - {{ name: NCCL_IB_ADAPTIVE_ROUTING, value: "1" }}
        - {{ name: NCCL_IB_QPS_PER_CONNECTION, value: "4" }}
        - {{ name: NCCL_IB_TC, value: "52" }}
        - {{ name: NCCL_IB_FIFO_TC, value: "84" }}
        - {{ name: NCCL_NVLS_ENABLE, value: "0" }}
        - {{ name: NCCL_MNNVL_ENABLE, value: "{MNNVL}" }}
        - {{ name: NCCL_IB_MERGE_NICS, value: "0" }}
        - {{ name: NCCL_CUMEM_ENABLE, value: "1" }}
        - {{ name: NCCL_IB_GID_INDEX, value: "3" }}
        - {{ name: EP_BUFFER_DEBUG, value: "1" }}
        - {{ name: NCCL_DEBUG, value: "INFO" }}
        - {{ name: NCCL_DEBUG_SUBSYS, value: "INIT,NET,ENV" }}
        - {{ name: TRITON_PTXAS_PATH, value: "/usr/local/cuda/bin/ptxas" }}
        - {{ name: NVSHMEM_BOOTSTRAP, value: "MPI" }}
        - {{ name: NVSHMEM_BOOTSTRAP_MPI_PLUGIN, value: "nvshmem_bootstrap_torch.so" }}
        - {{ name: POD_NAME, valueFrom: {{ fieldRef: {{ fieldPath: metadata.name }} }} }}
        command: ["/bin/bash", "-c"]
        args:
{master_entry}
      resourceClaims:
{rdma_resource_claim_decl}      - {{ name: compute-domain-channel, resourceClaimTemplateName: {channel_tpl(d)} }}""")

with open(OUT, "w") as f:
    f.write("\n".join(out) + "\n")

print(f"Wrote {OUT}: {N} pod ({D} STS × {PER_DOM} replicas), {N*4} GPU, master={gl(1)}-0, tests=[{', '.join(t for t, _ in TESTS)}]")
