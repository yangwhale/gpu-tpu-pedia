#!/usr/bin/env python3
"""Generate k8s134-nccl-{N}node-{D}domain[-{mode}]-sts.yaml using StatefulSet (~10x less rows).

Usage: gen-nccl-multinode-sts.py <N> <D> [<mode>] <out-file>
  N: total nodes (e.g. 18, 36)
  D: number of compute domains (1 = all same clique, 2 = split into 2 cliques)
  mode (optional, default '4coll'):
    '4coll'    — all_reduce + all_gather + reduce_scatter + alltoall (4 collective chain, has Phase 13 alltoall chain pollution issue at 144 GPU)
    '3coll'    — all_reduce + all_gather + reduce_scatter (3 collective, no alltoall, avoids chain pollution)
    'alltoall' — alltoall only (single pass vanilla, avoids chain pollution)

Phase 13 discovered: chain 4 collective at 144 GPU causes NCCL state pollution
(transport/p2p.cc Cuda failure 400) on the 4th alltoall. Split into 3coll + alltoall yaml works fine.
"""
import sys

args = sys.argv[1:]
N = int(args[0])
D = int(args[1])
if len(args) == 4:
    MODE = args[2]
    OUT = args[3]
else:
    MODE = "4coll"
    OUT = args[2]
assert D in (1, 2), "domains must be 1 or 2"
assert N % D == 0
assert MODE in ("4coll", "3coll", "alltoall"), f"mode must be 4coll/3coll/alltoall, got {MODE}"

PER_DOM = N // D
SUFFIX = "" if MODE == "4coll" else f"-{MODE}"
PREFIX = f"nccl-{N}n{SUFFIX}"

if MODE == "4coll":
    COLLS = "all_reduce all_gather reduce_scatter alltoall"
    MODE_DESC = "4 collective chain (has Phase 13 alltoall chain pollution at 144 GPU)"
elif MODE == "3coll":
    COLLS = "all_reduce all_gather reduce_scatter"
    MODE_DESC = "3 collective (no alltoall, avoids Phase 13 chain pollution)"
else:
    COLLS = "alltoall"
    MODE_DESC = "alltoall single-pass vanilla (Phase 13 100% PASS 1M-16G)"

GIB_IMAGE = "us-docker.pkg.dev/gce-ai-infra/gpudirect-gib/nccl-plugin-gib-diagnostic-arm64:v1.1.2"

def group_label(d):
    return f"{PREFIX}-g{d}"  # nccl-18n-g1, nccl-36n-g1, nccl-36n-g2

def cd_name(d):
    return f"{PREFIX}-cd-g{d}"

def channel_tpl(d):
    return f"{cd_name(d)}-channel"

# Pod hostname is "{group}-{ordinal}", e.g. nccl-18n-g1-0
# Master is g1-0 (ordinal 0 of g1), it ssh-launches mpirun across all hosts

# Build ALL_HOSTS for master (g1-0): hostname + all other peer hostnames
# Using StatefulSet headless service DNS: {pod-name}.{service-name}.{ns}.svc.cluster.local
# Within same namespace, "{pod-name}.{service-name}" 足够 resolve

def pod_dns(d, i):
    return f"{group_label(d)}-{i}.{group_label(d)}"

all_hosts = []
for d in range(1, D+1):
    for i in range(PER_DOM):
        all_hosts.append(pod_dns(d, i))
ALL_HOSTS_STR = " ".join(all_hosts)

out = []
out.append(f"""# NCCL Test — {N} node × {D} ComputeDomain (StatefulSet version, ~10x less rows)
# mode: {MODE} — {MODE_DESC}
# 1 master (g1 ordinal 0) + {N-1} worker (sshd-only). Auto-gen by gen-nccl-multinode-sts.py.
# {'1 个 ComputeDomain, 全 ' + str(N) + ' pod 同 clique (MNNVL)' if D == 1 else '2 个 ComputeDomain, g1/g2 各 ' + str(PER_DOM) + ' pod 跨 clique'}
""")

# ComputeDomains
for d in range(1, D+1):
    out.append(f"""---
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata: {{ name: {cd_name(d)} }}
spec:
  numNodes: 0
  channel:
    resourceClaimTemplate: {{ name: {channel_tpl(d)} }}""")

# Per-group: 1 headless service + 1 RDMA RCT + 1 StatefulSet
for d in range(1, D+1):
    gl = group_label(d)
    rct_rdma = f"rdma-nics-{gl}"

    # ResourceClaimTemplate (per-group RDMA)
    out.append(f"""---
apiVersion: resource.k8s.io/v1
kind: ResourceClaimTemplate
metadata: {{ name: {rct_rdma} }}
spec:
  spec:
    devices:
      requests:
      - {{ name: rdma-nics, exactly: {{ deviceClassName: rdma-devices, count: 4 }} }}""")

    # Headless service (governs StatefulSet DNS)
    out.append(f"""---
apiVersion: v1
kind: Service
metadata: {{ name: {gl} }}
spec:
  selector: {{ app: {gl} }}
  clusterIP: None
  publishNotReadyAddresses: true""")

    # Affinity:
    #   intra-group: podAffinity{app=gl, clique} → 该 group 18 pod 同 clique
    #   inter-group (g2 only): podAntiAffinity{app=g1, clique} → g2 跟 g1 不同 clique
    #   spread: podAntiAffinity{app=PREFIX, hostname} → 全 N pod 各 host
    if d == 1:
        inter_anti = ""
    else:
        # g2 跟 g1 不同 clique
        inter_anti = f"""
          - labelSelector: {{ matchExpressions: [{{ key: app, operator: In, values: [{group_label(1)}] }}] }}
            topologyKey: nvidia.com/gpu.clique"""

    # ALL_PEERS for master pod (g1-0 only): all hosts except itself
    # We'll set ALL_HOSTS for everyone; master picks self vs peers in entry script.
    master_entry = f"""        - |
          set -eu
          source /usr/local/gib/scripts/set_nccl_env.sh 2>/dev/null || true
          mkdir -p /run/sshd
          sed -i 's/^#\\?Port .*/Port 222/' /etc/ssh/sshd_config
          /usr/sbin/sshd

          ORDINAL=$(hostname | sed 's/.*-//')
          ALL_HOSTS="{ALL_HOSTS_STR}"
          SELF="$(hostname).{gl}"

          if [ "$ORDINAL" = "0" ] && [ "{gl}" = "{group_label(1)}" ]; then
            echo "=== [{PREFIX}-master] ${{SELF}} IMEX=$(ls /dev/nvidia-caps-imex-channels/ 2>&1) ==="
            # {N}-pod barrier (skip self)
            for peer in ${{ALL_HOSTS}}; do
              [ "$peer" = "$SELF" ] && continue
              for i in $(seq 1 60); do
                if ssh -p 222 -o ConnectTimeout=3 -o BatchMode=yes -o StrictHostKeyChecking=no $peer echo ok 2>/dev/null; then echo "Peer $peer ssh ok"; break; fi
                sleep 5
              done
            done
            echo "[{PREFIX}-master] all peers sshd ready, sleep 5 buffer"
            sleep 5

            for COLL in {COLLS}; do
              echo ""
              echo "=== [{PREFIX}] ${{COLL}} {N*4} GPU (1M..16G) ==="
              /usr/local/gib/scripts/run_nccl_tests.sh \\
                -t ${{COLL}} -b 1M -e 16G -f 2 -p 222 -g 4 \\
                ${{ALL_HOSTS}} 2>&1
            done

            echo "=== [{PREFIX}] DONE ==="
            sleep 3600
          else
            echo "=== [${{SELF}}] worker sshd:222 ready (ordinal=$ORDINAL, group={gl}) ==="
            sleep 3600
          fi"""

    out.append(f"""---
apiVersion: apps/v1
kind: StatefulSet
metadata: {{ name: {gl} }}
spec:
  serviceName: {gl}
  replicas: {PER_DOM}
  podManagementPolicy: Parallel
  selector:
    matchLabels: {{ app: {gl} }}
  template:
    metadata:
      labels: {{ app: {gl}, app-wide: {PREFIX} }}
    spec:
      tolerations: [{{ operator: "Exists" }}]
      nodeSelector: {{ feature.node.kubernetes.io/pci-10de.present: "true" }}
      affinity:
        podAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector: {{ matchExpressions: [{{ key: app, operator: In, values: [{gl}] }}] }}
            topologyKey: nvidia.com/gpu.clique
        podAntiAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
          - labelSelector: {{ matchExpressions: [{{ key: app-wide, operator: In, values: [{PREFIX}] }}] }}
            topologyKey: kubernetes.io/hostname{inter_anti}
      volumes:
      - {{ name: shared-memory, emptyDir: {{ medium: "Memory", sizeLimit: 250Gi }} }}
      containers:
      - image: {GIB_IMAGE}
        name: nccl-test
        resources:
          limits: {{ nvidia.com/gpu: 4 }}
          claims:
          - {{ name: rdma-nics }}
          - {{ name: compute-domain-channel }}
        securityContext: {{ privileged: true }}
        volumeMounts: [{{ name: shared-memory, mountPath: /dev/shm }}]
        env:
        - {{ name: LD_LIBRARY_PATH, value: "/usr/local/gib/lib64:/usr/local/nvidia/lib64" }}
        - {{ name: NCCL_MNNVL_ENABLE, value: "2" }}
        - {{ name: NCCL_CUMEM_ENABLE, value: "1" }}
        - {{ name: NCCL_IB_GID_INDEX, value: "3" }}
        command: ["/bin/bash", "-c"]
        args:
{master_entry}
      resourceClaims:
      - {{ name: rdma-nics, resourceClaimTemplateName: {rct_rdma} }}
      - {{ name: compute-domain-channel, resourceClaimTemplateName: {channel_tpl(d)} }}""")

with open(OUT, "w") as f:
    f.write("\n".join(out) + "\n")

print(f"Wrote {OUT}: {N} pod ({D} StatefulSet × {PER_DOM} replicas), {N*4} GPU, master={group_label(1)}-0")
