# A4 GPU Testbed - DeepEP on GKE

Deploy and test [DeepEP](https://github.com/deepseek-ai/DeepEP) (DeepSeek Expert Parallelism) on [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine) with A4 (B200) GPUs.

## What's Included

- **Docker image** (`nemo-deepep:25.11`): NeMo 25.11 + NVSHMEM v3.5.19-1 (IBGDA) + DeepEP with GPU-NIC mapping
- **Cloud Build config**: One-command image build via `gcloud builds submit`
- **Helm chart**: JobSet-based multi-node deployment with topology-aware scheduling
- **Launcher script**: Automated SSH setup, node discovery, and physical topology sorting

## Tech Stack

- **Base image**: `nvcr.io/nvidia/nemo:25.11`
- **NVSHMEM**: v3.5.19-1 with IBGDA (GPU-initiated data access for RDMA)
- **DeepEP**: PR #466 (GPU-NIC explicit mapping), commit `8a07e7e`
- **GPU arch**: SM 10.0 (B200)
- **Orchestration**: GKE + Kubernetes JobSet + Helm

## Prerequisites

- GKE cluster with A4 node pool (DENSE deployment type)
- Artifact Registry for Docker images
- GCS bucket for logs
- Kueue + JobSet APIs installed
- Kueue ClusterQueue `nominalQuota` must cover total GPU count (e.g. 16 for 2 nodes)
- If ResourceFlavor has `topologyName` set, all nodes must be in the same topology block — or remove the topology constraint

## Quick Start

### 1. Build the Image

```bash
cd gpu-tpu-pedia/gpu/testbed-a4/docker

export ARTIFACT_REGISTRY=<your-registry-url>
# e.g. asia-docker.pkg.dev/your-project/your-repo

gcloud builds submit \
    --config cloudbuild.yml \
    --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY \
    --timeout "2h" \
    --machine-type=e2-highcpu-32 \
    --quiet --async
```

The image will be tagged as `${ARTIFACT_REGISTRY}/nemo-deepep:25.11`.

### 2. Deploy to GKE

```bash
export CLUSTER_NAME=<cluster-name>
export REGION=<region>
export GCS_BUCKET=<bucket-name>
export WORKLOAD_IMAGE=${ARTIFACT_REGISTRY}/nemo-deepep:25.11

gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION

cd gpu-tpu-pedia/gpu/testbed-a4
helm install -f gke-runtime/values.yaml \
    --set-file workload_launcher=gke-runtime/launchers/torchrun-stratup.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "workload.gpus"=16 \
    --set "queue=a4" \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    --set "volumes.nfs.ip"=<filestore-ip> \
    --set "volumes.nfs.region"=<filestore-zone> \
    --set "volumes.nfs.instance"=<filestore-instance> \
    --set "volumes.nfs.volume"=<filestore-share> \
    --set "volumes.nfs.storage"=1024Gi \
    $USER-deepep \
    gke-runtime/jobset
```

Key parameters:
- `workload.gpus`: Total GPU count (must be multiple of 8, matching available nodes)
- `queue`: Kueue LocalQueue name (must exist and have sufficient quota)
- `volumes.nfs.*`: Filestore instance details (get via `gcloud filestore instances list`)

### 3. Run DeepEP Tests

**Option A: Automated** — set `RUN_DEEPEP_TEST=true` in values.yaml (or `--set`) before deploying. The launcher script runs the test automatically via MPI.

**Option B: Manual** — exec into the coordinator pod:

```bash
COORD_POD=$(kubectl get pods -l jobset.sigs.k8s.io/jobset-name=$USER-deepep \
    --sort-by=.metadata.name -o jsonpath='{.items[0].metadata.name}')

kubectl exec -it $COORD_POD -c workload -- bash
```

Inside the pod:

```bash
source /opt/deepep/unified-env.sh
source /usr/local/gib/scripts/set_nccl_env.sh

# Create per-node hostfile (1 process per node — test_internode.py spawns 8 GPU procs internally)
sed 's/slots=8/slots=1/g' /etc/job-worker-services.txt > /tmp/hostfile-1pernode.txt

mpirun --allow-run-as-root \
  --hostfile /tmp/hostfile-1pernode.txt \
  --mca orte_keep_fqdn_hostnames 1 \
  --mca plm_rsh_agent "ssh -q -o LogLevel=ERROR -o StrictHostKeyChecking=no -p 222" \
  -np $NNODES \
  -x NVSHMEM_REMOTE_TRANSPORT -x NVSHMEM_IBGDA_NIC_HANDLER \
  -x DEEP_EP_DEVICE_TO_HCA_MAPPING -x NVSHMEM_DISABLE_CUDA_VMM \
  -x LD_PRELOAD -x LD_LIBRARY_PATH -x NVSHMEM_IB_GID_INDEX \
  -x NCCL_SOCKET_IFNAME=eth0,eth1 \
  -x NCCL_TUNER_CONFIG_PATH=/usr/local/gib/configs/tuner_config_a4.txtpb \
  bash -c "export WORLD_SIZE=$NNODES && export RANK=\$OMPI_COMM_WORLD_RANK && \
    export MASTER_ADDR=$MASTER_ADDR && export MASTER_PORT=29500 && \
    source /opt/deepep/unified-env.sh && source /usr/local/gib/scripts/set_nccl_env.sh && \
    cd /opt/deepep/DeepEP && python3 tests/test_internode.py"
```

Expected: 32/32 tests pass (BF16/FP8 × dispatch/combine × sync/async × with/without top-k).

> **Important**: `test_internode.py` uses `torch.multiprocessing.spawn` to manage GPU processes internally. MPI must launch only 1 process per node (`slots=1`), not 8. `WORLD_SIZE` means number of nodes (not total GPUs) for DeepEP's `init_dist()`.

## Directory Structure

```
testbed-a4/
├── docker/
│   ├── testbed.Dockerfile      # DeepEP image: NeMo + NVSHMEM + DeepEP
│   ├── cloudbuild.yml          # Cloud Build config
│   └── README.md               # Docker build details
└── gke-runtime/
    ├── values.yaml             # Helm values (GPU count, image, volumes)
    ├── jobset/                 # Helm chart templates
    └── launchers/
        └── torchrun-stratup.sh # Node discovery, SSH setup, topology sorting
```

## Key Environment Variables (DeepEP Runtime)

Set automatically by `/opt/deepep/unified-env.sh`:

| Variable | Value | Purpose |
|----------|-------|---------|
| `NVSHMEM_REMOTE_TRANSPORT` | `ibgda` | GPU-initiated RDMA |
| `NVSHMEM_IBGDA_NIC_HANDLER` | `gpu` | GPU handles NIC directly |
| `DEEP_EP_DEVICE_TO_HCA_MAPPING` | `0:mlx5_0:1,...` | GPU-NIC affinity mapping |
| `NVSHMEM_DISABLE_CUDA_VMM` | `1` | Required for IBGDA |
| `LD_PRELOAD` | `libnvshmem_host.so.3` | NVSHMEM runtime |

## Monitoring

```bash
# Check pods
kubectl get pods | grep deepep

# Follow coordinator logs
kubectl logs -f <coordinator-pod>

# Check JobSet status
kubectl get jobset
```

## Cleanup

```bash
helm uninstall $USER-deepep
```
