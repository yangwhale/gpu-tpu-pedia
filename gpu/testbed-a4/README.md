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
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    $USER-deepep \
    gke-runtime/jobset
```

### 3. Run DeepEP Tests

SSH into the coordinator pod and run:

```bash
# Source DeepEP runtime environment
source /opt/deepep/unified-env.sh

# Run internode tests (from any node)
cd /opt/deepep/DeepEP
python3 tests/test_internode.py
```

Expected: 32/32 tests pass (BF16/FP8 × dispatch/combine × sync/async × with/without top-k).

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
