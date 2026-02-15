# A4 GPU Testbed on GKE - Docker Build and NCCL Testing

This recipe provides a comprehensive testbed for A4 GPU workloads on [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine). It includes automated Docker image building, Helm-based deployment, and NCCL performance testing capabilities.

## Overview

The testbed is designed to:
- Build custom Docker images for A4 GPU workloads using Google Cloud Build
- Deploy distributed workloads on GKE using Helm charts and JobSet
- Run NCCL performance tests to validate GPU communication
- Provide a foundation for developing and testing A4 GPU applications

## Orchestration and deployment tools

This recipe uses the following technology stack:

- **Orchestration** - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine)
- **Container Build** - [Google Cloud Build](https://cloud.google.com/build) with custom Dockerfile
- **Job Management** - [Kubernetes JobSet](https://kubernetes.io/blog/2025/03/23/introducing-jobset) deployed via Helm charts
- **GPU Communication** - NCCL with gIB plugin for optimized A4 performance

## Test environment

This recipe has been optimized for and tested with the following configuration:

### GKE Cluster Requirements
- [Regional standard cluster](https://cloud.google.com/kubernetes-engine/docs/concepts/configuration-overview) version: 1.31.7-gke.1265000 or later
- GPU node pool with [a4-highgpu-8g](https://cloud.google.com/compute/docs/accelerator-optimized-machines#a4-high-vms) nodes provisioned using the DENSE deployment type
- [Workload Identity Federation for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/workload-identity) enabled
- [Cloud Storage FUSE CSI driver for GKE](https://cloud.google.com/kubernetes-engine/docs/concepts/cloud-storage-fuse-csi-driver) enabled
- [DCGM metrics](https://cloud.google.com/kubernetes-engine/docs/how-to/dcgm-metrics) enabled
- [Kueue](https://kueue.sigs.k8s.io/docs/reference/kueue.v1beta1/) and [JobSet](https://jobset.sigs.k8s.io/docs/overview/) APIs installed
- Kueue configured to support [Topology Aware Scheduling](https://kueue.sigs.k8s.io/docs/concepts/topology_aware_scheduling/)

### Storage Requirements
- Regional Google Cloud Storage (GCS) bucket for logs and artifacts
- Google Artifact Registry for storing custom Docker images

To prepare the required environment, see [GKE environment setup guide](../../../../docs/configuring-environment-gke-a4.md).

## Docker Container Image

This recipe builds a custom Docker image optimized for A4 GPU workloads:

**Base Image**: `us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo25.04-gib1.0.6-A4`

The image includes:
- NVIDIA NeMo 25.04 framework
- NCCL gIB plugin v1.0.6 for A4 GPU optimization
- All necessary dependencies for distributed training and testing

## Key Features

### Automated Docker Build
- Uses Google Cloud Build for scalable image building
- Configurable build parameters via substitutions
- High-performance build machines (e2-highcpu-32)
- Asynchronous build process with 2-hour timeout

### Distributed Job Management
- Helm-based deployment with customizable values
- JobSet for managing multi-node workloads
- Automatic node discovery and SSH configuration
- Physical topology-aware scheduling

### NCCL Performance Testing
- Multi-node NCCL all-reduce performance tests
- Optimized NCCL settings for A4 GPUs
- Comprehensive communication parameter tuning
- Performance validation across different data sizes

## Run the Recipe

From your client workstation, complete the following steps:

### Configure Environment Settings

Set the environment variables to match your environment:

```bash
export PROJECT_ID=<PROJECT_ID>
export REGION=<REGION>
export CLUSTER_NAME=<CLUSTER_NAME>
export GCS_BUCKET=<GCS_BUCKET>
export KUEUE_NAME=<KUEUE_NAME>
export ARTIFACT_REGISTRY=<ARTIFACT_REGISTRY>
```

Replace the following values:
- `<PROJECT_ID>`: your Google Cloud project ID
- `<REGION>`: the region where your cluster is located (e.g., us-central1)
- `<CLUSTER_NAME>`: the name of your GKE cluster
- `<GCS_BUCKET>`: the name of your Cloud Storage bucket (without `gs://` prefix)
- `<KUEUE_NAME>`: the name of the Kueue local queue (default: `a4-high`)
- `<ARTIFACT_REGISTRY>`: your Artifact Registry URL

Set the default project:

```bash
gcloud config set project $PROJECT_ID
```

### Get the Recipe

Clone the `gpu-recipes` repository and set references to the recipe folder:

```bash
git clone -b a4-early-access https://github.com/yangwhale/gpu-recipes.git
cd gpu-recipes
export REPO_ROOT=`git rev-parse --show-toplevel`
export RECIPE_ROOT=$REPO_ROOT/training/a4/testbed
cd $RECIPE_ROOT
```

### Build Custom Docker Image

Build the custom Docker image using Google Cloud Build:

```bash
cd $REPO_ROOT/training/a4/testbed/docker
gcloud builds submit --region=${REGION} \
    --config cloudbuild.yml \
    --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY \
    --timeout "2h" \
    --machine-type=e2-highcpu-32 \
    --quiet \
    --async
```

**Note**: The build process runs asynchronously. You can monitor the build status in the Google Cloud Console or use `gcloud builds list` to check progress.

### Get Cluster Credentials

```bash
gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION
```

### Deploy and Run the Testbed

Deploy the testbed workload using Helm:

```bash
cd $RECIPE_ROOT
helm install -f $RECIPE_ROOT/gke-runtime/values.yaml \
    --set-file workload_launcher=$REPO_ROOT/training/a4/testbed/gke-runtime/launchers/torchrun-stratup.sh \
    --set "workload.image"=$WORKLOAD_IMAGE \
    --set "volumes.gcsMounts[0].bucketName"=${GCS_BUCKET} \
    $USER-testbed \
    $REPO_ROOT/training/a4/testbed/gke-runtime/jobset
```

Where `$WORKLOAD_IMAGE` should be set to your built image:
```bash
export WORKLOAD_IMAGE=us-central1-docker.pkg.dev/supercomputer-testing/chrisya-docker-repo-supercomputer-testing-uc1/testbed:nemo25.04-gib1.0.6-A4
```

## Testing and Validation

The testbed automatically runs the following tests:

### 1. Basic Connectivity Test
Verifies that all nodes can communicate via MPI:
```bash
mpirun --allow-run-as-root -np 16 -hostfile /etc/job-worker-services.txt \
--mca orte_keep_fqdn_hostnames 1 hostname
```

### 2. NCCL Performance Test
Runs comprehensive NCCL all-reduce performance testing:
```bash
#run multi-node NCCL test
mpirun --allow-run-as-root \
--hostfile /etc/job-worker-services.txt \
-wdir /third_party/nccl-tests \
-mca plm_rsh_no_tree_spawn 1 \
--mca orte_keep_fqdn_hostnames 1 \
--map-by slot \
--mca plm_rsh_agent "ssh -q -o LogLevel=ERROR -o StrictHostKeyChecking=no" \
bash -c "source /tmp/export_init_env.sh && ./build/all_reduce_perf -b 2M -e 16G -f 2 -n 1 -g 1 -w 10"
```

### NCCL Optimization Parameters

The testbed uses the following optimized NCCL settings for A4 GPUs:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NCCL_NET` | `gIB` | Use gIB plugin for network communication |
| `NCCL_CROSS_NIC` | `0` | Disable cross-NIC communication |
| `NCCL_NET_GDR_LEVEL` | `PIX` | GPU Direct RDMA level |
| `NCCL_P2P_NET_CHUNKSIZE` | `131072` | P2P network chunk size |
| `NCCL_NVLS_CHUNKSIZE` | `524288` | NVLS chunk size |
| `NCCL_IB_ADAPTIVE_ROUTING` | `1` | Enable adaptive routing |
| `NCCL_IB_QPS_PER_CONNECTION` | `4` | Queue pairs per connection |
| `NCCL_IB_TC` | `52` | Traffic class |
| `NCCL_IB_FIFO_TC` | `84` | FIFO traffic class |
| `NCCL_MIN_NCHANNELS` | `32` | Minimum number of channels |

## Monitor the Job

To check the status of pods in your job:

```bash
kubectl get pods | grep $USER-testbed
```

To get logs from a specific pod:

```bash
kubectl logs POD_NAME
```

To follow logs from the main coordinator pod:

```bash
kubectl logs -f $USER-testbed-workload-0-0-xxxxx
```

## Analyze Results

### NCCL Performance Metrics

The NCCL test outputs performance metrics including:
- **Bandwidth**: Data transfer rate in GB/s
- **Latency**: Communication latency in microseconds
- **Efficiency**: Percentage of theoretical peak performance

Example output:
```
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
     2097152        524288     float     sum      -1    123.4   17.01   31.89      0    123.4   17.01   31.89      0
     4194304       1048576     float     sum      -1    234.5   17.89   33.54      0    234.5   17.89   33.54      0
```

### Log Collection

Logs are automatically collected in the configured GCS bucket:

```
gs://${GCS_BUCKET}/testbed-logs/
├── nccl-test-results.txt
├── connectivity-test.log
└── pod-logs/
    ├── coordinator.log
    └── worker-*.log
```

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check Cloud Build logs: `gcloud builds list`
   - Verify Artifact Registry permissions
   - Ensure sufficient build quota

2. **Pod Startup Issues**
   - Check pod status: `kubectl describe pod POD_NAME`
   - Verify image pull permissions
   - Check node resource availability

3. **NCCL Communication Failures**
   - Verify network connectivity between nodes
   - Check gIB plugin installation
   - Review NCCL debug logs

4. **SSH Connection Issues**
   - Verify SSH public key configuration
   - Check SSH service startup in pods
   - Ensure proper network policies

### Debug Commands

```bash
# Check JobSet status
kubectl get jobset

# Check pod events
kubectl get events --sort-by=.metadata.creationTimestamp

# Check node GPU status
kubectl describe nodes -l cloud.google.com/gke-accelerator=nvidia-a4-high-8g

# Test NCCL manually
kubectl exec -it POD_NAME -- /third_party/nccl-tests/build/all_reduce_perf -b 1M -e 1M -i 1
```

## Customization

### Environment Variables

The testbed supports the following environment variables in [`values.yaml`](training/a4/testbed/gke-runtime/values.yaml:1):

| Variable | Default | Description |
|----------|---------|-------------|
| `SSH_PUBLIC_KEY` | Pre-configured | SSH public key for inter-node communication |
| `SLEEP_INFINITY` | `true` | Keep containers running for debugging |
| `HF_TOKEN` | Pre-configured | HuggingFace token for model access |

### Scaling Configuration

To modify the number of GPUs or nodes, update the [`values.yaml`](training/a4/testbed/gke-runtime/values.yaml:46):

```yaml
workload:
  gpus: 32  # Total number of GPUs (must be multiple of 8)
```

### Custom Launcher Scripts

You can provide custom launcher scripts by modifying the `--set-file workload_launcher` parameter in the Helm command.

## Uninstall the Helm Release

To clean up the testbed resources:

```bash
helm uninstall $USER-testbed
```

## Next Steps

This testbed provides a foundation for:
- Developing custom A4 GPU applications
- Performance benchmarking and optimization
- Distributed training workload development
- NCCL communication pattern analysis

For more advanced use cases, consider:
- Integrating with MLOps pipelines
- Adding custom performance metrics collection
- Implementing automated scaling policies
- Developing application-specific test suites
