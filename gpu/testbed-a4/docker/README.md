# DeepEP Docker Image

Custom Docker image for running DeepEP on GKE A4 (B200) GPU nodes.

## Image Contents

**Base**: `nvcr.io/nvidia/nemo:25.11`

| Component | Version | Details |
|-----------|---------|---------|
| NVSHMEM | v3.5.19-1 | IBGDA enabled, no GDRCopy, installed to `/opt/deepep/nvshmem` |
| DeepEP | PR #466 (commit `8a07e7e`) | GPU-NIC explicit mapping, built for SM 10.0 |
| SSH | Port 222 | RSA 4096-bit key, root login enabled |
| Build tools | cmake, ninja-build | For NVSHMEM/DeepEP compilation |

## Build

```bash
export ARTIFACT_REGISTRY=<your-registry-url>

gcloud builds submit \
    --config cloudbuild.yml \
    --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY \
    --timeout "2h" \
    --machine-type=e2-highcpu-32 \
    --quiet --async
```

Output image: `${ARTIFACT_REGISTRY}/nemo-deepep:25.11`

## Build Notes

- `CPLUS_INCLUDE_PATH=/usr/local/cuda/targets/x86_64-linux/include/cccl` is needed because NVSHMEM headers reference `cuda/std/tuple` from CCCL — nvcc finds it automatically but g++ (for .cpp files) needs the explicit path
- `libmlx5.so` symlink is created manually because cmake expects the unversioned `.so` but only `libmlx5.so.1` exists
- NVSHMEM is built without MPI/SHMEM support (not needed for DeepEP)
- `TORCH_CUDA_ARCH_LIST=10.0` targets B200 GPUs only

## Runtime

Source the environment script before running DeepEP:

```bash
source /opt/deepep/unified-env.sh
```

This sets up `LD_LIBRARY_PATH`, `LD_PRELOAD`, NVSHMEM transport config, and GPU-NIC mapping.

## Files

- `testbed.Dockerfile` — Image definition
- `cloudbuild.yml` — Cloud Build pipeline (single docker build step)
