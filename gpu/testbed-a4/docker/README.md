# A4 GPU Testbed Docker Image

This directory contains the Docker build configuration for the A4 GPU testbed container image, optimized for running distributed workloads and NCCL testing on Google A4 GPUs.

## Overview

The testbed Docker image is built on top of NVIDIA's NeMo framework with gIB plugin optimization for A4 GPUs. It provides a complete environment for:
- Distributed GPU workload testing
- NCCL performance benchmarking
- Multi-node communication via SSH
- Integration with Google Cloud services

## Files

- [`testbed.Dockerfile`](testbed.Dockerfile): Main Dockerfile defining the container image build process
- [`cloudbuild.yml`](cloudbuild.yml): Google Cloud Build configuration for automated build pipeline

## Base Image

```dockerfile
FROM us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo25.04-gib1.0.6-A4
```

This base image includes:
- **NVIDIA NeMo 25.04**: Complete framework for large language model training
- **NCCL gIB Plugin v1.0.6**: Optimized for A4 GPU communication
- **PyTorch**: Deep learning framework with GPU support
- **CUDA 12.8**: GPU computing platform
- **All necessary dependencies**: For distributed training and inference

## Key Features

### SSH Configuration for Multi-Node Communication
The image is configured with SSH on port 222 to enable secure communication between pods in a distributed setup:

```dockerfile
# Configure SSH on port 222
RUN cd /etc/ssh/ && sed --in-place='.bak' 's/#Port 22/Port 222/' sshd_config && \
    sed --in-place='.bak' 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' sshd_config

# Generate SSH key pair for inter-node communication
RUN ssh-keygen -t rsa -b 4096 -q -f /root/.ssh/id_rsa -N ""
RUN touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
```

### Optimized for A4 GPU Architecture
- Pre-configured NCCL settings for optimal A4 performance
- gIB plugin integration for high-speed GPU-to-GPU communication
- Support for topology-aware scheduling

## Building the Image

### Prerequisites

Before building, ensure you have:
- Google Cloud SDK installed and configured
- Appropriate permissions for Cloud Build and Artifact Registry
- Access to the base image repository

### Environment Variables

Set the following environment variables:

```bash
export PROJECT_ID=<your-project-id>
export REGION=<your-region>  # e.g., us-central1
export ARTIFACT_REGISTRY=<your-artifact-registry-url>
```

Example:
```bash
export PROJECT_ID=supercomputer-testing
export REGION=us-central1
export ARTIFACT_REGISTRY=us-central1-docker.pkg.dev/supercomputer-testing/chrisya-docker-repo-supercomputer-testing-uc1
```

### Build Command

Build the image using Google Cloud Build:

```bash
cd training/a4/testbed/docker
gcloud builds submit --region=${REGION} \
    --config cloudbuild.yml \
    --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY \
    --timeout "2h" \
    --machine-type=e2-highcpu-32 \
    --quiet \
    --async
```

### Build Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--region` | Google Cloud region for the build | Required |
| `--timeout` | Maximum build time | 2h |
| `--machine-type` | Build machine specification | e2-highcpu-32 |
| `--async` | Run build asynchronously | - |

### Image Tag

After successful build, the image will be tagged as:
```
${_ARTIFACT_REGISTRY}/testbed:nemo25.04-gib1.0.6-A4
```

## Usage

This container image is designed to be used with the A4 testbed deployment system. It serves as the base for:

1. **NCCL Performance Testing**: Validates GPU communication performance
2. **Distributed Workload Execution**: Runs multi-node training and inference
3. **Development Environment**: Provides a consistent environment for A4 GPU development

### Integration with Testbed

The image is automatically used by the testbed Helm chart when deployed via:

```bash
helm install -f values.yaml \
    --set "workload.image"=${ARTIFACT_REGISTRY}/testbed:nemo25.04-gib1.0.6-A4 \
    --set-file workload_launcher=launchers/torchrun-stratup.sh \
    testbed-deployment \
    jobset/
```

## Technical Specifications

### Container Configuration
- **Working Directory**: `/workspace`
- **SSH Port**: 222 (configured for inter-pod communication)
- **Root Access**: Enabled with SSH key authentication
- **Base OS**: Ubuntu-based (inherited from NeMo image)

### Networking
- Supports host networking for optimal GPU communication
- Pre-configured for gIB plugin usage
- Compatible with Kubernetes pod-to-pod communication

### Security
- SSH access secured with RSA 4096-bit keys
- Root login restricted to key-based authentication
- Minimal attack surface with only necessary services

## Monitoring Build Status

### Check Build Progress

```bash
# List recent builds
gcloud builds list --limit=10

# Get specific build details
gcloud builds describe BUILD_ID

# Follow build logs
gcloud builds log BUILD_ID --stream
```

### Build Artifacts

Successful builds produce:
- Container image in Artifact Registry
- Build logs and metadata
- Security scan results (if enabled)

## Troubleshooting

### Common Build Issues

1. **Permission Denied**
   ```bash
   # Ensure proper IAM roles
   gcloud projects add-iam-policy-binding $PROJECT_ID \
       --member="user:your-email@domain.com" \
       --role="roles/cloudbuild.builds.editor"
   ```

2. **Base Image Access**
   ```bash
   # Verify access to base image
   docker pull us-central1-docker.pkg.dev/deeplearning-images/reproducibility/pytorch-gpu-nemo-nccl:nemo25.04-gib1.0.6-A4
   ```

3. **Build Timeout**
   ```bash
   # Increase timeout for large builds
   --timeout "3h"
   ```

### Debug Commands

```bash
# Check Artifact Registry permissions
gcloud artifacts repositories list --location=$REGION

# Verify Cloud Build API is enabled
gcloud services list --enabled | grep cloudbuild

# Test local Docker build
docker build -f testbed.Dockerfile -t testbed:local .
```

## Customization

### Adding Custom Dependencies

To add custom packages or configurations, modify the [`testbed.Dockerfile`](testbed.Dockerfile):

```dockerfile
# Add custom packages
RUN apt-get update && apt-get install -y \
    your-package-name \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install your-python-package
```

### Environment Variables

Add runtime environment variables in the Dockerfile:

```dockerfile
ENV YOUR_VARIABLE=value
ENV NCCL_DEBUG=INFO
```

### Custom SSH Configuration

Modify SSH settings for specific requirements:

```dockerfile
# Custom SSH configuration
RUN echo "ClientAliveInterval 60" >> /etc/ssh/sshd_config
RUN echo "ClientAliveCountMax 3" >> /etc/ssh/sshd_config
```

## Version History

### nemo25.04-gib1.0.6-A4
- Based on NVIDIA NeMo 25.04
- Includes gIB plugin v1.0.6 for A4 optimization
- SSH configured on port 222
- Optimized for distributed workloads

## Related Documentation

- [A4 Testbed Main README](../README.md)
- [NVIDIA NeMo Documentation](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/)
- [Google Cloud Build Documentation](https://cloud.google.com/build/docs)
- [Artifact Registry Documentation](https://cloud.google.com/artifact-registry/docs)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)

## License

Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.