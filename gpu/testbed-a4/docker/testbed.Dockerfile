# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM nvcr.io/nvidia/nemo:25.11
WORKDIR /workspace

# SSH: enable inter-node communication
RUN cd /etc/ssh/ && sed --in-place='.bak' 's/#Port 22/Port 222/' sshd_config && \
    sed --in-place='.bak' 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' sshd_config
RUN ssh-keygen -t rsa -b 4096 -q -f /root/.ssh/id_rsa -N ""
RUN touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys

# Build tools for NVSHMEM and DeepEP
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends cmake ninja-build libibverbs-dev rdma-core ibverbs-utils && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

# NVSHMEM v3.5.19-1 with IBGDA (no GDRCopy)
ENV CUDA_HOME=/usr/local/cuda
RUN git clone --depth 1 --branch v3.5.19-1 https://github.com/NVIDIA/nvshmem.git /tmp/nvshmem-src && \
    mkdir -p /tmp/nvshmem-src/build && cd /tmp/nvshmem-src/build && \
    cmake .. \
        -DCMAKE_INSTALL_PREFIX=/opt/deepep/nvshmem \
        -DCMAKE_CUDA_ARCHITECTURES=100 \
        -DNVSHMEM_IBGDA_SUPPORT=ON \
        -DNVSHMEM_MPI_SUPPORT=OFF \
        -DNVSHMEM_SHMEM_SUPPORT=OFF \
        -DCUDA_HOME=$CUDA_HOME && \
    make -j$(nproc) && \
    mkdir -p /opt/deepep/nvshmem/{lib,include,bin} && \
    cp -a src/lib/*.so* /opt/deepep/nvshmem/lib/ && \
    cp -a src/lib/*.a /opt/deepep/nvshmem/lib/ 2>/dev/null || true && \
    cp -r /tmp/nvshmem-src/src/include/* /opt/deepep/nvshmem/include/ && \
    rm -rf /tmp/nvshmem-src

# DeepEP with PR #466 (GPU-NIC explicit mapping), pinned to commit 8a07e7e
# CPLUS_INCLUDE_PATH: NVSHMEM headers reference cuda/std/tuple from CCCL,
# nvcc finds it automatically but g++ (for .cpp files) needs explicit path
ENV CPLUS_INCLUDE_PATH=/usr/local/cuda/targets/x86_64-linux/include/cccl
RUN git clone https://github.com/deepseek-ai/DeepEP.git /opt/deepep/DeepEP && \
    cd /opt/deepep/DeepEP && \
    git fetch origin pull/466/head:pr-466 && \
    git checkout 8a07e7e && \
    TORCH_CUDA_ARCH_LIST=10.0 NVSHMEM_DIR=/opt/deepep/nvshmem \
    python3 setup.py build_ext --inplace

# DeepEP runtime environment script
RUN printf '#!/bin/bash\n\
export NVSHMEM_HOME=/opt/deepep/nvshmem\n\
export CUDA_HOME=/usr/local/cuda\n\
export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}\n\
export LD_PRELOAD="${NVSHMEM_HOME}/lib/libnvshmem_host.so.3"\n\
export NVSHMEM_REMOTE_TRANSPORT=ibgda\n\
export NVSHMEM_IB_ENABLE_IBGDA=1\n\
export NVSHMEM_IBGDA_NIC_HANDLER=gpu\n\
export NVSHMEM_HCA_PREFIX=mlx5\n\
export NVSHMEM_IB_GID_INDEX=3\n\
export NVSHMEM_DISABLE_CUDA_VMM=1\n\
export NVSHMEM_ENABLE_NIC_PE_MAPPING=1\n\
export DEEP_EP_DEVICE_TO_HCA_MAPPING=0:mlx5_0:1,1:mlx5_1:1,2:mlx5_2:1,3:mlx5_3:1,4:mlx5_4:1,5:mlx5_5:1,6:mlx5_6:1,7:mlx5_7:1\n\
export PYTHONPATH=/opt/deepep/DeepEP:${PYTHONPATH:-}\n' > /opt/deepep/unified-env.sh && \
    chmod +x /opt/deepep/unified-env.sh

ENV PYTHONPATH=/opt/deepep/DeepEP:${PYTHONPATH:-}

WORKDIR /workspace
