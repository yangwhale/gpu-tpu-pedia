FROM nvcr.io/nvidia/pytorch:25.06-py3 as base

# Build the image
# nvidia-docker build -f B200.dockerfile --build-arg --rm --network host -t gitlab-master.nvidia.com/denliu/dockers:pytorch2506-TE2.8-deepep1.2.1-x86 .
ENV SHELL /bin/bash

RUN rm /opt/megatron-lm -rf
RUN apt-get update
RUN apt-get install -y sudo gdb pstack bash-builtins git zsh autojump tmux curl
RUN pip install debugpy dm-tree torch_tb_profiler einops wandb
RUN pip install sentencepiece tokenizers transformers torchvision ftfy modelcards datasets tqdm pydantic==2.2.1
RUN pip install nvidia-pytriton py-spy yapf darker pytest-cov pytest_mock
# envsubst used for model_params substitution
RUN apt-get install -y gettext

# NOTE: CUDA_HOME is already correctly set by the base image
ENV CPATH=/usr/local/mpi/include:$CPATH
ENV LD_LIBRARY_PATH=/usr/local/mpi/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/lib/${ARCHITECTURE}-linux-gnu:$LD_LIBRARY_PATH

#
# Install grouped gemm
#
RUN --mount=type=cache,target=/root/.cache/pip \
    TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0" pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4


# Install TE
RUN pip install nvidia-mathdx
ARG COMMIT=dd7ab71
ARG TE="git+https://github.com/NVIDIA/TransformerEngine.git@$COMMIT"
RUN unset PIP_CONSTRAINT && NVTE_CUDA_ARCHS="90;100" NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch pip install --no-cache-dir --no-build-isolation $TE

RUN rm -rf /root/.cache /tmp/*

## the dependency of IBGDA
RUN ln -s /usr/lib/x86_64-linux-gnu/libmlx5.so.1 /usr/lib/x86_64-linux-gnu/libmlx5.so

## Clone and build deepep and deepep-nvshmem
WORKDIR /home/dpsk_a2a
RUN git clone https://github.com/deepseek-ai/DeepEP.git ./deepep
RUN cd ./deepep && git checkout v1.2.1 && cd /home/dpsk_a2a
RUN wget https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.3.9_cuda12-archive.tar.xz -O nvshmem_src.tar.xz
RUN tar -xvf nvshmem_src.tar.xz && mv libnvshmem-linux-x86_64-3.3.9_cuda12-archive deepep-nvshmem

ENV NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem/
ENV LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH
ENV PATH=${NVSHMEM_DIR}/bin:$PATH

## Build deepep
WORKDIR /home/dpsk_a2a/deepep
ENV TORCH_CUDA_ARCH_LIST="10.0"
RUN NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem python setup.py develop
RUN NVSHMEM_DIR=/home/dpsk_a2a/deepep-nvshmem python setup.py install

#
# Install Megatron
#
RUN git clone --depth 1 --branch core_r0.14.0 https://github.com/NVIDIA/Megatron-LM.git /workspace/Megatron-LM
ENV MEGATRON_PATH=/workspace/Megatron-LM
ENV PYTHONPATH="${PYTHONPATH}:${MEGATRON_PATH}"

# Install necessary dependencies
RUN pip install transformers tokenizers "huggingface_hub[cli]" \
    sentencepiece modelcards datasets py-spy debugpy einops wandb

## Change the workspace
WORKDIR /home/
