# veRL GRPO 训练验证：Qwen3-30B-A3B MoE

基于 veRL 官方 recipe `run_qwen3_30b_a3b_megatron.sh` 的端到端复刻验证。

## 目标

在单节点 8 GPU (B200) 上，使用 DAPO-style GRPO 算法训练 Qwen3-30B-A3B MoE 模型。Colocate 模式（actor + rollout 共享 GPU）。验证 veRL 官方 recipe 的完整流程，对标官方 benchmark MFU=0.4。

## 官方推荐配置（8 GPU / 1 node）

来源：[veRL 文档 - Training DeepSeek 671b](https://verl.readthedocs.io/en/latest/perf/dpsk.html) Qwen3-30B-A3B 部分

| 参数 | 值 | 说明 |
|---|---|---|
| NNODES | 1 | 单节点 |
| NGPUS_PER_NODE | 8 | |
| ACTOR_TP | 1 | 训练 tensor parallel |
| ACTOR_PP | 1 | 训练 pipeline parallel（无 bubble） |
| ACTOR_EP | 8 | 训练 expert parallel（8 expert 组各占 1 GPU） |
| ALL_OFFLOAD | True | CPU offload（optimizer state + param + grad） |
| ROLLOUT_TP | 4 | vLLM 推理 tensor parallel |
| MFU (官方) | 0.4 | 目标对标值 |

## 环境信息

| 项目 | 值 |
|---|---|
| 机器 | chrisya-b200-spot-mig-ase1 (asia-southeast1-b) |
| GPU | 8× NVIDIA B200, 180GB HBM3e each |
| CPU/RAM | ~3.8TB RAM |
| OS | Ubuntu 24.04.3 LTS |
| Driver | 580.126.09 |
| PyTorch | 2.9.1+cu129（预装） |
| Disk | 1.9TB |

## Step 1: 安装 Docker + 拉 veRL 官方镜像

> **关键教训（踩坑 #1-8 总结）**：在 B200 GCE 裸机上用 pip 安装 veRL 会陷入依赖地狱——系统预装的 PyTorch/boto3/OpenSSL/numpy 与 veRL 依赖互相冲突，Megatron 后端在稳定版不可用，dev 分支有未发布依赖。**直接用 veRL 官方 Docker 镜像**是唯一可靠路径。

```bash
# 1.1 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 1.2 安装 NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 1.3 验证 GPU 在 Docker 中可见
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi

# 1.4 拉 veRL 官方镜像（含 Megatron + vLLM + DeepEP 全套）
docker pull verlai/verl:vllm023.dev1

# 1.5 验证
docker run --rm --gpus all verlai/verl:vllm023.dev1 \
  python3 -c "import verl; import vllm; import megatron; print('ALL OK')"
```

## Step 2: 下载模型和数据

> **踩坑 #2**：veRL 脚本中的 HF dataset repo ID `BytedTsinghua/DAPO-Math-17k` 不存在。正确 ID 为 `BytedTsinghua-SIA/DAPO-Math-17k`（多了 `-SIA`）。AIME-2024 使用 `HuggingFaceH4/aime_2024`。
>
> **踩坑 #3**：AIME-2024 的 parquet 文件名为 `train-00000-of-00001.parquet`，而 veRL 脚本期望 `aime-2024.parquet`。需要 rename 或修改 VAL_FILES 环境变量。

```bash
# 2.1 下载 Qwen3-30B-A3B-Base 模型（~60GB，公开无需 token）
huggingface-cli download Qwen/Qwen3-30B-A3B-Base --local-dir ~/models/Qwen3-30B-A3B-Base

# 2.2 下载训练数据 DAPO-Math-17k（注意 repo ID 带 -SIA）
huggingface-cli download BytedTsinghua-SIA/DAPO-Math-17k --repo-type dataset --local-dir ~/data/DAPO-Math-17k

# 2.3 下载验证数据 AIME-2024
huggingface-cli download HuggingFaceH4/aime_2024 --repo-type dataset --local-dir ~/data/AIME-2024

# 2.4 重命名 AIME parquet 以匹配 veRL 脚本期望
mkdir -p ~/data/AIME-2024/data
cp ~/data/AIME-2024/data/train-00000-of-00001.parquet ~/data/AIME-2024/data/aime-2024.parquet 2>/dev/null || true

# 2.5 验证
ls ~/data/DAPO-Math-17k/data/dapo-math-17k.parquet
ls ~/data/AIME-2024/data/aime-2024.parquet
ls ~/models/Qwen3-30B-A3B-Base/*.safetensors | wc -l
```

## Step 3: 克隆 veRL 仓库获取官方脚本

```bash
cd /home/chrisya
git clone --depth 1 https://github.com/verl-project/verl.git
cd verl
git submodule update --init --recursive recipe
```

## Step 4: 运行 GRPO 训练（Docker + Colocate 模式）

```bash
# 启动 veRL 容器（挂载模型、数据、代码目录）
docker run --rm -it --gpus all --shm-size=256g \
  --network=host \
  -v ~/models:/models \
  -v ~/data:/data \
  -v ~/verl:/workspace/verl \
  -w /workspace/verl \
  -e WANDB_MODE=disabled \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  verlai/verl:vllm023.dev1 \
  bash -c "
    # 官方推荐 8 GPU 并行度
    export MODEL_PATH=/models/Qwen3-30B-A3B-Base
    export TRAIN_FILES=/data/DAPO-Math-17k/data/dapo-math-17k.parquet
    export VAL_FILES=/data/AIME-2024/data/aime-2024.parquet
    export ACTOR_TP=1 ACTOR_PP=1 ACTOR_EP=8
    export REF_TP=1 REF_PP=1 REF_EP=8
    export ALL_OFFLOAD=True
    export ROLLOUT_TP=4 GEN_MOE_TP=2 GEN_MOE_EP=2
    export NNODES=1 NGPUS_PER_NODE=8
    export TOTAL_EPOCHS=3 TEST_FREQ=1 SAVE_FREQ=100
    bash examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh
  "
```

## Step 5: 性能对标

### 官方 Benchmark（来源 veRL 文档）

| 指标 | 8 GPU 官方值 |
|---|---|
| MFU | 0.4 |

### 我方实测

| 指标 | 实测值 | 对比 | 备注 |
|---|---|---|---|
| MFU | — | — | |
| step time (s) | — | — | |
| rollout time (s) | — | — | |
| GPU memory (GB) | — | — | |
| CPU memory (GB) | — | — | |

## 踩坑记录

（在执行过程中填写）

| 序号 | 问题 | 根因 | 解决方法 |
|---|---|---|---|
| 1 | `pip install` 被拒绝 (PEP 668) | Ubuntu 24.04 externally-managed-environment | 用 venv（推荐）或加 `--break-system-packages` |
| 2 | DAPO-Math-17k 下载 404 | veRL 脚本写的 `BytedTsinghua/` 不存在 | 正确 repo: `BytedTsinghua-SIA/DAPO-Math-17k` |
| 3 | AIME-2024 parquet 文件名不匹配 | HF 自动命名 `train-00000-of-00001.parquet` | cp rename 为 `aime-2024.parquet` 或改 VAL_FILES 环境变量 |
| 4 | accelerate 循环 import | B200 GCE 镜像的系统包（boto3/OpenSSL/numpy）与 veRL 依赖互相冲突 | **不用** `--system-site-packages`，创建干净 venv 从头装 |
| 5 | numpy ABI + OpenSSL + boto3 连锁冲突 | `--system-site-packages` venv 导致系统包泄漏进 venv | 干净 venv + `pip install torch==2.9.1` 从 PyPI 装 |
| 6 | `transfer_queue` 模块找不到 | veRL git main 分支 (0.9.0.dev0) 引入了未发布模块 | 用 pip 装稳定版 `verl==0.8.0`，不从 clone 目录运行 |
| 7 | transformers 5.x breaking changes | `--force-reinstall` 拉到了 transformers 5.12.1 | 固定 `transformers>=4.56.0,<5` |
| 8 | verl 0.8.0 不含 Megatron 后端 | Megatron backend 是 dev 分支 preview，PyPI 稳定版未注册 | **用 Docker 镜像**（含 Megatron + vLLM + DeepEP 全套）|
| 9 | veRL 文档推荐的镜像不支持 B200 | `app-verl0.4-*` 基于 NGC 24.08 (Hopper only)，检测到 B200 直接拒绝启动 | 用最新镜像 `verlai/verl:vllm023.dev1`（CUDA 13.0+，支持 Blackwell）|

## 参考

- veRL 官方 recipe: `examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh`
- veRL 文档: https://verl.readthedocs.io/en/latest/perf/dpsk.html
- DAPO 论文: https://arxiv.org/abs/2503.14476
- Qwen3-30B-A3B 模型: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base
