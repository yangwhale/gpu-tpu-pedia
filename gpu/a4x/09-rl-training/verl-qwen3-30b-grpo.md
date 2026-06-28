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

## Step 1: 安装 Docker + 拉镜像 + 容器内装 veRL

> **关键教训（踩坑 #1-14 总结）**：
> 1. 裸机 pip install veRL 会陷入依赖地狱（系统包冲突、版本不匹配）→ **必须用 Docker**
> 2. veRL 文档推荐的镜像 `app-verl0.4-*` 是 Hopper only → **用 `verlai/verl:vllm023.dev1`**（CUDA 13，支持 Blackwell）
> 3. `vllm023.dev1` 是 base image 不含 verl → **容器内从 git clone main 安装**
> 4. veRL main 分支依赖未发布的 `transfer_queue` 模块 → **创建 stub 绕过**
> 5. B200 GCE 镜像 dpkg 损坏 → `apt-mark hold linux-image-*` 后再装 Docker

### 1.1 安装 Docker（B200 GCE 特殊处理）

```bash
# B200 GCE 镜像的 kernel dkms 可能损坏，先 hold 再装
sudo apt-mark hold linux-image-* linux-headers-*
sudo dpkg --configure -a --force-all
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 1.2 拉镜像

```bash
# 拉 veRL base image（CUDA 13 + vLLM 0.23 + Megatron）
docker pull verlai/verl:vllm023.dev1

# 验证 GPU
docker run --rm --gpus all verlai/verl:vllm023.dev1 \
  python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

### 1.3 准备容器启动脚本

veRL main 分支依赖未发布的 `transfer_queue`，需要 stub。将以下脚本保存为 `~/verl-run.sh`：

```bash
#!/bin/bash
# Install veRL from latest main
cd /tmp && git clone --depth 1 -q https://github.com/verl-project/verl.git
cd /tmp/verl && pip install -e . -q
pip install -q "transformers>=4.56.0,<5"
pip install -q -U "git+https://github.com/ISEEKYAN/mbridge.git"

# Create transfer_queue stub (未发布模块的最小替代)
mkdir -p /tmp/verl/transfer_queue
cat > /tmp/verl/transfer_queue/__init__.py << 'STUB'
class KVBatchMeta:
    pass
class TransferQueue:
    pass
def init(config):
    pass
def close():
    pass
def get_queue():
    return None
STUB

python3 -c "import verl; print('verl:', verl.__version__)"

# Run training
cd /tmp/verl
exec bash examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh "$@"
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
docker run -d --name verl-grpo --gpus all --shm-size=256g \
  --network=host \
  -v ~/models:/models \
  -v ~/data:/data \
  -v ~/verl-run.sh:/tmp/verl-run.sh \
  -e WANDB_MODE=disabled \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e HYDRA_FULL_ERROR=1 \
  -e MODEL_PATH=/models/Qwen3-30B-A3B-Base \
  -e TRAIN_FILES=/data/DAPO-Math-17k/data/dapo-math-17k.parquet \
  -e VAL_FILES=/data/AIME-2024/data/aime-2024.parquet \
  -e ACTOR_TP=1 -e ACTOR_PP=1 -e ACTOR_EP=8 \
  -e REF_TP=1 -e REF_PP=1 -e REF_EP=8 \
  -e ALL_OFFLOAD=True \
  -e ROLLOUT_TP=8 -e GEN_MOE_TP=1 -e GEN_MOE_EP=8 \
  -e NNODES=1 -e NGPUS_PER_NODE=8 \
  -e TOTAL_EPOCHS=3 -e TEST_FREQ=1 -e SAVE_FREQ=100 \
  -e TRAIN_BATCH_SIZE=64 \
  verlai/verl:vllm023.dev1 \
  bash /tmp/verl-run.sh

# 查看日志
docker logs -f verl-grpo
```

> **Rollout 并行度注意**：8 GPU 单节点时 ROLLOUT_TP=8、GEN_MOE_EP=8，确保 `EP == TP * DP`（8 = 8 × 1）。设其他值（如 TP=4, EP=2）会触发 assertion 错误。

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
| 10 | verl v0.8.0 weight resharding 不兼容 vLLM 0.23 | MoE weight loader `shard_dim=0` ValueError | 用 veRL main 分支（git clone latest），配合 `transfer_queue` stub |
| 11 | Docker 镜像 `vllm023.dev1` 不含 verl | 这是 base image，需要自己装 verl | 容器内 `pip install -e .` from git clone |
| 12 | `transfer_queue` 需要 `KVBatchMeta` class | stub 只有 `init/close` 不够 | stub 补全 `KVBatchMeta` + `TransferQueue` class |
| 13 | rollout EP 配置 assertion | `expert_parallel_size != TP * DP` | ROLLOUT_TP=8, GEN_MOE_EP=8（8 卡 = 1 个推理组）|
| 14 | B200 GCE dpkg 状态损坏 | kernel dkms 编译失败阻塞所有 apt install | `apt-mark hold linux-image-*` 绕过 |

## 参考

- veRL 官方 recipe: `examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh`
- veRL 文档: https://verl.readthedocs.io/en/latest/perf/dpsk.html
- DAPO 论文: https://arxiv.org/abs/2503.14476
- Qwen3-30B-A3B 模型: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base
