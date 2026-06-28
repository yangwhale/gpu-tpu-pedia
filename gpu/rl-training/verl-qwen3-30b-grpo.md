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
# 拉 veRL 验证过的镜像（vLLM 0.11 + PyTorch 2.8 + CUDA 12.8）
# 注意：不要用 vllm023.dev1（vLLM 0.23 的 MoE weight loader 与 veRL 不兼容）
docker pull verlai/verl:vllm011.latest

# 验证 GPU
docker run --rm --gpus all --entrypoint bash verlai/verl:vllm011.latest \
  -c "python3 -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0))'"
```

### 1.3 容器内安装 veRL

> **核心教训**：用 `pip install verl==0.8.0`（稳定版），**不要 `git clone` main 分支**。main 分支依赖未发布的 `transfer_queue` 模块，会导致 import 错误。稳定版一行命令开箱即用。

容器内只需两行：
```bash
pip install verl==0.8.0
pip install datasets   # 用于下载 GSM8K
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

## Step 4: 运行 GRPO 训练（Docker + FSDP2 Colocate）

> **实测验证的完整命令**（2026-06-28 在 B200 8 卡上跑通）

```bash
docker run -d --name verl-grpo --gpus all --shm-size=256g \
  --network=host --entrypoint bash \
  -v ~/models:/models \
  verlai/verl:vllm011.latest \
  -c '
    # 安装 veRL 稳定版
    pip install verl==0.8.0 datasets -q

    # 准备 GSM8K 数据（data_source 必须是 "openai/gsm8k"）
    python3 -c "
import datasets, os, pandas as pd
ds = datasets.load_dataset(\"openai/gsm8k\", \"main\")
os.makedirs(\"/tmp/gsm8k\", exist_ok=True)
for split in [\"train\", \"test\"]:
    data = []
    for ex in ds[split]:
        answer = ex[\"answer\"].split(\"####\")[-1].strip()
        data.append({
            \"data_source\": \"openai/gsm8k\",
            \"prompt\": [{\"role\": \"user\", \"content\": ex[\"question\"]}],
            \"reward_model\": {\"style\": \"rule\", \"ground_truth\": answer},
            \"extra_info\": {\"answer\": answer}
        })
    pd.DataFrame(data).to_parquet(f\"/tmp/gsm8k/{split}.parquet\")
    print(f\"{split}: {len(data)}\")
"

    # 运行 GRPO（FSDP2 后端 + vLLM rollout colocate）
    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
      data.train_files=/tmp/gsm8k/train.parquet \
      data.val_files=/tmp/gsm8k/test.parquet \
      data.train_batch_size=64 \
      data.max_prompt_length=512 \
      data.max_response_length=512 \
      data.prompt_key=prompt \
      data.return_raw_chat=True \
      actor_rollout_ref.model.path=/models/Qwen2.5-7B-Instruct \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.actor.ppo_mini_batch_size=64 \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
      actor_rollout_ref.actor.strategy=fsdp2 \
      algorithm.kl_ctrl.kl_coef=0.001 \
      trainer.logger=console \
      trainer.val_before_train=False \
      trainer.n_gpus_per_node=8 \
      trainer.nnodes=1 \
      trainer.save_freq=100 \
      trainer.test_freq=5 \
      algorithm.adv_estimator=grpo \
      actor_rollout_ref.rollout.n=4 \
      trainer.total_epochs=2
  '

# 查看日志
docker logs -f verl-grpo
```

## Step 5: 性能对标

### 官方 Benchmark（来源 veRL 文档）

| 指标 | 8 GPU 官方值 |
|---|---|
| MFU | 0.4 |

### 我方实测（B200 8 卡，FSDP2 + Qwen2.5-7B-Instruct）

> **注**：首次验证使用 Qwen2.5-7B dense 模型（非 MoE），目的是先跑通 RL 链路。MoE 模型待后续验证。

| 指标 | 实测值 | 备注 |
|---|---|---|
| MFU (actor) | **0.14** | FSDP2 后端，单节点 8 卡 |
| step time | **~8.5 秒** | 包含 rollout + train + weight sync |
| throughput | **~1,480 tokens/s** | |
| rollout 生成 | ~3.3 秒 | vLLM TP=1，gpu_mem_util=0.5 |
| actor 训练 | ~1.8 秒 | FSDP2 + gradient checkpointing |
| weight 更新 | ~2.0 秒 | FSDP→vLLM reshard |
| GPU 显存 | 101 GB / 178 GB (57%) | 每卡 |
| CPU 显存 | 102 GB | optimizer offload |
| response 长度 | ~280 tokens (avg) | max 512 |
| 模型 | Qwen2.5-7B-Instruct (7.62B) | dense，非 MoE |
| 数据集 | GSM8K (7473 train / 1319 test) | 数学推理 |
| 算法 | GRPO (no critic) | adv_estimator=grpo |

**环境组合（最终可工作的配置）**：
- 镜像：`verlai/verl:vllm011.latest`（vLLM 0.11, PyTorch 2.8+cu128）
- veRL：`pip install verl==0.8.0`（稳定版，不用 git main）
- 后端：FSDP2（不用 Megatron）
- rollout：vLLM colocate，TP=1

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
| 15 | Docker 默认 entrypoint 是 vLLM api_server | `vllm011.latest` 镜像的 entrypoint 不是 bash | 加 `--entrypoint bash` |
| 16 | GSM8K 数据缺 `prompt` 列 | 原始 HF 数据列名是 `question` | 预处理时转成 `[{"role":"user","content":...}]` 格式 |
| 17 | GSM8K `data_source` 不匹配 reward 注册表 | 用了 `"gsm8k"` 但 verl 注册的是 `"openai/gsm8k"` | data_source 必须跟 verl 源码中的注册名完全一致 |
| 18 | veRL git main 分支有未发布 `transfer_queue` 依赖 | main 分支正在开发新功能 | **用 `pip install verl==0.8.0`，不 clone git main** |

## 参考

- veRL 官方 recipe: `examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh`
- veRL 文档: https://verl.readthedocs.io/en/latest/perf/dpsk.html
- DAPO 论文: https://arxiv.org/abs/2503.14476
- Qwen3-30B-A3B 模型: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base
