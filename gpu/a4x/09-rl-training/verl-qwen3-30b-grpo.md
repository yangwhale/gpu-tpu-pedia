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

## Step 1: 安装 veRL 和依赖

```bash
# 1.1 安装 veRL（从 PyPI 或 source）
pip install verl

# 1.2 安装 vLLM（rollout 引擎）
pip install vllm

# 1.3 安装 Megatron-LM bridge（Megatron 后端必需）
pip install -U git+https://github.com/ISEEKYAN/mbridge.git

# 1.4 安装 Ray（分布式调度）
pip install "ray[default]"

# 1.5 安装其他依赖
pip install wandb flash-attn

# 1.6 验证安装
python3 -c "import verl; print('verl OK')"
python3 -c "import vllm; print('vLLM OK')"
python3 -c "import megatron; print('Megatron OK')" 2>/dev/null || echo "Megatron via mbridge"
```

## Step 2: 下载模型和数据

```bash
# 2.1 下载 Qwen3-30B-A3B-Base 模型（~60GB）
# 需要 huggingface-cli 登录或 HF_TOKEN
export HF_TOKEN=<your_token>
huggingface-cli download Qwen/Qwen3-30B-A3B-Base --local-dir /home/chrisya/models/Qwen3-30B-A3B-Base

# 2.2 下载训练数据 DAPO-Math-17k
huggingface-cli download BytedTsinghua/DAPO-Math-17k --repo-type dataset --local-dir /home/chrisya/data/DAPO-Math-17k

# 2.3 下载验证数据 AIME-2024
huggingface-cli download MaxwellYoung/AIME-2024 --repo-type dataset --local-dir /home/chrisya/data/AIME-2024

# 2.4 验证数据文件存在
ls /home/chrisya/data/DAPO-Math-17k/data/dapo-math-17k.parquet
ls /home/chrisya/data/AIME-2024/data/aime-2024.parquet
```

## Step 3: 克隆 veRL 仓库获取官方脚本

```bash
cd /home/chrisya
git clone --depth 1 https://github.com/verl-project/verl.git
cd verl
git submodule update --init --recursive recipe
```

## Step 4: 运行 GRPO 训练（Colocate 模式）

```bash
cd /home/chrisya/verl

# 使用官方推荐的 8 GPU 配置覆盖脚本默认值
export MODEL_PATH=/home/chrisya/models/Qwen3-30B-A3B-Base
export DATA_DIR=/home/chrisya
export TRAIN_FILES=/home/chrisya/data/DAPO-Math-17k/data/dapo-math-17k.parquet
export VAL_FILES=/home/chrisya/data/AIME-2024/data/aime-2024.parquet

# 官方推荐 8 GPU 并行度
export ACTOR_TP=1
export ACTOR_PP=1
export ACTOR_EP=8
export REF_TP=1
export REF_PP=1
export REF_EP=8
export ALL_OFFLOAD=True

# Rollout（vLLM）
export ROLLOUT_TP=4
export GEN_MOE_TP=2
export GEN_MOE_EP=2

# 单节点
export NNODES=1
export NGPUS_PER_NODE=8

# 减少 epoch 用于验证（官方默认 1000）
export TOTAL_EPOCHS=5
export TEST_FREQ=1
export SAVE_FREQ=100

# 关闭 wandb（或设置 WANDB_API_KEY）
export WANDB_MODE=disabled

# 运行
bash examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh
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
| 1 | — | — | — |

## 参考

- veRL 官方 recipe: `examples/grpo_trainer/run_qwen3_30b_a3b_megatron.sh`
- veRL 文档: https://verl.readthedocs.io/en/latest/perf/dpsk.html
- DAPO 论文: https://arxiv.org/abs/2503.14476
- Qwen3-30B-A3B 模型: https://huggingface.co/Qwen/Qwen3-30B-A3B-Base
