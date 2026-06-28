# veRL GRPO 训练：从零到收敛的端到端指南

在 NVIDIA B200 8 卡上使用 veRL + FSDP2 + vLLM 运行 GRPO 强化学习训练，训练 Qwen2.5-7B 做数学推理。

**实测结果**：Score 从 77.7% → 98.4%（55 steps），训练收敛，端到端验证通过（2026-06-28）。

---

## 前置条件

- 一台有 GPU 的机器（本指南使用 8×B200，H100/H200/A100 同样适用）
- Docker + NVIDIA Container Toolkit 已安装
- HuggingFace 访问（下载模型，Qwen2.5-7B 无需 token）

## Step 1: 安装 Docker + NVIDIA Container Toolkit

```bash
# 如果 dpkg 状态损坏（B200 GCE 镜像常见），先 hold kernel 包
sudo apt-mark hold linux-image-* linux-headers-* 2>/dev/null
sudo dpkg --configure -a --force-all 2>/dev/null

# 安装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 安装 NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
sudo apt-get update -qq && sudo apt-get install -y -qq nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu24.04 nvidia-smi --query-gpu=name --format=csv,noheader
```

## Step 2: 下载模型

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ~/models/Qwen2.5-7B-Instruct
```

> 模型约 15GB，公开无需 token。如需使用更大模型（如 Qwen2.5-32B），替换 model path 即可。

## Step 3: 拉取 Docker 镜像

```bash
docker pull verlai/verl:vllm011.latest
```

> **镜像选择关键**：
> - 用 `verlai/verl:vllm011.latest`（vLLM 0.11 + PyTorch 2.8 + CUDA 12.8）
> - **不要**用 `app-verl0.4-*`（Hopper only，B200 会被拒绝启动）
> - **不要**用 `vllm023.dev1`（vLLM 0.23 的 MoE weight loader 与 verl 不兼容）

## Step 4: 一键启动训练

```bash
docker run -d --name verl-grpo --gpus all --shm-size=256g \
  --network=host --entrypoint bash \
  -v ~/models:/models \
  verlai/verl:vllm011.latest \
  -c '
    # ===== 1. 安装依赖 =====
    pip install verl==0.8.0 datasets -q

    # ===== 2. 准备 GSM8K 数据 =====
    # 关键：data_source 必须是 "openai/gsm8k"（与 verl 内置 reward function 注册名匹配）
    # 关键：prompt 必须加 system 提示模型用 "#### <数字>" 格式结尾（否则 reward 全 0）
    python3 -c "
import datasets, os, pandas as pd
ds = datasets.load_dataset(\"openai/gsm8k\", \"main\")
os.makedirs(\"/tmp/gsm8k\", exist_ok=True)
SYSTEM = \"You are a math tutor. Solve the problem step by step. End your answer with: #### <final number>\"
for split in [\"train\", \"test\"]:
    data = []
    for ex in ds[split]:
        answer = ex[\"answer\"].split(\"####\")[-1].strip()
        data.append({
            \"data_source\": \"openai/gsm8k\",
            \"prompt\": [
                {\"role\": \"system\", \"content\": SYSTEM},
                {\"role\": \"user\", \"content\": ex[\"question\"]}
            ],
            \"reward_model\": {\"style\": \"rule\", \"ground_truth\": answer},
            \"extra_info\": {\"answer\": answer}
        })
    pd.DataFrame(data).to_parquet(f\"/tmp/gsm8k/{split}.parquet\")
    print(f\"{split}: {len(data)} examples\")
"

    # ===== 3. 运行 GRPO 训练 =====
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
```

```bash
# 查看训练日志
docker logs -f verl-grpo
```

## Step 5: 判断训练是否成功

训练日志中每个 step 会输出 metrics。关注三个指标：

**1. `critic/score/mean`（准确率）** — 应该从低逐步上升

**2. `actor/entropy`（输出多样性）** — 应该缓慢下降但不归零

**3. `actor/grad_norm`（梯度稳定性）** — 应该稳定，不突然飙高

### 我方实测训练曲线（8×B200, Qwen2.5-7B, GSM8K, 55 steps）

| Step | Score | Entropy | Grad Norm |
|---|---|---|---|
| 1 | 0.777 | 0.175 | 0.54 |
| 5 | 0.930 | 0.188 | 0.22 |
| 10 | 0.895 | 0.197 | 0.21 |
| 18 | 0.969 | 0.197 | 0.23 |
| 20 | 0.957 | 0.199 | 0.32 |
| 30 | 0.934 | 0.151 | 0.42 |
| 40 | 0.906 | 0.149 | 0.31 |
| 50 | 0.941 | 0.128 | 0.43 |
| 55 | **0.984** | 0.113 | 0.30 |

**结论**：Score 从 77.7% 提升到 98.4%，entropy 从 0.175 下降到 0.113（未归零），grad_norm 稳定在 0.2-0.4。**RL 训练有效，模型在收敛。**

### 实验 2：MATH-500 竞赛题（8×B200, Qwen2.5-7B, 35 steps）

切换到 MATH-500 数据集（高中数学竞赛：代数、几何、数论、组合），验证 RL 在更难任务上的表现。

| Step | Score | Entropy | Grad Norm |
|---|---|---|---|
| 1 | 0.707 | 0.117 | 0.168 |
| 5 | 0.691 | 0.107 | 0.137 |
| 8 | 0.605 | 0.095 | 0.134 |
| 10 | 0.773 | 0.090 | 0.177 |
| 15 | 0.602 | 0.092 | 0.141 |
| 20 | 0.734 | 0.097 | 0.134 |
| 25 | 0.711 | 0.100 | 0.147 |
| 30 | 0.684 | 0.093 | 0.190 |
| 35 | 0.703 | 0.094 | 0.153 |

**结论**：Score 在 0.60-0.79 之间波动，整体从均值 0.69 微升到 0.71，提升不到 2 个百分点。Entropy 从 0.117 下降到 0.094（-20%），模型在变得更确定但能力没有显著提升。

### GSM8K vs MATH 对比分析

| 维度 | GSM8K（小学数学） | MATH-500（竞赛数学） |
|---|---|---|
| 初始 Score | 0.777 (77.7%) | 0.707 (70.7%) |
| 最终 Score | **0.984 (98.4%)** | 0.703 (70.3%) |
| Score 提升 | **+20.7%** | +0%（无明显提升） |
| Entropy 下降 | 0.175 → 0.113 (-35%) | 0.117 → 0.094 (-20%) |
| Grad Norm | 0.2-0.4（活跃） | 0.12-0.19（较低） |
| 训练效果 | 明显收敛 | 波动无趋势 |

### 经验教训

1. **RL 不能突破模型的能力天花板**。GSM8K 是小学数学，7B 模型本身有能力做对，RL 帮它从 77% 提到 98%——把"会但不稳定"变成"稳定做对"。MATH 是竞赛题，7B 模型很多题根本不会解，RL 无法教会它新知识。

2. **数据难度要匹配模型能力**。最佳 RL 训练效果出现在模型"似懂非懂"的区间——有一定比例能做对（提供正向信号），也有一定比例做错（提供学习空间）。GSM8K 对 7B 刚好在这个区间；MATH 太难了，大部分题 group 内所有 response 全错，GRPO 算不出有效的 advantage。

3. **Entropy 是早期预警指标**。MATH 上 entropy 一开始就比 GSM8K 低（0.117 vs 0.175），说明模型输出更"模板化"——对竞赛题没有探索的余地，直接套固定模式作答。

4. **Grad norm 反映学习信号强度**。GSM8K 的 grad_norm 0.2-0.4 说明模型每步都在被有效纠正；MATH 的 0.12-0.19 说明梯度信号弱——要么 reward 太稀疏（全错=0 advantage），要么模型已经在能力极限无法进一步优化。

5. **想在竞赛题上看到 RL 效果**，需要：更大的模型（32B+）、更长的 response length（4096+）、更多训练步数（500+）、或者用 Instruct 版 + 更难的 prompt 引导 chain-of-thought。

### 性能指标

| 指标 | 值 |
|---|---|
| step time | ~8 秒 |
| throughput | ~1,400-1,500 tokens/s |
| MFU (actor) | 0.10-0.14 |
| GPU 显存 | 101 GB / 178 GB (57%) per card |
| 模型参数 | 7.62B (Qwen2.5-7B-Instruct) |
| 训练后端 | FSDP2 (PyTorch native) |
| 推理引擎 | vLLM 0.11, colocate mode, TP=1 |

## 关键配置说明

| 配置 | 值 | 为什么 |
|---|---|---|
| `actor.strategy=fsdp2` | FSDP2 | PyTorch 原生，兼容性最好。Megatron 后端在 verl 0.8.0 未完全支持 |
| `rollout.tensor_model_parallel_size=1` | TP=1 | 7B 模型单卡放得下，TP=1 最简单 |
| `rollout.gpu_memory_utilization=0.5` | 50% | colocate 模式需要给训练留显存 |
| `rollout.n=4` | 4 | 每个 prompt 生成 4 个 response 做 group 对比 |
| `algorithm.adv_estimator=grpo` | GRPO | 无 Critic 模型，用 group 内相对比较计算 advantage |
| `data.return_raw_chat=True` | True | prompt 是 chat message 格式，需要 tokenizer apply_chat_template |
| `data_source="openai/gsm8k"` | 在数据中 | 必须与 verl 内置 reward function 的注册名完全匹配 |

## 数据格式要求（重要）

verl 的 GSM8K reward function 用 **strict 模式**匹配答案：只认 `#### <数字>` 格式。如果模型的输出不包含 `####`，score 会全是 0，梯度为 0，模型不会学习。

解法：在 system prompt 中明确告诉模型输出格式：
```
You are a math tutor. Solve the problem step by step. End your answer with: #### <final number>
```

数据 parquet 必须包含以下列：
```python
{
    "data_source": "openai/gsm8k",      # 必须与 verl reward 注册名一致
    "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "问题"}],
    "reward_model": {"style": "rule", "ground_truth": "42"},
    "extra_info": {"answer": "42"}
}
```

## 踩坑记录（18 个，按发现顺序）

| # | 问题 | 解决方法 |
|---|---|---|
| 1 | Ubuntu 24.04 PEP 668 拒绝 pip install | 用 Docker，不在裸机装 |
| 2 | DAPO-Math-17k repo ID 错误 | 正确: `BytedTsinghua-SIA/DAPO-Math-17k` |
| 3 | AIME-2024 parquet 文件名不匹配 | rename 或改 VAL_FILES |
| 4 | accelerate 循环 import | 用 Docker，不在裸机装 |
| 5 | numpy ABI 不兼容 | 用 Docker，不在裸机装 |
| 6 | verl git main 有未发布 transfer_queue 依赖 | **用 pip install verl==0.8.0** |
| 7 | transformers 5.x breaking changes | Docker 镜像里版本锁定 |
| 8 | verl 0.8.0 不含 Megatron 后端 | **用 FSDP2 后端** |
| 9 | veRL 文档推荐的镜像不支持 B200 | **用 verlai/verl:vllm011.latest** |
| 10 | verl 0.8.0 weight resharding 不兼容 vLLM 0.23 | 用 vllm011 镜像不用 vllm023 |
| 11 | vllm023 镜像不含 verl | 容器内 pip install |
| 12 | transfer_queue 需要 KVBatchMeta class | 不用 git main，用 pip 0.8.0 |
| 13 | rollout EP 配置 assertion | 用 FSDP2 不涉及 |
| 14 | B200 GCE dpkg 损坏 | apt-mark hold linux-image-* |
| 15 | Docker entrypoint 是 vLLM api_server | 加 --entrypoint bash |
| 16 | GSM8K 数据缺 prompt 列 | 预处理转 chat message 格式 |
| 17 | data_source 不匹配 reward 注册表 | 用 "openai/gsm8k" 不是 "gsm8k" |
| 18 | Score 全 0，grad 全 0 | 加 system prompt 教模型用 #### 格式 |

> **总结**：18 个坑中，用 Docker + pip 稳定版可以避免 #1-15。真正需要注意的只有 #16-18（数据格式）。

## 后续方向

- [ ] 用更大模型（Qwen2.5-32B）在 MATH 上验证 RL 收敛
- [ ] 替换为 MoE 模型（Qwen3-30B-A3B）验证 MoE RL 训练
- [ ] 迁移到 NVL72 (GB200) 多节点训练
- [ ] 对比 OpenRLHF 和 NeMo-RL 框架
- [ ] 调优 optimizer offload 策略（动态 CPU↔GPU 搬运 vs 全程 GPU）
- [x] ~~使用更难的数据集（MATH）观察收敛~~ → 已完成，7B 模型在 MATH 上无明显收敛

## 参考

- [veRL on GKE B200 教程](https://discuss.google.dev/t/tutorial-scaling-reinforcement-learning-with-verl-on-gke/336370)
- [NeMo-RL on GKE B200](https://discuss.google.dev/t/accelerating-reinforcement-learning-on-google-cloud-using-nvidia-nemo-rl/269579)
- [veRL GitHub](https://github.com/verl-project/verl)
- [GRPO 论文 (DeepSeek)](https://arxiv.org/abs/2402.03300)
- [GSM8K 数据集](https://huggingface.co/datasets/openai/gsm8k)
