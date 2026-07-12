# GB300 (A4X Max) RL Training

强化学习训练（GRPO / PPO / DPO）在 GB300 上的测试。

> 待 GB300 基础训练测试（07-megatron-training）完成后开展。

## 与 GB200 的差异

GB300 的 288 GB HBM3e 对 RL 训练有显著优势：
- Rollout 阶段可放更大的模型和更长的 sequence
- Policy/Value model 可以放在同一组 GPU 上（减少跨节点通信）
- MBS 翻倍降低 gradient accumulation 步数

## 测试计划

| # | 场景 | 模型 | GPU 数 | 状态 |
|---|------|------|--------|------|
| 1 | GRPO 单节点 | Qwen3 30B | 4 | 待测 |
| 2 | GRPO 多节点 | Qwen3 235B | 64 | 待测 |

## GB200 参考

GB200 RL Training 文档: [a4x/09-rl-training/](../../a4x/09-rl-training/)
