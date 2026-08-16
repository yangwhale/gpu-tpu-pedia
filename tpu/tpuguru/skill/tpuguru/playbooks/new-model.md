# playbook：拿到一个新模型，怎么定起始配置

**目标**：给出一份**起点合理、不撞已知坑**的配置 + 一个能直接跑的测试脚本。
不追求最优 —— 最优靠 AOT 二分 + 真机验证。

## 1. 从 model config 里抽这几个数

`num_hidden_layers` · `hidden_size` · `num_experts` · `moe_intermediate_size`
· `num_experts_per_tok` · `vocab_size` · `max_position_embeddings`

抽不到就问用户，**不要猜**。

## 2. 定并行

- `ici_fsdp_parallelism=-1`（吃满 device），`ici_data_parallelism=1`
- `ici_expert_parallelism=1` —— 实测大幅负收益，除非用户明确要求
- **若用户要开跨卡量化收集**：FSDP 必须能整除 `num_experts`，否则直接报错。
  算一下能整除的最大宽度，并提醒「这会浪费一半分片能力」

## 3. 定 tile

- 三个分块不得超过对应维度：`batch_seq` 自由、`embed_dim ≤ hidden_size`、
  `mlp_dim ≤ moe_intermediate_size`
- 安全起点：`(512, min(2048, hidden_size/2), moe_intermediate_size)`
- **`embed_dim` 取满整个维度会撞 Mosaic 的向量化限制**，别取满

## 4. 估 batch 起点（只作为二分的起点，真值问 AOT）

```
每卡常驻 ≈ (总参数 × 主权重字节数 + 优化器状态) / FSDP宽度
可用 ≈ 94.74 GB − 每卡常驻 − 固定开销
batch 起点 ≈ 可用 / 单条序列激活估算
```

**算完就交给 AOT 二分**，别在这上面较真。

## 5. 跑 lint

把生成的配置直接喂 `scripts/lint.py`，把命中的规则一并展示。

## 6. 产出

1. 填好的表单（前端可直接渲染）
2. 格式化的 AOT 命令
3. **每条选择的一句话理由**，标明是「实测依据」还是「推断」

## 注意

- 层数先用 **4–8 层**把配置跑通（快 3 倍，错误照样暴露），
  **问显存时必须换回生产层数**
- 镜像 tag 必须与目标生产环境一致，否则结论不可迁移
