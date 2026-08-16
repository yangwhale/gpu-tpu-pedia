# MoE kernel 分支与实测性能

## 四条分支

```
if use_tokamax_gmm:
    if quantization or use_gmm_v2:  →  megablox 封装 + tokamax 后端（带跨卡量化收集）
    else:                           →  tokamax.ragged_dot（裸路径）
elif megablox:                      →  megablox 原生 Pallas kernel
else:                               →  jax.lax.ragged_dot
```

**同一个开关在两个精度下走的是完全不同的代码** —— 这是最容易误判的地方：

| 精度 | 开 `use_tokamax_gmm` 后实际走到哪 | 实测 |
|---|---|---|
| BF16 | 裸 `ragged_dot` | **慢 12 倍** |
| FP8 | megablox 封装 + 跨卡量化收集 | 677.0 per-chip |

所以「BF16 下 native 赢、FP8 下 tokamax 赢」不是矛盾，是两条不同的代码。

## megablox 原生 kernel 的设计

**不是「融合的 MoE kernel」** —— 它只是一个分组矩阵乘，路由/排序/反排序都在 kernel 外面。

精髓是一套类似 CSR 的组元数据：

1. 对每个专家的 token 数做前缀和，得到每组的起止行
2. 把起止**向上向下对齐到分块**（硬件分块固定，而每个专家分到的 token 数不齐）
3. 展开成两张查找表：grid 第 i 步用哪个专家、处理哪个 m 分块
4. **边界分块跑两趟**（前后两个专家各一次），用 store mask 只写回属于自己的行

表长 = `m 分块数 + 组数`，多出来的就是边界重叠。
**「宁可多算再筛」** —— 比拆箱简单，且一个 token 都不丢（这是它相对 padding 方案的核心卖点）。

计算部分：三层 grid（n / 活跃分块 / k），**累加器在 VMEM 用 fp32，只在 k 走完才写回 HBM**。

**kernel 内部没有任何跨卡通信原语** —— 没有远程 DMA、没有 `axis_index`、没有 `all_gather`。
所以「kernel 自己按需收专家」在这个实现里不可能。

## 关键：耗时由「要扫过的行数」决定，不是组数

实测（单独计时，同样 229,376 行）：

| 配置 | 耗时 |
|---|---|
| 192 组全活跃 | 5,154 us |
| 3 组活跃（行数塞满） | 3,730 us |
| 权重只有 3 个槽位（真实只覆盖 3,582 行） | **788 us** |

前两行差别小，是因为 grid 骨架由 `m 分块数`（448）主导；
第三行快，是因为**真的只算了 1/64 的行**。

满负荷实测 **559–774 TFLOP/s**（峰值的 48–67%），对分组矩阵乘属正常水平。
