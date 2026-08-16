# XLA flags

## 必须整套照抄，**不能挑着开**（这一族有依赖关系）

这些 flag **有依赖关系**。实测漏掉 SparseCore 聚合器那一个，编译器直接拒绝：

```
INVALID_ARGUMENT: Latency hiding layer scheduler requires
                  sparse core collective aggregator to be enabled
```

## SparseCore 卸载族（8 个）

把集合通信卸载到 SparseCore，与 TensorCore 计算并行。
前面几个是「把某类通信交给 SparseCore」，**聚合器是真正在上面做规约的部件** ——
没有它，前面的卸载无处落地。

## 通信能不能藏住，取决于是谁插的

实测同一条权重 all-gather：

| 谁插的 | 每步暴露耗时 |
|---|---|
| 编译器按分片规格插（在权重刚读出来的地方） | **34.6 ms** |
| 手写在 kernel 入口（依赖链中间） | **6,170 ms** |

**178 倍。** 关键证据是执行次数：编译器插的那条**每步只发一次**
（80 层的收集被合并提升出了循环），手写的**每层发一次**，一层都提不出去。

> 展开见 `11-tokamax-vs-native.md`。

## dvfs_p_state

**默认 3**，合法 `[0,7]`，每档约 +2.4%，单调无拐点，**7 已顶格**。
实测从默认到 7：BF16 +8.6%、FP8 +8.0%，**显存一字节没涨**。零代价，必开。
