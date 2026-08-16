# knowledge — 领域知识

与 [PARAMS.md](../../../PARAMS.md) **同源**：前端渲染问号读 PARAMS.md，
agent 回答追问读这里。内容一致、形式不同。

| 文件 | 内容 |
|---|---|
| **`00-scope.md`** | **⚠️ 先读：这些知识什么时候不适用 + 三级置信度** |
| `01-hardware.md` | 硬件常数、AOT 可信度、成本对比 |
| **`02-silent-failures.md`** | **★ 会静默出错的组合与通用判据** |
| `03-moe-kernels.md` | 四条 kernel 分支、megablox 设计、耗时由什么决定 |
| `04-sharding.md` | 切哪个轴、跨卡量化收集的约束链、EP |
| `05-quantization.md` | 精度链、主权重为什么必须 fp32、校准方式 |
| `06-memory.md` | 显存怎么算、非单调、两种 OOM、batch 上限表 |
| `07-xla-flags.md` | flag 族依赖、通信能不能藏住、dvfs |
| **`08-profiling.md`** | **★ 读 profile 的五个陷阱** |
| **`09-verification.md`** | **★ 怎么验「还是同一个模型」** |
| `10-baselines.md` | 实测基线与已作废的数字 |
| **`11-tokamax-vs-native.md`** | **★ 两条 kernel 路线的特性/优劣/决策树（问得最多）** |

| `99-maintenance.md` | 知识怎么维护、什么时候必须回来改、推翻旧结论的规矩 |

**每条都必须带出处**（哪次实测、什么规模、什么日期）。
没有出处的数字不要写进来 —— 它会被当成事实引用出去。
