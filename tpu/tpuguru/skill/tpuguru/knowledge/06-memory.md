# 显存

## 账怎么算

主要三块：**主权重 + 优化器状态**（∝ 1/FSDP）、**激活**（∝ batch × seq）、**临时缓冲**。

以 290 B 参数的 MoE 权重、fp32 主权重为例：

| FSDP | 每卡常驻（仅主权重） |
|---|---|
| 128 | **9.06 GB** |
| 64 | **18.12 GB** |

**加宽 FSDP 是省显存性价比最高的手段**，远大于调重算策略。

## ⚠️ 显存不随 batch 单调

实测：某配置 batch=13 超 0.77 G，**降到 12 反而超 1.26 G**。
不同尺寸让编译器选了不同的融合/排布方案。

**逐档实测或逐档问 AOT，不要外推。**

## 两种 OOM 要分清

| 报错 | 含义 | 好不好治 |
|---|---|---|
| `HLO temporaries (X) exceeds available HBM (Y)` | 运行期临时缓冲超 | 降 batch 通常有效 |
| `CompileTimeHbmOom ... Exceeded hbm capacity by X` | **连排布方案都找不到** | 更严重，未必线性 |
| `CompileTimeScopedVmemOom` | VMEM 不够 | 调 tile 或 `scoped_vmem_limit_kib` |

## 实测的 batch 上限（64 芯片，80 层，seq 4096）

| 配置 | 上限 | per-chip |
|---|---|---|
| BF16 | 13 | 666.6 |
| FP8 + `fixed` | **13** | **727.0** |
| FP8 + `absmax` | **11** | 670.8 |
| FP8 + 跨卡量化收集（FSDP 锁 64） | 7 | 677.0 |

**FP8 相对 BF16 只快约 9%** —— 因为权重那部分（fp32 主权重）不随计算精度缩小，
FP8 只压缩了激活和矩阵乘输入。
