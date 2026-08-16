# playbook：OOM 了怎么办

## 1. 先分清是哪一种

| 报错特征 | 含义 | 严重度 |
|---|---|---|
| `total memory required for HLO temporaries (X) exceeds available HBM (Y)` | 运行期临时缓冲超了 | 降 batch 通常能解决 |
| `CompileTimeHbmOom ... Exceeded hbm capacity by X` | **连排布方案都找不到** | 更严重，降 batch 未必线性有效 |
| `CompileTimeScopedVmemOom` | VMEM 不够，跟 tile 大小与 `scoped_vmem_limit_kib` 有关 | 调 tile 或提高上限 |

## 2. 二分找上限

**全在 CPU 上做，不占卡。** 从当前值开始向下二分，每档跑一次 AOT。

## 3. ⚠️ 不要假设「降 batch 一定省显存」

实测遇到过**非单调**：某配置在 batch=13 超 0.77 G，降到 12 反而超 1.26 G
（不同尺寸让编译器选了不同的排布方案）。**逐档实测，不要外推。**

## 4. 降 batch 之外的手段（按性价比排序）

1. **加宽 FSDP** —— 每卡常驻权重 ∝ 1/FSDP，收益最大
2. 检查是否被某个开关逼着用了更窄的 FSDP（如跨卡量化收集的整除约束）
3. 优化器状态精度（一阶动量、梯度降到 bf16）
4. 重算 / offload 策略
5. ❌ **不要动主权重精度** —— 会毁训练

## 5. 汇报格式

给出：差多少、二分出的上限、以及**上一档的实测吞吐**，让用户自己权衡。
