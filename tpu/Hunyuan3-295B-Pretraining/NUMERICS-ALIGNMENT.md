# 混元 3：GPU（Megatron）与 TPU（MaxText）逐段对齐清单

本页从两边代码出发，按数据流逐段回答四个问题：哪一步应该**一位不差**，哪一步**注定对不齐末位**，各用什么手段检查。

**对齐目标是 GPU。** 模型在 GPU 上已经训稳，迁移要的是 TPU 跑得和 GPU 一样，所以 GPU 就是标准答案：模块级、loss 级一律 TPU 直接对 GPU 比，门槛见[第 5 节](#5-门槛以-gpu-为标准答案)。FP64 不是目标，只在单个算子上当诊断工具，用来判断 TPU 对 GPU 的差是谁造成的。

> **状态**：这是读代码得出的对齐方案，还没有跑过。两边现有的跑法都是性能测试配置（见第 0 步），不能直接拿来比 loss。

代码基准：

- TPU：`yangwhale/maxtext` 分支 `hunyuan3`，commit `1afaa0f`；启动参数见 [QUICKSTART-v7.md](QUICKSTART-v7.md)
- GPU：Megatron-Bridge `HYV3Bridge` + Megatron-Core 上游 main；启动见 [07f-gb300-hy3-295b-gke](../../gpu/a4x-max/07-megatron-training/07f-gb300-hy3-295b-gke/README.md)。GB300 镜像里的 Megatron 版本**没有逐行核对**，下文行为以上游 main 为准

---

## 一页总表

| 段 | 两边做法 | 期望 | 检查手段 |
|---|---|---|---|
| token 数据 | 都要换成同一份真实数据 | **一位不差** | 逐字节比前 N 步 token；核对每台机器拿到的那一份 |
| 权重转换 | HF → Megatron / HF → MaxText，只有拆分、拼接、转置 | **一位不差** | 全量逐字节比，不许抽样；转过去再转回 HF 也要一位不差 |
| embedding 查表 | GPU gather；TPU `use_iota_embed=True`（one-hot 矩阵乘） | **一位不差** | one-hot 只乘 1 和 0，FP32 累加精确，结果等于直接查表；输出直接逐位比 |
| RMSNorm / QK norm | GPU：TE 融合 kernel（cuDNN 后端），FP32 算完舍一次；TPU 源码：FP32 归一化后舍成 BF16，再乘 BF16 缩放 | 末位对不齐 | 看 TPU 编译后 HLO 实际舍几次；降精度复现，目标逐位等于 GPU |
| RoPE | 都是前后两半配对（GPU `rotary_interleaved=False`，TPU `_rotate_half`），约定一致；频率算法不同 | 末位对不齐 | sin / cos 单独对比；降精度复现 |
| 所有矩阵乘 | 都是 BF16 乘、FP32 累加；切块和加法顺序不同 | 末位对不齐 | 逐层注入后对 GPU 比 rel_L2 / cosine（第 5 节门槛）；超线再用 FP64 判是谁的差 |
| attention | GPU cuDNN fused attention；TPU tokamax splash | 末位对不齐 | 同上；块大小不同，softmax 权重舍成 BF16 的位置不同 |
| 路由打分 | 公式逐行一致（见下） | 分数末位对不齐 | FP32 路径，对 GPU 比 rel_L2，能逐位就逐位 |
| **挑专家（top-8）** | 离散决策 | **大多数一致，几乎并列处可能不同** | 一致比例 + 不一致处的分差；router replay 切断累积 |
| 专家计算与合并 | GPU TE grouped GEMM（CuTe DSL）；TPU tokamax GMM | 末位对不齐 | 在 replay 下逐层比 |
| 路由偏置更新 | 都是数每个专家分到几个 token，按符号加减 0.001 | 选的专家一样时**一位不差**【推断】 | 逐位比每层 bias |
| cross entropy | 都在 FP32 里算；实现不同 | 末位对不齐 | 整网 loss 门（第 5 节） |
| 优化器一步 | 公式一样，状态精度当前不一样（见第 0 步） | 末位对不齐 | 从 GPU 的 FP32 主权重出发跑一步，逐参数比；选学习率大于零的那一步 |

---

## 0. 先统一配置

两边现有配置是为测性能写的，以下几处不统一就比，差出来的不是精度问题：

| 项 | GPU 现状 | TPU 现状 | 对齐时改成 |
|---|---|---|---|
| 路由 | `moe_router_force_load_balancing=True`（强制均匀，benchmark 专用） | 正常路由 | 两边都正常路由 |
| 数据 | mock 数据 + NullTokenizer | `dataset_type=synthetic` | 同一份真实数据、同一个 tokenizer |
| Adam 一阶动量 | Megatron 默认 `exp_avg_dtype=fp32`（qwen3 骨架 recipe 是否改过未核） | `mu_dtype=bfloat16` | 都用 FP32。一阶动量用 BF16，小更新会被抹回原值 |
| MTP | 对齐脚本 `--mtp-layers 0` | 模型配置 `mtp_num_layers: 1` | 先都关，主干对齐后再开 |
| FP8 | 对齐脚本有 FP8 / BF16 两档 | 生产配置有 FP8 | 先全 BF16 |
| 梯度通信精度 | `grad_reduce_in_fp32=False`（BF16） | `grad_dtype=bfloat16` | 一致，可保留 |
| 确定性 | `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0` | 编译期排定顺序，本来就确定 | 两边先各自重跑一次，量出自己的噪声 |

---

## 1. 应该一位不差的四段

**token、权重、embedding、路由偏置更新。** 这些地方要么不做计算（搬数据、拆拼权重），要么计算结果精确（one-hot 乘法只有乘 1、加 0），要么只数个数（偏置更新）。差一位就是 bug，不用讨论精度。

- **权重**：Megatron 把 QKV 按 KV 组交错拼成一个矩阵、把 gate 和 up 拼成 `linear_fc1`；MaxText 分开存。转换只做拆拼和转置，所以要求逐字节相等。另外核对专家偏置：两边都存 FP32（GPU `expert_bias` 强制 FP32；TPU `weight_dtype=float32`），转换脚本不能把它降成 BF16。
- **embedding**：TPU 开了 `use_iota_embed`，用 one-hot 矩阵乘代替查表。每个输出只有一个乘积非零，等于原值，所以和 GPU 的 gather 应该一位不差。不一致就查 token 或权重。
- **路由偏置更新**：只依赖每个专家分到的 token 数（整数），在 FP32 里按符号加减更新率。只要两边挑的专家完全一样，更新就应该一位不差【推断：两边更新公式在源码层面一致，未实测】。

检查：算哈希或全量逐位比。工具先喂一份故意改坏的数据，确认它会报错。

## 2. 注定对不齐末位的计算

### RMSNorm 与 QK norm

- TPU（`layers/normalizations.py`）：输入转 FP32 → 求均方 → `x * rsqrt(mean + eps)` → **舍成 BF16** → 乘 BF16 缩放系数。源码上舍两次。
- GPU：TE 融合 RMSNorm（`NVTE_NORM_FWD_USE_CUDNN=1`），整段在 FP32 里算完，出口舍一次。
- 源码上差一次舍入，但 XLA 融合后可能只舍一次。**看编译后的 HLO 定**，不要看源码定。eps 两边都是 `1e-5`。

### RoPE

- 配对约定一致：两边都是前一半和后一半对应位置配对。
- 频率算法不同：GPU 先算 `inv_freq = 1 / base^(2i/d)`，再 `torch.outer(位置, inv_freq)`，是**乘法**；TPU 先算 `timescale = base^(2i/d)`，再 `位置 / timescale`，是**除法**。数学相同，舍入不同，sin / cos 可能差末位。
- cos / sin 都先舍成 BF16 再乘（GPU 非融合写法 `rope_utils.py:133`，TPU `embeddings.py:361`）。如果 GPU 开了融合 RoPE，在 FP32 里算，这一点会不同（GB300 实际是否开启未核）。
- 混元 3 的 `rope_theta = 11158840`，低频分量的角度很小，对频率末位更敏感。

检查：单独比 sin / cos 表；用降精度复现，在 TPU 上照 GPU 的写法（先求倒数再乘）算一遍，应能和 GPU 一位不差。

### 矩阵乘与 attention

所有 BF16 矩阵乘两边都是 **BF16 乘、FP32 累加**，不存在「TPU 有 FP32 累加器、GPU 没有」。末位差来自：

1. 切块大小和加法顺序不同；
2. GPU 端 cuBLAS 拆 K 维并行时，中间结果允许低精度合并（PyTorch 默认允许，本环境未单独确认）；
3. attention 里 softmax 后的权重两边都先舍成 BF16 再乘 V，但块大小不同（TPU 用 2048 的块），舍入落点不同。

检查：逐层注入（把 GPU 上一层的输出喂给 TPU 这一层），对 GPU 比 rel_L2 和余弦，门槛见第 5 节。超线时再从同一组 BF16 输入算一份 FP64 真值：两边离真值一样近，是平台固有差异；TPU 明显更远，是 TPU 的问题。

## 3. 路由：写 FP32 保住的是哪一段

**打分公式两边逐行一致**：

```
s = sigmoid(logits)             # FP32
idx = top8(s + expert_bias)     # 选人用加了偏置的分数
w = s[idx]                      # 权重用不加偏置的原始分数
w = w / (sum(w) + 1e-20) * 2.826
```

GPU：`moe_utils.topk_routing_with_score_function`；TPU：`moe.py` 的 `deepseek_routing` + `deepseek_scale_weights`。

**路由矩阵乘的精度两边也一样**：

- TPU：`float32_gate_logits: True` 把输入和权重转成 FP32，但 `matmul_precision` 是默认档，矩阵单元把它们舍回 BF16 再乘，FP32 累加、FP32 输出。
- GPU：`moe_router_dtype=fp32`，但混合精度下权重被统一转成 BF16，hidden state 本来就是 BF16；`te_general_gemm(..., out_dtype=fp32)` 同样是 BF16 乘、FP32 累加、FP32 输出。

所以「路由写 FP32」保住的是**累加、输出分数、sigmoid 和挑前 8 名**，不是乘法本身：192 个分数挨得很近，存成 BF16 只剩两三位有效数字，名次会排错。真要乘法也是 FP32，TPU 要把精度开到 `highest`，GPU 要让权重保持 FP32 进入矩阵乘；只有在要和一个真 FP32 的参考对答案时才需要。

### 挑专家的检查方法

logits 和 sigmoid 都有末位差，所以挑出的专家不保证每个 token 都一样。不用「接近」来判，分两步：

1. **一致比例**：每层统计 top-8 集合完全一致的 token 占比；
2. **看不一致的那些**：计算它们第 8 名和第 9 名（加偏置后）的分差。都小到末位量级，是正常平台差异；有分差明显大于末位还挑错的，是 bug。

### 切断累积：router replay

Megatron 有 `RouterReplay`（`moe/router_replay.py`），能把挑选结果录下再回放。把 GPU 每层挑的专家喂给 TPU，两边走同一批专家，之后专家计算、合并和后续层的差才是它们自己的。

## 4. 推荐顺序

1. 按第 0 步统一配置，两边各自重跑一次，量出自己的噪声；
2. token、权重、embedding 逐位比，必须全过；
3. 小算子单测：RMSNorm、RoPE、路由打分，对 GPU 比，能降精度复现的做到逐位一致；
4. 逐层比前向，路由先看一致比例，再开 replay 逐层往下；
5. 整网第 0 步只跑前向；
6. 跑 1000 步，除主 loss 外逐步画每层路由偏置、梯度范数；
7. 主干过了，再开 MTP，最后开 FP8，FP8 按 FP8 对 BF16、FP8 对 FP8 分开比。

---

## 5. 门槛：以 GPU 为标准答案

以下门槛来自一次 GPU → TPU 大模型迁移的 CI 实践，可直接作为起点；在自己的模型上先量 GPU 重跑噪声再微调。

### 指标怎么算

记 a = TPU，b = GPU，逐元素对应。**指标本身在 FP64 里算**（相近的数相减精确，丢精度的是平方和与除法）。

| 指标 | 公式 | 说明 |
|---|---|---|
| rel_L2 | ‖a − b‖₂ / ‖b‖₂ | 逐对相减、平方、求和、开根号，除以 GPU 那组数平方求和再开根号。分母**不是元素个数，也不是值的总和**；等价于「差的均方根 ÷ 参考的均方根」 |
| cosine | a·b / (‖a‖₂ ‖b‖₂) | 只管方向；1 − cos ≈ ½·rel_L2²，不是独立证据，用来分辨「方向对、幅度错」 |
| \|Δmean\|/std | \|mean(a) − mean(b)\| / std(b) | 系统偏移 |
| ULP 距离 | BF16 位模式转单调整数后相减 | 回到原始 dtype 数；跨 0 无意义；近零设下限（参考值 < 0.0625 按 0.0625 算）；报 0 / 1 / 2–4 / 5–8 / 9–16 / 17+ 的分布 |
| loss 逐步相对差 | dₜ = (TPUₜ − GPUₜ) / GPUₜ | 判**带符号均值** mean(dₜ)，同报 MARE mean(\|dₜ\|)、第 0 步、最大单步 |

### 门槛

| 级别 | 判据 | 门槛 |
|---|---|---|
| 数据、权重 | 逐 bit，全量 | atol = 0 |
| MoE 模块（逐层注入） | rel_L2 + cosine + max_abs | rel_L2 < 0.01，cosine > 0.999，max_abs < 0.1 |
| attention 模块（逐层注入；原为 MLA 的门槛，Hy3 是 GQA，借用） | rel_L2 + 超容差比例 + 均值差 | rel_L2 < 0.01，超容差元素 < 1%，mean_diff < 0.01，允许 2 个 BF16 ULP |
| fused MoE 反向 | rel_L2 + 归一化最大差 | rel_L2 ≤ 2⁻¹⁰，归一化 max ≤ 2⁻⁷ |
| FP8 MoE（对 GPU FP8） | rel_L2 + cosine | rel_L2 ≤ 0.05，cosine ≥ 0.995 |
| 整网第 0 步 | loss 相对差 | BF16 0.025%–0.05% |
| 整网 1..N 步（BF16） | 带符号均值 | ≤ 0.02% |
| 整网 1..N 步（FP8） | 软门 + 硬限 | 软门 0.1%（只报警）；任一步 > 2% 或均值 > 1% 判失败；NaN/Inf 判失败 |

### 用法

- 模块门槛**只在逐层注入下有效**；整网串跑后几层必然超线，只用来找第一个台阶。
- 模块全过不等于 loss 对齐，两级都要过。
- **ULP 只诊断不当门槛**：跨芯片健康的累加顺序差常有几个到上百个 ULP，真 bug 有时只差 1–2 个。看 0 ULP 比例在哪一层突然下降。
- loss 门槛不能低于噪声：GPU 同配置重跑两遍可差 0.04% 量级，是 0.02% 门槛的两倍，贴着门槛的单次结果说明不了什么。
