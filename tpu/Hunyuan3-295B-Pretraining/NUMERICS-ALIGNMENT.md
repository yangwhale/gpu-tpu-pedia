# 混元 3：GPU（Megatron）与 TPU（MaxText）逐段对齐清单

本页从两边代码出发，按数据流逐段回答四个问题：哪一步应该**一位不差**，哪一步**注定对不齐末位**，各用什么手段检查。

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
| RMSNorm / QK norm | GPU：TE 融合 kernel（cuDNN 后端），FP32 算完舍一次；TPU 源码：FP32 归一化后舍成 BF16，再乘 BF16 缩放 | 末位对不齐 | 看 TPU 编译后 HLO 实际舍几次；降精度复现 |
| RoPE | 都是前后两半配对（GPU `rotary_interleaved=False`，TPU `_rotate_half`），约定一致；频率算法不同 | 末位对不齐 | sin / cos 单独对比；降精度复现 |
| 所有矩阵乘 | 都是 BF16 乘、FP32 累加；切块和加法顺序不同 | 末位对不齐 | FP64 第三方裁判 + 相对误差门槛 |
| attention | GPU cuDNN fused attention；TPU tokamax splash | 末位对不齐 | 同上；块大小不同，softmax 权重舍成 BF16 的位置不同 |
| 路由打分 | 公式逐行一致（见下） | 分数末位对不齐 | FP64 裁判 |
| **挑专家（top-8）** | 离散决策 | **大多数一致，几乎并列处可能不同** | 一致比例 + 不一致处的分差；router replay 切断累积 |
| 专家计算与合并 | GPU TE grouped GEMM（CuTe DSL）；TPU tokamax GMM | 末位对不齐 | 在 replay 下逐层比 |
| 路由偏置更新 | 都是数每个专家分到几个 token，按符号加减 0.001 | 选的专家一样时**一位不差**【推断】 | 逐位比每层 bias |
| cross entropy | 都在 FP32 里算；实现不同 | 末位对不齐 | FP64 裁判 |
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

检查：单独比 sin / cos 表；用降精度复现，在 TPU 上照 GPU 的写法（先求倒数再乘）算一遍，应能一位不差。

### 矩阵乘与 attention

所有 BF16 矩阵乘两边都是 **BF16 乘、FP32 累加**，不存在「TPU 有 FP32 累加器、GPU 没有」。末位差来自：

1. 切块大小和加法顺序不同；
2. GPU 端 cuBLAS 拆 K 维并行时，中间结果允许低精度合并（PyTorch 默认允许，本环境未单独确认）；
3. attention 里 softmax 后的权重两边都先舍成 BF16 再乘 V，但块大小不同（TPU 用 2048 的块），舍入落点不同。

检查：在 CPU 上用 FP64 算标准答案，看两边离它是不是一样近；用相对 L2 和余弦按层设门槛。

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
3. 小算子单测：RMSNorm、RoPE、路由打分，用 FP64 裁判，再做降精度复现；
4. 逐层比前向，路由先看一致比例，再开 replay 逐层往下；
5. 整网第 0 步只跑前向；
6. 跑 1000 步，除主 loss 外逐步画每层路由偏置、梯度范数；
7. 主干过了，再开 MTP，最后开 FP8，FP8 按 FP8 对 BF16、FP8 对 FP8 分开比。
