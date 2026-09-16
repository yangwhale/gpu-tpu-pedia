# LiveAvatar on TPU — v6e 移植准备

阿里 Quark 开源的 **LiveAvatar**（音频驱动数字人，Wan2.2-S2V-14B + LoRA）往 **TPU v6e**
的移植准备。参照 [`../Wan2.1`](../Wan2.1) / [`../Wan2.2`](../Wan2.2) 的做法。

> **现状：还没开始在 TPU 上跑。** 本页是移植前的结构分析和工作项清单 ——
> 把 GPU 侧已经量清楚的瓶颈翻译成 TPU 侧要解决的问题，避免上来就照搬。
>
> 📌 **对标目标与测量口径：[`B200-BASELINE.md`](B200-BASELINE.md)** ——
> 自带完整配置、分项拆解、kernel 级分解和三条测量禁忌。
> **量之前先读那一页的「测量口径」一节，否则数字不可比。**
> GPU 侧完整实测记录见 [`../../gpu/inference/LiveAvatar/`](../../gpu/inference/LiveAvatar/)。

## 为什么这个模型值得搬到 TPU

它跟我们已经搬过的 Wan 系列**共用底座**：基座就是 `Wan2.2-S2V-14B`。
`../Wan2.1` / `../Wan2.2` 里那套 torchax 分阶段执行、Splash Attention、
VAE 参数从 config 动态加载的做法**可以直接复用**，不是从零开始。

差异在于 LiveAvatar 多出来的三样：

1. **因果 + KV cache**：不是一次性出整段，而是按 block 流式滚动
2. **4 步采样的流水线并行**（官方叫 TPP）：4 步被切到 4 个 device 上，每个只算一步
3. **音频条件注入**：`wav2vec2` 特征在 40 层里的 12 层注入（`audio_inject_layers`）

## GPU 侧量到了什么（决定 TPU 侧该先打哪）

实测在 5×B200 上（4 卡 DiT + 1 卡 VAE），每块 12 帧 = 0.48 s 视频：

| 组件 | 每块耗时 | 占比 | 性质 |
|---|---|---|---|
| DiT 单级前向 | 0.206 s | **主瓶颈** | 26,617 个 kernel，平均 7.7 µs；算数只占 29%，搬数据占 28% |
| VAE decode（已优化） | 0.091 s | 31% | 原 0.193 s，靠 `channels_last_3d` 降下来的 |
| VAE → host 拷贝 | 0.020 s | — | 不是问题 |

⚠️ **GPU 侧那个最有效的优化（`channels_last_3d`）在 TPU 上大概率无意义** ——
它解决的是 cuDNN 内部 NDHWC 与外部 NCDHW 之间来回转换（一次解码 312 次）。
XLA 有自己的 layout assignment，不存在这个中间层。**不要照搬，要重新量。**

⚠️ **CUDA Graph 那条路在 GPU 上被 KV cache 的原地修改堵死了**
（`skipping cudagraphs due to mutated inputs`）。TPU 侧对应的问题是
**donation / aliasing**：滚动 KV cache 必须走 `donate_argnums` 或 `input_output_aliases`，
否则每步都会复制一整份 cache。这是 TPU 侧要提前设计的，不是事后调优。

## 工作项

### 一 · 结构移植（先让它能跑）

- [ ] 复用 `../Wan2.2` 的三阶段拆分（Encoder / Transformer / Decoder）与 safetensors 中间缓存
- [ ] LoRA 合并：GPU 侧是 `--load_lora --lora_path_dmd` 运行时加载，
      TPU 侧建议**离线 merge 进权重**再转，避免多一层 indirection
- [ ] `wav2vec2` 音频编码器：一次性预处理，不进 TPU 主循环
- [ ] **CausalConv3d 的流式 cache**：GPU 侧靠 `cat` + `constant_pad_nd` 拼历史帧
      （占 VAE 解码 51.9%）。TPU 上要改成固定形状的 ring buffer —— **变长 cat 会触发反复重编译**
- [ ] fp8：GPU 侧用 `--fp8`（`_scaled_mm` + absmax）。v6e 上先跑 bf16 基线，
      量化另起一轮（参考 `../DeepSeek-V3.2-Training` 的口径）

### 二 · 并行策略（决定能不能实时）

GPU 侧是 **4 级流水线 × 1 卡/级**，`ulysses_size=1`，序列没切。
v6e 上有两条路，**需要实测才能定**：

- **A. 照搬流水线**：4 步切到 4 个 device。好处是跟 GPU 侧同构，
  坏处是每级只有一个 device，单级延迟 0.206 s 这个数在 v6e 上会更难看
- **B. 换成序列/张量并行**：v6e 的 ICI 拓扑更适合在单步内切 —— dim 5120、40 层、
  40 头，切头或切序列都有空间。**GPU 侧 `ulysses_size` 从来没试过 >1，
  所以这条路两边都没有数据**

> 先跑 A 拿基线，再用 B 对比。别一上来就上 B。

### 三 · Attention

GPU 侧热路径是 cuDNN 的 sm_100 flash kernel（32.9 ms / 40 次）。
TPU 侧对应 `../Wan2.2/splash_attention_utils.py` 里那套 Splash Attention。
**但 LiveAvatar 是因果 + KV cache + attention sink 的组合**，
不是 Wan2.2 那种一次性全序列 attention —— mask 形态不一样，
`splash_attention` 的 block mask 要重新构造。

### 四 · 判据

搬完之后拿这几个数跟 GPU 侧对：

完整对标表、配置对齐项和测量口径见 **[`B200-BASELINE.md`](B200-BASELINE.md)**。核心两个数：

| 指标 | GPU 侧实测（5×B200） | TPU v6e |
|---|---|---|
| ★ 生成 1 s 视频耗时 | **0.604 s** | 待测 |
| ★ 实时倍数 | **1.66×** | 待测 |
| 每块周期（12 帧 = 0.48 s 视频） | 0.290 s | 待测 |

**全部是纯生成时间，启动/加载/编译不计** —— 常驻服务只在部署时付一次。

成功档位：**能用** ≥1.0× ｜ **追平** ≥1.66× ｜ **超过** ≥2.33×（B200 当前并行策略的理论上限）。
长时稳定性也要复验 —— B200 侧已验证 118 秒连续生成身份零漂移。

⚠️ **量的是整条生成链路，不是某一段。**
GPU 侧踩过这个坑：采样进度条稳态 22 it/s 看着飞快，但那只量 DiT 去噪，
不含 VAE、音频特征、motion frame，据此得出的「比实时快」结论是错的。
口径是 `生成循环耗时 ÷ 产出视频长度`。

## 相关

- GPU / B200 侧完整记录：[`../../gpu/inference/LiveAvatar/`](../../gpu/inference/LiveAvatar/)
- 同底座的已有移植：[`../Wan2.1`](../Wan2.1) · [`../Wan2.2`](../Wan2.2)
- 上游：<https://github.com/Alibaba-Quark/LiveAvatar> · 官方样片：<https://liveavatar.github.io/>
