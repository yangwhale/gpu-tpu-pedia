# LiveAvatar on B200 · 照着做一次跑通

> 这份是**操作手册**：从一台干净的 8×B200 机器到出视频，照抄即可。
> 想知道每个结论怎么来的、以及走过哪些弯路 → [OPTIMIZATION-JOURNAL.md](OPTIMIZATION-JOURNAL.md)。
>
> 环境全部经 2026-09-17 在 `a4-highgpu-8g`（Ubuntu 24.04 / 驱动 580.126.09）实测。

## 0. 三件事先记住，能省你半天

| ⛔ 别做 | 会怎样 | 正确做法 |
|---|---|---|
| venv 带 `--system-site-packages` | pip 判「已满足」什么都不装。**包能 import，但入口脚本 / CUDA 配套库不在** | 建**干净** venv |
| 装 FlashAttention **3** | 五个 rank 一起崩：`flash-attention/hopper/...: no kernel image is available` | 装 **FA2 2.8.3** |
| 照 `requirements.txt` 装完就跑 | `import peft` 和 `from transformers import PreTrainedModel` 直接炸 | 装完**卸掉 deepspeed** |

三条都是「pip 返回 0、`import` 也成功、真跑才炸」。

## 1. 环境（一条命令）

```bash
bash worker-bootstrap.sh          # 幂等，可反复跑；--check 只体检
```

脚本在 [liveavatar-gateway/scripts/worker-bootstrap.sh](https://github.com/yangwhale/liveavatar-gateway/blob/main/scripts/worker-bootstrap.sh)，
做的事按顺序是：

1. clone fork `b200-realtime`
2. **干净 venv**（不共享 system site-packages）
3. `torch==2.8.0 torchvision==0.23.0 --index-url .../cu128`
4. `pip install -r requirements.txt`
5. **`pip uninstall deepspeed`**
6. `pip install flash-attn==2.8.3`（**不是 FA3**）
7. **装完真的 import 一遍**，不过就 `exit 1`
8. 后台拉权重（47 GB，两个 HF 仓库）

> 为什么是脚本不是手敲：worker 跑在 Spot 上，**实例说没就没**，回来是一台
> 全新机器。手敲的东西在这种机器上等于没装。

跑完必须看到 `VERIFY-PASS`。

## 2. 为什么是这几个版本

| 组件 | 版本 | 不这么配会怎样 |
|---|---|---|
| torch | **2.8.0 + cu128** | 2.9.x 上 transformers 4.51.3 在 `modeling_opt` 循环导入，`from transformers import PreTrainedModel` 直接 RuntimeError |
| flash-attn | **2.8.3** | FA3 的 wheel 是 **Hopper（sm_90）** 编的，B200 是 **Blackwell（sm_100）**，没有本卡机器码 |
| deepspeed | **卸掉** | `transformers/modeling_utils.py:158` 检测到它就 `import deepspeed` → 它反手取 `transformers.models.opt.modeling_opt` → 循环 |

### FA2 为什么还是得装（虽然主干走 cuDNN）

`wan_2_2/modules/attention.py` 的 `attention()` 确实 **cuDNN 优先**
（`cudnn_require()`）。但 **cross-attention 在 `model.py:175` 直接调
`flash_attention()`，绕过了那个入口**：

```
model.py:175  x = flash_attention(q, k, v, k_lens=context_lens)
                → attention.py:143  assert FLASH_ATTN_2_AVAILABLE   ← 一个 FA 都没装就死在这
```

所以三种配置的实测结果是：

| 配置 | 结果 |
|---|---|
| FA3 | ❌ `no kernel image`（Hopper kernel） |
| 一个都不装 | ❌ `AssertionError: FLASH_ATTN_2_AVAILABLE` |
| **FA2 2.8.3** | ✅ **唯一能跑的** |

> 想做到零 FA：把 `model.py:175`（以及 `causal_model_s2v.py:153/176`、
> `wan_base/modules/model.py` 那几处）的 `flash_attention(` 换成
> `attention(`，让它们也走 cuDNN 分支。**一行的事，但那是改上游行为，
> 要单独验数值和性能**，还没做。

## 3. 跑

```bash
# 五卡（4 DiT + 1 VAE）—— 官方脚本原样
bash infinite_inference_multi_gpu.sh

# 单卡 —— 必须降分辨率 + 裁历史，否则过不了实时线
export LA_TRIM_K=4
bash infinite_inference_single_gpu.sh      # 内含 --size "384*256"
```

## 4. 关键参数（查文档前先看这张表，别自己 grep）

| 参数 | 值 | 出处 / 坑 |
|---|---|---|
| `sample_fps` | **25** | `wan_2_2/configs/shared_config.py`。⚠️ `wan_base/configs/` 里是 **16**，**s2v-14B 不走那棵树** |
| `infer_frames` | **48** | 官方两个 `.sh` 都传 48（函数签名默认是 80，别被骗） |
| `num_frames_per_block` | 3 | 一个 block = 3 latent = **12 帧视频 = 0.48 s** |
| 音频嵌入帧率 | 30 Hz | wav2vec 出 50 Hz，`audio_encoder.py:86` 重采样到 30。**别拿 PCM 采样率换算** |
| 一个 clip | 48 帧 = **1.92 s** | 音频侧 48×(30/25) = 57.6 帧 ÷ 30 Hz = 1.92 s，两边对得上 |

## 5. 性能（实测，口径：纯生成，不含加载/编译）

| 路径 | 分辨率 | 实时倍数 |
|---|---|---|
| 单卡 + `LA_TRIM_K=4` | 384×256 | **1.357×** |
| 单卡 + `LA_TRIM_K=4` | 704×384 | 0.507× ❌ 过不了线 |
| 5 卡 TPP | 720×400 | **1.610×** |
| 5 卡 TPP | 384×256 | 1.687× |

**每卡效率：单卡 1.357 vs 5 卡 0.337，差 4.0 倍。**
同样 8 张卡，**跑 8 路独立会话 ≫ 跑 1 路快 5 倍**。

5 卡几乎不吃分辨率红利（像素砍到 36% 只快 4.8%）—— 它是**启动受限**，
DiT 一次前向 26,617 个 kernel，这个数不随分辨率变。

> 墙上时间会比这些数字差很多：`ENABLE_COMPILE=true` 时 dynamo 会一直编到
> 生成开始好几分钟之后。**第一次跑别拿总时长当性能**。

## 6. 两处本 fork 的改动

| 改动 | 效果 | 默认行为 |
|---|---|---|
| VAE 切 `channels_last_3d` | 解码 189.5 → **75.4 ms（2.51×）** | 纯布局优化，不改数值 |
| `LA_TRIM_K` 只喂必要历史 latent | 单卡 0.591× → **1.357×** | 不设或设 0 = 上游行为 |

`TRIM_K=4` 画质：与全量版逐帧 PSNR 27.7–39.0 dB，肉眼无差异。
差异来自解码结果沿 motion 反馈累积的运动相位分叉，**不是画质损失**。

## 7. 想要「流式出帧」要知道的

**官方两个入口都是批式的** —— 整段生成完返回 tensor、存 mp4：

| pipeline | `yield` | 谁在用 |
|---|---|---|
| `causal_s2v_pipeline`（单卡） | 0 | 官方入口 |
| `causal_s2v_pipeline_tpp`（5 卡） | 0 | 官方入口 |
| `causal_s2v_pipeline_tpp_blockwise` | **2** | **没有任何地方 import 它** |

唯一带流式出帧的 `blockwise` 是个**孤儿文件**。好消息是它的
**构造函数和 `generate()` 签名跟 `causal_s2v_pipeline_tpp` 逐字一致** ——
纯 drop-in，换个 import 就能用，区别只是它 `yield` 而不是 return。

它内部用的是 `vae_streaming.py` 的**带状态流式解码器**（`stream_decode`），
在 block 循环里逐块吐 RGB。

⚠️ **别想着把单卡也改成流式** —— 试过，**慢 3.7×**。单卡是延迟受限，
批量解码 GPU 利用率高得多，多算 9 倍反而更快。两条路径长得不一样
**是量出来的结果，不是疏忽**。

## 8. 排障速查

| 症状 | 真因 |
|---|---|
| `no kernel image is available` + 路径含 `flash-attention/hopper` | 装了 FA3。换 FA2 2.8.3 |
| `AssertionError` 在 `attention.py` 的 `assert FLASH_ATTN_2_AVAILABLE` | 一个 FA 都没装。cross-attention 绕过了 cuDNN 分支 |
| `from transformers import PreTrainedModel` → RuntimeError 循环导入 | deepspeed 没卸 |
| `ImportError: libcusparseLt.so.0` | venv 用了 `--system-site-packages`，cu128 配套库没装进来 |
| `hf` / `huggingface-cli` 命令不存在但包能 import | 同上 |
| 帧数 / 时长对不上 | `sample_fps` 拿成了 16。s2v-14B 是 **25** |
