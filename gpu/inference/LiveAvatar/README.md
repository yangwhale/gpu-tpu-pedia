# LiveAvatar on B200 — 实时音频驱动数字人

阿里 Quark 开源的 **LiveAvatar**（ECCV 2026 Spotlight）在 **8×B200** 上的完整跑通记录：
可复现的安装与运行步骤、五个必踩的坑、kernel 级性能分解，以及一个已验证有效的优化
（VAE 解码 **2.51×**）。

> **同名陷阱**：LiveKit 官方插件库里有个叫 `LiveAvatar` 的插件，那是 **HeyGen 的商业产品**
> （需要 `LIVEAVATAR_API_KEY`），跟本文的阿里开源项目**同名但毫无关系**。

## 目录

- [它是什么](#它是什么)
- [架构：五卡流水线](#架构五卡流水线)
- [安装](#安装)
- [运行](#运行)
- [五个坑](#五个坑)
- [性能实测](#性能实测)
- [优化：channels_last_3d](#优化channels_last_3d)
- [试过但无效的三条](#试过但无效的三条)
- [还没走的路](#还没走的路)

---

## 它是什么

| 项 | 内容 |
|---|---|
| 输入 | **一张参考图 + 一段音频**（+ 可选文字提示） |
| 输出 | 25 fps 视频，**时长与音频完全一致**（实测 9 条，误差在一帧内） |
| 基座 | `Wan-AI/Wan2.2-S2V-14B`（46 GB），Apache 2.0 |
| LoRA | `Quark-Vision/Live-Avatar`（1.3 GB），Apache 2.0 |
| 音频编码器 | `wav2vec2-large-xlsr-53-english` —— 名字带 english，**实测中文完全没问题** |
| 形象 | **完全自定义**，不是预设库。喂一张图即可，不用先录视频训练替身 |

**它驱动的是声学能量，不是音素。**实测喂鼾声进去，嘴一样张得很大（真人打鼾不会这样）。
好处是**语言无关**；代价是急速语音下口型幅度会被平滑。

## 架构：五卡流水线

官方叫 **TPP**（pipeline parallel）。**4 步采样被切到 4 张卡上，每张只算一步**，
latent 依次流过，第 5 张卡做 VAE 解码：

```
block ──▶ GPU0 ──▶ GPU1 ──▶ GPU2 ──▶ GPU3 ──▶ GPU4 ──▶ 帧
         step0    step1    step2    step3     VAE
         ←──────── 四级重叠，吞吐 = 最慢一级 ────────→
```

实测**确实在重叠**：同一块穿过四级的**延迟** 1.55 s，但**吞吐周期**只有 0.29 s。
代码里 `if i != dist.get_rank(): continue` 就是这个映射。

每块产出 **12 帧 = 0.48 s 视频**。

> 代码里自带一个瓶颈探测器：VAE 那级如果 `dist.recv` 在 10 ms 内返回，说明数据早就在等，
> 打印 `WARNING: VAE serves as a bottleneck!`。优化前 15% 的块命中。

## 安装

实测**整个环境 4 分 17 秒装完**，46 GB 基座模型下载只用 45 秒。

```bash
sudo apt-get install -y ffmpeg git-lfs build-essential && git lfs install
git clone --depth 1 https://github.com/Alibaba-Quark/LiveAvatar.git && cd LiveAvatar

# uv 建 py3.10 venv，比 conda 快一个数量级
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.10 .venv && source .venv/bin/activate

uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

# ⚠️ B200 是 Blackwell(sm_100) 不是 Hopper。README 推荐的 FlashAttention 3 那条 wheel
#    是给 H800/H200 的；Blackwell 装 FA2。实测 2.8.3 可用。
#    （但见坑 #6：主干 attention 走的是 cuDNN，FA2 几乎不进热路径）
uv pip install flash-attn==2.8.3 --no-build-isolation

uv pip uninstall deepspeed          # ⛔ 必须，见坑 #2

uv pip install "huggingface_hub[cli]"
hf download Wan-AI/Wan2.2-S2V-14B --local-dir ./ckpt/Wan2.2-S2V-14B
hf download Quark-Vision/Live-Avatar --local-dir ./ckpt/LiveAvatar
```

自检：

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"
# 期望 2.8.0+cu128 (10, 0)
.venv/bin/python -c "from transformers.models.opt.modeling_opt import OPTModel; print('import 链 OK')"
```

## 运行

```bash
export TORCHINDUCTOR_CACHE_DIR=/path/to/persistent/cache   # 跨次复用，省掉每次 ~34 s 编译预热
export ENABLE_COMPILE=true NCCL_DEBUG=WARN

CUDA_VISIBLE_DEVICES=0,1,2,3,4 .venv/bin/torchrun --nproc_per_node=5 --master_port=29102 \
  minimal_inference/s2v_streaming_interact.py \
  --task s2v-14B --ulysses_size 1 \
  --size "720*400" \                                       # ⛔ 面积有上限，见坑 #3
  --training_config liveavatar/configs/s2v_causal_sft.yaml \
  --offload_model False --convert_model_dtype \
  --prompt "<描述画面内容与光照风格，不是台词>" \
  --image  "<参考图，长宽比决定输出画幅>" \
  --audio  "<16 kHz 单声道 wav>" \
  --infer_frames 48 --load_lora --lora_path_dmd "Quark-Vision/Live-Avatar" \
  --sample_steps 4 --sample_guide_scale 0 \
  --num_clip 120 \                                         # 按音频长度给足：clip 数 ≈ 秒数 / 1.92
  --num_gpus_dit 4 --sample_solver euler --enable_vae_parallel \
  --ckpt_dir ckpt/Wan2.2-S2V-14B/ --fp8 \
  --save_file "/绝对路径/输出.mp4"                           # ⛔ 必须全路径带扩展名，见坑 #4
```

- `--prompt` 描述的是**画面**（人物、场景、光照、风格），不是要念的台词
- `--size` 是**面积预算**，输出长宽比跟着参考图走：竖图出 384×704，横图出 704×384
- `--num_clip` 给小了会**静默截断**音频

## 五个坑

### 坑 1 · 激活 venv 后不要再改 PATH
`source .venv/bin/activate` 之后又 `export PATH="$HOME/.local/bin:$PATH"`，
系统 `torchrun` 会盖掉 venv 的，用系统 python 跑，venv 里几百个包一个都看不见。
**症状是一串 `ModuleNotFoundError`，看着像依赖没装全。**
判据：看 traceback 里 site-packages 的 python 版本。用绝对路径 `.venv/bin/torchrun` 最稳。

### 坑 2 · `deepspeed` × `transformers` 循环 import（官方 requirements 的坑）
照 README 装完**直接跑不起来**：

```
transformers/modeling_utils.py:158        import deepspeed
  → deepspeed/runtime/hybrid_engine.py:26  transformers.models.opt.modeling_opt...
    → 回头要 PreTrainedModel，而 modeling_utils 还在初始化中
ImportError: cannot import name 'PreTrainedModel' from partially initialized module
```

deepspeed 在 requirements 里是给训练用的，而**训练代码还没开源**，推理不需要。卸掉即通。

> 迷惑性在于报错说「circular import」，像是 transformers 自己的 bug。
> 单独 `import transformers` 是好的，**得 `import transformers.modeling_utils` 才看得到第一因**。

### 坑 3 · `--size` 有面积上限
配 LiveAvatar LoRA 的管线里 **KV cache 长度写死成 3000**，面积大到 token 数超了就崩：

```
RuntimeError: The expanded size of the tensor (3000) must match the existing size (3640)
```

四组对照（每组只变一个量）：

| 参考图 | size | 实际输出 | 结果 |
|---|---|---|---|
| 官方样例 | `720*400` | 704×384 | ✅ |
| 自制横图 | `720*400` | 704×384 | ✅ |
| 自制竖图 | `720*400` | 384×704 | ✅ |
| 自制横图 | `704*384` | 704×384 | ✅ |
| 官方样例 | `832*480` | — | ❌ 3000 vs **3360** |
| 自制竖图 | `480*832` | — | ❌ 3000 vs **3640** |

**变量是面积，不是图片也不是构图** —— 官方自己的图换大尺寸照样炸。
另外 `720*400` 和 `704*384` 对同一张图 snap 到同一分辨率，其实是一档。

### 坑 4 · `--save_file` 传短名会写丢
源码是 `args.save_file = args.save_dir + args.save_file + suffix` ——
**字符串直接拼、中间没有分隔符**，且只在 `save_file is None` 时执行。
传 `--save_file out` 的结果是写到当前目录、文件名就叫 `out`、**没有 .mp4 扩展名**，
`--save_dir` 被完全忽略。传绝对路径 + 扩展名；要用 `--save_dir` 则**必须带结尾斜杠**。

### 坑 5 · `pkill -f` 会杀掉自己
`pkill -f s2v_streaming_interact` 里那串也在你自己的命令行里，ssh 会话当场被杀（退出码 255）。
写成字符类：`pkill -f "s2v_streami[n]g"`。

### 坑 6（认知坑）· FA2 装了也基本没用
`causal_model_s2v.py` 导的是 `wan_2_2.modules.attention`，里面 `cudnn_require()` 的条件是
「没有滑窗 且 head_dim ≤ 256」——**默认恒为真，所以主干 attention 一直走 cuDNN**。
profile 证实：热路径是 `cudnn_generated_fort_native_sdpa_sm100_flash_fprop`（32.9 ms / 40 次），
而 `flash_attn` 只有 4.04 ms / 52 次。

微基准（B200，40 头 ×128，bf16）说明为什么这是对的：

| seq len | FA2 | **cuDNN** | flash-SDPA | mem-efficient |
|---|---|---|---|---|
| 1560 | 0.155 ms | **0.056 ms** | 0.165 ms | 0.343 ms |
| 3000 | 0.486 ms | **0.172 ms** | 0.525 ms | 1.149 ms |
| 3640 | 0.666 ms | **0.251 ms** | 0.716 ms | 1.698 ms |

**Blackwell 上 cuDNN 比 FA2 快 2.8×** —— FA2 没有 sm_100 原生 kernel，cuDNN 9.x 有。

## 性能实测

⚠️ **下列全是纯视频生成时间，启动/加载/编译一概不计。**
数字人是常驻服务，那笔只在部署时付一次 —— 没人会为了出一段实时视频去等启动。
口径：在生成循环里量稳态每 block 周期，丢掉前 2 块预热。

| 每块（12 帧 = 0.48 s 视频） | 优化前 | 优化后 |
|---|---|---|
| VAE decode | 0.1930 s | **0.0912 s** |
| VAE → CPU 拷贝 | 0.0198 s | 0.0198 s |
| DiT 单级前向 | 0.206 s | 0.206 s |
| **每块周期** | 0.318 s | **0.290 s** |
| VAE 占周期 | 61% | 31% |
| **生成 1 s 视频需要** | 0.663 s | **0.604 s** |
| **实时倍数 / FPS** | 1.51× / 37.7 | **1.66× / 41.4** |

**显存**：DiT 4 张各 47.6 GB，VAE 1 张 38.8 GB。B200 单卡 183 GB ——
**显存远不是瓶颈**，卡数由流水线切分决定，不是被容量逼的。

对比官方宣称「多卡 H800 45 FPS」（= 1.80× 实时）：**B200 仍没跑赢**。
官方没写「多卡」具体几张、怎么切，对比不严格，但方向清楚：**买的算力没被这个实现吃下去。**

### kernel 级分解

| VAE 一次解码（189 ms，7,120 kernel） | 占比 |
|---|---|
| `triton_poi_fused__to_copy_cat_constant_pad_nd_convolution`（57 次） | 51.9% |
| `cudnn_convolution`（115 次） | 26.3% |
| cutlass `sm100_tensorop` implicit gemm（45 次） | 14.8% |
| **NCHW↔NHWC 布局转换（312 次）** | 8.6% |

| DiT 一次前向（206 ms，**26,617 kernel**） | 耗时 |
|---|---|
| cuDNN sm100 flash attention（40 次） | 32.9 ms |
| fp8 GEMM · nvjet Blackwell kernel（344 次） | 27.2 ms |
| **copy / mul / add / cat / elementwise（1,900+ 次）** | ~58 ms |
| fp8 absmax 量化（40 次） | 2.9 ms |

DiT 平均每 kernel **7.7 µs**（VAE 是 26.6 µs）—— 粒度碎 3.5 倍，
真正算数的只占 29%，纯搬数据占 28%。

## 优化：channels_last_3d

**唯一验证有效的一条。** 独立基准 **189.50 → 75.43 ms（2.51×）**，
流水线内 0.193 → 0.0912 s。

**原理**：cuDNN 的 3D 卷积内部就是 NDHWC。喂 NCDHW 进去，它每层前后各转一次布局 ——
profile 里 `nchwToNhwc` + `nhwcToNchw` **一次解码调 312 次**。改掉之后这些全消失，
而且卷积本身也能直接吃原生布局（所以收益远大于那 8.6%）。

补丁见 [`patches/channels_last_3d.py`](patches/channels_last_3d.py)，核心是：

```python
# ⚠️ 只能转 5D 权重 —— 对 1D/4D 张量调 channels_last_3d 会直接抛
#    "required rank 5 tensor"，整个 .to(memory_format=) 挂掉
for mod in self.vae.model.modules():
    for _, p in list(mod.named_parameters(recurse=False)):
        if p.dim() == 5:
            p.data = p.data.to(memory_format=torch.channels_last_3d)
```

解码输入也要转，否则每次调用还要转一遍：

```python
decode_latents = block_latents.unsqueeze(0).contiguous(memory_format=torch.channels_last_3d)
```

**整体只提升 8.8%**，因为瓶颈换人了：VAE 从 61% 降到 31%，现在卡在 DiT 每级 0.206 s。
理论下限 0.48 / 0.206 = **2.33× 实时**。**VAE 这条路已经榨到头。**

## 试过但无效的三条

| 方案 | 结果 | 为什么 |
|---|---|---|
| 去掉 Upsample 的 `.float()` | 189.55 ms（**零变化**） | 那个 fp32 cast 根本不在热路径上 |
| VAE 上 CUDA Graph | 187.47 ms（1%） | VAE 是大 kernel、GPU 受限，启动开销本来就不是问题 |
| DiT 上 `torch.compile(mode="reduce-overhead")` | **被 inductor 拒绝捕获** | 见下 |

DiT 的 CUDA Graph 被拒，日志里两个原因各 14 次：

```
skipping cudagraphs due to mutated inputs (2 instances)
skipping cudagraphs due to cpu device (arg0_1 / view_6)
```

`mutated inputs` 就是 **KV cache** —— 因果流式模型每块都要原地改它，
而 CUDA Graph 要求输入不可变。**架构冲突，不是开关问题。**
代价还是负的：前向从 0.206 s 变成 39 s（反复重编译）。

> ⚠️ **测量方法论上的一个坑，值得单独记。**
> 中途用子类做 Upsample 探针，**那个子类打断了 `torch.compile`**，
> 基线从 189 ms 变成 1035 ms。在那个瘸腿基线上，channels_last 测出来是「慢 5.4 倍」——
> **跟真相完全相反**。探针本身会改变被测对象；发现它是因为多看了一眼 baseline 对不上。
>
> 同理，**别拿采样进度条下吞吐结论**：那个条稳态 22 it/s 看着飞快，
> 但它只量 DiT 去噪一段，不含 VAE、音频特征、motion frame。
> 口径是 `生成循环耗时 ÷ 产出视频长度` —— 量整条生成链路，但**不含启动**。

## 还没走的路

| 路线 | 预期 | 成本 |
|---|---|---|
| **序列并行**：`--ulysses_size` 现在是 1，把 DiT 序列切到 2 卡 | **唯一有可能推到 3× 以上** | 要改拓扑：4 级 ×2 = 8 卡，VAE 得跟某级共卡（它只占 31%，可行） |
| 修 `cpu device` 那个编译打断点 | 减少 kernel 碎片，收益未知 | 低，属微优化 |

8 卡只用了 5 张，**3 张一直闲着** —— 最大的一块未动用资源。

## 相关

- TPU 侧移植准备：[`../../../tpu/LiveAvatar/`](../../../tpu/LiveAvatar/)
- 上游：<https://github.com/Alibaba-Quark/LiveAvatar> · 官方样片：<https://liveavatar.github.io/>
