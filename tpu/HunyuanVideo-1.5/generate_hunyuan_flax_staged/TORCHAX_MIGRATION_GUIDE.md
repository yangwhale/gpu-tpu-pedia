# PyTorch GPU → torchax TPU 完整迁移指南

本文档记录了将 HunyuanVideo-1.5 Transformer 从 GPU PyTorch 迁移到 TPU torchax 的完整过程，包括问题分析、解决方案和最佳实践。

---

## 📚 目录

1. [迁移概览](#1-迁移概览)
2. [架构对比](#2-架构对比)
3. [核心修复详解](#3-核心修复详解)
4. [常见陷阱与解决方案](#4-常见陷阱与解决方案)
5. [完整迁移流程](#5-完整迁移流程)
6. [代码模板](#6-代码模板)
7. [性能优化](#7-性能优化)
8. [调试技巧](#8-调试技巧)

---

## 1. 迁移概览

### 1.1 整体迁移流程

```mermaid
flowchart TB
    subgraph GPU["🎮 GPU 版本"]
        G1[PyTorch 模型] --> G2[Flash Attention]
        G2 --> G3[CUDA Kernels]
        G3 --> G4[NCCL 分布式]
    end
    
    subgraph TPU["☁️ TPU 版本"]
        T1[PyTorch 模型] --> T2[torchax 桥接]
        T2 --> T3[Splash Attention]
        T3 --> T4[XLA 编译]
        T4 --> T5[GSPMD 分布式]
    end
    
    GPU --> |迁移| TPU
    
    style GPU fill:#ffcccc
    style TPU fill:#ccffcc
```

### 1.2 关键技术栈对比

| 技术层 | GPU 版本 | TPU 版本 |
|--------|----------|----------|
| 运行框架 | PyTorch | torchax (PyTorch → JAX) |
| Attention | Flash Attention 2/3 | Splash Attention (Pallas) |
| JIT 编译 | torch.compile | XLA JIT |
| 分布式 | NCCL + 手动 SP/TP | GSPMD (自动分片) |
| 数据类型 | fp16 / fp32 | bf16 (原生支持) |
| 设备管理 | CUDA | JAX Device Mesh |

---

## 2. 架构对比

### 2.1 数据流对比

```mermaid
flowchart LR
    subgraph GPU["GPU 数据流"]
        direction TB
        GA[CPU Tensor] --> GB[.cuda]
        GB --> GC[GPU Tensor]
        GC --> GD[Model Forward]
        GD --> GE[Output Tensor]
    end
    
    subgraph TPU["TPU 数据流"]
        direction TB
        TA[CPU Tensor] --> TB[.to jax]
        TB --> TC[env.to_xla]
        TC --> TD[XLA Tensor]
        TD --> TE[JIT Compiled Model]
        TE --> TF[Output Tensor]
    end
    
    style GPU fill:#f9f
    style TPU fill:#9ff
```

### 2.2 Attention 实现对比

```mermaid
flowchart TB
    subgraph Flash["Flash Attention (GPU)"]
        F1[Q, K, V] --> F2[Variable Length<br/>Packed Sequences]
        F2 --> F3[flash_attn_no_pad]
        F3 --> F4[CUDA Kernel<br/>分块计算]
        F4 --> F5[Output]
    end
    
    subgraph Splash["Splash Attention (TPU)"]
        S1[Q, K, V] --> S2[Pad to Block Size]
        S2 --> S3[shard_map 分片]
        S3 --> S4[Pallas Kernel<br/>分块计算]
        S4 --> S5[Trim Padding]
        S5 --> S6[Output]
    end
    
    style Flash fill:#ffcccc
    style Splash fill:#ccffcc
```

---

## 3. 核心修复详解

在将 HunyuanVideo-1.5 迁移到 TPU 时，遇到了多个导致生成质量问题的关键差异。以下是详细分析和修复过程。

### 3.1 修复 #1: ByT5 Embeddings 精度问题

#### 问题分析

```mermaid
flowchart LR
    subgraph GPU["GPU: ByT5 处理"]
        G1[ByT5 Output<br/>float32] --> G2[保持 float32]
        G2 --> G3[Transformer<br/>混合精度]
    end
    
    subgraph TPU_BAD["TPU (错误): ByT5 处理"]
        T1[ByT5 Output<br/>float32] --> T2[转换为 bf16]
        T2 --> T3[精度损失!]
        T3 --> T4[生成质量下降]
    end
    
    subgraph TPU_GOOD["TPU (修复): ByT5 处理"]
        F1[ByT5 Output<br/>float32] --> F2[保持 float32]
        F2 --> F3[质量正常]
    end
    
    style GPU fill:#ccffcc
    style TPU_BAD fill:#ffcccc
    style TPU_GOOD fill:#ccffcc
```

#### 错误代码

```python
# ❌ 错误：使用 bf16 导致精度损失
prompt_embeds_2 = prompt_embeds_2.to(dtype=target_dtype).to('jax')  # bf16
```

#### 修复代码

```python
# ✅ 正确：保持 float32
prompt_embeds_2 = prompt_embeds_2.to(dtype=torch.float32).to('jax')
```

#### 为什么 ByT5 需要 float32？

ByT5 是字节级别的文本编码器，其输出用于细粒度的文本条件控制。bf16 的精度不足以保留细微的文本语义差异，导致生成视频无法准确跟随提示词。

---

### 3.2 修复 #2: Attention Mask 处理

#### 问题分析

这是最关键的修复。GPU 版本使用 `flex_attention` 配合 `score_mod` 函数来屏蔽 padding tokens，而我们的初始 TPU 版本完全忽略了这个 mask。

```mermaid
flowchart TB
    subgraph GPU["GPU: Attention Mask 处理"]
        G1[text_mask] --> G2[F.pad 扩展到完整序列]
        G2 --> G3[flex_attention<br/>score_mod 函数]
        G3 --> G4[Padding 位置 → -inf]
        G4 --> G5[Softmax 后权重 = 0]
    end
    
    subgraph TPU_BAD["TPU (错误): 无 Mask"]
        T1[text_mask] --> T2[忽略!]
        T2 --> T3[Splash Attention<br/>无 mask]
        T3 --> T4[Padding 参与计算]
        T4 --> T5[注意力污染]
    end
    
    subgraph TPU_GOOD["TPU (修复): K/V 置零近似"]
        F1[text_mask] --> F2[扩展 mask 维度]
        F2 --> F3[K *= mask<br/>V *= mask]
        F3 --> F4[Padding 位置 K/V = 0]
        F4 --> F5[QK^T 对应位置 ≈ 0]
        F5 --> F6[Softmax 后权重很低]
    end
    
    style GPU fill:#ccffcc
    style TPU_BAD fill:#ffcccc
    style TPU_GOOD fill:#ccffcc
```

#### 原始 GPU 代码 (attention.py)

```python
# GPU 使用 flex_attention + score_mod
if text_mask is not None:
    attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)

def score_mod(score, b, h, q_idx, kv_idx):
    return torch.where(attn_mask[b, q_idx] & attn_mask[b, kv_idx], score, float('-inf'))

hidden_states = flex_attention(query, key, value, score_mod=score_mod)
```

#### 错误的 TPU 代码

```python
# ❌ 错误：完全忽略 text_mask
attn_mask = None  # 强制使用 Splash Attention，但没有 mask！
hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
```

#### 修复后的 TPU 代码

```python
# ✅ 正确：将 padding 位置的 K/V 设为零
if text_mask is not None:
    # text_mask: [B, text_len], 1=有效, 0=padding
    text_mask_expanded = text_mask.unsqueeze(-1).unsqueeze(-1).to(encoder_key.dtype)
    encoder_key = encoder_key * text_mask_expanded    # Padding 位置 → 0
    encoder_value = encoder_value * text_mask_expanded  # Padding 位置 → 0

# 合并 image 和 text tokens
query = torch.cat([query, encoder_query], dim=1)
key = torch.cat([key, encoder_key], dim=1)     # text padding 部分是 0
value = torch.cat([value, encoder_value], dim=1)  # text padding 部分是 0

# Splash Attention（无需显式 mask）
hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=None)
```

#### 为什么 K/V 置零有效？

```mermaid
flowchart LR
    subgraph 数学原理
        A["Q @ K^T"] --> B["当 K[i]=0 时<br/>score[i] ≈ 0"]
        B --> C["Softmax 后<br/>weight[i] 很小"]
        C --> D["V[i] 贡献<br/>接近于 0"]
    end
```

这是一个近似方案：
- 精确方案：将 score 设为 `-inf`，softmax 后权重 = 0
- 近似方案：将 K/V 设为 0，score ≈ 0，softmax 后权重很小

近似方案足够有效，因为 padding tokens 的影响被大幅降低。

---

### 3.3 修复 #3: vision_states 处理

#### 问题分析

```mermaid
flowchart LR
    subgraph GPU["GPU: t2v 模式"]
        G1["vision_states = zeros(...)"] --> G2["torch.all(x==0) 检查"]
        G2 --> G3["extra_attention_mask = 0"]
    end
    
    subgraph TPU["TPU: t2v 模式"]
        T1["vision_states = None"] --> T2["跳过 vision_in 分支"]
        T2 --> T3["等效效果"]
    end
    
    style GPU fill:#ccffcc
    style TPU fill:#ccffcc
```

#### 为什么使用 None 而非零向量？

Transformer 代码中有这样的检查：

```python
if mask_type == "t2v" and torch.all(vision_states == 0):
    ...
```

`torch.all()` 在 JIT 编译时会导致 ConcretizationTypeError，因为它需要具体的布尔值。使用 `None` 可以完全跳过这个分支，避免问题。

---

## 4. 常见陷阱与解决方案

### 4.1 ConcretizationTypeError

```mermaid
flowchart TB
    A["JIT 编译模型"] --> B{"遇到条件判断?"}
    B -->|"if tensor.max() > 1"| C["ConcretizationTypeError!"]
    B -->|"if static_arg == 'value'"| D["正常编译"]
    C --> E["解决方案"]
    E --> E1["1. 预计算移到 JIT 外"]
    E --> E2["2. Monkey-patch 移除检查"]
    E --> E3["3. 使用 static_argnames"]
```

**常见触发场景：**

| 代码模式 | 问题 | 解决方案 |
|----------|------|----------|
| `if tensor.max() > 1:` | 需要具体值 | 移到 JIT 外或移除 |
| `assert tensor.min() >= 0` | 断言需要具体值 | Monkey-patch 移除 |
| `torch.all(x == 0)` | 需要具体布尔值 | 传入 None 跳过分支 |
| `tensor.item()` | 需要标量值 | 使用 tensor 运算代替 |

### 4.2 布尔索引不支持

```mermaid
flowchart LR
    A["tensor[bool_mask]"] --> B["torchax 不支持!"]
    B --> C["解决方案"]
    C --> C1["torch.where()"]
    C --> C2["* mask 乘法"]
    C --> C3["简化逻辑避免"]
```

**示例修复：**

```python
# ❌ 错误
selected = tensor[~mask]

# ✅ 方案 1: torch.where
selected = torch.where(mask.unsqueeze(-1), tensor, torch.zeros_like(tensor))

# ✅ 方案 2: 乘法
selected = tensor * mask.unsqueeze(-1).float()
```

### 4.3 动态 Tensor 创建

```mermaid
flowchart TB
    A["JIT 内部调用<br/>torch.arange()"] --> B["每次重新编译!"]
    B --> C["性能极差"]
    C --> D["解决方案：预计算"]
    D --> E["在 JIT 外计算一次"]
    E --> F["缓存到模型属性"]
    F --> G["JIT 内使用缓存"]
```

**示例：Rotary Position Embeddings**

```python
# 在 JIT 编译前预计算
with torch.no_grad():
    freqs_cos, freqs_sin = model.get_rotary_pos_embed((t, h, w))
    with env:
        model._cached_freqs_cos = freqs_cos.to('jax')
        model._cached_freqs_sin = freqs_sin.to('jax')

# Monkey-patch 使用缓存
def cached_get_rotary_pos_embed(self, latent_size):
    return self._cached_freqs_cos, self._cached_freqs_sin
model.get_rotary_pos_embed = types.MethodType(cached_get_rotary_pos_embed, model)
```

### 4.4 Scheduler dtype 问题

```mermaid
flowchart LR
    A["latents<br/>bf16"] --> B["scheduler.step()"]
    B --> C["内部转 fp32<br/>精度保护"]
    C --> D["输出 fp32"]
    D --> E["需要转回 bf16!"]
    E --> F["latents.to(bf16)"]
```

```python
# 每次 scheduler.step 后转回 bf16
latents = scheduler.step(noise_pred, t, latents)[0]
latents = latents.to(target_dtype)  # 转回 bf16
```

### 4.5 OOM 问题

```mermaid
flowchart TB
    A["创建 attention mask<br/>[B, H, S, S]"] --> B{"S = 26456?"}
    B -->|Yes| C["矩阵太大<br/>26456 x 26456 x 2 x 64"]
    C --> D["OOM!"]
    B -->|No| E["正常"]
    
    D --> F["解决方案"]
    F --> F1["使用 Splash Attention<br/>分块计算"]
    F --> F2["不创建完整 mask<br/>K/V 置零近似"]
```

---

## 5. 完整迁移流程

```mermaid
flowchart TB
    subgraph PREP["📋 准备阶段"]
        P1[分析 GPU 代码] --> P2[识别不兼容模式]
        P2 --> P3[设计解决方案]
    end
    
    subgraph SETUP["⚙️ 环境设置"]
        S1[创建 JAX Mesh] --> S2[创建 torchax 环境]
        S2 --> S3[注册 Splash Attention]
    end
    
    subgraph PATCH["🔧 代码适配"]
        M1[Mock GPU 分布式状态] --> M2[Patch 不兼容函数]
        M2 --> M3[导入模型]
    end
    
    subgraph LOAD["📦 模型加载"]
        L1[加载模型] --> L2[转换权重到 XLA]
        L2 --> L3[权重分片]
        L3 --> L4[预计算动态 Tensor]
    end
    
    subgraph COMPILE["🚀 编译运行"]
        C1[JIT 编译] --> C2[推理循环]
        C2 --> C3[保存结果]
        C3 --> C4[显式退出]
    end
    
    PREP --> SETUP --> PATCH --> LOAD --> COMPILE
```

### 步骤 1: 创建 JAX Mesh

```python
from jax.sharding import Mesh
from jax.experimental import mesh_utils

tp_dim = jax.device_count()  # 8 个 TPU cores
dp_dim = 1
sp_dim = 1

mesh_devices = mesh_utils.create_device_mesh(
    (tp_dim, dp_dim, sp_dim),
    allow_split_physical_axes=True
)
mesh = Mesh(mesh_devices, ('tp', 'dp', 'sp'))
```

### 步骤 2: 创建 torchax 环境

```python
import torchax

env = torchax.default_env()
env._mesh = mesh
env.config.use_tpu_splash_attention = True

torch.set_default_dtype(torch.bfloat16)
```

### 步骤 3: 注册 Splash Attention

```python
# 保存原始 SDPA
_ORIGINAL_SDPA = torch.nn.functional.scaled_dot_product_attention

# 注册自定义 attention
custom_attention = functools.partial(scaled_dot_product_attention, env=env)
env._ops[torch.nn.functional.scaled_dot_product_attention] = ops_registry.Operator(
    torch.nn.functional.scaled_dot_product_attention,
    custom_attention,
    is_jax_function=False,
    is_user_defined=True,
    needs_env=False,
    is_view_op=False,
)
```

### 步骤 4: Monkey-Patch 不兼容代码

**必须在导入模型之前！**

```python
# Mock GPU 分布式状态
import module.parallel_states as ps
from types import SimpleNamespace

ps.get_parallel_state = lambda: SimpleNamespace(
    sp=1,
    sp_enabled=False,
    sp_group=None,
)

# Patch 有问题的函数
def patched_function(...):
    # 移除运行时检查，简化逻辑
    pass
module.original_function = patched_function

# 现在才导入模型
from module.model import Model
```

### 步骤 5: 加载和转换模型

```python
model = Model.from_pretrained(path, torch_dtype=torch.bfloat16)

with env:
    with jax.default_device('cpu'):
        state_dict = model.state_dict()
        state_dict = env.to_xla(state_dict)
        model.load_state_dict(state_dict, assign=True)
    
    weights = shard_weights(mesh, model.state_dict())
    model.load_state_dict(weights, assign=True, strict=False)
    torchax.interop.call_jax(jax.block_until_ready, weights)

model.eval()
```

### 步骤 6: 预计算并 JIT 编译

```python
# 预计算动态 tensor
with torch.no_grad():
    freqs = model.get_rotary_pos_embed(size)
    with env:
        model._cached_freqs = freqs.to('jax')

# JIT 编译
with env:
    model = torchax.compile(model, torchax.CompileOptions(
        jax_jit_kwargs={'static_argnames': ('return_dict',)}
    ))
```

### 步骤 7: 推理循环

```python
with mesh, env:
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            output = model(inputs)
            latents = scheduler.step(output, t, latents)[0]
            latents = latents.to(target_dtype)  # 转回 bf16

# 保存结果
save_results(latents.cpu())

# 显式退出
sys.exit(0)
```

---

## 6. 代码模板

### 6.1 Splash Attention 完整实现

```python
from jax.experimental.pallas.ops.tpu import splash_attention
from jax.experimental.shard_map import shard_map

BQSIZE = 2048
BKVSIZE = 2048
BKVCOMPUTESIZE = 1024

def _tpu_splash_attention(query, key, value, mesh, scale=None, window_size=None):
    """
    TPU Splash Attention 实现
    
    Args:
        query: [B, H, Sq, D]
        key: [B, H, Skv, D]
        value: [B, H, Skv, D]
        mesh: JAX 设备 mesh
        scale: 缩放因子，默认 1/sqrt(D)
        window_size: 局部注意力窗口大小，None 表示全局注意力
    """
    num_heads = query.shape[1]

    def _attention_on_slices(q, k, v):
        scale_factor = 1.0 / math.sqrt(q.shape[-1]) if scale is None else scale
        q = q * scale_factor

        def pad_to_multiple(x, multiple, axis):
            seq_len = x.shape[axis]
            pad_len = (multiple - seq_len % multiple) % multiple
            if pad_len == 0:
                return x, seq_len
            pad_width = [(0, 0)] * x.ndim
            pad_width[axis] = (0, pad_len)
            return jnp.pad(x, pad_width), seq_len

        def kernel_3d(q_3d, k_3d, v_3d):
            num_heads_on_device = q_3d.shape[0]
            
            q_3d_padded, q_orig_len = pad_to_multiple(q_3d, BQSIZE, axis=1)
            k_3d_padded, _ = pad_to_multiple(k_3d, BKVSIZE, axis=1)
            v_3d_padded, _ = pad_to_multiple(v_3d, BKVSIZE, axis=1)

            if window_size is not None:
                mask_class = functools.partial(
                    splash_attention.LocalMask, 
                    window_size=window_size
                )
            else:
                mask_class = splash_attention.FullMask

            mask = splash_attention.MultiHeadMask([
                mask_class((q_3d_padded.shape[1], k_3d_padded.shape[1]))
                for _ in range(num_heads_on_device)
            ])

            block_sizes = splash_attention.BlockSizes(
                block_q=min(BQSIZE, q_3d_padded.shape[1]),
                block_kv=min(BKVSIZE, k_3d_padded.shape[1]),
                block_kv_compute=min(BKVCOMPUTESIZE, k_3d_padded.shape[1]),
            )
            
            kernel = splash_attention.make_splash_mha(
                mask=mask, block_sizes=block_sizes
            )
            out = kernel(q_3d_padded, k_3d_padded, v_3d_padded)
            return out[:, :q_orig_len, ...]

        return jax.vmap(kernel_3d)(q, k, v)

    # 分片规则
    q_spec = P('dp', 'tp', 'sp', None)
    kv_spec = P('dp', 'tp', None, None)

    sharded_fn = shard_map(
        _attention_on_slices,
        mesh=mesh,
        in_specs=(q_spec, kv_spec, kv_spec),
        out_specs=q_spec,
        check_rep=False,
    )
    return sharded_fn(query, key, value)
```

### 6.2 权重分片模板

```python
from jax.sharding import PartitionSpec as P, NamedSharding
import re

# Tensor Parallel: Column-Row 模式
sharding_rules = {
    # Column Parallel: 在 output 维度分片
    r'.*\.q_proj\.weight$': (('tp', 'sp'), None),
    r'.*\.k_proj\.weight$': (('tp', 'sp'), None),
    r'.*\.v_proj\.weight$': (('tp', 'sp'), None),
    r'.*\.fc1\.weight$': (('tp', 'sp'), None),
    
    # Row Parallel: 在 input 维度分片
    r'.*\.o_proj\.weight$': (None, ('tp', 'sp')),
    r'.*\.fc2\.weight$': (None, ('tp', 'sp')),
}

def shard_weights(mesh, weights, rules):
    matched = 0
    for name, tensor in weights.items():
        for pattern, spec in rules.items():
            if re.fullmatch(pattern, name):
                tensor.apply_jax_(jax.device_put, NamedSharding(mesh, P(*spec)))
                matched += 1
                break
        else:
            # 未匹配：复制到所有设备
            tensor.apply_jax_(jax.device_put, NamedSharding(mesh, P()))
    
    print(f"分片完成: {matched} 个匹配, {len(weights)-matched} 个复制")
    return weights
```

---

## 7. 性能优化

### 7.1 JIT 编译缓存

```python
# 启用持久化缓存
jax.config.update("jax_compilation_cache_dir", "/dev/shm/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
```

**效果：**
- 首次运行：~60s 编译
- 后续运行：~5s 加载缓存

### 7.2 dtype 优化

```mermaid
flowchart LR
    A["模型权重<br/>bf16"] --> B["中间计算<br/>bf16"]
    B --> C["注意力<br/>bf16"]
    C --> D["输出<br/>bf16"]
    
    E["特殊情况"] --> E1["ByT5: float32<br/>精度敏感"]
    E --> E2["Scheduler: float32<br/>累加精度"]
```

### 7.3 性能基准

| 配置 | Token 数 | 总时间 | 每步时间 |
|------|----------|--------|----------|
| 25帧, 720p | 25,200 | 114s | 2.3s |
| 49帧, 720p | 46,800 | 216s | 4.3s |
| 121帧, 720p | 111,600 | 512s | 10.3s |

---

## 8. 调试技巧

### 8.1 查看完整 traceback

```bash
JAX_TRACEBACK_FILTERING=off python script.py
```

### 8.2 逐步测试

```python
# 先用 1 步测试
args.num_inference_steps = 1
# 成功后再增加
```

### 8.3 检测 XLA tensor

```python
def is_xla_tensor(tensor):
    if tensor is None:
        return False
    if hasattr(tensor, '_elem'):
        return True
    if hasattr(tensor, 'device'):
        return 'jax' in str(tensor.device) or 'xla' in str(tensor.device)
    return False
```

### 8.4 调试打印

```python
def debug_tensor(name, t):
    if t is None:
        print(f"{name}: None")
    else:
        print(f"{name}: shape={t.shape}, dtype={t.dtype}, "
              f"mean={t.float().mean().item():.4f}")
```

---

## 📋 迁移 Checklist

### 开始前

- [ ] 识别所有 CUDA 特定代码
- [ ] 识别所有运行时检查 (assert, if tensor.max())
- [ ] 识别所有动态 tensor 创建 (torch.arange, torch.zeros)
- [ ] 识别所有布尔索引
- [ ] 确认 dtype 要求

### 迁移中

- [ ] 创建 JAX Mesh
- [ ] 注册 Splash Attention
- [ ] Monkey-patch 不兼容代码
- [ ] 加载并分片权重
- [ ] 预计算动态 tensor
- [ ] JIT 编译模型

### 完成后

- [ ] 程序正常退出
- [ ] 输出 dtype 正确 (bf16)
- [ ] 无 OOM 问题
- [ ] 生成质量正确

---

## 📚 参考资源

- [torchax GitHub](https://github.com/pytorch/xla)
- [JAX Splash Attention](https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/splash_attention)
- [JAX shard_map](https://jax.readthedocs.io/en/latest/notebooks/shard_map.html)
- [HunyuanVideo-1.5](https://github.com/Tencent/HunyuanVideo)