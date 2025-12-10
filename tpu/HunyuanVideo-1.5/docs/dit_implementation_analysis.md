# HunyuanVideo-1.5 DiT 实现对比分析

本文档深入对比分析两个 DiT (Diffusion Transformer) 实现的异同：
1. `dit_gpu.py` - 性能测试脚本
2. `generate_hunyuan_staged/stage2_transformer_explained.py` - 完整推理实现

---

## 📋 目录

1. [HunyuanVideo 1.5 整体架构详解](#hunyuanvideo-15-整体架构详解)
2. [概述对比](#概述对比)
3. [架构流程图](#架构流程图)
4. [核心组件对比](#核心组件对比)
5. [输入准备对比](#输入准备对比)
6. [执行流程对比](#执行流程对比)
7. [功能差异详解](#功能差异详解)
8. [设计理念分析](#设计理念分析)

---

## HunyuanVideo 1.5 整体架构详解

基于官方架构图，HunyuanVideo 1.5 采用多模态输入的扩散 Transformer 架构：

### 系统架构总览

```mermaid
flowchart TB
    subgraph inputs["🎯 多模态输入"]
        TEXT["📝 文本 Prompt<br/>'A drop of rich black ink falls...'"]
        IMG_REF["🖼️ 参考图像<br/>(可选, i2v 模式)"]
        IMG_COND["🎬 条件图像/视频<br/>(i2v 第一帧)"]
    end
    
    subgraph encoders["🔧 编码器层"]
        direction TB
        MLLM["<b>MLLM</b><br/>(LLaVA)<br/>多模态大语言模型"]
        BYT5["<b>Glyph ByT5</b><br/>字节级文本编码器"]
        SIGLIP["<b>SigLip</b><br/>视觉语言模型"]
        VAE_ENC["<b>VAE Encoder</b><br/>视频压缩编码"]
    end
    
    subgraph projectors["📐 投影层"]
        TOKEN_REF["Token Refiner<br/>(1000, 3584) → (1000, 3072)"]
        BYT5_PROJ["ByT5 Proj<br/>(256, 1472) → (256, 3072)"]
        VIS_PROJ["Vision Proj<br/>(729, 1152) → (729, 3072)"]
        PATCH_EMB["Patch Emb<br/>(T×H×W, 32) → (T×H×W, 3072)"]
    end
    
    subgraph transformer["🧠 Dual Stream Transformer"]
        direction TB
        ROPE["3D RoPE<br/>时空旋转位置编码"]
        ATTN["Self-Attention /<br/>Sparse-Attention"]
        MLP1["MLP (Text Stream)"]
        MLP2["MLP (Video Stream)"]
        BLOCKS["× 53 Blocks"]
    end
    
    subgraph output["📤 输出"]
        LATENTS["去噪 Latents<br/>(B, 16, T, H, W)"]
    end
    
    TEXT --> MLLM --> TOKEN_REF
    TEXT --> BYT5 --> BYT5_PROJ
    IMG_REF -.-> SIGLIP --> VIS_PROJ
    IMG_COND --> VAE_ENC --> PATCH_EMB
    
    TOKEN_REF --> transformer
    BYT5_PROJ --> transformer
    VIS_PROJ -.-> transformer
    PATCH_EMB --> transformer
    
    ROPE --> ATTN
    ATTN --> MLP1
    ATTN --> MLP2
    MLP1 --> BLOCKS
    MLP2 --> BLOCKS
    
    transformer --> LATENTS
    
    style inputs fill:#e3f2fd
    style encoders fill:#fff3e0
    style projectors fill:#f3e5f5
    style transformer fill:#e8f5e9
    style output fill:#fce4ec
```

### 四路输入详解

| 输入模态 | 编码器 | 投影层 | 输出维度 | 颜色标识 | 用途 |
|---------|--------|--------|----------|----------|------|
| **文本 (语义)** | MLLM (LLaVA) | Token Refiner | (1000, 3072) | 🟣 紫色 | 高层语义理解 |
| **文本 (字符)** | Glyph ByT5 | ByT5 Proj | (256, 3072) | 🟣 浅紫 | 精确字符渲染 |
| **参考图像** | SigLip | Vision Proj | (729, 3072) | 🔵 蓝色 | 视觉风格参考 (可选) |
| **视频/图像 Latent** | VAE Encoder | Patch Emb | (T×H×W, 3072) | 🟠 橙色 | 条件帧 + 噪声帧 |

### 编码器详细规格

#### 1. MLLM (多模态大语言模型)

```mermaid
flowchart LR
    subgraph mllm["MLLM Pipeline"]
        A["文本 Prompt"] --> B["LLaVA Tokenizer"]
        B --> C["LLaVA Model<br/>(~14GB)"]
        C --> D["prompt_embeds<br/>(1, 1000, 3584)"]
    end
    
    subgraph refiner["Token Refiner"]
        D --> E["LayerNorm"]
        E --> F["Linear(3584→3072)"]
        F --> G["时间步调制"]
        G --> H["(1, 1000, 3072)"]
    end
```

**特点**:
- 使用 LLaVA 作为语言理解骨干
- 输出 1000 个 tokens，每个 3584 维
- Token Refiner 包含时间步条件注入

#### 2. Glyph ByT5 (字节级编码器)

```mermaid
flowchart LR
    A["文本 Prompt"] --> B["UTF-8 字节序列"]
    B --> C["ByT5 Encoder<br/>(~5GB)"]
    C --> D["byt5_text_states<br/>(1, 256, 1472)"]
    D --> E["Linear Proj"]
    E --> F["(1, 256, 3072)"]
```

**为什么需要 ByT5?**
```
问题场景:
  Prompt: "生成带有 'HUNYUAN' 文字的海报"
  
  MLLM 理解: "某个品牌/名称的海报" (语义级别)
  ByT5 理解: H-U-N-Y-U-A-N 每个字符 (字节级别)
  
结果: 配合使用可以准确渲染文字
```

#### 3. SigLip (视觉语言模型)

```mermaid
flowchart LR
    A["参考图像<br/>224×224"] --> B["SigLip ViT<br/>(~400MB)"]
    B --> C["vision_states<br/>(1, 729, 1152)"]
    C --> D["Linear Proj"]
    D --> E["(1, 729, 3072)"]
```

**Token 数量解析**:
- 729 = 27 × 27
- 来源: 224 / patch_size(8) = 28，去掉 CLS 或边界 = 27

#### 4. VAE Encoder (视频压缩)

```mermaid
flowchart LR
    subgraph input["输入"]
        A["视频帧<br/>(B, 3, T, H, W)<br/>如 (1, 3, 121, 720, 1280)"]
    end
    
    subgraph vae["VAE 压缩"]
        B["时间压缩 4x"]
        C["空间压缩 16x"]
        D["通道扩展 3→32"]
    end
    
    subgraph output["输出"]
        E["Latent<br/>(1, 32, 31, 45, 80)"]
    end
    
    A --> B --> C --> D --> E
```

**压缩公式**:
```python
latent_frames = (video_frames - 1) // 4 + 1  # 121 → 31
latent_height = video_height // 16           # 720 → 45
latent_width = video_width // 16             # 1280 → 80
latent_channels = 32                         # 不是 16!
```

### Dual Stream Block 详解

这是 HunyuanVideo 1.5 的核心创新之一：

```mermaid
flowchart TB
    subgraph input["输入 Tokens"]
        TXT["Text Tokens<br/>(紫+浅紫)<br/>1256 tokens"]
        IMG["Video Tokens<br/>(橙)<br/>~111,600 tokens"]
    end
    
    subgraph prep["预处理"]
        TXT --> TXT_QKV["txt_qkv_proj"]
        IMG --> IMG_QKV["img_qkv_proj"]
        TXT_QKV --> TXT_Q["txt_q, txt_k, txt_v"]
        IMG_QKV --> IMG_Q["img_q, img_k, img_v"]
    end
    
    subgraph rope["3D RoPE"]
        ROPE_T["时间维度 RoPE"]
        ROPE_H["高度维度 RoPE"]
        ROPE_W["宽度维度 RoPE"]
        IMG_Q --> ROPE_T --> ROPE_H --> ROPE_W
    end
    
    subgraph attention["Joint Attention"]
        direction TB
        CAT_Q["Concat Q<br/>[img_q, txt_q]"]
        CAT_K["Concat K<br/>[img_k, txt_k]"]
        CAT_V["Concat V<br/>[img_v, txt_v]"]
        
        ATTN["Self-Attention<br/>或 Sparse-Attention"]
        
        CAT_Q --> ATTN
        CAT_K --> ATTN
        CAT_V --> ATTN
        
        ATTN --> SPLIT["Split Output"]
        SPLIT --> IMG_ATTN["img_attn"]
        SPLIT --> TXT_ATTN["txt_attn"]
    end
    
    subgraph dual_mlp["双流 MLP (独立权重)"]
        IMG_ATTN --> IMG_ADD1["⊕ Residual"]
        IMG_ADD1 --> IMG_MLP["img_mlp"]
        IMG_MLP --> IMG_ADD2["⊕ Residual"]
        
        TXT_ATTN --> TXT_ADD1["⊕ Residual"]
        TXT_ADD1 --> TXT_MLP["txt_mlp"]
        TXT_MLP --> TXT_ADD2["⊕ Residual"]
    end
    
    subgraph output["输出"]
        IMG_ADD2 --> IMG_OUT["Video Tokens<br/>(更新后)"]
        TXT_ADD2 --> TXT_OUT["Text Tokens<br/>(更新后)"]
    end
    
    ROPE_W --> CAT_Q
    TXT_Q --> CAT_Q
    
    style attention fill:#e8f5e9
    style dual_mlp fill:#fff3e0
```

### 为什么是 "Dual Stream"?

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 传统 Cross-Attention (如 Stable Diffusion):                             │
│                                                                          │
│   Text ─────────────────────────────────────────────────→ 不变           │
│                        ↓ (只作为 K,V)                                    │
│   Image ──── Q ───→ Attention ──→ 更新后的 Image                        │
│                                                                          │
│   问题: Text tokens 不会被更新，信息流是单向的                           │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ HunyuanVideo Dual Stream (Joint Attention + Dual MLP):                  │
│                                                                          │
│   Text ──┬── Q,K,V ──→ ┌──────────────┐ ──→ txt_attn ──→ txt_mlp ──→ 更新 │
│          │             │    Joint     │                                  │
│          └─────────────│  Attention   │─────────────────────────────────│
│   Video ─┬── Q,K,V ──→ │              │ ──→ img_attn ──→ img_mlp ──→ 更新 │
│          │             └──────────────┘                                  │
│                                                                          │
│   优势:                                                                  │
│   1. 双向交互: Text 和 Video 互相 attend                                │
│   2. 独立 MLP: 各自的特征变换 (不共享权重)                               │
│   3. 更强表达力: Text tokens 也会根据 Video 调整                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3D RoPE (旋转位置编码)

```mermaid
flowchart LR
    subgraph dims["三个维度"]
        T["时间 T<br/>dim: 16"]
        H["高度 H<br/>dim: 56"]
        W["宽度 W<br/>dim: 56"]
    end
    
    subgraph combine["组合"]
        T --> ROPE["3D RoPE<br/>total: 128"]
        H --> ROPE
        W --> ROPE
    end
    
    subgraph apply["应用"]
        ROPE --> Q["Query 旋转"]
        ROPE --> K["Key 旋转"]
    end
```

**rope_dim_list = [16, 56, 56]**:
- 时间维度: 16 维 (视频帧间的位置关系)
- 空间维度: 56 + 56 = 112 维 (2D 空间位置)
- 总计: 128 维 = head_dim

### Token 数量计算

```python
# 720p, 121 帧的例子
video_tokens = T_latent × H_latent × W_latent
             = 31 × 45 × 80
             = 111,600 tokens

# 文本 tokens
mllm_tokens = 1000   # LLaVA 输出
byt5_tokens = 256    # ByT5 输出
text_tokens = 1256   # 拼接后

# 总 tokens (一次 attention)
total_tokens = 111,600 + 1,256 = 112,856 tokens!
```

### 稀疏注意力 (Sparse Attention)

由于 token 数量巨大，HunyuanVideo 使用稀疏注意力:

```mermaid
flowchart TB
    subgraph full["Full Attention (不可行)"]
        A["112,856 × 112,856<br/>= 12.7 Billion 元素<br/>≈ 100GB 显存"]
    end
    
    subgraph sparse["Sparse Attention"]
        B["分块处理<br/>tile_size = [6, 8, 8]"]
        C["SSTA<br/>(Sparse Spatial-Temporal Attention)"]
        D["Top-K 采样<br/>ssta_topk = 4096"]
    end
    
    full --> |"优化"| sparse
```

### 完整 Transformer 结构

```
HunyuanVideo_1_5_DiffusionTransformer:
├── embedders:
│   ├── txt_in: SingleTokenRefiner (LLaVA → 3072)
│   ├── byt5_in: Linear (1472 → 3072)
│   ├── img_in: PatchEmbedder (32 → 3072)
│   └── time_in: TimestepEmbedder (用于 AdaLN)
│
├── double_blocks: × 53
│   ├── img_mod: ModulateDiT (时间步调制 for img)
│   ├── txt_mod: ModulateDiT (时间步调制 for txt)
│   ├── img_norm1/2: RMSNorm
│   ├── txt_norm1/2: RMSNorm
│   ├── img_attn_qkv: Linear (3072 → 3072×3)
│   ├── txt_attn_qkv: Linear (3072 → 3072×3)
│   ├── img_attn_proj: Linear (3072 → 3072)
│   ├── txt_attn_proj: Linear (3072 → 3072)
│   ├── img_mlp: MLP (3072 → 12288 → 3072)
│   └── txt_mlp: MLP (3072 → 12288 → 3072)
│
├── final_layer: Linear (3072 → 32×patch_size^3)
│
└── params: ~13B (DiT 部分)
```

### 条件注入机制 (AdaLN)

```mermaid
flowchart TB
    subgraph time["时间步编码"]
        T["timestep t"] --> TE["TimestepEmbedder"]
        TE --> VEC["vec (B, 3072)"]
    end
    
    subgraph mod["Modulation"]
        VEC --> IMGMOD["img_mod"]
        VEC --> TXTMOD["txt_mod"]
        
        IMGMOD --> |".chunk(6)"| IMG_PARAMS["shift1, scale1, gate1<br/>shift2, scale2, gate2"]
        TXTMOD --> |".chunk(6)"| TXT_PARAMS["shift1, scale1, gate1<br/>shift2, scale2, gate2"]
    end
    
    subgraph apply["应用"]
        X["输入 x"] --> NORM["LayerNorm(x)"]
        NORM --> MODULATE["x × (1 + scale) + shift"]
        MODULATE --> LAYER["Attention/MLP"]
        LAYER --> GATE["output × gate"]
        GATE --> RES["x + gated_output"]
    end
```

**AdaLN 公式**:
```python
# 调制 (before attention/mlp)
x_modulated = LayerNorm(x) * (1 + scale) + shift

# 门控 (after attention/mlp)
output = x + gate * layer_output
```

---

## 概述对比

### 文件定位

| 维度 | `dit_gpu.py` | `stage2_transformer_explained.py` |
|------|-------------|-----------------------------------|
| **目的** | 性能测试/基准测试 | 真实视频生成推理 |
| **代码行数** | ~426 行 | ~2779 行 |
| **复杂度** | 简单 | 复杂（含详细注释） |
| **是否生成视频** | ❌ 只测试速度/显存 | ✅ 生成真实视频 |
| **输入数据** | 随机张量 | 真实 Text Embeddings |
| **Scheduler** | 不使用 | FlowMatchDiscreteScheduler |
| **去噪循环** | ❌ 单次前向 | ✅ 多步迭代去噪 |

### 代码量对比

```
dit_gpu.py:
├── 性能测试函数: ~150 行
├── 模型加载: ~100 行
├── 工具函数: ~80 行
└── 命令行参数: ~50 行

stage2_transformer_explained.py:
├── 详细文档注释: ~1000 行
├── 核心推理逻辑: ~400 行
├── 辅助函数: ~300 行
├── 概念解释附录: ~200 行
└── 其他: ~100 行
```

---

## 架构流程图

### dit_gpu.py 执行流程

```mermaid
flowchart TB
    subgraph init["初始化阶段"]
        A[命令行参数解析] --> B[分布式环境初始化]
        B --> C[CUDA 设备设置]
    end
    
    subgraph model["模型加载"]
        D[确定模型路径] --> E[from_pretrained 加载]
        E --> F[移动到 GPU]
        F --> G[设置 eval 模式]
    end
    
    subgraph input["输入准备"]
        H[创建随机 latents] --> I[创建随机 text_states]
        I --> J[创建随机 mask]
        J --> K[构造 hidden_states]
        K --> L[准备 ByT5 embeddings]
    end
    
    subgraph test["性能测试"]
        M[record_time 开始计时] --> N[Transformer 前向传播]
        N --> O[torch.cuda.synchronize]
        O --> P[记录峰值显存]
        P --> Q{是否继续?}
        Q -->|是| M
        Q -->|否| R[统计结果]
    end
    
    init --> model --> input --> test
    
    style init fill:#e3f2fd
    style model fill:#fff3e0
    style input fill:#e8f5e9
    style test fill:#fce4ec
```

### stage2_transformer_explained.py 执行流程

```mermaid
flowchart TB
    subgraph init["初始化阶段"]
        A[模块级并行状态初始化] --> B[命令行参数解析]
        B --> C[推理状态初始化]
    end
    
    subgraph load["加载 Stage 1 输出"]
        D[加载 config.json] --> E[加载 embeddings.safetensors]
        E --> F[提取 LLaVA embeddings]
        F --> G[提取 ByT5 embeddings]
    end
    
    subgraph model["模型加载"]
        H[加载 Transformer] --> I[加载 Scheduler]
        I --> J[设置 flow_shift]
    end
    
    subgraph prepare["输入准备"]
        K[计算分辨率] --> L[同步随机种子]
        L --> M[设置时间步]
        M --> N[准备 text embeddings + CFG]
        N --> O[生成随机 latents]
        O --> P[准备 cond_latents + mask]
        P --> Q[准备 vision_states]
    end
    
    subgraph denoise["去噪循环"]
        R[for t in timesteps] --> S[拼接 latents + cond_latents]
        S --> T[CFG: 复制输入]
        T --> U[Scheduler 缩放]
        U --> V[处理 Meanflow timestep_r]
        V --> W[Transformer 前向传播]
        W --> X[CFG 公式应用]
        X --> Y[Scheduler.step 更新]
        Y --> Z{最后一步?}
        Z -->|否| R
        Z -->|是| AA[完成去噪]
    end
    
    subgraph save["保存输出"]
        BB[保存 latents.safetensors] --> CC[更新 config.json]
    end
    
    init --> load --> model --> prepare --> denoise --> save
    
    style init fill:#e3f2fd
    style load fill:#fff9c4
    style model fill:#fff3e0
    style prepare fill:#e8f5e9
    style denoise fill:#fce4ec
    style save fill:#e1bee7
```

### 两者的核心差异流程

```mermaid
flowchart LR
    subgraph dit_gpu["dit_gpu.py (性能测试)"]
        A1[随机输入] --> A2[1次前向传播]
        A2 --> A3[测量时间/显存]
    end
    
    subgraph stage2["stage2_transformer.py (推理)"]
        B1[真实 Embeddings] --> B2[N次迭代去噪]
        B2 --> B3[保存 Latents]
    end
    
    style dit_gpu fill:#ffebee
    style stage2 fill:#e8f5e9
```

---

## 核心组件对比

### 1. 模型加载方式

```python
# dit_gpu.py - 可选使用 SageAttention
transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_mode=attn_mode,  # "flash" 或 "sageattn"
).to(DEVICE)

# stage2_transformer_explained.py - 固定配置
transformer = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(
    transformer_path,
    torch_dtype=transformer_dtype,
    low_cpu_mem_usage=True,
)
```

### 2. 分布式初始化

```python
# 两者都使用相同的初始化模式
parallel_dims = initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
```

### 3. Scheduler 使用

| 特性 | `dit_gpu.py` | `stage2_transformer_explained.py` |
|------|-------------|-----------------------------------|
| Scheduler 类型 | 无 | FlowMatchDiscreteScheduler |
| 时间步设置 | 固定 t=999 | 动态 timesteps 序列 |
| flow_shift | 无 | 从 PIPELINE_CONFIGS 获取 |
| solver | 无 | euler |

---

## 输入准备对比

### dit_gpu.py 的输入构造

```mermaid
graph TD
    subgraph inputs["输入张量构造"]
        A["latents<br/>(B, 32, T_latent, H, W)<br/>随机噪声"]
        B["cond_latents<br/>(B, 32, T_latent, H, W)<br/>全零"]
        C["mask<br/>(B, 1, T_latent, H, W)<br/>全零"]
        
        A --> D["torch.cat([latents, cond_latents, mask])"]
        B --> D
        C --> D
        D --> E["hidden_states<br/>(B, 65, T_latent, H, W)"]
    end
    
    subgraph text["文本嵌入"]
        F["text_states<br/>(B, 1000, 3584)<br/>随机"]
        G["byt5_text_states<br/>(B, 256, 1472)<br/>全零"]
    end
    
    E --> H["Transformer"]
    F --> H
    G --> H
```

### stage2_transformer_explained.py 的输入构造

```mermaid
graph TD
    subgraph embeddings["真实 Embeddings (来自 Stage 1)"]
        A1["prompt_embeds<br/>(1, 1000, 3584)<br/>LLaVA 输出"]
        A2["negative_prompt_embeds<br/>(1, 1000, 3584)"]
        A3["prompt_embeds_2<br/>(1, 256, 1472)<br/>ByT5 输出"]
    end
    
    subgraph cfg["CFG 处理"]
        B1["torch.cat([negative, positive])"]
        A1 --> B1
        A2 --> B1
        B1 --> B2["(2, 1000, 3584)"]
    end
    
    subgraph latents["Latents 准备"]
        C1["prepare_latents()<br/>随机噪声"]
        C2["prepare_cond_latents()<br/>条件 + mask"]
        C1 --> C3["torch.cat([latents, cond])"]
        C2 --> C3
        C3 --> C4["(1, 33, T, H, W)"]
    end
    
    subgraph loop["去噪循环"]
        D1["for t in timesteps"]
        D2["CFG: cat x 2"]
        D3["Transformer(...)"]
        D4["CFG 公式"]
        D5["scheduler.step()"]
        
        C4 --> D1
        B2 --> D3
        D1 --> D2 --> D3 --> D4 --> D5
        D5 --> D1
    end
```

### 输入参数对比表

| 参数 | dit_gpu.py | stage2_transformer_explained.py |
|------|------------|----------------------------------|
| `hidden_states` | 随机 (B, 65, T, H, W) | 拼接后 (2B, 33, T, H, W) |
| `timestep` | 固定 999 | 动态 timesteps 序列 |
| `text_states` | 随机 | 真实 LLaVA embeddings |
| `text_states_2` | None | None (720p 模式) |
| `encoder_attention_mask` | 全 1 | 真实 prompt_mask |
| `byt5_text_states` | 全零 | 真实 ByT5 embeddings |
| `byt5_text_mask` | 全零 | 真实 mask |
| `timestep_r` | 不传 | Meanflow 下一步时间 |
| `vision_states` | 不传 | 全零 (t2v 模式) |
| `guidance` | 不传 | None |
| `return_dict` | False | False |

---

## 执行流程对比

### 单次调用 vs 迭代去噪

```mermaid
sequenceDiagram
    participant C as 调用者
    participant T as Transformer
    participant S as Scheduler
    
    Note over C,S: dit_gpu.py (性能测试)
    rect rgb(255, 235, 238)
        C->>T: forward(latents, t=999, ...)
        T-->>C: noise_pred
        Note right of C: 完成! 只测速度
    end
    
    Note over C,S: stage2_transformer_explained.py (推理)
    rect rgb(232, 245, 233)
        loop 50 步
            C->>T: forward(latents, t, ...)
            T-->>C: noise_pred
            C->>C: CFG 公式处理
            C->>S: step(noise_pred, t, latents)
            S-->>C: updated_latents
        end
        Note right of C: 保存 final_latents
    end
```

### CFG 处理差异

```python
# dit_gpu.py - 可选 CFG，简单复制
batch = 2 if enable_cfg else 1
# 如果启用，batch 翻倍，但不做 CFG 公式计算

# stage2_transformer_explained.py - 完整 CFG 实现
if do_classifier_free_guidance:
    # 1. 输入翻倍
    latent_model_input = torch.cat([latents_concat] * 2)
    
    # 2. 模型输出分离
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    
    # 3. 应用 CFG 公式
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
```

---

## 功能差异详解

### 1. Meanflow 支持

```mermaid
graph LR
    subgraph dit_gpu["dit_gpu.py"]
        A1[不支持 Meanflow]
        A2[不传 timestep_r]
    end
    
    subgraph stage2["stage2_transformer.py"]
        B1[检测 use_meanflow 配置]
        B2{是最后一步?}
        B3[timestep_r = 0]
        B4[timestep_r = timesteps[i+1]]
        
        B1 --> B2
        B2 -->|是| B3
        B2 -->|否| B4
    end
```

### 2. 多任务支持 (i2v vs t2v)

```python
# dit_gpu.py - 只测试 t2v 模式
cond_latents = torch.zeros(...)  # 全零
mask = torch.zeros(...)           # 全零

# stage2_transformer_explained.py - 支持 i2v
def prepare_cond_latents(task_type, image_cond, latents, multitask_mask):
    if image_cond is not None and task_type == 'i2v':
        # i2v: 第一帧是条件图像
        latents_concat = image_cond.repeat(1, 1, latents.shape[2], 1, 1)
        latents_concat[:, :, 1:, :, :] = 0.0  # 后续帧清零
    else:
        # t2v: 全零
        latents_concat = torch.zeros_like(latents)
```

### 3. 性能测量 vs 实际推理

```mermaid
graph TB
    subgraph dit_gpu["dit_gpu.py 测量模式"]
        A1[torch.cuda.synchronize] --> A2[记录开始时间]
        A2 --> A3[前向传播]
        A3 --> A4[torch.cuda.synchronize]
        A4 --> A5[记录结束时间]
        A5 --> A6[计算峰值显存]
        A6 --> A7[统计平均/最大/最小]
    end
    
    subgraph stage2["stage2_transformer.py 推理模式"]
        B1[time.perf_counter] --> B2[去噪循环]
        B2 --> B3[每10步打印GPU内存]
        B3 --> B4[记录总耗时]
        B4 --> B5[保存结果]
    end
```

### 4. SageAttention 支持

| 特性 | dit_gpu.py | stage2_transformer_explained.py |
|------|------------|----------------------------------|
| SageAttention | ✅ 可选 `--use_sage_attn` | ❌ 不支持 |
| attn_mode 参数 | `"flash"` 或 `"sageattn"` | 默认 |
| 动态检测 | ✅ 检测 SAGE_ATTN_AVAILABLE | ❌ 无 |

---

## 设计理念分析

### 1. dit_gpu.py 设计思想

```
目标: 快速、可重复地测量 DiT 模型性能
      ↓
设计决策:
├── 输入: 随机张量（无需真实数据）
├── 单次前向: 不需要完整去噪
├── 精确计时: CUDA synchronize 确保准确
├── 内存测量: reset_peak_memory_stats
├── 多次运行: 统计稳定性
└── 最小依赖: 不需要 text encoder, VAE 等
```

### 2. stage2_transformer_explained.py 设计思想

```
目标: 正确执行 Stage 2 推理，生成高质量 latents
      ↓
设计决策:
├── 三阶段分离: 节省内存，灵活调度
├── 真实 embeddings: 从 Stage 1 加载
├── 完整去噪循环: scheduler + CFG
├── 多任务支持: t2v / i2v
├── Meanflow: 改善时间一致性
├── SP 支持: 多 GPU 长视频生成
└── 详细文档: 便于理解和维护
```

### 3. 代码复用分析

```mermaid
graph TB
    subgraph shared["共享组件"]
        A[HunyuanVideo_1_5_DiffusionTransformer]
        B[initialize_parallel_state]
        C[torch.cuda.set_device]
    end
    
    subgraph dit_only["dit_gpu.py 独有"]
        D[record_time/record_peak_memory]
        E[SageAttention 检测]
        F[统计函数 print_results]
    end
    
    subgraph stage2_only["stage2_transformer.py 独有"]
        G[FlowMatchDiscreteScheduler]
        H[prepare_latents/prepare_cond_latents]
        I[get_task_mask]
        J[safetensors 加载/保存]
        K[CFG 完整实现]
    end
    
    A --> dit_only
    A --> stage2_only
    B --> dit_only
    B --> stage2_only
```

---

## 使用场景建议

### 使用 dit_gpu.py 当：

1. ✅ 需要快速评估模型在不同硬件上的性能
2. ✅ 测试新的 attention 优化（如 SageAttention）
3. ✅ 比较不同配置（帧数、分辨率）的性能
4. ✅ 不关心生成质量，只关心速度/显存

### 使用 stage2_transformer_explained.py 当：

1. ✅ 需要生成真实视频
2. ✅ 需要理解 HunyuanVideo 的推理机制
3. ✅ 需要在多 GPU 上生成长视频
4. ✅ 需要调试或修改推理逻辑

---

## 性能参考数据

### dit_gpu.py 典型输出

```
=== DiT 测试结果 (帧数: 121) ===
运行次数: 3

峰值显存 (MB):
  平均值: 28456.78
  最小值: 28432.12
  最大值: 28489.34

执行时间 (ms):
  平均值: 1523.45
  最小值: 1498.23
  最大值: 1567.89
```

### stage2_transformer_explained.py 典型输出

```
开始 Transformer 推理...
  使用 Meanflow: True
  使用 CFG: True
  SP 状态: sp_enabled=True, sp_size=8

  步骤 1/50
    GPU allocated: 28.45GB
  步骤 11/50
    GPU allocated: 28.67GB
  ...
  步骤 50/50
    GPU allocated: 28.52GB

✓ Transformer 推理完成，耗时: 264.32 秒
```

---

## 总结

| 方面 | dit_gpu.py | stage2_transformer_explained.py |
|------|------------|----------------------------------|
| **定位** | 性能基准测试 | 生产推理 |
| **复杂度** | 简单 | 复杂 |
| **输入** | 随机 | 真实 |
| **输出** | 性能指标 | 视频 latents |
| **适用场景** | 性能调优 | 视频生成 |
| **可读性** | 代码简洁 | 详细注释 |
| **扩展性** | 易于修改测试参数 | 易于理解完整流程 |

两个文件互为补充：
- `dit_gpu.py` 用于快速验证硬件性能
- `stage2_transformer_explained.py` 用于实际生成视频并学习架构