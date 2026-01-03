# ComfyUI TPU Custom Node 开发指南

本文档介绍如何将 diffusers 的 torchax 优化模型 porting 成 ComfyUI custom node。

## 目录

1. [架构概述](#架构概述)
2. [文件结构](#文件结构)
3. [核心组件](#核心组件)
4. [开发步骤](#开发步骤)
5. [Torchax 算子注册](#torchax-算子注册)
6. [常见问题](#常见问题)

---

## 架构概述

### 工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                         ComfyUI                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │ Text Encoder │ → │ TPU Sampler  │ → │ TPU VAE Decoder      │ │
│  │    (CPU)     │   │   (TPU)      │   │      (TPU)           │ │
│  └──────────────┘   └──────────────┘   └──────────────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    TorchaxEnvManager                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ - 单例模式管理 torchax 环境                                  │ │
│  │ - 算子注册 (unflatten, rms_norm, group_norm, dropout...)   │ │
│  │ - pause()/resume() 用于模型加载                             │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                       TPU (JAX/XLA)                              │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ - 8 核 TPU v5e/v6e                                          │ │
│  │ - 权重分片 (Tensor Parallelism)                             │ │
│  │ - Splash Attention (长序列优化)                             │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 关键原则

1. **模型加载在 torchax 环境外进行** - safetensors 加载与 torchax 冲突
2. **单例环境** - 避免多次 enable/disable 导致状态不一致
3. **按需注册算子** - 根据模型需要添加缺失的 aten 算子

---

## 文件结构

```
ComfyUI-TPU/custom_nodes/ComfyUI-{ModelName}-TPU/
├── __init__.py           # 模块初始化，JAX/TPU 设置
├── nodes.py              # ComfyUI 节点定义
├── utils.py              # 工具函数（权重分片等）
├── splash_attention.py   # TPU Splash Attention（可选）
└── test_comfyui_api.py   # API 测试脚本
```

### `__init__.py`

```python
"""
ComfyUI {ModelName} TPU Nodes
"""
import os
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh

# JAX 编译缓存
jax_cache_dir = os.path.expanduser("~/.cache/jax_cache")
os.makedirs(jax_cache_dir, exist_ok=True)
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", jax_cache_dir)
print(f"✓ JAX 编译缓存: {jax_cache_dir}")

# 创建 TPU Mesh
try:
    devices = jax.devices('tpu')
    print(f"[{__name__}] Detected {len(devices)} TPU cores")
except:
    devices = jax.devices('cpu')
    print(f"[{__name__}] WARNING: No TPU detected, falling back to CPU")

tp_dim = len(devices)
mesh = Mesh(mesh_utils.create_device_mesh((tp_dim,), allow_split_physical_axes=True), ("tp",))
print(f"[{__name__}] Created Mesh: tp={tp_dim}")

# 注册 PyTree 节点（如果模型需要）
# from diffusers.xxx import XxxOutput
# jax.tree_util.register_static(XxxOutput)

# 导出节点
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
```

### `utils.py`

```python
"""
工具函数：权重分片、XLA 转换等
"""
import re
import jax
from jax.sharding import PartitionSpec as P, NamedSharding

# 权重分片规则（根据模型结构调整）
# 格式: { "正则表达式": PartitionSpec }
TRANSFORMER_SHARDINGS = {
    # 注意力权重沿头维度分片
    r".*\.attn\..*\.weight": P("tp", None),
    # FFN 权重
    r".*\.ff\.net\.0\.proj\.weight": P(None, "tp"),
    r".*\.ff\.net\.2\.weight": P("tp", None),
    # 其他权重复制
    r".*": P(),
}

VAE_DECODER_SHARDINGS = {
    r".*": P(),  # VAE 通常复制
}


def shard_weight_dict(params, sharding_rules, mesh):
    """对权重字典应用分片规则"""
    def get_sharding(name):
        for pattern, spec in sharding_rules.items():
            if re.match(pattern, name):
                return NamedSharding(mesh, spec)
        return NamedSharding(mesh, P())
    
    sharded_count, replicated_count = 0, 0
    sharded_size, replicated_size = 0, 0
    
    def shard_param(path, param):
        nonlocal sharded_count, replicated_count, sharded_size, replicated_size
        name = ".".join(str(p) for p in path)
        sharding = get_sharding(name)
        
        if hasattr(param, '_elem'):
            jax_param = param._elem
        else:
            return param
        
        size = jax_param.size * jax_param.dtype.itemsize / 1e9
        if sharding.spec == P():
            replicated_count += 1
            replicated_size += size
        else:
            sharded_count += 1
            sharded_size += size
        
        return jax.device_put(jax_param, sharding)
    
    result = jax.tree_util.tree_map_with_path(shard_param, params)
    print(f"  分片统计: {sharded_count} 个分片 ({sharded_size:.2f}GB), {replicated_count} 个复制 ({replicated_size:.2f}GB)")
    return result


def move_module_to_xla(env, module):
    """将 PyTorch 模块移动到 XLA"""
    with jax.default_device("cpu"):
        state_dict = env.to_xla(module.state_dict())
        module.load_state_dict(state_dict, assign=True)
```

---

## 核心组件

### TorchaxEnvManager（单例模式）

```python
class TorchaxEnvManager:
    """Torchax 环境的单例管理器"""
    _env = None
    _initialized = False
    
    @classmethod
    def get_env(cls, mesh_obj=None):
        """获取或创建全局 env"""
        if not cls._initialized:
            print("[TorchaxEnvManager] Initializing...")
            torchax.enable_globally()
            cls._env = torchax.default_env()
            _register_operators_on_env(cls._env, mesh_obj)
            cls._initialized = True
        return cls._env
    
    @classmethod
    def pause(cls):
        """临时暂停（用于模型加载）"""
        if cls._initialized:
            torchax.disable_globally()
    
    @classmethod
    def resume(cls):
        """恢复环境"""
        if cls._initialized:
            torchax.enable_globally()


class TorchaxContext:
    """Torchax 上下文管理器"""
    def __init__(self, mesh_obj=None):
        self.mesh = mesh_obj
    
    def __enter__(self):
        self.env = TorchaxEnvManager.get_env(self.mesh)
        return self
    
    def __exit__(self, *args):
        pass  # 默认不 disable
```

### ComfyUI 节点模板

```python
class MyTPUNode:
    """TPU 推理节点模板"""
    
    _cached_model = None
    _cached_model_id = None
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "height": ("INT", {"default": 1024}),
                "width": ("INT", {"default": 1024}),
                "steps": ("INT", {"default": 50}),
                "seed": ("INT", {"default": 42}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "model/path"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "TPU/MyModel"
    
    def generate(self, prompt, height, width, steps, seed, model_id="model/path"):
        model = self._get_or_create_model(model_id)
        
        with TorchaxContext() as ctx:
            # 推理逻辑
            result = model(...)
        
        return (result,)
    
    def _get_or_create_model(self, model_id):
        if self._cached_model is not None:
            return self._cached_model
        
        # 关键：加载模型前暂停 torchax
        was_initialized = TorchaxEnvManager._initialized
        if was_initialized:
            TorchaxEnvManager.pause()
        
        try:
            model = load_model(model_id)
        finally:
            if was_initialized:
                TorchaxEnvManager.resume()
        
        # 转换到 XLA 并编译
        with TorchaxContext() as ctx:
            move_module_to_xla(ctx.env, model)
            model = torchax.compile(model)
            model.params = shard_weight_dict(model.params, SHARDINGS, mesh)
        
        self._cached_model = model
        return model


# 注册节点
NODE_CLASS_MAPPINGS = {"MyTPUNode": MyTPUNode}
NODE_DISPLAY_NAME_MAPPINGS = {"MyTPUNode": "My TPU Node"}
```

---

## 开发步骤

### 1. 参考 diffusers torchax 实现

查看 `gpu-tpu-pedia/tpu/{Model}/generate_diffusers_torchax_staged/` 目录：

- `stage1_text_encoder.py` - 文本编码（通常 CPU）
- `stage2_transformer.py` - Transformer 采样（TPU）
- `stage3_vae_decoder.py` - VAE 解码（TPU）

### 2. 创建 custom node 目录

```bash
mkdir -p ComfyUI-TPU/custom_nodes/ComfyUI-{ModelName}-TPU
cd ComfyUI-TPU/custom_nodes/ComfyUI-{ModelName}-TPU
```

### 3. 复制并修改模板文件

从 `ComfyUI-Flux-TPU` 复制模板：

```bash
cp ../ComfyUI-Flux-TPU/__init__.py .
cp ../ComfyUI-Flux-TPU/utils.py .
cp ../ComfyUI-Flux-TPU/splash_attention.py .  # 如果需要
```

### 4. 修改 `utils.py` 中的分片规则

根据新模型的结构调整 `TRANSFORMER_SHARDINGS`。

### 5. 实现节点 (`nodes.py`)

1. 定义 `_register_operators_on_env()` 注册缺失算子
2. 实现各个节点类（Sampler、VAE Decoder、Text Encoder 等）
3. 注册节点到 `NODE_CLASS_MAPPINGS`

### 6. 测试

```bash
# 启动 ComfyUI
cd ComfyUI-TPU && python3 main.py --cpu --port 8189

# 运行测试
python3 custom_nodes/ComfyUI-{ModelName}-TPU/test_comfyui_api.py
```

---

## Torchax 算子注册

### 常见缺失算子

当遇到 `OperatorNotFound` 错误时，需要手动实现该算子：

```python
def _register_operators_on_env(env, mesh_obj):
    def override_op(op, impl):
        env._ops[op] = ops_registry.Operator(
            op, impl, is_jax_function=False, is_user_defined=True,
            needs_env=False, is_view_op=False,
        )
    
    # 示例：unflatten
    def unflatten_impl(input, dim, sizes, env=env):
        jinput = env.t2j_iso(input)
        # ... JAX 实现 ...
        return env.j2t_iso(result)
    
    override_op(torch.ops.aten.unflatten.int, 
                functools.partial(unflatten_impl, env=env))
```

### 已实现的算子列表

| 算子 | 用途 |
|------|------|
| `unflatten.int` | 维度展开 |
| `rms_norm` | RMS 归一化 |
| `layer_norm` / `native_layer_norm` | Layer 归一化 |
| `group_norm` / `native_group_norm` | Group 归一化（VAE） |
| `dropout` / `native_dropout` | Dropout（推理时直接返回） |
| `chunk` | 张量分块 |
| `cartesian_prod` | 笛卡尔积 |
| `conv2d` | 卷积 |
| `scaled_dot_product_attention` | 注意力（可用 Splash） |

### 算子实现模板

```python
def my_op_impl(input, ..., env=env):
    # 1. 转换到 JAX
    jinput = env.t2j_iso(input)
    
    # 2. JAX 实现
    result = jnp.xxx(jinput, ...)
    
    # 3. 转换回 torchax
    return env.j2t_iso(result)
```

---

## 常见问题

### Q1: 模型加载失败 / safetensors 错误

**原因**: torchax 环境启用时加载模型会冲突

**解决**: 加载前暂停 torchax
```python
TorchaxEnvManager.pause()
model = load_model(...)
TorchaxEnvManager.resume()
```

### Q2: OperatorNotFound 错误

**原因**: torchax 未实现该 aten 算子

**解决**: 在 `_register_operators_on_env()` 中添加实现

### Q3: TPU 被占用

**错误**: `The TPU is already in use by process with pid xxx`

**解决**: 杀掉占用进程
```bash
pkill -9 -f python
```

### Q4: 编译很慢

**原因**: JIT 编译需要时间

**解决**: 
- 设置 JAX 编译缓存: `JAX_COMPILATION_CACHE_DIR=~/.cache/jax_cache`
- 第二次运行会快很多

### Q5: 内存不足

**解决**: 
- 减小 batch size
- 使用更小的分辨率进行测试
- 确保权重正确分片

---

## Wan2.1 / Wan2.2 移植提示

参考 `gpu-tpu-pedia/tpu/Wan2.1/` 和 `gpu-tpu-pedia/tpu/Wan2.2/` 目录：

1. 查看 `generate_diffusers_torchax_staged/` 了解推理流程
2. 确认使用的 diffusers 模型类（如 `WanTransformer2DModel`）
3. 调整分片规则以匹配模型结构
4. 根据报错添加缺失算子

---

## 参考资源

- [torchax 文档](https://github.com/pytorch/xla/tree/master/torchax)
- [diffusers torchax 实现](https://github.com/huggingface/diffusers)
- [JAX 文档](https://jax.readthedocs.io/)
- [Splash Attention](https://github.com/jax-ml/jax/tree/main/jax/experimental/pallas/ops/tpu)
