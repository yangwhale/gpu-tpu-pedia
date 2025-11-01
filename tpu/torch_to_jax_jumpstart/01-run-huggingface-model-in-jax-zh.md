# 如何在 JAX 中运行 Hugging Face 模型（第一部分）

Hugging Face 最近从其 `transformers` 库中移除了对 JAX 和 TensorFlow 的原生支持，旨在精简其代码库。这一决定让许多 JAX 用户感到困惑：如何在不重新实现所有功能的情况下，继续利用 Hugging Face 丰富的模型集合？

本文探讨了一个解决方案：使用 JAX 输入运行基于 PyTorch 的 Hugging Face 模型。这种方法为依赖 Hugging Face 模型的 JAX 用户提供了一个宝贵的"出路"。

## 背景与方法

作为 [torchax](https://github.com/pytorch/xla/tree/master/torchax) 的作者——一个专为 JAX 和 PyTorch 之间无缝互操作而设计的新兴库，这次探索正好可以作为 `torchax` 的一次优秀压力测试。让我们深入了解！

-----

## 环境设置

我们将从标准的 Hugging Face 快速入门设置开始。如果您还没有设置环境，请执行以下操作：

```bash
# 创建虚拟环境 / conda 环境；激活等
pip install huggingface-cli
huggingface-cli login # 设置您的 Hugging Face token
pip install -U transformers datasets evaluate accelerate timm flax
```

接下来，直接从最新的开发版本安装 `torchax`：

```bash
pip install torchax
pip install jax[tpu] # 如果使用 GPU，请使用 jax[cuda12] https://docs.jax.dev/en/latest/installation.html
```

### 环境说明

- **huggingface-cli**: Hugging Face 的命令行工具，用于登录和管理模型
- **transformers**: Hugging Face 的核心库，包含大量预训练模型
- **torchax**: PyTorch 和 JAX 之间的互操作层
- **jax[tpu]** 或 **jax[cuda12]**: 根据您的硬件选择合适的 JAX 版本

-----

## 第一次尝试：Eager 模式

让我们从实例化一个模型和分词器开始。创建一个名为 `jax_hg_01.py` 的脚本，包含以下代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import jax # 导入 jax，稍后使用

# 加载 PyTorch 模型和分词器
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="bfloat16", device_map="cpu")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 对输入进行分词，请求返回 JAX 数组
model_inputs = tokenizer(["The secret to baking a good cake is "], return_tensors="jax")
print(model_inputs)
```

### 关键要点

注意分词器调用中的关键参数 `return_tensors="jax"`。这会指示 Hugging Face 直接返回 JAX 数组，这对于我们使用 JAX 输入运行 PyTorch 模型的目标至关重要。

运行上述脚本将输出：

```
{'input_ids': Array([[    1,   450,  7035,   304,   289,  5086,   263,  1781,   274,
         1296,   338, 29871]], dtype=int32), 'attention_mask': Array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=int32)}
```

可以看到，我们得到的是 JAX 的 `Array` 类型，而不是 PyTorch 的 `Tensor`。

### 将 PyTorch 模型转换为 JAX 可调用函数

现在，让我们使用 `torchax` 将这个 PyTorch 模型转换为 JAX 可调用函数。修改您的脚本如下：

```python
import torchax
# ... (之前的代码)

weights, func = torchax.extract_jax(model)
```

`torchax.extract_jax` 函数做了两件事：
1. 将模型的 `forward` 方法转换为与 JAX 兼容的可调用函数
2. 将模型的权重作为 JAX 数组的 Pytree 返回（本质上是将 `model.state_dict()` 转换为 JAX 数组）

### 调用转换后的函数

有了 `func` 和 `weights`，我们现在可以调用这个 JAX 函数了。调用约定是：
- 第一个参数：`weights`（权重）
- 第二个参数：位置参数的元组（`args`）
- 第三个参数（可选）：关键字参数的字典（`kwargs`）

让我们在脚本中添加调用：

```python
# ... (之前的代码)

print(func(weights, (model_inputs.input_ids, )))
```

执行此代码将产生以下输出，展示了成功的 eager 模式执行：

```
In [2]: import torchax

In [3]: weights, func = torchax.extract_jax(model)
WARNING:root:Duplicate op registration for aten.__and__

In [4]: print(func(weights, (model_inputs.input_ids, )))
CausalLMOutputWithPast(loss=None, logits=Array([[[-12.950611  ,  -7.4854484 ,  -0.42371067, ...,  -6.819363  ,
          -8.073828  ,  -7.5583534 ],
        [-13.508438  , -11.716616  ,  -6.9578876 , ...,  -9.135823  ,
         -10.237023  ,  -8.56888   ],
        [-12.8517685 , -11.180469  ,  -4.0543456 , ...,  -7.9564795 ,
         -11.546011  , -10.686134  ],
        ...,
        [ -2.983235  ,  -5.621302  ,  11.553352  , ...,  -2.6286669 ,
          -2.8319468 ,  -1.9902805 ],
        [ -8.674949  , -10.042385  ,   3.4400458 , ...,  -3.7776647 ,
          -8.616567  ,  -5.7228904 ],
        [ -4.0748825 ,  -4.706395  ,   5.117742  , ...,   6.7174563 ,
           0.5748794 ,   2.506649  ]]], dtype=float32), past_key_values=DynamicCache(), hidden_states=None, attentions=None)
```

### 传递关键字参数

如果要向函数传递关键字参数（kwargs），只需将它们作为第三个参数添加：

```python
print(func(weights, (model_inputs.input_ids, ), {'use_cache': False}))
```

虽然这展示了基本功能，但 JAX 的真正威力在于其 **JIT 编译**。即时编译（JIT）可以显著加速计算，特别是在 GPU 和 TPU 等加速器上。因此，我们的下一步是对函数应用 `jax.jit`。

-----

## JIT 编译 - 处理 Pytree

在 JAX 中，JIT 编译就像用 `jax.jit` 包装您的函数一样简单。让我们试试：

```python
import jax
# ... (之前的代码)

func_jit = jax.jit(func)
res = func_jit(weights, (model_inputs.input_ids,))
```

运行此代码很可能会导致 `TypeError`：

```
The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/hanq_google_com/learning_machine/jax-huggingface/script.py", line 18, in <module>
    res = func_jit(weights, (model_inputs.input_ids,))
TypeError: function jax_func at /home/hanq_google_com/pytorch/xla/torchax/torchax/__init__.py:52 traced for jit returned a value of type <class 'transformers.modeling_outputs.CausalLMOutputWithPast'>, which is not a valid JAX type
```

### 理解错误

错误消息表明 JAX 不理解 `CausalLMOutputWithPast` 类型。当您对函数应用 `jax.jit` 时，JAX 要求所有输入和输出都是"JAX 类型"——这意味着它们可以使用 `jax.tree.flatten` 被展平成 JAX 理解的元素列表。

### Pytree 系统

**Pytree（Python Tree）** 是 JAX 中的核心概念。Pytree 是嵌套的数据结构（如元组、列表和字典），JAX 可以遍历并对其应用转换。通过注册自定义类型，我们告诉 JAX 如何将其分解为组成部分（children）并重新构建它。

要解决这个问题，我们需要向 **JAX 的 Pytree 系统**注册这些自定义类型。在脚本中添加以下内容：

```python
from jax.tree_util import register_pytree_node
from transformers import modeling_outputs

def output_flatten(v):
  """将输出对象展平为元组"""
  return v.to_tuple(), None

def output_unflatten(aux, children):
  """从元组重建输出对象"""
  return modeling_outputs.CausalLMOutputWithPast(*children)

register_pytree_node(
  modeling_outputs.CausalLMOutputWithPast,
  output_flatten,
  output_unflatten,
)
```

### 注册 Pytree 的工作原理

- `output_flatten`: 定义如何将对象转换为其内部组件的元组
- `output_unflatten`: 定义如何从这些组件重建对象
- `register_pytree_node`: 将这两个函数注册到 JAX 的 Pytree 系统

### 第二个 Pytree 错误

然而，再次运行脚本后，您会遇到类似的错误：

```
Traceback (most recent call last):
  File "/home/hanq_google_com/learning_machine/jax-huggingface/script.py", line 33, in <module>
    res = func_jit(weights, (model_inputs.input_ids,))
TypeError: function jax_func at /home/hanq_google_com/pytorch/xla/torchax/torchax/__init__.py:52 traced for jit returned a value of type <class 'transformers.cache_utils.DynamicCache'> at output component [1], which is not a valid JAX type
```

同样的 Pytree 注册技巧也适用于 `transformers.cache_utils.DynamicCache`：

```python
from transformers import cache_utils

def _flatten_dynamic_cache(dynamic_cache):
  """展平 DynamicCache 对象"""
  return (
      dynamic_cache.key_cache,
      dynamic_cache.value_cache,
  ), None

def _unflatten_dynamic_cache(aux, children):
  """重建 DynamicCache 对象"""
  cache = cache_utils.DynamicCache()
  cache.key_cache, cache.value_cache = children
  return cache

register_pytree_node(
  cache_utils.DynamicCache,
  _flatten_dynamic_cache,
  _unflatten_dynamic_cache,
)
```

通过这些注册，我们解决了 Pytree 类型问题。但是，又会出现另一个常见的 JAX 错误：

```
jax.errors.ConcretizationTypeError: Abstract tracer value encountered where concrete value is expected: traced array with shape bool[]
This occurred in the item() method of jax.Array
The error occurred while tracing the function jax_func at /home/hanq_google_com/pytorch/xla/torchax/torchax/__init__.py:52 for jit. This concrete value was not available in Python because it depends on the value of the argument kwargs['use_cache'].

See https://docs.jax.dev/en/latest/errors.html#jax.errors.ConcretizationTypeError
```

-----

## 静态参数

### 理解 ConcretizationTypeError

这个 `ConcretizationTypeError` 是 JAX 的经典问题。当您对函数应用 `jax.jit` 时，JAX 会追踪其执行以构建计算图。在追踪期间，它将所有输入视为 *tracers*（追踪器）——值的符号表示——而不是它们的具体值。

错误的产生是因为代码中的条件语句 `if use_cache and past_key_values is None:` 试图读取 `use_cache` 的实际布尔值，但这在追踪期间是不可用的。

### 解决方案

有两种主要方法可以解决这个问题：

1. 在 `jax.jit` 中使用 `static_argnums` 显式告诉 JAX 哪些参数是编译时常量
2. 使用**闭包（closure）**将常量值"烘焙"进去

对于这个例子，我们将演示闭包方法。我们将定义一个新函数来封装常量 `use_cache` 值，然后对该函数进行 JIT 编译：

```python
import time
# ... (之前的代码，包括 jax.tree_util 导入和 pytree 注册)

def func_with_constant(weights, input_ids):
  """封装了常量 use_cache 值的函数"""
  res = func(weights, (input_ids, ), {'use_cache': False}) # 将 use_cache 作为固定值传递
  return res

jitted_func = jax.jit(func_with_constant)
res = jitted_func(weights, model_inputs.input_ids)
print(res)
```

### 为什么使用闭包？

闭包允许我们在函数定义时"捕获"常量值。这样，JAX 在追踪时就知道 `use_cache` 是一个常量，不会尝试将其作为变量处理。

运行此更新后的脚本最终会产生预期的输出，与我们的 eager 模式结果匹配：

```
CausalLMOutputWithPast(loss=Array([[[-12.926737  ,  -7.455758  ,  -0.42932802, ...,  -6.822556  ,
          -8.060653  ,  -7.5620213 ],
        [-13.511845  , -11.716769  ,  -6.9498663 , ...,  -9.14628   ,
         -10.245605  ,  -8.572137  ],
        [-12.842418  , -11.174898  ,  -4.0682483 , ...,  -7.9594035 ,
         -11.54412   , -10.675278  ],
        ...,
        [ -2.9683495 ,  -5.5914016 ,  11.563716  , ...,  -2.6254666 ,
          -2.8206763 ,  -1.9780521 ],
        [ -8.675585  , -10.044738  ,   3.4449315 , ...,  -3.7793014 ,
          -8.6158495 ,  -5.729558  ],
        [ -4.0751734 ,  -4.69619   ,   5.111123  , ...,   6.733637  ,
           0.57132554,   2.524692  ]]], dtype=float32), logits=None, past_key_values=None, hidden_states=None, attentions=None)
```

### 验证 JIT 编译的性能优势

我们已经成功地将 PyTorch 模型转换为 JAX 函数，使其与 `jax.jit` 兼容，并执行了它！

JIT 编译函数的一个关键特征是其性能特性：
- **第一次运行**：由于编译而较慢
- **后续运行**：由于使用缓存的编译图而显著更快

让我们通过计时几次运行来验证这一点：

```python
for i in range(3):
  start = time.time()
  res = jitted_func(weights, model_inputs.input_ids)
  jax.block_until_ready(res) # 确保计算完成
  end = time.time()
  print(i, end - start, 'seconds')
```

### 性能结果

在 Google Cloud TPU v6e 上，结果清楚地展示了 JIT 的优势：

```
0 4.365400552749634 seconds
1 0.01341700553894043 seconds
2 0.013022422790527344 seconds
```

**性能分析**：
- 第一次运行耗时超过 4 秒（包括编译时间）
- 后续运行在毫秒内完成（仅执行时间）
- 加速比约为 **300 倍**！

这就是 JAX 编译的威力！

### 完整示例

此示例的完整脚本可以在附带仓库中的 `jax_hg_01.py` 中找到。

-----

## 深入理解：Pytree 注册的重要性

让我们更详细地了解为什么需要 Pytree 注册：

### 什么是 Pytree？

Pytree 是 JAX 用来表示嵌套数据结构的概念。例如：
- 简单的 Pytree: `(1, 2, 3)`
- 嵌套的 Pytree: `{'a': [1, 2], 'b': {'c': 3}}`
- 包含 JAX 数组的 Pytree: `(jnp.array([1, 2]), {'x': jnp.array([3, 4])})`

### JAX 如何处理 Pytree

当您调用 `jax.jit` 或其他 JAX 转换时，JAX 需要：
1. **展平（Flatten）**：将嵌套结构转换为平面的叶子列表
2. **转换（Transform）**：对每个叶子应用操作
3. **重建（Unflatten）**：将结果重新组合成原始结构

### 自定义类型的问题

默认情况下，JAX 只知道如何处理基本的 Python 类型（tuple、list、dict）和 JAX 数组。对于自定义类（如 `CausalLMOutputWithPast`），JAX 不知道如何：
- 提取其内部数据
- 重新构建对象

### 注册过程

通过注册 Pytree 节点，我们教会 JAX：

```python
# 展平：告诉 JAX 如何提取数据
def output_flatten(v):
  # 返回 (children, aux_data)
  # children: 需要被 JAX 处理的数据
  # aux_data: 不需要被转换的元数据
  return v.to_tuple(), None

# 重建：告诉 JAX 如何重新组合
def output_unflatten(aux, children):
  # 使用 aux_data 和 children 重建对象
  return modeling_outputs.CausalLMOutputWithPast(*children)

# 注册
register_pytree_node(
  modeling_outputs.CausalLMOutputWithPast,
  output_flatten,
  output_unflatten,
)
```

### 实际例子

假设我们有一个简单的自定义类：

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

# 注册为 Pytree
def point_flatten(point):
    return (point.x, point.y), None

def point_unflatten(aux, children):
    return Point(*children)

register_pytree_node(Point, point_flatten, point_unflatten)

# 现在可以在 JAX 中使用
p = Point(jnp.array([1.0]), jnp.array([2.0]))
jitted_func = jax.jit(lambda p: Point(p.x * 2, p.y * 2))
result = jitted_func(p)  # 现在可以工作了！
```

-----

## 静态参数的其他处理方法

除了使用闭包，我们还可以使用 `static_argnums` 参数：

```python
# 方法 1：使用 static_argnums
def func_with_kwargs(weights, input_ids, use_cache):
  res = func(weights, (input_ids,), {'use_cache': use_cache})
  return res

# 告诉 JAX 第三个参数（use_cache）是静态的
jitted_func = jax.jit(func_with_kwargs, static_argnums=(2,))
res = jitted_func(weights, model_inputs.input_ids, False)
```

### static_argnums 的工作原理

- JAX 会为每个不同的静态参数值编译一个单独的版本
- 静态参数在编译时是已知的，可以在控制流中使用
- 如果静态参数改变，JAX 会重新编译函数

### 选择哪种方法？

- **闭包**: 适合参数值固定的情况
- **static_argnums**: 适合需要为不同参数值编译多个版本的情况

-----

## 额外示例：完整的推理流程

让我们创建一个完整的示例，展示如何使用 JIT 编译的模型进行文本生成：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import jax
import jax.numpy as jnp
import torchax
from jax.tree_util import register_pytree_node
from transformers import modeling_outputs, cache_utils
import time

# Pytree 注册（同上）
def output_flatten(v):
  return v.to_tuple(), None

def output_unflatten(aux, children):
  return modeling_outputs.CausalLMOutputWithPast(*children)

register_pytree_node(
  modeling_outputs.CausalLMOutputWithPast,
  output_flatten,
  output_unflatten,
)

def _flatten_dynamic_cache(dynamic_cache):
  return (
      dynamic_cache.key_cache,
      dynamic_cache.value_cache,
  ), None

def _unflatten_dynamic_cache(aux, children):
  cache = cache_utils.DynamicCache()
  cache.key_cache, cache.value_cache = children
  return cache

register_pytree_node(
  cache_utils.DynamicCache,
  _flatten_dynamic_cache,
  _unflatten_dynamic_cache,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf", 
    torch_dtype="bfloat16", 
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 转换为 JAX
weights, func = torchax.extract_jax(model)

# 创建 JIT 编译的函数
def forward_pass(weights, input_ids):
  return func(weights, (input_ids,), {'use_cache': False})

jitted_forward = jax.jit(forward_pass)

# 准备输入
prompt = "The secret to baking a good cake is "
model_inputs = tokenizer([prompt], return_tensors="jax")

# 性能测试
print("预热（包含编译时间）:")
start = time.time()
res = jitted_forward(weights, model_inputs.input_ids)
jax.block_until_ready(res)
print(f"第一次运行: {time.time() - start:.4f} 秒")

print("\n实际推理（使用缓存的编译）:")
for i in range(5):
    start = time.time()
    res = jitted_forward(weights, model_inputs.input_ids)
    jax.block_until_ready(res)
    print(f"运行 {i+1}: {time.time() - start:.6f} 秒")

# 获取预测的下一个词
logits = res.logits
next_token_logits = logits[0, -1, :]
next_token_id = jnp.argmax(next_token_logits)
next_token = tokenizer.decode([int(next_token_id)])
print(f"\n预测的下一个词: '{next_token}'")
```

-----

## 结论

本文展示了在 JAX 中运行来自 Hugging Face 的 `torch.nn.Module` 确实是可行的，尽管需要解决一些"粗糙的边缘"。主要挑战包括：

1. **Pytree 注册**: 将 Hugging Face 的自定义输出类型注册到 JAX 的 Pytree 系统
2. **静态参数管理**: 处理 JIT 编译的静态参数

### 关键要点

- ✅ 可以在 JAX 中使用 PyTorch 的 Hugging Face 模型
- ✅ `torchax` 提供了便捷的互操作层
- ✅ JIT 编译可以带来显著的性能提升（~300倍）
- ⚠️ 需要注册自定义 Pytree 类型
- ⚠️ 需要处理静态参数

### 未来展望

将来，一个适配器库可以预先注册常见的 Hugging Face Pytree，并为 JAX 用户提供更流畅的集成体验。

## 下一步

我们已经打下了基础！在下一篇文章中，我们将深入探讨：

* **解码句子**: 演示如何在这个 JAX-PyTorch 设置中使用 `model.generate` 进行文本生成
* **张量并行**: 展示如何扩展此解决方案以在多个 TPU（例如 8 个 TPU）上运行以加速推理

敬请期待！

-----

## 常见问题解答

### Q: 为什么 Hugging Face 移除了 JAX 支持？

A: Hugging Face 团队决定精简代码库，专注于 PyTorch 实现。维护多个框架的原生支持需要大量资源，而 PyTorch 是社区中使用最广泛的框架。

### Q: torchax 的性能如何？

A: 通过 JIT 编译，性能可以达到原生 JAX 实现的水平。在本文的例子中，我们看到了约 300 倍的加速（相比第一次运行）。

### Q: 我需要修改 Hugging Face 模型的代码吗？

A: 不需要！这就是这种方法的美妙之处。您可以直接使用预训练的 Hugging Face 模型，无需任何修改。

### Q: 所有 Hugging Face 模型都支持吗？

A: 理论上，任何基于 PyTorch 的 Hugging Face 模型都可以使用这种方法。但是，您可能需要为特定模型的输出类型注册额外的 Pytree。

### Q: 我可以在 GPU 上使用这种方法吗？

A: 当然可以！只需安装相应的 JAX 版本（`jax[cuda12]`）并确保您的 GPU 驱动程序正确配置即可。

### Q: 与原生 JAX/Flax 实现相比如何？

A: 性能应该是相似的，因为底层计算都是通过 JAX 进行的。主要区别在于：
- **优势**: 可以使用所有 Hugging Face 的预训练模型和工具
- **劣势**: 可能需要额外的 Pytree 注册和处理

### Q: 我可以训练模型吗？

A: 本文主要关注推理。训练需要额外的工作来处理梯度和优化器状态。这将在未来的文章中探讨。