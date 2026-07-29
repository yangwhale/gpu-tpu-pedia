# 把一个模型移植到 MaxText — 从 Hunyuan3 总结出来的范式

这份文档不讲 Hy3 的性能，只讲**「往 MaxText 里加一个新模型」这件事本身有什么规律**。
写给下一个要做同样事情的人：你会改多少文件、每一处在干什么、怎么判断自己有没有漏。

样本是腾讯混元 3（295B-A21B）。它的价值在于**结构上一半像 Qwen3、一半像 DeepSeek V3**，
所以几乎把 MaxText 里所有「按模型家族分叉」的地方都踩了一遍。

> 配套产物：[`maxtext-hunyuan3/`](maxtext-hunyuan3/) —— 模型文件、两个配置、以及
> [`port.py`](maxtext-hunyuan3/port.py)（把这套改动重新应用到任意上游 checkout）。

---

## 一、先看总量

| | 数量 | 说明 |
|---|---|---|
| **新增文件** | **3** | 1 个模型层文件（约 160 行）+ 2 个 yml 配置 |
| **改动上游文件** | **12** | 全部是「让框架认识这个模型」，没有一处是算法实现 |
| 真正的新代码 | **< 200 行** | 且**零新数学**：注意力继承 Qwen3，MoE 复用 DeepSeek |

**关键认知：工作量不在「实现模型」，在「跟框架里按名字分叉的地方打交道」。**

---

## 二、根本规律：MaxText 拿「模型家族名」当行为的代理变量

这是理解全部 12 个文件改动的**唯一一把钥匙**。

框架里到处是这样的判断：

```python
if config.decoder_block == DecoderBlockType.DEEPSEEK:
    ...
if config.model_name.startswith("deepseek3"):
    ...
```

但这些地方**真正想问的**不是「你是不是 DeepSeek」，而是：

| 代码问的 | 实际想问的 |
|---|---|
| `decoder_block == DEEPSEEK` | 你是不是「稠密首层 + 后续 MoE 层」的两段式结构？ |
| `model_name.startswith("deepseek3")` | 你路由是不是 sigmoid + 专家偏置？打分要不要取偏置前的值？ |
| `decoder_block in (DEEPSEEK, LLAMA4, ...)` | 你有没有共享专家？算 FLOP 时要不要算它？ |

**这些问题的答案配置里全都有**（`first_num_dense_layers`、`routed_score_func`、`shared_experts`…），
框架却用家族名当了替身。

所以「加一个模型」的本质是：**把所有问错问题的地方找出来，在答案名单里加上自己的名字。**
这就是为什么改的是 12 个文件，而不是填一个 config。

---

## 三、三类改动，没有第四类

### 类 A · 身份登记（4 处，纯机械）

告诉框架「世界上有这么个模型」。不动脑子，但**漏一处就崩，且报的错跟漏的地方常常无关**。

| 位置 | 做什么 | 漏了会怎样 |
|---|---|---|
| 解码块枚举 | 加一个枚举值 | 配置解析就过不去 |
| 配置白名单（pydantic `Literal`） | 加模型名 | `Input should be 'default', 'llama2-7b', ...` |
| 分派表 ×2（linen 侧） | 名字 → 层类 | 拿不到层类 |
| 分派表 ×1（nnx 侧） | 同上，**在另一个文件里** | `Incorrect decoder_block name` |

> ⚠️ **第三张分派表是最容易漏的一处**。前两张挨在一起，很容易以为改完了。
> 漏它报的错看起来跟分派表毫无关系。

### 类 B · 行为归类（7 处，真正要思考的部分）

每一处都是一道判断题：「我这个模型在**这个具体方面**，跟 DeepSeek 是不是一路的？」

| 方面 | 判断 | Hy3 的答案 |
|---|---|---|
| 路由数学（sigmoid / 偏置前打分 / 缩放系数） | 5 处名字门 | ✅ 是，血统就是 DSV3 |
| 两段式结构（稠密段 + MoE 段）的层分组 | scan 判定、流水线层数推导 | ✅ 是 |
| 权重导出时分组展开 | 3 处 | ✅ 是 —— **不加会导出错误结构** |
| RL 参数同步的分组 | 1 处 | ✅ 是 |
| Norm 类型白名单 | 1 处 | ✅ 是（RMSNorm）—— **else 是 `raise`** |
| 算力统计（共享专家 / 专家宽度） | 4 处 | ✅ 是 —— 不加**报表虚高约 5 倍** |
| MTP + batch split 的重分片 | 1 处 | ✅ 是 |
| 逐层量化、muon 优化器白名单 | 2 处 | ✅ 是 |
| SwiGLU 激活截断 | 1 处 | ❌ **否** —— 这是 DeepSeek V4 的特性，Hy3 config 里没有对应字段 |
| vLLM 权重映射表 | 1 处 | ❌ **否** —— DeepSeek 用 MLA，Hy3 是 GQA，**套用会错** |

### 类 C · 真 bug（1 处）

训练主循环里，无梯度专家偏置的更新路径把 DeepSeek 的**模块属性名写死**了：

```python
new_state.model.decoder.moe_layers.DeepSeekMoeBlock_0.MoeBlock_0.gate.bias
```

相当于收件地址里写死了「三号楼」。任何不叫这个名字的模型，只要开了
`routed_bias_update_rate`，**配置能过、训练能起、第一步 AttributeError**。

这跟 Hy3 没关系，是所有非 DeepSeek 模型的共同问题，**应该单独发一个上游 PR**。

---

## 四、下次加模型：五步范式

### 第 1 步 · 找齐所有分叉点

```bash
grep -rn "DecoderBlockType\.<某个已有家族>" --include=*.py src/
grep -rn 'startswith(("<某个已有家族>' --include=*.py src/
grep -rn "DecoderBlockType\." --include=*.py src/ | grep -E "in \(|in \[|in \{"   # 白名单
```

挑一个**跟你结构最接近的已有家族**当探针。Hy3 用的是 DeepSeek，因为 MoE 同源。

### 第 2 步 · 逐处翻译成行为问题

对每一处问：**这段代码实际在问什么行为？我有没有这个行为？**
不要问「我是不是 DeepSeek」——那是框架的错误提问方式，不该被继承。

### 第 3 步 · 三种处置，「不适用」必须写理由

| 处置 | 何时 | 要求 |
|---|---|---|
| 加进名单 | 行为相同 | — |
| 明确不适用 | 行为确实不同 | **必须写下可验证的理由** |
| 判为上游 bug | 该处根本不该按名字判断 | 单独提 PR，不要混进模型 PR |

> ⚠️ **「我们没开这个开关，所以不用改」不是理由。** 开关随时会被打开，
> 到那时炸在生产环境里。可接受的理由长这样：
> 「我的 config 里根本没有 `mlp_activations_limit` 这个字段」——这是可验证的事实。

### 第 4 步 · 每处改动带命中数断言

不要断言「文件被改了」，要断言「**正好改了 N 处**」：

```python
def edit(rel, fn, expect):
  s = fn(open(path).read())
  n = s.upper().count("<你的模型名>")
  assert n == expect, f"{rel}: 命中 {n} 处，期望 {expect} 处——上游锚点可能变了"
```

**这条是从真实教训来的**：早期版本只断言「文件变了」，结果某个文件三处改动里有一处
正则没匹配上（上游把单行 tuple 拆成了多行），**静默漏掉**，一直到复现审计才发现。

### 第 5 步 · 从全新 checkout 验证，然后真跑

```bash
git clone --depth=1 <upstream> /tmp/fresh
cp <你的模型文件> /tmp/fresh/...
MAXTEXT_ROOT=/tmp/fresh/src/maxtext python3 port.py    # 只跑脚本，不做任何手工修补
# 然后真的在硬件上跑一次缩层冒烟
```

> ⚠️ **这一步不能省，也不能用「我本地那棵调试树」代替。**
> 我们在这一步抓出了两个自己产物里的 bug：配置白名单少注册了一个模型名、
> 模型文件里的属性名没跟补丁脚本对齐。两处在调试树里都是对的，
> 所有 benchmark 都跑在那棵树上，**所以整整一晚上零征兆**。
>
> 「补丁能打进去」≠「按文档从零走一遍能跑」。

---

## 五、验证清单（缩层冒烟该看什么）

用一份**结构与正式配置完全一致、只把层数砍到 4 层**的配置，在最小规模上跑 8 步：

| 检查项 | 通过标准 |
|---|---|
| 配置能加载 | 无 pydantic `Value error` |
| 层能构造 | 无 `Incorrect decoder_block name` / 无缺参数 |
| 前反向通 | loss 单调下降 |
| 数值健康 | 0 NaN、0 skipped |
| MTP（若有） | `mtp_loss` 单独打出且下降 |
| 无梯度偏置更新（若有） | 不报 `AttributeError` |
| 参数量 | 与解析式对得上 |

---

## 六、读日志的三条规矩

多机 TPU 上，**报错的表象和病因经常不在一处**。这三条能省掉大量时间：

1. **先确认 `N/N Running` 再看日志。** TPU 切片全有全无，人不齐时活着的 pod 会报
   `GetSliceInfo can only be invoked after a slice is built` —— 那是症状不是病因。
2. **判错看最早那条，不是日志尾。** 配置非法会**先把 TPU 拉起来再退**，
   真正的报错（`MAXTEXT CONFIG ERROR` / pydantic 的 `Value error`）在日志上方。
3. **step 0 含编译，step 1/2 是 JAX 异步派发的假读数**，稳态取 step ≥ 3。

---

## 七、Hy3 的最终状态（供对照）

34 处按 DeepSeek 名字分叉的判断点：

| | 数量 | 说明 |
|---|---|---|
| 已覆盖 | **31** | 加进名单 |
| 刻意不覆盖 | **3** | DeepSeek 自己的分派项；V4 专属的激活截断；vLLM 权重映射（GQA ≠ MLA，套用会错） |

`port.py` 一次跑完输出每个文件的命中数，一眼看得出有没有漏：

```
已改: common_types(2), decoders(11), moe(6), maxtext_utils(4), types(5),
      nnx_decoders(8), generate_param_only_checkpoint(3), grpo_utils(1),
      linears(1), multi_token_prediction(1), layerwise_quantization(1),
      trainers/pre_train/train.py
```

---

*样本：腾讯混元 3（295B-A21B）· 实测平台 TPU v5p 256 芯片 / v7 Ironwood 64 芯片*
