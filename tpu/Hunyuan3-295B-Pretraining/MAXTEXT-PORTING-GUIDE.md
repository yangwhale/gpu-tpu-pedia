# 把一个模型移植到 MaxText — 从 Hunyuan3 总结出来的范式

这份文档不讲 Hy3 的性能，只讲**「往 MaxText 里加一个新模型」这件事本身有什么规律**。
写给下一个要做同样事情的人：你会改多少文件、每一处在干什么、怎么判断自己有没有漏。

样本是腾讯混元 3（295B-A21B）。它的价值在于**结构上一半像 Qwen3、一半像 DeepSeek V3**，
所以几乎把 MaxText 里所有「按模型家族分叉」的地方都踩了一遍。

> 配套产物：代码在 [`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3)；
> 跑测试的脚本在 [`maxtext-hunyuan3/`](maxtext-hunyuan3/)。

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
| SwiGLU 激活截断 | 1 处 | ✅ 是 —— **见下方「差点判错的一处」** |
| vLLM 权重映射表 | 1 处 | ❌ **否** —— DeepSeek 用 MLA，Hy3 是 GQA，**套用会错** |

> #### ⚠️ 差点判错的一处：「我 config 里没这个字段」也不是理由
>
> SwiGLU 激活截断那处，我一开始判为「不适用」，理由是「Hy3 的 config 里没有
> `mlp_activations_limit` 这个字段」。这看起来是可验证的事实，其实**跟「我们没开这个开关」
> 是同一个错误**——那是个**调优旋钮**，将来试参数时随时会加上，加上的那天它会静默走进
> 未截断的分支。
>
> **可接受的「不适用」只有一种：这个行为在你的模型上永远不成立。**
> 比如 vLLM 权重映射那处——DeepSeek 是 MLA、Hy3 是 GQA，映射表结构根本不同，
> 套用一定错；那不是「暂时不需要」，是「用了就是 bug」。

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

### 第 4 步 · 改动落在 fork 的分支上，不要维护补丁脚本

**在上游的 fork 上开一个分支，所有改动作为 commit 落在分支上。** 跟上游用 `git rebase`。

我们一开始走了弯路：写了个补丁脚本，用字符串锚点去改上游文件，每处断言「正好命中 N 处」。
断言本身是对的（它抓到过一次静默漏改：上游把单行 tuple 拆成多行，正则没匹配上），
但**整条路线是错的**：

| | 补丁脚本 | 分支 + rebase |
|---|---|---|
| 上游改了附近代码 | 锚点漂了，靠你**事先写对**断言才发现 | **明确报冲突**，必须处理 |
| 真相来源 | 脚本一份、实际跑的树一份 → **会分叉** | 只有分支一份 |
| 提 PR | 还要再转换一次 | 直接就是 commit |

> ⚠️ **两个真相来源是真会咬人的。** 我们同时维护「仓库里的模型文件 + 补丁脚本」和
> 「分支」，结果在一处改对、另一处没跟上，而所有测试恰好跑在改对的那份上——
> **零征兆，直到从全新 checkout 走一遍才暴露**（见第 5 步）。

### 第 5 步 · 从全新 checkout 验证，然后真跑

```bash
git clone --depth=1 -b <你的分支> <你的 fork> /tmp/fresh
# 不做任何手工修补，直接用这棵树去跑
```

> ⚠️ **这一步不能省，也不能用「我本地那棵调试树」代替。**
> 我们在这一步抓出了两个自己产物里的 bug：配置白名单少注册了一个模型名、
> 模型文件里的属性名跟训练循环里的查找表对不上。两处在调试树里都是对的，
> 所有 benchmark 都跑在那棵树上，**所以整整一晚上零征兆**。
>
> 「代码看着对」≠「从零走一遍能跑」。

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
| 已覆盖 | **32** | 加进名单 |
| 刻意不覆盖 | **2** | ① DeepSeek 自己的分派项（我们有自己的一条）② vLLM 权重映射（GQA ≠ MLA，**套用一定错**，需要单独写一份映射，属 TODO 不属「不适用」） |

涉及的 12 个上游文件：

```
common/common_types.py          configs/types.py
layers/decoders.py              layers/nnx_decoders.py
layers/moe.py                   layers/linears.py
layers/multi_token_prediction.py
utils/maxtext_utils.py          utils/generate_param_only_checkpoint.py
utils/layerwise_quantization.py experimental/rl/grpo_utils.py
trainers/pre_train/train.py     ← 这个是上游 bug，应单独提 PR
```

---

*样本：腾讯混元 3（295B-A21B）· 实测平台 TPU v5p 256 芯片 / v7 Ironwood 64 芯片*
