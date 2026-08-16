# 换了 kernel，模型还是同一个模型吗 —— 用 KL 散度做等价性验证

**一句话**：换 kernel / 换量化 / 换并行切法之后，要证明「算的还是同一个模型」，
**唯一站得住的判据是在 logits 上比 KL 散度**。比 loss 太粗，比逐位太严，比中间激活比错了地方。

这是生产级验证流程里的一步 —— 性能提升报出去之前，先过这一关。

---

## 1. 什么时候需要做这件事

任何**「数学上应该等价、实现上完全不同」**的改动落地前：

| 改动 | 为什么危险 |
|---|---|
| 换 MoE kernel（如 megablox ↔ 其它 GMM 实现） | 分组矩阵乘的分组/掩码逻辑不同，容易少算或多算专家 |
| 开量化（FP8 / INT8） | 校准范围、量化粒度、是否量化后再通信，都会改变数值 |
| 改并行切法（FSDP 宽度、专家并行） | 通信被省掉或被重复，肉眼在 profile 上看不出来 |
| 改分块参数（tile / block size） | 只应改变分块顺序，但实现可能顺手改了边界处理 |
| 换编译器 flag、开融合 | 融合可能改写累加顺序甚至跳过一段计算 |

**共同特征**：跑得通、loss 看着正常、profile 显示更快 —— 但可能算少了。
「更快」和「算少了」在性能数字上长得一模一样。

---

## 2. 三个看起来合理、实际不成立的判据

这三条都试过，都不行。**别再走一遍。**

### ❌ 判据一：比 loss

`loss` 是在整个 batch 上归约成的一个标量，还只打印到小数点后 3 位。
两个明显不同的分布可以给出同一个 loss。

它能证伪（loss 发散 = 一定有问题），**不能证实**。

### ❌ 判据二：比逐位相同（bitwise）

这条对实现细节过敏，**在数学上就不该期待它成立**：

- 浮点加法不满足结合律，换个分块就换个累加顺序
- 不同 kernel 的规约树形状不同
- 编译器融合会改写运算顺序

要求「换实现后逐位相同」比要求「换分块后逐位相同」还严格。
**它失败不代表实现错了，它成功也只是运气。**

> 有意思的是，实测里两个不同 kernel 的 logits 反而**真的逐位相同**（见 §6）。
> 那是个惊喜，不是应该拿来当门槛的标准。

### ❌ 判据三：比中间层激活的张量指纹

思路是：在每层入口出口对张量的原始位算个哈希，两次跑对一对。

**这条会给出假的差异**，原因是分片：

- MoE 的分组矩阵乘通常跑在 `shard_map`（或等价的手动分片）里面
- 在 `shard_map` **内部**做的任何归约，拿到的是**本卡那一片**的值，不是全局值
- `jax.debug.print` 只在部分进程上打印，哪张卡先打、打哪一片都不确定

于是两次跑的哈希天然对不上 —— **那是采样噪声，不是数值差异**。

实测踩过：这套方法报出「换个 tile 大小就 0/30 位相同」，
而同一组配置在 logits 上其实是逐位相同的。**结论完全相反。**

另外，如果模型开了激活 offload（把层输入换出到 host 内存），
在被 offload 的张量上插位操作还会直接把编译器搞崩：

```
Bitcast cannot have different memory spaces of output (5) and operand (0)
```

---

## 3. 正确判据：logits 上的 KL 散度

**比的是模型对下一个 token 的概率分布。**

$$D_{KL}(P \parallel Q) = \sum_{v \in V} p_v (\log p_v - \log q_v)$$

其中 $P$ 是参考实现（已知正确的那个）的输出分布，$Q$ 是新实现的。

**为什么是它**：

- 它问的是**模型行为**是否等价，不是某个中间张量的比特
- 它对「算少了一部分贡献」极其敏感 —— 少算会让分布明显平掉或偏移
- 它对「累加顺序不同带来的末位抖动」几乎不敏感 —— 那种差异在 softmax 后被压没了
- 它有**天然的参考尺度**：跟「当前分布 vs 均匀分布的 KL」比一比，就知道差异是大是小

### 判据阈值

| KL（nats/token） | 结论 |
|---|---|
| `0` ~ `1e-7` | 等价。可以放行 |
| `1e-7` ~ `1e-4` | 数值噪声级。BF16/FP8 换实现的正常范围，看 top-1 是否一致 |
| `1e-4` ~ `1e-2` | 可疑。要查是不是量化校准或边界处理不同 |
| `> 1e-2` | **算错了**。少算一部分专家/头/token 就落在这一档 |

**必须同时报一个对照尺度**，否则数字没有意义：

```
H_ref = log(|V|) + Σ p log p      # 当前分布相对均匀分布的 KL
```

实测中 `H_ref ≈ 0.4992 nats`。如果新实现只算了 1/64 的专家，
KL 会落在 `1e-1 ~ 1e1`，跟 `0` 差好几个数量级，一眼能分。

---

## 4. 实现

三件事必须做对，否则算出来的 KL 是错的。

### 4.1 归一化必须在图里做

logits 的词表维通常是**切片**的。把 logits 分片存盘再离线 softmax 是错的 ——
每张卡只有词表的一段，各自归一化出来的不是同一个分布。

**在图里先算 `log_softmax`**，让编译器自己插入跨分片的 max / sum：

```python
lsm = jax.nn.log_softmax(jnp.asarray(logits, jnp.float32), axis=-1)
```

存下来的就已经是正确归一化过的对数概率了。

### 4.2 分片按序一一对应

两次 run 用同一个 mesh、同一份分片规格，落盘的分片顺序就是一致的。
于是 KL 可以逐片求和 —— 每片贡献 `Σ_v p_v (log p_v − log q_v)`，
加起来就是整个词表上的精确 KL。

### 4.3 注意模块的双导入别名

**同一个 `.py` 文件被以两条路径 import，会得到两个不同的模块对象。**

```python
>>> import maxtext.utils.max_utils as A
>>> import src.maxtext.utils.max_utils as B
>>> A is B
False
>>> A.__file__ == B.__file__
True
```

打补丁只打其中一个，会**静默不生效** —— 补丁装上了、打印了「已安装」、
一条数据都出不来。两个都打。

### 4.4 完整的注入补丁

挂在交叉熵函数上（它是唯一同时拿到 logits 和 targets 的地方）：

```python
# kl_probe.py —— 在 logits 处抓分布
import os, importlib
import numpy as np
import jax, jax.numpy as jnp

TAG = os.environ.get("KLTAG", "x")
OUT = f"/tmp/kl-{TAG}"
os.makedirs(OUT, exist_ok=True)
_cnt = {"n": 0}

def _save(arr):
    i = _cnt["n"]; _cnt["n"] += 1
    if i < 64:
        np.save(f"{OUT}/lsm_{i:03d}.npy", np.asarray(arr))
        print(f"@@@KL saved lsm_{i:03d} shape={np.asarray(arr).shape}", flush=True)

def _hook(logits):
    # 关键：归一化在图里做，跨分片的 max/sum 交给编译器
    lsm = jax.nn.log_softmax(jnp.asarray(logits, jnp.float32), axis=-1)
    jax.debug.callback(_save, lsm[:1, :8, :])       # 只留 8 个 token，够用

for modname in ("maxtext.utils.max_utils", "src.maxtext.utils.max_utils"):
    try:
        M = importlib.import_module(modname)
    except Exception as e:
        print(f"@@@KL import {modname} failed: {e}", flush=True); continue
    orig = getattr(M, "cross_entropy_with_logits", None)
    if orig is None or getattr(orig, "_kl_wrapped", False):
        continue
    def make(o):
        def w(logits, targets, *a, **k):
            _hook(logits)
            return o(logits, targets, *a, **k)
        w._kl_wrapped = True
        return w
    M.cross_entropy_with_logits = make(orig)
    print(f"@@@KL patched {modname}", flush=True)
```

用一个薄壳把它塞进训练入口，**不改仓库代码**：

```python
# run_with_kl.py
import sys, runpy
sys.path.insert(0, '/deps')
exec(open('/tmp/kl_probe.py').read())
sys.argv = ['train'] + sys.argv[1:]
runpy.run_module('src.maxtext.trainers.pre_train.train', run_name='__main__')
```

跑两次，只改被测的那一个开关，其余参数逐字相同：

```bash
KLTAG=ref  python3 run_with_kl.py <config> ... steps=1        # 参考实现
KLTAG=new  python3 run_with_kl.py <config> ... steps=1 <被测开关>
```

> `steps=1` 就够。等价性看的是同一份输入下的输出，不需要训练。
> 层数也可以调小（省编译时间），但**至少要包含一层被测的结构**。

### 4.5 离线算 KL

```python
import numpy as np

def load(tag, i):
    return np.load(f"/tmp/kl-{tag}/lsm_{i:03d}.npy").astype(np.float64)

a, b = load("ref", 0), load("new", 0)
assert np.allclose(np.exp(a).sum(-1), 1.0, atol=1e-5), "归一化不对，检查 §4.1"

p  = np.exp(a)
kl = (p * (a - b)).sum(-1).ravel()              # 逐 token KL(ref‖new)
h_ref = np.log(a.shape[-1]) + (p * a).sum(-1).mean()   # 对照尺度

print(f"KL(ref‖new)   = {kl.mean():.3e} nats  逐 token: {kl}")
print(f"max|Δ log p|  = {np.abs(a - b).max():.3e}")
print(f"top-1 一致    = {(a.argmax(-1) == b.argmax(-1)).all()}")
print(f"对照尺度 H_ref = {h_ref:.4f} nats")
```

---

## 5. 三个必须做的完整性检查

拿到一个漂亮的数字之前，先证明这次测量不是自欺欺人。

**① 证明开关真的生效了。**
从两次 run 的日志里把被测参数抓出来对一眼，再看性能数字有没有变：

```bash
grep -m1 "Config param <被测开关>" ref.log new.log
grep -m1 "completed step: 0," ref.log new.log
```

两边配置相同 = 这次对比没有意义。实测中被测 kernel 慢了 13×，
**性能差异本身就是「开关确实生效」的旁证**。

**② 证明被测的那条代码路径真的被走到了。**

这一条最容易漏，而且漏了会给出**假的通过**。

实测教训：被测的差异（量化后再 all-gather）在源码里的开关是

```python
def explicitly_weight_ag(shard_exp_on_fsdp):
    if shard_exp_on_fsdp:
        rule = get_current_rule("gmm")
        if rule and rule.weight_calibration_method.startswith("fixed"):
            return True
    return False
```

第一轮用 BF16 测，没开量化 → `rule is None` → 开关为 False →
**两条分支里那段 all-gather 都被跳过**，喂进 kernel 的权重完全一样。
KL 漂亮地等于 0，**但它压根没测到要测的东西**。

换成 FP8（开关生效）重测，KL 立刻从 `0` 变成 `1e-3`。

> **做法**：把被测代码路径上的**每一个前置条件**列出来，逐个从日志里确认为真。
> 光确认「我传了 `--use_new_kernel=True`」不够 —— 那个 flag 可能被另一个条件否决。

**③ 证明两份输出不是同一个文件。**
`KL = 0` 的第一嫌疑是文件拷串了。在**产出端**校验，不是在拷贝之后：

```bash
kubectl exec <pod> -- md5sum /tmp/kl-ref/*.npy /tmp/kl-new/*.npy
stat -c '%n %y' /tmp/kl-ref/lsm_000.npy /tmp/kl-new/lsm_000.npy   # 时间戳应不同
```

**④ 证明输入是同一份。**
合成数据要确认它是**确定性**的。很多框架的合成数据用固定种子生成一次、
之后每步复用同一个 batch —— 那正好，输入天然一致。
真实数据集则要固定 shuffle seed 和起始 step。

顺带确认权重初始化种子也固定。

---

## 6. 一个真实样例

对象是一个 295B 规模的 MoE 模型（192 个专家），被测项是两种分组矩阵乘 kernel。

**背景**：新 kernel 快 50%，但 profile 里看到它在循环内的操作数形状是 `[3, ...]`，
而老 kernel 是 `[192, ...]`，`192 / 3 = 64` 正好等于 FSDP 宽度 ——
怀疑它把专家切了却没规约回来，也就是**只算了 1/64**。

**测量**（词表 120,832，8 个 token，对照尺度 H_ref ≈ 0.4992 nats）：

| 精度 | 被测路径是否走到 | KL(ref ‖ new) | max\|Δ log p\| | top-1 |
|---|---|---|---|---|
| BF16 | ❌ 没走到（见 §5②） | 0.000e+00 | 0.000 | 8/8 |
| **FP8** | ✅ 走到了 | **1e-3** | 0.21 | 7~8/8 |

第一轮 BF16 的 `KL = 0` 是**假的通过** —— 前置条件不满足，
被测的那段 all-gather 两边都没执行。换 FP8 重测才拿到真数字。

**结论**：`1e-3` 是 H_ref 的 0.2%，比「少算 1/64」该有的量级低 2~4 个数量级 ——
**不是算少了**，是 FP8 舍入路径不同。但也**不是完全等价**，
BF16 下两者逐位相同，FP8 下有 1/8 的 token top-1 翻转。那个 `3` 和 `192` 连维序都不一样
（`[3, 4096, 1536]` vs `[192, 1536, 4096]`），本来就不是同一个张量的两种切法 ——
**从一开始就在比错东西**。

作为佐证，32 步的 loss 曲线也重合：第 0 步逐位相同，第 31 步差 0.004（0.04%），
中段最大偏离 0.197 之后**重新收敛** —— 真少算 64 倍会单调发散，不会先分开再合回来。

> **这个案例的教训不在结论，在过程**：性能数字（多次复现、profile 自证）是可信的，
> 但**机理归因连错了三次**，每次都是「看到一个合理机制就当答案」。
> 归因必须能被独立证据钉死，否则就标成未决。

---

## 7. 这一关放进流程里的位置

```
改动落地
  ↓
① 冒烟：跑得起来吗（小切片，几分钟）
  ↓
② 等价性：KL 散度 == 0 或落在噪声档     ← 本文
  ↓
③ 性能：吞吐 / MFU，多次复现
  ↓
④ 收敛：几十步 loss 曲线与基线对拍
  ↓
⑤ 长跑：真实数据集
```

**② 一定要排在 ③ 前面。** 先测性能会让人对着一个可能算错的实现调参 ——
调出来的每一个数都得推倒重来。

---

## 8. 这个方法答不了什么

诚实划边界：

- **只覆盖前向。** 反向的梯度是否等价，KL 不管。要另测（比梯度范数，或看长跑 loss）。
- **只覆盖抽样到的那几个 token。** 8 个 token 上 KL 为 0，不等于全 batch 为 0。
  怀疑边界情况（比如某些 token 路由到特定专家）就要扩大抽样。
- **只覆盖测的那个精度和那个层数。** BF16 过了不代表 FP8 过，
  2 层过了不代表 80 层过 —— 层数会放大累积误差。
- **不覆盖动态行为。** 比如某些实现只在特定 batch 形状下出问题。

配套的做法是把它跟 §7 里的 ④⑤ 一起用：KL 管「同一时刻是否等价」，
loss 曲线管「累积下来是否等价」。

---

## 附：为什么不用别的散度

| 度量 | 为什么不选 |
|---|---|
| L2 / max abs（在 logits 上） | logits 有任意平移不变性，`+C` 不改变分布却会让 L2 变大 |
| 余弦相似度 | 对小概率尾部不敏感，尾部塌了看不出来 |
| JS 散度 | 有界（≤ log 2），差异大的时候会饱和，反而看不出量级 |
| 困惑度差 | 又归约成标量了，回到 §2 的判据一 |

KL 非对称，**方向要选对**：`KL(参考 ‖ 新实现)` ——
以参考分布为权重，新实现在参考认为重要的地方出错，惩罚才大。
两个方向都报出来更稳妥。

---

## 9. 附录：当 profile 的算子名对不上时，用 trace annotation

实测中遇到一个把人带进沟里的现象：**按算子名统计，MoE 的分组矩阵乘只占一步的 1.3%**
（100 ms / 7,850 ms）。而理论计算量要求它至少占 1.80 s —— **比物理下限还低 18 倍**。

原因不是时间消失，是**名字消失**：编译器把大量 MoE 相关计算融进了别的算子，
融合后的算子不再叫 `gmm`，按名字 grep 就漏了。

修法是给源码打 `jax.named_scope`，让归属信息留在 profile 的 `Framework op name` 字段里：

```python
import jax, importlib
for mn in ("pkg.layers.moe", "src.pkg.layers.moe"):        # 注意双导入别名，见 §4.3
    M = importlib.import_module(mn)
    C = M.RoutedMoE
    for meth, tag in (("sparse_matmul","MOE_sparse_matmul"),
                      ("permute","MOE_permute"), ("unpermute","MOE_unpermute"),
                      ("get_topk","MOE_router_topk")):
        o = getattr(C, meth)
        def mk(o, t):
            def w(self, *a, **k):
                with jax.named_scope(t):
                    return o(self, *a, **k)
            return w
        setattr(C, meth, mk(o, tag))
```

再按该字段聚合，真相立刻出来（同一份 profile，同一步）：

| 统计口径 | MoE 每步耗时 | 占比 |
|---|---|---|
| 按算子名 grep `gmm`/`tgmm` | 100 ms | 1.3% ❌ |
| **按 `named_scope` 聚合** | **2,565 ms** | **33%** ✅ |

按作用域再拆一层，才看得见钱花在哪：

| 类别 | 每步 | 说明 |
|---|---|---|
| loop fusion | 1,464 ms | 量化的截断/转换、算 scale 的 absmax 归约 |
| async-done | 603 ms | 激活 offload 换回来的等待 |
| sort | 161 ms | 路由后的 token 排序 |
| **custom-call（Pallas kernel 本体）** | **100 ms** | 真正的矩阵乘 |

**结论：优化方向完全变了。** 原本以为该调 kernel，实际大头在量化辅助算子和 offload 等待上。

### 顺带回答一个常见疑问

**XProf 能看出一个 Pallas kernel 跑了多久吗？** 能。
`pallas_call` 编成一个 custom-call，`Avg. self time` 就是它在 TPU 核上的真实执行时长。
**看不到的是 kernel 内部** —— 分块循环走了几轮、VMEM 命中、MXU 利用率，
那些要 LLO / Mosaic 层的 dump。

### 一个仍未定案的疑点（诚实记录）

上表里 Pallas kernel 的 100 ms/step，**低于该计算量的物理下限 1.80 s 达 18 倍**。
两种可能尚未分离：① kernel 只算了本地那 1/N 的专家；② XProf 对 custom-call 的
自耗时归因偏低。判据实验：**改变 FSDP 宽度让每卡专家数翻倍，看 kernel 耗时是否同步翻倍。**
