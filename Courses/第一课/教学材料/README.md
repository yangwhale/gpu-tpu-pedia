# 第一课 · 教学材料

这一课要用到的东西**都放在这里**，不散在别处。

每样材料都标三件事：**哪来的、什么许可、能不能进公开仓库**。
最后一列很重要 —— 这个仓库是 public 的，往里放东西等于对外发布。

---

## 清单

| 目录 / 文件 | 是什么 | 出处 | 许可 | 进公开库？ |
|---|---|---|---|---|
| `3b1b文字版/` | 4 章官方文字版 | 3blue1brown.com（Justin Sun 改编） | CC BY-NC-SA 4.0 | ✅ 入库 |
| `3b1b图/` | 自己渲的 **28 张** PNG | 3b1b/videos 场景代码 | CC BY-NC-SA 4.0 | ✅ 入库 |
| 见下 · CS336 | 从零手写 Transformer | Stanford CS336 | 课程材料 | 🔗 只引用 |
| 见下 · 场景源码 | manim 场景 | 3b1b/videos | CC BY-NC-SA 4.0 | 🔗 只引用 |

**许可已定（2026-08-21）**：`Courses/` 整体采用 **CC BY-NC-SA 4.0**，
与 3Blue1Brown 的材料同一许可，SA 条款自然满足。
全文见 [`../../LICENSE.txt`](../../LICENSE.txt)。

---

## 1 · `3b1b文字版/` — 四章正文

| 文件 | 对应视频 | 这一课用哪部分 |
|---|---|---|
| `3b1b-mini-llm.md` | Large Language Models explained briefly | **整篇 = 第一课骨架** |
| `3b1b-gpt.md` | Ch5 Transformers, the tech behind LLMs | 第 2 节 embedding、第 5 节参数账与 softmax |
| `3b1b-attention.md` | Ch6 Attention, step-by-step | **第 3 节主线** |
| `3b1b-mlp.md` | Ch7 How might LLMs store facts | **第 4 节主线** + superposition |

抓取方式：`curl https://r.jina.ai/https://3blue1brown.com/lessons/<slug>`，
末尾的赞助者名单已剔除。**这是原文，不是我们的改写** —— 引用时要转成自己的话。

## 2 · `3b1b图/` — 28 张已渲好的图

全部 1920×1080，合计 6.1 MB。按第一课的五节归位：

**第 1 节 · 会接话的机器**

| 文件 | 用来讲 |
|---|---|
| `SimpleAutogregression.png` | 一个词一个词往外蹦 |
| `AnnotateNextWord.png` | 下一个词的概率分布 |
| `TweakedMachine.png` | 参数就是一堆旋钮 |
| `DistinguishWeightsAndData.png` | 权重与数据是两回事 |
| `AthleteCompletion.png` | 模型「知道」很多事实 |

**第 2 节 · 词变成向量**

| 文件 | 用来讲 |
|---|---|
| `DiscussTokenization.png` | 一句话被切成 token |
| `IntroduceEmbeddingMatrix.png` | 嵌入矩阵：每词一列 |
| `ThreeDSpaceExample.png` | 词落在高维空间里 |
| `ManyIdeasManyDirections.png` | 一个方向 = 一个概念 |
| `DotProducts.png` | 点积在量什么 |

**第 3 节 · Attention**

| 文件 | 用来讲 |
|---|---|
| `AttentionPatterns.png` | ⭐ 主图：权重矩阵 + ΔE + E′ |
| `QueryMap.png` / `KeyMap.png` | Q / K 各自在干什么 |
| `ShowMasking.png` | causal mask |
| `DescribeAttentionEquation.png` | 那条公式逐项拆 |

**第 4 节 · MLP**

| 文件 | 用来讲 |
|---|---|
| `BasicMLPWalkThrough.png` | ⭐ 全流程 3D 图，带公式 |
| `BreakDownThreeSteps.png` | ⭐ 升维 → ReLU → 降维 |
| `Superposition.png` | ⭐ 叠加：exp(ε·N) 个方向 |
| `MLPIcon.png` | MLP 的形状 |
| `NonlinearityOfLanguage.png` | 为什么必须有非线性 |
| `ShowAngleRange.png` | 89°–91°：什么叫近似垂直 |

**第 5 节 · 参数账**

| 文件 | 用来讲 |
|---|---|
| `CountMatrixParameters.png` | attention 参数怎么数 |
| `ShowGPT3Numbers.png` | GPT-3 的整体数字 |
| `SoftmaxBreakdown.png` | softmax 与温度 |

**备用**：`AlmostOrthogonal` `StackOfVectors` `ClassicNeuralNetworksPicture` `LastTwoChapters`

复现：`../setup-render-env.sh` 搭环境，`../render-scenes.sh` 批量渲。
踩过的坑见 `../RENDER.md`。

> **还差一张**：Michael Jordan 那个例子在 `BasicMLPWalkThrough` 的**动画中段**，
> `-s` 只取最后一帧，要用 `-n <序号>` 抓中间帧。第 4 节需要。
>
> **渲不出来的**：`attention/IntroduceValueMatrix` —— 上游场景与当前 manimgl
> 版本不兼容（`FadeTransform` 往 VGroup 里塞 ImageMobject），不是我们的问题。

## 3 · CS336 —— 引用，不复制

Stanford CS336 Assignment 1 的完整中文翻译 + 代码 + 测试，在**另一个仓库**：

| 内容 | 路径 |
|---|---|
| 讲义中文版（38 题 / 6 个 Section） | `~/learning/cs336/` |
| 代码骨架 + 测试套件 + 原版 PDF | `~/learning/cs336-assignment1/` |
| 线上原仓库 | `github.com/stanford-cs336/assignment1-basics` |

**故意不复制过来。** 复制等于分叉，那边一更新这边就对不上。
第一课只会用到它的 Section 3（Transformer 架构）作为练习入口。

> 顺带一提，CS336 讲义里的「低资源提示」说：参考实现在 M4 Max 上
> MPS 不到 5 分钟、CPU 约 30 分钟就能训出会讲儿童故事的小模型。
> **第一课的练习门槛因此可以定为零** —— 一台笔记本就够。

## 4 · manim 场景源码 —— 引用，不复制

稀疏 clone 在 `~/3b1b-videos/`（1.4 MB，只含 `_2024/transformers/` + 根目录必需文件）。

| 文件 | 场景数 | 内容 |
|---|---|---|
| `mlp.py` | 12 | MLP、Michael Jordan、superposition |
| `attention.py` | 19 | 注意力机制（第 3 节） |
| `embedding.py` | 19 | 词向量、方向有含义 |
| `ml_basics.py` | 12 | 参数即旋钮、线性回归类比 |
| `auto_regression.py` | 10 | 一个词一个词往外蹦 |
| `chm.py` / `supplements.py` | 26 / 76 | 杂项与补充 |

同样不复制 —— 它是个独立 git repo，`git pull` 就能跟上游同步。

## 5 · 我们自己的东西（还没有）

第一课最终要产出的、**完全属于我们**的材料，将来放这里：

- [ ] 改写后的正文（中文，用自己的话）
- [ ] 自己画或自己渲的图（换中文标注的版本）
- [ ] 练习题与参考答案
- [ ] 讲的时候用的 slides

这一格现在是空的。**它填满的那天，这一课才算真的是我们的。**

---

## 加新材料时

1. 放进对应子目录，**在上面的清单里加一行**
2. 三件事必须写清楚：哪来的、什么许可、能不能公开
3. 拿不准许可就先标 ⏸，别直接 push
4. **产物入不入库看重建成本** —— 渲一张图要先搭 LaTeX + OpenGL 环境，所以图入库；
   但无论入不入库，**生成它的脚本必须在库里**
