# 第一课 · 教学材料

这一课要用到的东西**都放在这里**，不散在别处。

每样材料都标三件事：**哪来的、什么许可、能不能进公开仓库**。
最后一列很重要 —— 这个仓库是 public 的，往里放东西等于对外发布。

---

## 清单

| 目录 / 文件 | 是什么 | 出处 | 许可 | 进公开库？ |
|---|---|---|---|---|
| `3b1b文字版/` | 4 章官方文字版 | 3blue1brown.com（Justin Sun 改编） | CC BY-NC-SA 4.0 | ✅ 入库 |
| `3b1b图/` | 自己渲的 6 张 PNG | 3b1b/videos 场景代码 | CC BY-NC-SA 4.0 | ✅ 入库 |
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
| `3b1b-gpt.md` | Ch5 Transformers, the tech behind LLMs | 整体结构、embedding、softmax、参数账 |
| `3b1b-attention.md` | Ch6 Attention, step-by-step | 只取参数账，机制留第二课 |
| `3b1b-mlp.md` | Ch7 How might LLMs store facts | 智商从哪来 + superposition |

抓取方式：`curl https://r.jina.ai/https://3blue1brown.com/lessons/<slug>`，
末尾的赞助者名单已剔除。**这是原文，不是我们的改写** —— 引用时要转成自己的话。

## 2 · `3b1b图/` — 已渲出的图

| 文件 | 画的是什么 | 打算用在 |
|---|---|---|
| `MLPIcon.png` | 经典神经网络图（输入 → 4× 宽中间层 → 输出） | 第 3 节开篇 |
| `BreakDownThreeSteps.png` | **MLP 三步全图**（升维 → ReLU → 降维） | 第 3 节主图 |
| `AlmostOrthogonal.png` | 近似垂直的向量 | 第 3 节 superposition |
| `StackOfVectors.png` | 一摞向量 | 备用 |
| `NonlinearityOfLanguage.png` | 语言的非线性 | 第 3 节引入 ReLU |
| `ClassicNeuralNetworksPicture.png` | 层与连线示意 | 备用 |

全部 1920×1080。渲染步骤见 `../RENDER.md`，脚本 `../render-scenes.sh`。

**这些是可以重新生成的产物**，所以不进版本库 —— 改一行代码重渲一遍就是了。
源头是场景代码，不是这些 PNG。

> 还差的图：`BasicMLPWalkThrough`（Michael Jordan 全程）、`Superposition`、
> `ShowAngleRange` —— 这三个 LaTeX 编译失败，原因未定位，见 `../RENDER.md`。

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
| `attention.py` | 19 | 注意力机制（第二课） |
| `embedding.py` | 19 | 词向量、方向有含义 |
| `ml_basics.py` | 12 | 参数即旋钮、线性回归类比 |
| `auto_regression.py` | 1 | 一个词一个词往外蹦 |
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
4. **能重新生成的产物（渲出的图、编译产物）不入库** —— 入库的是生成它的东西
