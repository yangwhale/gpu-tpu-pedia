# 自己渲 3Blue1Brown 的图 —— 可复现步骤

> ⚠️ **这套工具当前不在课程主线上。** 课程 2026-08-21 转向拆 DeepSeek V3，
> 3b1b 那批图讲的是标准 Transformer，跟 MLA / MoE 对不上，已从课程移除。
> 工具留着是因为它能一条命令重建全部 28 张图，以后要画图时可能还用得上。

2026-08-21 在 cc-tw（Ubuntu，无显示器，无 GPU）上跑通。
**28 个场景全部渲出**（`mlp` / `attention` / `embedding` / `ml_basics` / `auto_regression` 五个文件），
一张 1080p 静帧 3-80 秒，纯 CPU 软件渲染。

渲出来的不是截屏，是**原始矢量场景**，所以可以改：换中文标注、换数字、
挑任意一帧、调颜色、只保留想要的元素。

---

## 为什么能这么干

| 东西 | 在哪 | 许可 |
|---|---|---|
| manim（渲染引擎） | [3b1b/manim](https://github.com/3b1b/manim) | MIT |
| 场景代码（= 那些图） | [3b1b/videos](https://github.com/3b1b/videos) `_2024/transformers/` | CC BY-NC-SA 4.0 |

本课程整体采用同一许可，见 [`Courses/LICENSE.txt`](../../LICENSE.txt)。

---

## 快速开始

```bash
./setup-render-env.sh          # 一键搭环境（幂等，可重复跑）
./render-scenes.sh             # 批量渲那 28 个场景
./render-scenes.sh attention   # 只渲名字含 attention 的
```

默认路径：venv `~/manim-venv`、场景源码 `~/3b1b-videos`、
产物 `~/3b1b-render/frames`、日志 `~/3b1b-render/logs`。用同名环境变量可覆盖。

`setup-render-env.sh` 做四件事：装依赖 → 拉场景源码 → **修 manimgl 的一个真 bug**
→ 批量造齐作者私有素材的替身。下面是它背后的细节。

### 手动做的话

```bash
# 1 · 系统依赖
sudo apt-get install -y \
  libcairo2-dev libpango1.0-dev pkg-config python3-dev \
  libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev mesa-utils \
  xvfb ffmpeg \
  texlive texlive-latex-extra texlive-fonts-extra texlive-science dvisvgm

# 2 · 场景源码（稀疏 clone，1.4 MB）
git clone --filter=blob:none --sparse --depth 1 \
    https://github.com/3b1b/videos.git ~/3b1b-videos
cd ~/3b1b-videos && git sparse-checkout set _2024/transformers custom

# 3 · Python 环境
python3 -m venv ~/manim-venv
~/manim-venv/bin/pip install manimgl "setuptools<81"
~/manim-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu

# 4 · 渲一张
cd ~/3b1b-videos
xvfb-run -a -s "-screen 0 1920x1080x24" \
  ~/manim-venv/bin/manimgl _2024/transformers/mlp.py MLPIcon \
  -w -s --hd --video_dir ~/3b1b-render/frames
```

还要改 `custom_config.yml` 的 `base:`（原本写死作者的 Dropbox 路径），
以及打下面那个 numpy 补丁、造私有素材替身 —— 这些 setup 脚本都做了。

---

## 六个坑

每一个都会让你以为环境坏了。按踩到的顺序排。

### 🔴 1 · `-s` 单独用会挂住，必须 `-w -s` 一起

`--help` 把 `-s` 写成 "Save the last frame"，**但它不写文件** ——
它只是跳过动画、把最后一帧显示在交互窗口里，然后**停在那儿等键盘**。

headless 下的表现：进程活着、吃 19% CPU、零输出、永不退出。非常像卡死。

唯一的线索是日志里那行 `Press command + q or esc to quit`。
**加上 `-w` 就正常**：1.8 秒出图并退出。

> 我在这上面绕了三轮，先后怀疑过 OpenGL、llvmpipe 太慢、场景太复杂。
> 真正定位它的是 `ps -o wchan` 显示进程停在 `do_wait` —— **不是在算，是在等**。

### 🔴 2 · `setuptools` 必须 < 81

manimgl 1.7.2 顶层 `import pkg_resources`，setuptools 81 起移除了它。
默认装 84，于是一启动就 `ModuleNotFoundError: No module named 'pkg_resources'`。
`pip install setuptools` 反而装成更新的版本，**必须钉版本**。

### 🔴 3 · manimgl 自己的 bug：np.float32 不被当成标量

**这是真 bug，不是配置问题。** 影响所有带 `VFadeIn` / `VFadeOut` 的场景。

链条是这样的：

1. `get_stroke_opacity()` 从 **float32** 数组取值 → 返回 `np.float32`
2. `mobject.py` 里判断 `isinstance(opacity, (float, int))` → **False**
   （numpy 里只有 `np.float64` 是 python `float` 的子类，`np.float32` 不是）
3. 于是走 `np.array(np.float32(0.5))` → 得到一个 **0 维数组**
4. `resize_with_interpolation` 对它取 `len()` → `TypeError: len() of unsized object`

一行修好（`mobject.py:1363`）：

```python
- if not isinstance(opacity, (float, int)):
+ if not isinstance(opacity, (float, int, np.floating, np.integer)):
```

`setup-render-env.sh` 会自动打这个补丁并备份原文件。
**修完 `Superposition` 和 `BasicMLPWalkThrough` 立刻就通过了。**

### 🟡 4 · 装到一半的 texlive 会给出误导性的错误

这个坑最阴。第二轮测试时刚 `which latex` 确认存在就开跑，结果三个场景报
`LaTeX compilation failed`，于是我以为是场景内容有问题，还去手写 tex 文件
逐个验证宏包、验证 dvisvgm、验证 manim 自己的 `Tex` ——**全都通过**，越查越糊涂。

真相是：**apt 装完 latex 二进制 ≠ 宏包索引就绪**。等 texlive 完全装完再跑，
`ShowAngleRange` 自己就好了；剩下两个暴露出来的才是上面那个真 bug。

> 教训：**环境正在变化时测出来的失败不能当结论。**
> setup 脚本里显式跑了一次 `mktexlsr` 就是为了压掉这个窗口。

### 🟡 5 · 有些场景要作者私有的图片和数据

`custom_config.yml` 的 `base:` 指向他的 Dropbox，里面有 `images/raster`、
`pi_creature`、章节缩略图等**不在 repo 里**的素材。例如：

```
OSError: /Users/grant/.../Thumbnails/Chapter5_TN5 not Found
FileNotFoundError: .../data/athlete_sports.txt
```

**做替身即可**，setup 脚本已包含：

- 两张自制占位缩略图（黑底 + 章节标题，明确标着 placeholder）
- 一份 `athlete_sports.txt`（每行一句 "Michael Jordan plays basketball"）

这两样是**我们自己造的**，不涉及版权，而且课程本来也要换成自己的图。

### 🟡 6 · 场景代码里的绝对路径要改

`mlp.py:119` 硬编码了作者的 Dropbox 路径。setup 脚本把它改成
`Path.home() / "3b1b-render" / "thumbnails"`。

注意 **`mlp.py` 顶部没有 `import os`**，所以别用 `os.path.expanduser` ——
用 `Path`（它由 `helpers.py` 的 `import *` 带进来）。

---

## 实测结果 · 28 个场景

先在 `mlp.py` 上迭代把环境问题挨个清掉，再横向铺开到另外四个文件：

| 轮次 | 环境状态 | `mlp.py` 通过 |
|---|---|---|
| 1 | 无 LaTeX | 3 / 10 |
| 2 | texlive 装到一半 | 6 / 10（其中 1 个是假失败） |
| 3 | texlive 就绪 | 7 / 10 |
| 4 | **+ numpy 补丁 + 素材替身** | **10 / 10** ✅ |

铺开之后又补掉两类问题（缺 `gensim` / `transformers`；`np.product` 在 numpy 2 已移除），
最终 **28 张图**：

| 文件 | 场景 | 典型耗时 |
|---|---|---|
| `mlp.py` | 10 个（含 `BasicMLPWalkThrough` `BreakDownThreeSteps` `Superposition`） | 3-24 s |
| `attention.py` | 6 个（含 `AttentionPatterns` `ShowMasking` `DescribeAttentionEquation`） | 4-22 s |
| `embedding.py` | 5 个（`IntroduceEmbeddingMatrix` `DotProducts` 等） | 3-23 s |
| `ml_basics.py` | 4 个（`TweakedMachine` `ShowGPT3Numbers` `SoftmaxBreakdown` 等） | 11-35 s |
| `auto_regression.py` | 3 个（`SimpleAutogregression` 要加载 GPT-2，最慢） | 10-79 s |

全部 1920×1080，合计 6.1 MB，已收进 [`教学材料/3b1b图/`](教学材料/3b1b图/)，
当时按五节归过位。

### 两个渲不出来的（原因清楚，不修）

| 场景 | 原因 |
|---|---|
| `attention/IntroduceValueMatrix` | **上游场景与当前 manimgl 不兼容** —— `FadeTransform` 里往 VGroup 塞 `ImageMobject`。跟占位素材无关，用作者原图一样会挂 |
| `embedding/KingQueenExample` | 需要 `gensim.downloader` 拉 1.6 GB 词向量模型，成本不值。`ThreeDSpaceExample` / `ManyIdeasManyDirections` 表达同一个意思 |

---

## 还没做的

- [ ] **抓中间帧** —— `-s` 只存最后一帧。Michael Jordan 那个例子在
      `BasicMLPWalkThrough` 的动画中段，要用 `-n <动画序号>` 取。
- [ ] **占位素材换成我们自己的图** —— 现在那些标着 placeholder 的方块
      （`Dalle3_*`、`RiverBank`、`AttentionPaperStill` 等）能让场景跑起来，
      但上课时得换成真图或干脆裁掉
- [ ] 渲成 mp4 动画（不加 `-s`）—— 软件渲染下速度未知
- [ ] **中文标注** —— manim 的 `Text` 走 pango，理论上支持中文，需装中文字体后实测；
      `tex_templates.yml` 里另有一个 `ctex` 模板走 xelatex
- [ ] 还没碰的：`chm.py` 26 个、`supplements.py` 76 个、`generation.py`、`network_flow.py`
