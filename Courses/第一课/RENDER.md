# 自己渲 3Blue1Brown 的图 —— 可复现步骤

2026-08-21 在 cc-tw（Ubuntu，无显示器，无 GPU）上实测跑通。
**一张 1080p 静帧约 3-4 秒**，纯 CPU 软件渲染。

渲出来的不是截屏，是**原始矢量场景**，所以可以改：换中文标注、换数字、
挑任意一帧、调颜色、只保留想要的元素。

---

## 为什么能这么干

| 东西 | 在哪 | 许可 |
|---|---|---|
| manim（渲染引擎） | [3b1b/manim](https://github.com/3b1b/manim) | MIT |
| 场景代码（= 那些图） | [3b1b/videos](https://github.com/3b1b/videos) `_2024/transformers/` | CC BY-NC-SA 4.0 |

许可的取舍见 `README.md` 末尾那一节。**这里只讲技术怎么做。**

---

## 一次性环境搭建

### 1 · 拿场景代码（稀疏 clone，1.4 MB）

```bash
git clone --filter=blob:none --sparse --depth 1 \
    https://github.com/3b1b/videos.git ~/3b1b-videos
cd ~/3b1b-videos
git sparse-checkout set _2024/transformers custom
```

根目录的 `manim_imports_ext.py`、`custom_config.yml` 会自动带上 —— 这两个是必须的。

### 2 · 系统依赖

```bash
sudo apt-get install -y \
  libcairo2-dev libpango1.0-dev pkg-config python3-dev \
  libgl1-mesa-dev libglu1-mesa-dev libegl1-mesa-dev mesa-utils \
  xvfb ffmpeg \
  texlive texlive-latex-extra texlive-fonts-extra texlive-science dvisvgm
```

- **pango / cairo** 不装的话 `manimpango` 编译失败，manimgl 根本装不上
- **mesa + xvfb** 提供无显示器的 OpenGL（走 llvmpipe 软件渲染，够用）
- **texlive** 给公式用 —— 不装的话**一多半场景直接失败**，见下面的坑

### 3 · Python 环境

```bash
python3 -m venv ~/manim-venv
~/manim-venv/bin/pip install manimgl "setuptools<81" torch --index-url https://download.pytorch.org/whl/cpu
~/manim-venv/bin/pip install manimgl "setuptools<81"
```

> `setuptools<81` 不是可选的 —— 见下面的坑。
> `torch` 是 `mlp.py` 第一行就 import 的，装 CPU 版即可（不需要 GPU）。

### 4 · 改配置指向本地

`custom_config.yml` 里的 `base:` 写死了 Grant 自己的 Dropbox 路径，必须改：

```yaml
directories:
  removed_mirror_prefix: "/home/<你>/3b1b-videos/"
  base: "/home/<你>/3b1b-render/"
camera:
  resolution: (1920, 1080)     # 原本是 4K，软件渲染下没必要
```

---

## 渲一张图

```bash
cd ~/3b1b-videos
xvfb-run -a -s "-screen 0 1920x1080x24" \
  ~/manim-venv/bin/manimgl _2024/transformers/mlp.py MLPIcon \
  -w -s --hd --video_dir ~/3b1b-render/frames
```

产物：`~/3b1b-render/frames/2024/transformers/mlp/MLPIcon.png`

**批量渲的脚本**：`/tmp/render3b1b.sh`（成功/失败逐条打印，带耗时）。
稳定下来之后应该挪进这个目录一起进版本库。

---

## 四个坑，每一个都会让你以为环境坏了

### 🔴 1 · `-s` 单独用会挂住，必须 `-w -s` 一起

`--help` 把 `-s` 写成 "Save the last frame"，**但它不写文件** ——
它只是跳过动画、把最后一帧显示在交互窗口里，然后**停在那儿等键盘**。

在 headless 环境下的表现是：进程活着、吃着 19% CPU、什么都不输出、永远不退出。
非常像卡死，实际上它在等你按 `esc`。

唯一的线索是日志里那行：

```
Tips: Using the keys `d`, `f`, or `z` you can interact with the scene.
Press `command + q` or `esc` to quit
```

**加上 `-w` 就正常了**：1.8 秒出图并退出。

（我在这上面花了三轮，先后怀疑过 OpenGL、llvmpipe 太慢、场景太复杂，
甚至去装了 py-spy 想抓栈。真正定位它的是 `ps -o wchan` 显示在 `do_wait`
—— 不是在算，是在等。）

### 🔴 2 · `setuptools` 必须 < 81

manimgl 1.7.2 顶层 `import pkg_resources`，而 setuptools 81 起把它移除了。
默认装的是 84，于是一启动就：

```
ModuleNotFoundError: No module named 'pkg_resources'
```

`pip install setuptools` 反而装成更新的版本，问题依旧。必须钉版本。

### 🟡 3 · 没有 LaTeX，一多半场景直接失败

实测 10 个场景：**3 个成功，6 个死在 `FileNotFoundError: 'latex'`**。
凡是画公式、带数学符号的都要。装 texlive 就好。

### 🟡 4 · 有些场景要 Grant 自己的图片资源，拿不到

`custom_config.yml` 的 `base:` 指向他的 Dropbox，里面有 `images/raster`、
`images/vector`、`pi_creature` 等**不在 repo 里**的素材。

例如 `LastTwoChapters` 要读章节缩略图：

```
OSError: /Users/grant/.../Thumbnails/Chapter5_TN5 not Found
```

`helpers.py` 里还有个 `WORD_FILE = OWL3_Dictionary.txt`（词表），同样不在。

**这类场景只能改代码绕开**（把缺的素材换成自己的，或删掉那部分）。
好在多数核心示意图不依赖外部素材。

---

## 实测结果 · `mlp.py` 全部 10 个场景

装 LaTeX **前后各跑一轮**，脚本 `render-scenes.sh`：

| 场景 | 装 LaTeX 前 | 装 LaTeX 后 | 耗时 | 画的是什么 |
|---|---|---|---|---|
| `MLPIcon` | ✅ | ✅ | 4 s | 经典神经网络图（输入 → 4× 宽中间层 → 输出） |
| `AlmostOrthogonal` | ✅ | ✅ | 3 s | 近似垂直向量 |
| `ClassicNeuralNetworksPicture` | ✅ | ✅ | 4 s | 层与连线示意 |
| `BreakDownThreeSteps` | ❌ latex | ✅ | 25 s | **MLP 三步全图**，第一课第 3 节主图 |
| `StackOfVectors` | ❌ latex | ✅ | 7 s | 一摞向量 |
| `NonlinearityOfLanguage` | ❌ latex | ✅ | 4 s | 语言的非线性 |
| `ShowAngleRange` | ❌ latex | ❌ **仍失败** | — | 角度分布（89°–91°） |
| `Superposition` | ❌ latex | ❌ **仍失败** | — | 叠加 |
| `BasicMLPWalkThrough` | ❌ latex | ❌ **仍失败** | — | Michael Jordan 全程 |
| `LastTwoChapters` | ❌ 素材 | ❌ 素材 | — | 章节回顾（要缩略图） |

**6 / 10 通过。** 装 LaTeX 让成功率从 3 涨到 6。

两张关键图已确认质量：
- `MLPIcon` —— 1920×1080，红蓝连线俱全，跟视频里一模一样
- `BreakDownThreeSteps` —— 标题 "Multilayer Perceptron"，
  左边升维矩阵的行向量 $\vec{R}_0 … \vec{R}_{n-1}$、中间 ReLU 折线、
  右边降维矩阵的列向量 $\vec{C}_0 … \vec{C}_{m-1}$，公式排版正常

### 剩下 3 个 LaTeX 失败还没定位

**不是环境问题**，已排除：

- 手写一份含 manim 全部 preamble 宏包的 `.tex`，`latex` 编译**通过**
- `dvisvgm` 转 SVG **通过**
- manim 自己的 `Tex(R"\vec{M} + \vec{J}")` 场景**渲染通过**
- 把 `Superposition` 里的三段 `TexText` 抠出来单独渲，**三段全过**

所以是这几个场景里**别的某段** tex 内容的问题。
manim 的 `full_tex_to_svg` 抛的是空 `error_str`，看不到 LaTeX 的原始报错 ——
要定位得给 `tex_file_writing.py` 打个补丁把 `.log` 打出来。**留待后续。**

---

## 还没解决的

- [ ] 定位剩下 3 个场景的 LaTeX 失败（需要 patch manim 拿到原始报错）
- [ ] `LastTwoChapters` 这类依赖 Grant 私有素材的，改代码绕开
- [ ] 其余章节（`attention.py` 19 个、`embedding.py` 19 个、`chm.py` 26 个）还没跑
- [ ] 动画（不加 `-s`，渲成 mp4）能不能跑 —— 软件渲染下速度未知
- [ ] 需要 GPT-2 权重的场景（`embedding.py` 里有几个 import `transformers`）
- [ ] 中文标注怎么加 —— manim 的 `Text` 走 pango，理论上支持中文，
      需要装中文字体后实测；`tex_templates.yml` 里另有一个 `ctex` 模板走 xelatex

> 渲出的 PNG 已收进 `教学材料/3b1b图/`，随库分发 ——
> 重建一次要装 LaTeX + OpenGL + manim，不该让每个读者都走一遍。
> 渲染时的工作目录仍是 `~/3b1b-render/frames/`。
