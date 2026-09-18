# tools/manim ——&nbsp;只在「动起来才讲得清」的地方用

⛔ **先看结论，省得下一个人再试一遍：**
**manim 不能替换本仓库的静态图。** 它的 `--format` 只接受
`png / gif / mp4 / webm / mov` ——&nbsp;**没有 SVG**（2026-09-18 在 0.21.0 上实测）。

而我们那套图是**内联 SVG**，靠它换来的不只是好看：

- 文字可选中、可搜索、**屏幕阅读器能读**（每张图都有 aria 描述）
- 投屏无损缩放
- **全部 lint 都建立在「它是 SVG」上** ——&nbsp;字号（MAXPROSE）、
  版面撞车、图内节号、墨水体检
- 它是文本，**进 git 能 diff**

换成 PNG 这些全丢。所以：**静态图继续用 `topic03_draw`，manim 只拍动画。**

## 什么时候值得拍一段

判据一句话：**这件事「动起来」才讲得清吗？**

- ✅ 值得：**鞍点的三维形状**（二维只能切两个剖面）
- 候选：学习率锯齿「越弹越大」、动量冲过浅坑 ——&nbsp;都是**时间现象**
- ❌ 不值得：任何静态图已经说清楚的东西。**多一段视频就多一份维护。**

## 规矩 ——&nbsp;⛔ 不在这里，在 skill 里

**房规的唯一出处是 `manim-teaching-figures` 这个 skill 的 `SKILL.md`。**
这里**不再复述**，只留一个指针。

> ⛔⛔ **为什么改成指针：** 这一段原本抄了一份「五条规矩」。
> 2026-09-18 房规松绑（第①条从「一个字都不放」改成「文字为讲解服务」）之后，
> 我改了 skill，却**漏了这一份和另外八处 docstring** ——&nbsp;
> 于是派出去的 agent 读到这里，理直气壮地按旧规矩拒绝执行，**而且它是对的**。
>
> ⭐⭐⭐ 判据：**一条规矩被复制 N 份，就等于有 N 份会过期。**
> 正确形状是**一份权威 ＋ N 个指针**。这跟绘图里
> 「复位只能定义在一个地方」是同一条道理，只是换到了文档上。

⭐ 仍然属于**本仓库特有**、skill 里没有的两条，留在这儿：

1. **静态图必须排在视频上面**（本仓库的 figure 结构），视频加载不出来也要能读懂。
2. 配色跟 `topic03_draw` 的主色对齐（`BL / RD / GR / OR / PU / GY / INK`）。

其余（文字能不能进画面、长度与循环怎么选、aria-label 的义务、
首尾同帧怎么验）一律以 skill 为准。

⚠️ **改 skill 之后记得重新部署**：源在 `~/CloseCrab/skills/`，
跑着的 agent 读的是 `~/.claude/skills/` 下的拷贝，不 `cp -a` 过去就不生效。

## 怎么渲

```bash
python3 -m venv ~/.venvs/manim && ~/.venvs/manim/bin/pip install manim
# 本机 ffmpeg / latex / dvisvgm 已有；venv 是因为系统 python 是 PEP 668 托管的

cd Courses
~/.venvs/manim/bin/manim --format=mp4 -qh --media_dir /tmp/manim-out \
    tools/manim/topic04-anim-saddle.py Saddle

# ⭐ 一定要压一道再进仓库：1080p60 直出 2.9 MB，压到 960 宽 crf30 只有 276 KB
ffmpeg -y -i /tmp/manim-out/videos/topic04-anim-saddle/1080p60/Saddle.mp4 \
    -vf "scale=960:-2" -c:v libx264 -crf 30 -preset slow \
    -pix_fmt yuv420p -movflags +faststart -an \
    WebPages/media/topic04-saddle.mp4
```

⭐ 现在有更省事的一条命令（渲染 → 压缩 → 查首尾接缝一步走完）：

```bash
LOOP_BASELINE=$PWD/tools/manim/loop-baseline.json \
  bash ~/.claude/skills/manim-teaching-figures/scripts/render.sh \
  tools/manim/topic04-anim-saddle.py Saddle WebPages/media/topic04-saddle.mp4
# 加 --draft 出 480p 草稿，迭代阶段一律用它
```

⚠️ 渲染耗时看**同时在场的 mobject 数量**，不看时长：
简单场景 1080p 要 2–5 分钟，重场景（几百个 mobject）**连 480p 草稿都要三分半**。
⛔ 渲染**不在 `build-all.sh` 里**，产物直接进仓库，改脚本才需要手动重渲；
但 `build-all.sh` 会跑 `check-loop.py` 查已进仓库的 mp4 首尾接不接得上。
