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

## 规矩

1. **视频里一个字都不放。** 文字由页面图注和静态图承担 ——&nbsp;
   视频里的字既不能选中也不能被读屏。
2. **静态图必须排在视频上面**，视频加载不出来也要能读懂。
3. `<video>` 一定要带 `aria-label`，把画面里发生的事写清楚。
4. **转满整圈**（或首尾同帧），这样 `loop` 起来看不见接缝。
5. 配色跟 `topic03_draw` 的主色对齐。

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

⚠️ 渲染要 **2 分半**（1080p60 / 8 秒 / 3D 曲面），不在 `build-all.sh` 里 ——&nbsp;
**产物直接进仓库**，改脚本才需要手动重渲。
