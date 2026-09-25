# 10 · 用 Manim 画 3Blue1Brown 式的动画

> 课程里 25 支会动的图（专题四 7 支、专题五 18 支）都是这么做的：
> 用 3Blue1Brown 那个作者自己写的动画引擎 Manim，按他的原生风格（黑底、他那套配色），
> 每支 5–40 秒、无声、无缝循环，嵌在课件的静态图旁边（多步过程片每一步会自动停住，点「下一步」再走）。
>
> 这一篇从头讲一遍：什么时候值得做、环境怎么装、一支动画从写到上线的完整流程、
> 两种写法（附可运行模板）、怎么保证循环不跳、怎么借 3Blue1Brown 的讲法而不抄他的代码、踩过的坑。
>
> 这一篇是课程仓库里 Manim 规矩的**公开正本**；私有工作副本（agent 用的 skill）改了规矩要同步回来。

---

## 一、先判：它该不该动

**判据只有一句：增量只能来自「时间」或「第三维」。两个都不占，就画静态图。**

| 该动 | 不该动 |
|---|---|
| 「你得重复 N 遍」—— 重复本身就是时间（环形通信转三步） | 并列对比两种方案 |
| 「一步步降下去」—— 过程就是内容（梯度下降） | 结构图、数据通路 |
| 「这个矩阵把空间掰成什么样」 | 一组数字、一张口径表 |
| 曲面要转着看才知道是鞍还是碗（鞍点） | 流程先后（箭头就够） |
| 「看着它慢」—— 同样的活，一种摆法计时器多跳好多格 | 一个结论 |

还有两条：

- **Manim 不能输出 SVG**（只有 png／gif／mp4／webm／mov）。课程的静态图是内联 SVG，能选中、能搜索、读屏能读、所有 lint 都建立在它上面 ——
  所以**动画永远是静态图的补充，不是替代**。静态图排在上面，视频加载不出来也要能读懂。
- **多一段视频就多一份维护。** 静态图已经说清楚的，不要再拍。

每支都要在文件头或场景类前的注释里写明「为什么非动不可」（专题五有三支是 2026-09-25 对照本手册才补上的）。例：

| 动画 | 为什么非动不可 |
|---|---|
| `topic04-descend` 梯度下降 | 一维 → 二维 → 承认高维想不出来 → 改画一列条，这条叙事线只能在时间里展开 |
| `topic04-saddle` 鞍点 | 三维形状，二维只能切两个剖面 |
| `topic05-pipeline` 流水线气泡 | 先演 4 个 micro-batch 再演 8 个，灰色气泡从 3/4 缩到 3/8，「缩」只有放在时间里才看得见 |
| `topic05-tpsplit` 张量并行 | 静态图画得出结构，画不出「中间那一段真的没有通信，通信只在最后一下」 |
| `topic05-meshmap` 摆到机器上 | 两种摆法的计时器一格一格跳（17 对 73），差距是「看着它慢」出来的 |

---

## 二、环境

```bash
python3 -m venv ~/.venvs/manim && ~/.venvs/manim/bin/pip install manim   # Manim Community，实测 0.21.0
# 还需要：ffmpeg / ffprobe（压缩与查接缝）、LaTeX（只有用 MathTex 公式时才需要）
# 中文用 Text()，走 Pango，不需要 CJK LaTeX 宏包
```

`render.sh` 默认用 `~/.venvs/manim/bin/manim`，别的路径用 `MANIM_BIN=... render.sh ...` 覆盖。

---

## 三、一支动画的完整流程

```
① 判该不该动（第一节）
② 写场景：tools/manim/topic0N-anim-<名>.py，文件头写「为什么非动不可 ／ 数据从哪来 ／ 刻意没画什么」
③ 草稿：render.sh ... --draft        → 480p，几秒钟；构图、时序、接缝在草稿上都看得出来
④ 反复改到满意（草稿一版几秒，正式渲染一版几分钟，所以在草稿上试二十轮）
⑤ 正式渲染：render.sh ... WebPages/media/topic0N-<名>.mp4   → 1080p60 渲染、压到 960 宽、查首尾接缝
⑥ 看拼接图 /tmp/loopdiff-<名>.png（上＝首帧，下＝末帧）—— 必须看
⑦ 登记基线：check-loop.py --update，然后手改 loop-baseline.json 里这一条的 note，写上人眼判定结论
⑧ 嵌进课件（第七节），图注写「N 秒无声循环，Manim 渲染」，aria-label 复述画面里的每一句话
⑨ bash build-all.sh（check-loop 会跟基线比、会核对图注秒数）
```

具体命令（在 `Courses/` 下）：

```bash
# ③ 草稿
构建手册/脚本/render.sh tools/manim/topic05-anim-cut.py PDDisagg /tmp/draft/topic05-pd.mp4 --draft

# ⑤ 正式
构建手册/脚本/render.sh tools/manim/topic05-anim-cut.py PDDisagg WebPages/media/topic05-pd.mp4

# ⑦ 登记基线（新片子没有基线会报红，逼你看过拼接图再登记）
python3 tools/manim/check-loop.py --media WebPages/media --baseline tools/manim/loop-baseline.json --update
#   然后打开 tools/manim/loop-baseline.json，把这一条 note 里的「⚠️ 未经人眼判定」改成你的判定，例：
#   "2026-09-24 人眼看过拼接图：首末帧同一幕（标题和三条车道标签），差值是压缩噪声"
```

几个数：全课 25 支，每支 5–25 秒，压完 56–285 KB，可以直接进仓库。
渲染耗时看**同时在场的 mobject 数量**，不看时长：简单场景 1080p 两到五分钟，几百个 mobject 的重场景连 480p 草稿都要三分半。
**渲染不在 `build-all.sh` 里**（产物直接进仓库，改了脚本才手动重渲）；`build-all.sh` 只读 mp4 查接缝，1 秒多。

⚠️ 正式渲染放后台跑，并给它挂一个完成后会叫醒你的监控 —— 否则「渲完我再看」说出口就落空了。

---

## 四、房规

> 三条是身份，不是偏好：**原生黑底和配色、每一支都无缝循环、画面里的字全部在 aria-label 里复述。**

1. **原生黑底**，不写 `background_color`。
   ```python
   from manim import BLUE, RED, GREEN, YELLOW, WHITE, GREY, GREY_B
   #   BLUE #58C4DD   RED #FC6255   GREEN #83C167   YELLOW #F7D96F
   #   GREY #888888   GREY_B #BBBBBB      线宽默认 4，别往下调
   ```
   证据：扒 3Blue1Brown 神经网络系列三个源文件，`background_color` 出现 0 次；用色频率 YELLOW 103／WHITE 101／BLUE 78／RED 70／GREEN 44。
   白底一度被试过：黄是他的第一主色，白底上黄几乎看不见 —— 白底直接封杀了他最常用的颜色。黑白并排对照之后定了黑底。
   - 黄是**强调色**不是语义色，不能拿来当正负号（他讲分量符号用的也是蓝／红）。
   - 黑底的辅助线要比白底**亮**：`GREY_D #444444` 在黑底上整幕消失，用 `GREY #888888`。
   - 要跟静态图配色一致时，可以扩用 Manim 自带的其它颜色（专题五四张卡要对应静态图的蓝、橙、绿、紫，就用了 `ORANGE` 和 `PURPLE_B`）。
2. **全部循环，不给播放器**。唯一写法 `<video autoplay loop muted playsinline>`。
   长的叙事片也循环（最长的一支 24.6 秒）：**结尾把画面恢复成第 0 帧的样子**。
   > 原话：「这种能放一个动图搞定的东西，就不要放一个视频啦。」
   ⭐ 唯一的例外是第 7 条的「下一步」：多步过程片仍然 `autoplay loop`，只是在每一步落地时自动暂停，不给进度条和音量。
3. **文字为讲解服务**：可以放 `MathTex` 公式、`Text` 中文短句；但**视频里的字选不中、读屏读不到、搜索搜不到**，
   所以 `aria-label` 必须复述画面里出现过的每一句话。别把静态图已经说清的话再抄进画面。
4. **数据全部当场算**：等高线真二分、轨迹真迭代、调度真排、精度边界真从位模式读。参数也算数据 ——
   与其手调到「看着对」，不如把「要满足的条件」写成筛子去搜一遍，再用断言钉住。
5. **`rate_func` 分两种**：

   | play 在动什么 | `rate_func` |
   |---|---|
   | 一个物体（`mob.animate.*`、`FadeIn`、`Write`、`Indicate`……） | 默认 `smooth`，别显式写 linear |
   | 一个时钟（`ValueTracker.animate.set_value`，后面挂着 `always_redraw`） | **必须显式 `rate_func=linear`** |

   `smooth` 把 play 两端的速度压到零：动物体时是自然的收势，驱动时钟时是让时间停摆 —— 实测每个段界前画面整整一秒完全静止。
6. **「不够 fancy」的解药是信息密度，不是时长。** 动手前先判：它缺的是「解释」（观众看得懂画面但不知道在说什么概念）
   还是「铺陈」（动作太快、讲不完）？缺解释就加公式和字幕，时长不动；缺铺陈才放长。

7. **一步一步的过程，要让人看得出「第几步、谁发给谁」**（2026-09-25 现场纠正）。原话：
   「你怎么 A 往右，然后剩下都往左。它应该是环形的，大家都往右发，到头了转一圈回来……第一步慢点，还没看明白呢，停一下再跳第二步。」
   - **运动方向必须跟算法的方向一致。** 环上卡 3 → 卡 0 那一块原来是横穿画面往左飞的，看上去就成了「别人都往左」。
     改成：先往右出画面，沿卡片下面一条车道绕到最左，再往右进卡 0。**绕回也是往右**。
   - **别用「直接飞到目的地」的逻辑视图代替真实步骤。** AllGather／ReduceScatter 原来是块直接飞到终点，一半往左一半往右；
     改成按环一步一步走，跟 1.5 的环、跟通信库的实际做法一致。
   - **每一步落地后停住**，字幕写「第 s 步完成：……」；飞 1.8 秒、停 2 秒。
   - **停顿做成页面上的「下一步」**：场景里每一次停顿记下时刻，写进 `tools/manim/steps/<Scene>.json`；
     课件把它挂到 `<video data-pauses>`，播放到那里自动暂停，点「下一步」再走，勾「连续播放」就不停。图注秒数也从这份 json 取。
   - 落点被占就换布局：AllToAll 只用一列时，落点上还坐着没寄走的块（草稿里撞在一起过），改成每张卡「寄出」「收到」两列。

---

## 五、两种写法

### 写法 A：时钟式（推荐，专题四的主力）

一个 `ValueTracker` 当时钟，画面全部写成「第 t 秒长什么样」的纯函数（`always_redraw`），让时钟从 0 走到 T。

- 好处：每一帧都能由 t 算出来。**结尾把时钟拨回 0，首帧就回来了**，首尾同帧是免费的。
- 字幕也挂在时钟上，按区间切 opacity —— 复位时字幕自动回到第一句。不要用 `FadeIn／FadeOut` 序列切字幕，那样结尾还得手动补一遍复位。
- 复位**只在一个地方定义**（拨回时钟，或让时钟倒着走回去），不要在每个绘制函数里各写一份 `if`。

可以直接渲染的模板：[模板/anim-模板.py](模板/anim-模板.py)（环形 ReduceScatter，8.5 秒）。骨架：

```python
class RingReduceScatter(Scene):
    def construct(self):
        t = ValueTracker(0.0)                           # ① 唯一的时钟
        self.add(always_redraw(lambda: draw_at(t.get_value())))   # ② 画面 ＝ t 的纯函数

        texts = [Text(s, ...) for _, s in caps]          # ③ 字幕建一次，按区间切 opacity
        for i, m in enumerate(texts):
            show_caption(m, i)                           #    ⛔ 先手动算一次（见下）
            m.add_updater(lambda m, i=i: show_caption(m, i))
            self.add(m)

        for s in range(1, N):                            # ④ 走时钟，必须 linear
            self.play(t.animate.set_value(s * STEP_T), run_time=1.0, rate_func=linear)
            self.wait(STEP_T - 1.0)

        self.play(t.animate.set_value(0.0), run_time=1/30, rate_func=linear)   # ⑤ 复位
        self.wait(0.5)
```

⛔ 写这个模板时当场踩了一次：updater 要等第一次 `play` 才开始跑，第 0 帧三句字幕全是不透明的，叠成一团 ——
**拼接图一眼就看出来了，而不一致度只有 13%，看数字根本不会怀疑。** 所以加字幕 updater 时，先手动调一次把初始状态算对。

⛔ 另外两条：
- `always_redraw` 的返回值要留引用，否则之后再也拿不到它。
- 别把 `Text` 放进 `always_redraw` 里每帧重建：Pango 排版不便宜，16 秒 × 60 帧 ＝ 近千次重排。建一次，改 opacity。

### 写法 B：分幕式（专题五的大多数）

一串 `self.play`，一幕一幕往上加东西；配一个 `say()` 小函数换字幕。适合「一格一格长出来」的时间线（流水线、PD 分离）。

```python
sub = cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2)
self.add(sub)

def say(t, color=GREY_B):                # 换一句字幕：旧的淡出、新的淡入
    nonlocal sub
    n = cap_text(t, color, 24).next_to(title, DOWN, buff=0.2)
    self.play(FadeOut(sub), FadeIn(n), run_time=0.35)
    sub = n

shown = VGroup()                          # ⭐ 所有中途加进来的东西都收进这一个组
say("放在一起：大家一步一步 decode，每格出一个字")
...   self.play(FadeIn(tick)); shown.add(tick)   ...

# 收尾复位：清干净，再摆一个跟第 0 帧一模一样的静态件
self.play(FadeOut(shown), FadeOut(sub), run_time=0.6)
self.remove(shown, sub)                   # ⛔ FadeOut 之后补一刀 remove，别留残影
self.add(cap_text(" ", GREY_B, 24).next_to(title, DOWN, buff=0.2))   # ＝ 第 0 帧那个空字幕
self.wait(0.6)
```

分幕式的复位要自己保证。通用收尾模板：**清干净，再摆一个跟第 0 帧一模一样的静态件**。
不要试图去唤醒一个中途已经被移走的 `always_redraw` —— 二十几个 play 之后，你很难记清一个 mobject 还在不在场上。
确定性比聪明重要。

---

## 六、循环为什么必须看拼接图

`check-loop.py` 把首帧和末帧的差异量成一个百分比。但**一个标量分不开「末帧是另一幕」和「末帧偏了一个像素」**：

- 鞍点那支（`topic04-saddle`）只转了不到半度，就量出 51.2%，跟首末帧差着一整幕的梯度下降那支同一档。
- 首帧那颗小球没了，不一致度 22.9%，跟正常循环同一档 —— 一颗小球在整幅画里几乎不占面积。
- 本篇模板的字幕叠成一团，13.2%。

所以它做成**基线回归**，不是阈值判定：

- 第一次渲出来，人看拼接图，判定「首末帧是同一幕，差值是压缩噪声」之后登记基线，note 里写判定结论。
- 之后每次 `build-all.sh`，只问「有没有比记录的更糟」（容差 3%）。
- 新片子没有基线直接报红。
- 图注里写的「N 秒」也会拿实测时长去对（容差 0.6 秒）—— 改了时长、忘改图注，构建会拦。

它当初被接进构建，是因为发现三支已上线的片子**每轮循环都在跳**，而没人量过 ——
「构建全绿」只覆盖单帧的几何和文字，盖不到产物的时间维度。

---

## 七、嵌进课件

多支动画并排放在一个网格里，排在对应静态图下面：

```html
<div class="animgrid">
<figure class="animcell" id="anim-pd">
<video src="media/topic05-pd.mp4" autoplay loop muted playsinline
       aria-label="PD 分离动画。上面一行是放在一起……（复述画面里的每一句字幕）"></video>
<figcaption><b>一句话说看什么。</b>
  <span class="sub">（11 秒无声循环，Manim 渲染。）</span></figcaption></figure>
</div>
```

- `id` 用 `anim-` 开头：讲义里写 `anim-pd`，讲义构建时会断言这个 id 在课件里存在。
- 图注的补充部分写给读者看的「刻意没画」：「格数是示意」「画的是不带因果掩码的情形」。
- aria-label 最好从场景里的标题和字幕字符串生成，只留一个来源 —— 手抄会漏（专题五七支原语动画都漏了标题后半句，对账时补上）。
- 讲义里放动画那一步，写「先闭嘴等它转完一圈（11 秒）」—— 边放边讲，台下眼睛和耳朵会打架。

---

## 八、借 3Blue1Brown：借结构，不抄代码

**版权**：Manim 本身是 MIT，随便用；但 3Blue1Brown 视频仓库（`3b1b/videos`）里的内容是 **CC BY-NC-SA 4.0** ——
抄代码或画面会把这个协议传染给你的仓库。**可以借的是思想**：叙事顺序、编码手法、教学结构。借了在图注里写明出处。

一个完整的例子 —— 专题四梯度下降那支（`tools/manim/topic04-anim-descend.py`）：

- **骨架是从他那一集源码的类名序列读出来的**，不是抄画面。他那一集 61 个 Scene 排下来正好是一条叙事线：
  ```
  SingleVariableCostFunction   → 先在一维上讲通
  TwoVariableInputSpace        → 升到二维看形状
  CostSurface
  ConfusedAboutHighDimension   → 主动承认：再往上想不出来了
  NonSpatialGradientIntuition  → 于是不再画空间
  ```
- 最后那一步是整段最值得学的手法：**高维不画成空间，画成一列数** —— 颜色表示每个分量的符号，长度表示大小。
  「想象三千亿维」是不可能的，「看一列条在动」是可能的。
- 画面全是自己的：一维 loss 是自己造的曲线（要求是「有两个深浅不同的谷」，谷底让脚本自己扫出来并断言深浅拉得开）；
  二维那一幕真的在算等高线和梯度；第三幕那一列条的长度和颜色来自真实的梯度分量。

怎么读他的源码找结构：扒一集的 Scene 类名按文件顺序列出来，那就是他的分镜表；再看每个类的 `construct` 用了哪些编码（颜色表什么、长度表什么）。
Manim 自带的类别去手摆：`Matrix`、`LinearTransformationScene`、`VectorField`、`ThreeDScene`…… 按教学主题选类见 [参考/manim-选类.md](参考/manim-选类.md)。

---

## 九、踩过的坑（节选）

全部在 [参考/manim-踩坑.md](参考/manim-踩坑.md)，这里是最常撞的：

| 坑 | 症状 | 修法 |
|---|---|---|
| 字幕 updater 第 0 帧没跑 | 首帧几句字幕叠在一起 | 挂 updater 前先手动调一次 |
| 驱动时钟用了默认 `smooth` | 每个段界画面静止约一秒 | 驱动 `ValueTracker` 的 play 显式 `rate_func=linear` |
| 复位时 `set_opacity(1)` | 黑底上曲线被填成白色（它同时设了 fill） | 只恢复 stroke：`set_stroke(opacity=1)`、`set_fill(opacity=0)` |
| 复位去唤醒被移走的 `always_redraw` | 首帧那颗球回不来 | 清干净，摆静态件 |
| `self.wait()` 不一定驱动 updater | 末帧多了东西 | 用一个极短的 `play` 推一下时钟 |
| `FadeOut` 之后残影 | 下一幕还看得见上一幕的淡影 | `FadeOut` 之后补 `self.remove` |
| 造了一个偶函数当 loss | 两个谷必然等深，「落进不同的谷」没戏 | 断言挡下时先看它挡的是什么；加奇次项破对称 |
| `DecimalMatrix` 默认只留 1 位小数 | 0.31 显示成 0.3，讲精度的图论点被讲反 | 显式设小数位 |
| 下划线短名撞车 | `stroke_color=_W` 拿到了一个 numpy 数组 | 引入新名字前先 grep |
| 通用版 check-loop 默认扫 cwd | 守卫连着报绿，其实一支片子都没查 | 显式传 `--media` 和 `--baseline`；空集合不算通过 |

---

## 十、开工清单（一支动画）

- [ ] 判过：增量来自时间或第三维；静态图已经有了
- [ ] 文件头写了「为什么非动不可／数据从哪来／刻意没画什么」
- [ ] 原生黑底、房规配色；数据当场算，关键数有断言
- [ ] 驱动时钟的 play 全是 `rate_func=linear`
- [ ] 结尾恢复成第 0 帧（时钟拨回 0，或清干净摆静态件）
- [ ] 草稿看过；正式渲染后**看过拼接图**；登记了基线并写了 note
- [ ] 嵌进课件：`anim-` id、`autoplay loop muted playsinline`、aria-label 复述每一句、图注写秒数
- [ ] 讲义里写「先闭嘴等它转完一圈（N 秒）」
- [ ] `bash build-all.sh` 通过
