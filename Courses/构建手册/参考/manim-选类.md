> 本文件原是私有 skill「manim-teaching-figures」的 references/toolbox.md，2026-09-25 并入课程构建手册。文中 `scripts/…` 指 `构建手册/脚本/`（render.sh）和 `tools/manim/`（check-loop.py）。

# manim 工具箱 —— 按「要讲什么」找类，不是按 API 分类

> 环境实测（2026-09-18，cc-tw）：`~/.venvs/manim/bin/manim`，**Manim CE 0.21.0**，
> LaTeX 齐全（`/usr/bin/latex` `pdflatex` `dvisvgm`），所以 `Matrix` / `MathTex` 直接能用。
>
> ⛔ **manim 不能输出 SVG。** 静态图走项目自己的 SVG 基元，manim 只管会动的那部分。

## 目录
- [先问一句：这一格值不值得动起来](#先问一句这一格值不值得动起来)
- [按教学主题选类](#按教学主题选类)
- [已实测的用法](#已实测的用法)
- [还没实测的](#还没实测的)

---

## 先问一句：这一格值不值得动起来

⭐⭐ **判据：增量只能来自「时间」或「第三维」。** 两个都不占，就别做成视频 ——
静态 SVG 更清晰、能选中、能被读屏、不用维护。

| 该动 | 不该动 |
|---|---|
| 「你得重复 N 遍」——&nbsp;重复本身就是时间 | 并列对比两种方案 |
| 「一步一步降下去」——&nbsp;过程就是内容 | 一张结构图 / 数据通路 |
| 「这个矩阵把空间掰成什么样」——&nbsp;形变是连续的 | 一组数字、一张口径表 |
| 曲面要转着看才知道是鞍还是碗 | 流程的先后（用箭头就够） |

---

## 按教学主题选类

大模型/基础设施课里反复要画的东西，对应关系如下。
**加 ✅ 的是我在这台机器上真跑过的**，其余标注见下一节。

| 要讲的 | 用什么 | |
|---|---|---|
| 权重矩阵、QKV 投影、注意力打分表 | `Matrix` `DecimalMatrix` `IntegerMatrix` `MobjectMatrix` | ✅ |
| 「线性层就是把整个空间掰弯」 | `LinearTransformationScene` + `apply_matrix()` | ✅ |
| 单独给某个物体套一个矩阵（不要整场景） | `ApplyMatrix` 动画 | ✅ |
| 动量、梯度场、优化器为什么会打转 | `ArrowVectorField` + `StreamLines` | ✅ |
| loss 地形、鞍点、要转着看才懂的曲面 | `ThreeDScene` + `ThreeDAxes` + `Surface` + `begin_ambient_camera_rotation` | ✅ 生产在用 |
| 曲线、真实训练数据、精度/溢出边界 | `Axes` + `ParametricFunction` + numpy 当场算 | ✅ 生产在用 |
| **高维画不出来 → 改画一列分量条** | 一堆 `Rectangle` + `always_redraw`，颜色＝符号、长度＝大小 | ✅ 生产在用 |
| 逐 token 生成、KV cache 一格格长出来 | `VGroup` of `Square` + `ValueTracker` | ✅ 生产在用 |
| 柱状对比、表格 | `BarChart` `Table` `MathTable` `IntegerTable` | ⚠️ 没实测 |
| 计算图 / 依赖关系 | `Graph` `DiGraph` | ⚠️ 没实测 |

**动词（怎么让它动）** ——&nbsp;`Transform` `TransformMatchingShapes` `ApplyMatrix`
`MoveAlongPath` `Rotate` `Homotopy`；强调用 `Indicate` `Circumscribe` `Flash`
`Wiggle` `ShowPassingFlash`。

⭐⭐ 但**我们最常用的不是这些动词，是 `ValueTracker` + `always_redraw`**：
把「第 t 秒画面长什么样」写成一个纯函数，然后让一个 tracker 从 0 走到 T。
好处是**每一帧都能由 t 算出来**，于是「首尾同一帧」这类不变量可以直接被验证，
而一串 `self.play(...)` 拼出来的时间线做不到这点。

---

## 已实测的用法

### 矩阵

```python
from manim import Matrix, DecimalMatrix
Matrix([["w_{11}", "w_{12}"], ["w_{21}", "w_{22}"]])   # 元素是 LaTeX
DecimalMatrix(np.array([[0.31, -1.2], [2.0, 0.07]]),
              element_to_mobject_config={"num_decimal_places": 2})
```

⛔ **`DecimalMatrix` 默认只保留 1 位小数** ——&nbsp;`0.31` 会显示成 `0.3`、
`0.07` 显示成 `0.1`。讲数值精度的图里这是**会把论点讲反**的默认值。

### 线性变换（亮底版 · 非默认）

> ⛔ **2026-09-19 起默认是原生黑底**（见 `04-绘图.md` 四、房规第 1 条），下面这套亮底配置只在某一幕非用亮底不可、
> 并且做过黑白并排对照之后才用。它是黑底定稿之前的写法，留着是因为亮底参数实测过、不用再摸。

`LinearTransformationScene` 默认是深色主题、而且**自带一对基向量**。
要接我们这套亮底课件，构造函数得这么写（实测出图）：

```python
class Demo(LinearTransformationScene):
    def __init__(self, **kw):
        LinearTransformationScene.__init__(
            self,
            background_plane_kwargs=dict(
                background_line_style=dict(stroke_color="#c3c7cb", stroke_width=1),
                axis_config=dict(stroke_color="#202124")),
            foreground_plane_kwargs=dict(
                background_line_style=dict(stroke_color="#4285f4", stroke_width=1.6,
                                           stroke_opacity=0.65),
                axis_config=dict(stroke_color="#202124")),
            show_coordinates=False, **kw)
    def construct(self):
        self.camera.background_color = WHITE      # ⭐ 要在 construct 里设
        self.add_vector([1.0, 0.5], color="#d93025")
        self.apply_matrix([[1.5, 0.9], [0.1, 0.7]], run_time=1.2)
```

其它有用的开关：`show_basis_vectors=False`（不要那对绿/红基向量）、
`leave_ghost_vectors=True`（变换后留一份变换前的淡影，**讲「变成了什么」时很好用**）。

### 向量场 / 流线

```python
f = lambda p: np.array([-0.35 * p[1], 0.10 * p[0], 0])
self.add(ArrowVectorField(f))
s = StreamLines(f, stroke_width=2, max_anchors_per_line=30)
self.add(s); s.start_animation(warm_up=False, flow_speed=1.2)
```

⚠️ `ArrowVectorField` 默认按模长上一套彩虹色。教学图里通常要换成单色系
（传 `color=` 或 `colors=[...]`），否则**颜色在传达一个你没打算传达的量**。

---

## 还没实测的

`BarChart` `Table` 一家、`Graph`/`DiGraph`、`ComplexPlane`、`PolarPlane`、
`Prism`/`Torus`/`Cone` 这些 3D 体。要用之前**先起一个 `--draft` 冒烟**，
别直接写进正式图里。

