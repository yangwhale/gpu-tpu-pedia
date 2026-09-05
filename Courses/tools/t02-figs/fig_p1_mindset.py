# -*- coding: utf-8 -*-
"""图 P-1 —— 两种心智模型：你描述一个线程，还是描述整个数组。

这是第 5 节的第一张图，也是全课那个假设的正面陈述。前面三节都在讲
「硬件长什么样」，从这里开始讲「谁在安排」。

画法上刻意做了一件事：**左右两栏各放一段真代码，而且让差别出现在代码本身**，
不是出现在旁边的形容词里。CUDA 那段里有 `threadIdx`，JAX 那段里连下标都没有 ——
听众看一眼就知道「描述一个线程」和「描述整个数组」不是修辞。

下半部分是代价和红利各一栏。**两栏必须一样长** —— 只画代价会变成拉踩，
只画红利会变成宣传，而这一节的落点恰恰是「这是个取舍，不是个优劣」。
"""
from common import Fig, para, BL, RD, YL, GN, PU, INK, SUB, GREY, FILL

W = 1400
TOP = 84

LX, LW = 20, 676
RX, RW = 724, 656
COL_H = 268

CMP_Y = TOP + COL_H + 40
CMP_ROWS = [
    ("形状什么时候定",
     "运行时可变。同一份 kernel 喂什么形状都能跑",
     "<b>编译期必须确定</b>。形状一变就重新编译"),
    ("控制流",
     "随便写。<code>if</code> / <code>while</code> 想怎么分支怎么分支",
     "数据相关的分支<b>很贵</b>。MoE 路由这类写起来别扭"),
    ("优化发生在哪",
     "人写 kernel ＋ 调库。<b>优化是一处一处做的</b>",
     "编译器看得到整张图，<b>做全局优化</b>"),
    ("出问题怎么查",
     "打印、单步、profiler，<b>看得见每一行</b>",
     "读编译产物、读 profile。<b>看得见结果，看不见过程</b>"),
]
CMP_HH, CMP_RH = 40, 58
CMP_H = CMP_HH + len(CMP_ROWS) * CMP_RH

CARD_Y = CMP_Y + CMP_H + 42
CARD_H = 132
BAND_Y = CARD_Y + CARD_H + 26
H = BAND_Y + 108


def build():
    f = Fig(W, H, "两种心智模型对照：CUDA 描述单个线程的行为，XLA 描述整个数组的变换，"
                  "由此分出静态编译的代价与红利")
    f.title("同一件事，两种写法　—— 你描述<tspan font-weight=\"700\" fill=\"#d93025\">一个线程</tspan>"
            "，还是描述<tspan font-weight=\"700\" fill=\"#1e8e3e\">整个数组</tspan>")
    f.legend([(RD, "GPU / CUDA：SIMT，单线程视角"),
              (GN, "TPU / XLA：SPMD，整张图视角"),
              (GREY, "灰底＝这一侧付出的代价")])

    _col(f, LX, LW, RD, "GPU · CUDA　—— 写一个 kernel，描述<b>一个线程</b>干什么",
         ["__global__ void add(float* a, float* b, float* c, int n) {",
          "    int i = blockIdx.x * blockDim.x + threadIdx.x;",
          "    if (i < n) c[i] = a[i] + b[i];",
          "}"],
         "看这段代码里出现了什么：<r>blockIdx</r>、<r>threadIdx</r>、还有一句 <code>if</code> 边界判断。<b>「我是第几号、我负责哪一格」写在代码里</b>，"
         "所以切分方式是你决定的 —— 也只能由你决定。",
         "换个形状、换张卡，这段代码<b>照样跑</b>，但快不快就不好说了。")

    _col(f, RX, RW, GN, "TPU · JAX / XLA　—— 写整张图，描述<b>整个数组</b>怎么变换",
         ["c = a + b",
          "",
          "c = jax.lax.with_sharding_constraint(",
          "        c, NamedSharding(mesh, P('data', 'model')))"],
         "这段里<b>一个下标都没有</b>。你没说「谁算哪一格」，只说了"
         "<b>「这个数组按哪根轴切」</b> —— 剩下的切分、通信、重叠，全由编译器生成。",
         "换个形状，<r>整张图要重编译</r>。v7 上实测<b>十到十七分钟</b>。")

    _cmp(f)
    _cards(f)
    _band(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
def _col(f, x, w, c, head, code, note, cost):
    f.rect(x, TOP, w, COL_H, FILL[c], c, 1.4, 10)
    para(f, x + 18, TOP + 26, w - 36, head, "sec")

    ch = 22 * len(code) + 20
    f.rect(x + 18, TOP + 42, w - 36, ch, "#fff", "#e0e0e0", 1, 6)
    for i, ln in enumerate(code):
        # f.t() 内部已经 escape 过了，这里再手动转一次会变成 `&amp;lt;`，
        # 屏幕上显示的就是字面的 `&lt;`。2026-08-31 由 common.py 新加的
        # 实体自检抓出来 —— 在此之前它一直是错的。
        f.t(x + 32, TOP + 64 + i * 22, ln, "mono", INK)

    y = para(f, x + 18, TOP + 60 + ch, w - 36, note, "xs", 18)

    # 代价单独一行、灰底 —— 每一栏都要有，避免只夸不说代价
    f.rect(x + 18, y + 6, w - 36, 38, "#f1f3f4", None, 0, 6)
    para(f, x + 30, y + 30, w - 60, cost, "xs", 17)


# ══════════════════════════════════════════════════════════════════════
def _cmp(f):
    """中间那张四行对照表。

    ⭐ 2026-09-04 重画。Chris 的原话：**「表面上是图，其实是一个表格，
       左边一列右边一列，特别单薄，字还那么小。」** 这一张正是那个样子 ——
       三栏全是灰底细线，读者看不出「哪两栏是一对」。

    改的是三件事，每一件都对应一种层次：
      ① **身份**：两个答案栏各铺一条通到底的淡红／淡绿色带，
         跟上半张那两张卡是同一套颜色 ——&nbsp;
         于是「左红右绿」这个身份在整张图里只需要建立一次。
      ② **节奏**：斑马纹从 #fafafa 提到 #f6f7f9（原来那档在投影上等于没有），
         行高 50 → 58，行与行之间靠底色分组，不靠细线。
      ③ **主次**：问题那一栏加一条 3px 的靛蓝竖轨 ＋ 加粗 ——&nbsp;
         它是「行的名字」，不该跟答案同一个视觉重量。
    ⛔ 不要再往里加第四栏：这张表的全部意思就是「同一个问题，两种答案」。
    """
    cx = [20, 232, 812]
    cw = [206, 574, 568]
    f.t(20, CMP_Y - 12, "把差别摊成四行　—— 每一行都是上面那个选择的直接后果", "sec")

    # ① 两条通到底的身份色带，先铺底，后面所有东西压在它上面
    f.rect(cx[1], CMP_Y, cw[1], CMP_H, "#fdf3f2", None, 0, 0)
    f.rect(cx[2], CMP_Y, cw[2], CMP_H, "#f1f8f3", None, 0, 0)

    f.rect(20, CMP_Y, 1360, CMP_HH, "#eceff1", None, 0, 8)
    f.rect(20, CMP_Y + CMP_HH - 8, 1360, 8, "#eceff1", None, 0, 0)
    for i, (h, c) in enumerate([("问的是同一个问题", SUB),
                                ("GPU / CUDA", RD), ("TPU / XLA", GN)]):
        if i:                                   # 表头前面点一个色块，跟色带同色系
            f.rect(cx[i] + 14, CMP_Y + 13, 9, 9, c, None, 0, 2)
        f.t(cx[i] + (28 if i else 14), CMP_Y + 23, h, "lbl", c if i else INK)

    for r, (q, a, b) in enumerate(CMP_ROWS):
        y = CMP_Y + CMP_HH + r * CMP_RH
        if r % 2:                               # ② 斑马纹：整行压一层，色带也一起压深
            f.rect(20, y, 1360, CMP_RH, "#00000008", None, 0, 0)
        if r:
            f.line(cx[1], y, 1380, y, "#00000012", 1)
        f.rect(20, y + 9, 3, CMP_RH - 18, BL, None, 0, 2)   # ③ 问题栏的竖轨
        para(f, cx[0] + 16, y + 34, cw[0] - 30, "<b>%s</b>" % q, "lbl")
        para(f, cx[1] + 16, y + 28, cw[1] - 32, a, "xs", 18)
        para(f, cx[2] + 16, y + 28, cw[2] - 32, b, "xs", 18)
    f.line(cx[2], CMP_Y, cx[2], CMP_Y + CMP_H, "#00000014", 1)
    f.rect(20, CMP_Y, 1360, CMP_H, "none", "#dadce0", 1.2, 8)


# ══════════════════════════════════════════════════════════════════════
COST = ("<b>形状一变就重编译。</b>v7 上实测 10–17 分钟 —— 交互式调试基本别想。"
        "<br><b>变长序列只能绕。</b>分桶 ＋ padding，桶分粗了浪费算力，分细了编译次数爆炸。"
        "<br><b>数据相关的操作很别扭。</b>MoE 路由、动态 KV 分配，都要改写成静态形状。")

GAIN = ("<b>编译器看得到整张图。</b>跨算子融合、通信与计算重叠、内存复用，"
        "这三样不用你操心，也<b>不写在你的代码里</b>。"
        "<br><b>显存规划是确定的。</b>所以能<r>不占卡就提前算出会不会 OOM</r> —— "
        "这一条 GPU 侧没有对应物 —— <g>完整版 L300 §5.6 单讲。</g>"
        "<br><b>而且这三样是免费的。</b>你什么都不做就已经吃到了 —— 代价是<r>不想要的时候也关不掉</r>。")


def _cards(f):
    for x, w, c, t, body in [(20, 676, RD, "静态的代价", COST),
                             (724, 656, GN, "静态的红利", GAIN)]:
        f.rect(x, CARD_Y, w, CARD_H, FILL[c], c, 1.4, 10)
        f.t(x + 18, CARD_Y + 26, t, "sec")
        y = CARD_Y + 50
        for seg in body.split("<br>"):
            y = para(f, x + 18, y, w - 36, seg, "xs", 18) + 8


# ══════════════════════════════════════════════════════════════════════
def _band(f):
    f.rect(20, BAND_Y, 1360, 92, FILL[BL], BL, 1.4, 10)
    f.t(38, BAND_Y + 28, "一句话记住这一节", "sec")
    # para() 不认 <br>，长段落一律自己切段再逐段画 —— 这一点全库统一
    y = BAND_Y + 54
    for seg in [
            "<b>CUDA 把控制权交给你，XLA 把控制权交给编译器。</b>"
            "所以两边的痛苦完全不是同一种：一边是<r>「我写不出足够快的 kernel」</r>，"
            "一边是<r>「编译器不听话，而我插不上手」</r>。",
            "注意这里<b>没有哪一种痛苦更轻</b> —— 它们只是落在了不同的人身上："
            "前者要求你队里有人能写 kernel，后者要求你队里有人读得懂编译产物。"]:
        y = para(f, 38, y, 1324, seg, "xs", 19) + 4


if __name__ == "__main__":
    import sys
    open(sys.argv[1] if len(sys.argv) > 1 else "out/fig_p1_mindset.svg",
         "w", encoding="utf-8").write(build())
