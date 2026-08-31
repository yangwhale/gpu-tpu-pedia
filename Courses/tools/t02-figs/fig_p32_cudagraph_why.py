# -*- coding: utf-8 -*-
"""图 P-32 —— CUDA Graph 到底快在哪：省的是 CPU 那一侧，不是计算。

**这张图是被一个非常硬的反驳逼出来的。** 5.1–5.6 一路在说「GPU 把决定留到运行时」，
懂 GPU 的人听完第一反应一定是：<b>可训练和推理现在基本都走 CUDA Graph 啊，
那不就是编译期定死、一条道跑到黑了吗？而且它确实快好几倍。</b>
这个反驳<b>一半完全成立</b>，而且如果不正面接住，前面五节的论点就都悬着。

**接住它的第一步，是把「为什么快」讲准。** 最常见的误解是「因为没有分支了所以快」——
<b>不是</b>。CUDA Graph 省掉的是 <b>CPU 那一侧一条一条发 kernel 的开销</b>：
每次 launch CPU 端要花几微秒，而一个 MoE step 有上万个 kernel，
光是「发」就能吃掉几十毫秒。graph 把这上万次驱动调用合并成一次提交。
<b>计算本身一纳秒都没有变快。</b>

**把这一条钉死，后面两张图才立得住** —— 因为「省的是喂的动作、不是算的动作」
直接推出「kernel 内部一点没变」（P-33），也直接推出「硬件一个晶体管都省不掉」（P-34）。

**⚠️ 出处口径。** 「重复 launch 的直线型 graph 每个节点约省 60 ns」「首次 launch 仍要把
work description 上传到 GPU」出自 NVIDIA 开发者博客。
<b>单次 launch 5–10 微秒是量级，不是规格</b> —— 它随驱动、CPU、kernel 参数变。
「kernel 越碎收益越大、一个大 GEMM 几乎白开」<b>是我从机制推的</b>，图上标了。
"""
from common import Fig, para, BL, GN, RD, YL, PU, TL, INK, SUB, GREY, FILL

W = 1400

SC_Y, SC_H = 84, 280                        # ① 两种喂法并排
BILL_Y, BILL_H = SC_Y + SC_H + 22, 104      # ② 那笔账
SEC_Y, SEC_H = BILL_Y + BILL_H + 22, 104    # ③ 还有一小块
LAND_Y, LAND_H = SEC_Y + SEC_H + 22, 106    # ④ 落点
SRC_Y, SRC_H = LAND_Y + LAND_H + 22, 92    # ⑤ 出处
H = SRC_Y + SRC_H + 20

L_X, L_W = 20, 674
R_X, R_W = 706, 674


def build():
    f = Fig(W, H, "CUDA Graph 把上万次 CPU 端 kernel launch 合并成一次提交，"
                  "省掉的是主机侧的发射开销；GPU 上的计算本身没有变快")
    f.title("CUDA Graph 快在哪　—— 省掉的是 <tspan font-weight=\"700\">CPU "
            "一条一条喂</tspan>，不是计算变快了")
    f.legend([(GREY, "CPU 在忙（发指令）"), (BL, "GPU 在算（kernel）"),
              (RD, "空等：GPU 闲着，在等 CPU")])
    _scene(f)
    _bill(f)
    _second(f)
    _land(f)
    _src(f)
    return f.out()


# ══════════════════════════════════════════════════════════════════════
# 两条时间轴并排。**画法本身就是论点**：右边 GPU 那一行连成一片，
# 左边那一行被红色的空隙切碎 —— 碎掉的地方就是 CUDA Graph 赚回来的东西。
def _scene(f):
    KW, GAPW, PITCH, NK = 30, 34, 64, 9      # kernel 宽 / 空等宽 / 左边节距 / 个数
    SPAN = NK * PITCH                        # 不开 graph 的总跨度
    AX = 62                                  # 两个 panel 内轨道的左边距

    def frame(x0, w, ttl, sub, c):
        f.rect(x0, SC_Y, w, SC_H, FILL[c], c, 1.6, 10)
        f.t(x0 + 18, SC_Y + 28, ttl, "sec", c)
        f.t(x0 + 18, SC_Y + 50, sub, "xxs", SUB)
        ty = SC_Y + 84
        for dy, lab in ((0, "CPU"), (60, "GPU")):
            f.t(x0 + 16, ty + dy + 17, lab, "xxs", SUB)
            f.rect(x0 + AX, ty + dy, w - AX - 18, 24, "#fff", GREY, 0.8, 3)
        return ty, x0 + AX

    # ── 左：每发一条才跑一个，两个 kernel 之间 GPU 是真的闲着
    ty, bx = frame(L_X, L_W, "❌ 不开 graph：CPU 一条一条发",
                   "每个 kernel 都要走一遍驱动，GPU 干完就得等下一条", GREY)
    for i in range(NK):
        x = bx + i * PITCH
        f.rect(x, ty + 3, KW, 18, FILL[GREY], GREY, 0.9, 2)
        f.rect(x, ty + 63, GAPW, 18, FILL[RD], RD, 0.7, 2, dash="2 2")
        f.rect(x + GAPW, ty + 63, KW, 18, FILL[BL], BL, 0.9, 2)
    f.line(bx + SPAN, ty + 58, bx + SPAN, ty + 92, GREY, 1.2, dash="3 3")
    f.t(bx + SPAN - 4, ty + 104, "跑完", "xxs", SUB, anchor="end")

    # ── 右：一次提交；**同样 9 个、同样宽**，只是贴着排
    ty, bx = frame(R_X, R_W, "✅ 开 graph：一次提交，GPU 连着跑",
                   "整张 DAG 在 capture 时定死，之后 CPU 只发一条 replay", GN)
    f.rect(bx, ty + 3, 132, 18, FILL[GN], GN, 0.9, 2)
    f.t(bx + 140, ty + 17, "一条 replay", "xxs", GN)
    for i in range(NK):
        f.rect(bx + i * KW, ty + 63, KW, 18, FILL[BL], BL, 0.9, 2)
    # 同一把尺子：把左边那条终点线原样搬过来，差出来的就是省下的
    f.line(bx + SPAN, ty + 58, bx + SPAN, ty + 92, GREY, 1.2, dash="3 3")
    f.t(bx + NK * KW - 4, ty + 104, "跑完", "xxs", GN, anchor="end")
    f.line(bx + NK * KW, ty + 118, bx + SPAN, ty + 118, GN, 2.0)
    f.t((bx + NK * KW + bx + SPAN) / 2, ty + 134,
        "省下来的全是空等", "xxs", GN, anchor="middle")

    para(f, L_X + 18, SC_Y + 222, L_W - 36,
         "kernel 越小，<r>红色空隙占的比例越大</r>。"
         "推理 decode 一个 token 步里 kernel 又多又碎，"
         "<b>很容易走到「CPU 喂不过来、GPU 闲着」的地步</b>。", "xs", 18)
    para(f, R_X + 18, SC_Y + 222, R_W - 36,
         "<b>两边的蓝格子一样多、一样宽</b> —— <r>kernel 该跑多久还是跑多久</r>。"
         "右边早早收工，省的<b>全部</b>是格子之间那段空等。", "xs", 18)


# ══════════════════════════════════════════════════════════════════════
def _bill(f):
    f.rect(20, BILL_Y, 1360, BILL_H, FILL[YL], YL, 1.6, 10)
    f.t(38, BILL_Y + 26, "🔢 那笔账：为什么「发指令」能吃掉几十毫秒", "sec")
    y = para(f, 38, BILL_Y + 50, 1324,
             "单次 kernel launch，CPU 端<b>大致 5–10 微秒</b>（量级，不是规格）。　→　"
             "MoE 训练一个 step 有<b>上万个 kernel</b>（专家数 × 层数 × 前后向）　→　"
             "<b>光是发，就是几十毫秒的主机开销</b>。", "xs", 18)
    para(f, 38, y + 2, 1324,
         "<b>开 graph 之后这一整笔变成一次 replay。</b>所以收益跟"
         "<b>「kernel 有多碎、有多少个」成正比</b> —— "
         "<r>kernel 极碎的负载上，开与不开差到三倍以上是实测见过的量级</r>"
         "<g>（具体数字跟模型和集群强相关，这里只给量级）</g>。", "xs", 18)


# ══════════════════════════════════════════════════════════════════════
def _second(f):
    f.rect(20, SEC_Y, 1360, SEC_H, FILL[BL], BL, 1.4, 10)
    f.t(38, SEC_Y + 26, "＋ 还有一小块，在 GPU 那一侧", "sec", BL)
    y = para(f, 38, SEC_Y + 50, 1324,
             "直线型 graph 重复 launch 时，<b>节点之间的间隙也被优化掉一部分，"
             "约 60 纳秒一个节点</b>（NVIDIA 给的数）。"
             "上万个节点就是<b>毫秒级</b> —— 比不上主机那一笔，但不是零。", "xs", 18)
    para(f, 38, y + 2, 1324,
         "<g>代价也要说：<b>首次 launch 仍然要把整张图的 work description 上传到 GPU</b>，"
         "这笔一次性成本跑不掉，图改了还得再付一部分。所以 graph 只对"
         "「同一张图反复跑很多遍」的场景划算。</g>", "xxs", 17)


# ══════════════════════════════════════════════════════════════════════
def _land(f):
    f.rect(20, LAND_Y, 1360, LAND_H, FILL[RD], RD, 1.8, 10)
    f.t(38, LAND_Y + 28, "⭐ 落点：快的是「喂」这个动作，不是「算」这个动作", "sec", RD)
    y = para(f, 38, LAND_Y + 54, 1324,
             "这一条决定了后面两张图。<b>既然省的是主机侧的发射开销，"
             "那 kernel 内部就没有任何理由发生变化</b> —— "
             "事实也确实如此，下一张图专门拆开看。", "xs", 19)
    para(f, 38, y + 2, 1324,
         "<b>一个能当场用的推论</b>：<r>kernel 越碎，CUDA Graph 收益越大；"
         "反过来，一个从头跑到尾的大 GEMM，开 graph 几乎白开。</r>"
         "<g>（这一条是我从机制推的，不是谁的原话 —— 但它可以直接拿去测。）</g>", "xs", 19)


# ══════════════════════════════════════════════════════════════════════
def _src(f):
    f.rect(20, SRC_Y, 1360, SRC_H, "#fff", GREY, 1.4, 10)
    f.t(38, SRC_Y + 26, "⚠️ 出处分层", "sec")
    y = para(f, 38, SRC_Y + 50, 1324,
             "<b>查到的</b>：直线型 graph 重复 launch 每节点约省 60 ns；"
             "首次 launch 需上传 work description，这笔成本只付一次、图更新时补付一部分 —— "
             "出自 NVIDIA 开发者博客。", "xs", 18)
    para(f, 38, y + 2, 1324,
         "<b>量级不是规格</b>：单次 launch 5–10 微秒会随驱动、CPU、参数个数变。　"
         "<r>我推的</r>：「碎 kernel 收益大、大 GEMM 白开」是从机制推的结论。", "xxs", 17)


if __name__ == "__main__":
    import io
    io.open("out/fig_p32_cudagraph_why.svg", "w", encoding="utf-8").write(build())
    print("ok fig_p32_cudagraph_why")
