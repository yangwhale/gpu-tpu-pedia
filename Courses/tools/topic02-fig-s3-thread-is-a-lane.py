# -*- coding: utf-8 -*-
"""图 3-3 · 三个都叫「线程」，其实是三种东西。

⭐ **为什么要有这一张。** 2026-09-03 现场原话：

    「我感觉这一个线程好像跟平时应用级编程时候那个线程的概念不太一样，
      这一个线程就只能处理一个 lane 上的数据。这个跟 TPU 那边一个 lane
      是一条处理总线是一个概念。」

**这个判断在硬件层面是对的**，而且是自己想出来的 ——&nbsp;所以值得当成
一张图正式讲，不能让它停在「感觉」。

⛔ 但它在**编程模型层面断掉了**，而断掉的那一处恰好是这门课的主线：
   GPU 的 lane **有名字**（`threadIdx`），你写标量代码、写 if、写循环，
   就像在写一个真的线程；TPU 的 lane **没有名字**，你写不出
   「第 37 号 lane 干点别的」。同一块硅上的同一个东西，
   一边把它暴露给程序员，一边把它藏在编译器后面。

⭐ 所以这张图的形状是「三列对照」而不是「两列对照」——&nbsp;
   中间那列（CUDA thread）**两头都沾**：写法像左边，硬件像右边。
   把左右两根柱子立起来，中间那根的怪异之处才看得见。

📌 数：一个 SM 驻留 2,048 线程 × 148 SM ＝ 303,104，取整说「约 30 万」。
   TPU 向量寄存器 8 sublane × 128 lane ＝ 1,024 个 32-bit 格子
   （《TPU 显微镜》：8 × 128 × 32 bit ＝ 4 KiB）。
"""
import io

BL, OR, GR, RD, GY = "#1a73e8", "#e8710a", "#1e8e3e", "#d93025", "#5f6368"
W = 1400
C0, C1, C2, C3 = 0, 232, 556, 1046
RH, TOP = 74, 96

# (问题, OS 线程, CUDA thread, TPU 上对应的位置) —— 每格是一到两行
ROWS = [
    ("有没有自己的取指、发射",
     ["有，完全独立的控制流"],
     ["有自己的 PC（Volta 起），但<tspan font-weight=\"700\">发射单位始终是 warp</tspan>",
      "——&#160;所以它不能自己决定下一条指令是什么"],
     ["没有。它是一条向量指令覆盖到的一个位置"]),
    ("有没有「名字」",
     ["有 ——&#160;<tspan font-weight=\"700\">pthread_self()</tspan>"],
     ["<tspan font-weight=\"700\">有 ——&#160;threadIdx。这是 GPU 最特别的一点</tspan>",
      "你能写「第 37 号线程去干点别的」，而且它真的会去"],
     ["<tspan font-weight=\"700\">没有。</tspan>你写不出「第 37 号 lane 干点别的」",
      "——&#160;能写的只有整块整块的操作"]),
    ("换人要多少钱",
     ["把寄存器存下来再恢复 ——&#160;微秒级"],
     ["<tspan font-weight=\"700\">零。</tspan>寄存器物理常驻，从来不搬",
      "代价是那 256 KiB 一直占着 ——&#160;见「一个 SM 拆开看」那张"],
     ["不存在「换人」这回事 ——&#160;顺序编译期就排死了"]),
    ("一次处理多宽",
     ["你的代码说了算"],
     ["<tspan font-weight=\"700\">一条 32-bit lane。</tspan>"
      "访存可向量化到 128-bit，fp16 一拍两个",
      "但基本盘就是一格 ——&#160;跟你的直觉一致"],
     ["8 sublane × 128 lane 那张网格里的<tspan font-weight=\"700\">一格</tspan>"]),
    ("谁决定它什么时候跑",
     ["操作系统调度器，会被抢占"],
     ["<tspan font-weight=\"700\">warp scheduler，运行时挑</tspan>"
      "——&#160;谁的操作数到了就发谁"],
     ["<tspan font-weight=\"700\">编译器，编译期排死</tspan>。没有调度器"]),
    ("有多少个",
     ["几十到几百"],
     ["一个 SM 驻留 <tspan font-weight=\"700\">2,048</tspan>，"
      "全片 148 个 SM ＝ 约 <tspan font-weight=\"700\">30 万</tspan>"],
     ["一条向量指令一次盖住 <tspan font-weight=\"700\">1,024</tspan> 格"]),
]

H = TOP + len(ROWS) * RH + 128
p = ['<svg viewBox="0 0 %d %d" width="100%%" role="img" aria-label="'
     '操作系统线程、CUDA thread、TPU 的一个 lane 位置，三者逐项对照">' % (W, H)]
p.append('<text class="svglbl" x="0" y="17" fill="#202124" style="font-size:14px">'
         '三个都叫「线程」——&#160;<tspan font-weight="700">'
         '中间那个两头都沾：写法像左边，硬件像右边</tspan></text>')
p.append('<text class="svgsm" x="0" y="37">'
         'CUDA 的 thread 借了操作系统线程的名字和写法，'
         '骨子里却是一条 SIMD lane ——&#160;把左右两根柱子立起来，中间那根的怪异才看得见</text>')

# 中间那列整体加浅底：它是这张图的主角
p.append('<rect x="%d" y="%d" width="%d" height="%d" rx="10" fill="#e8f0fe"/>'
         % (C2 - 16, TOP - 34, C3 - C2, len(ROWS) * RH + 34))

for x, s, c in ((C1, '操作系统的线程', GY),
                (C2, 'CUDA 的 thread', BL),
                (C3, 'TPU 上对应的位置', OR)):
    p.append('<text class="svglbl" x="%d" y="%d" fill="%s" style="font-size:12px">%s</text>'
             % (x, TOP - 14, c, s))

for i, (q, a, b, c) in enumerate(ROWS):
    y = TOP + i * RH
    if i:
        p.append('<line x1="0" y1="%d" x2="%d" y2="%d" stroke="#e8eaed"/>' % (y, W, y))
    p.append('<text class="svglbl" x="0" y="%d" fill="#202124" style="font-size:12px">%s</text>'
             % (y + 26, q))
    for x, lines, col in ((C1, a, GY), (C2, b, "#174ea6"), (C3, c, "#7a3e00")):
        for j, s in enumerate(lines):
            p.append('<text class="svgsm" x="%d" y="%d" fill="%s">%s</text>'
                     % (x, y + 26 + j * 17, col, s))

YB = TOP + len(ROWS) * RH + 16
p.append('<rect x="0" y="%d" width="%d" height="94" rx="8" fill="#e6f4ea" stroke="%s"/>'
         % (YB, W, GR))
p.append('<text class="svglbl" x="16" y="%d" fill="#0b6b30" style="font-size:13px">'
         '⭐ 一句话：CUDA 的 thread 是「<tspan text-decoration="underline">有名字的 SIMD lane</tspan>」</text>'
         % (YB + 24))
p.append('<text class="svgsm" x="16" y="%d" fill="#0b6b30">'
         '你按操作系统线程的写法去写它 ——&#160;标量代码、if、循环，全都成立；'
         '但它<tspan font-weight="700">没有自己的取指</tspan>，'
         '三十二个一起听同一句口令。</text>' % (YB + 44))
p.append('<text class="svgsm" x="16" y="%d" fill="#0b6b30">'
         '<tspan font-weight="700">而这个「有名字」，就是两边的分界线。</tspan>'
         'GPU 把 lane 暴露给你，代价是它得配调度器、记分板、'
         '和那堆养着待命者的寄存器；</text>' % (YB + 62))
p.append('<text class="svgsm" x="16" y="%d" fill="#0b6b30">'
         'TPU 把 lane 藏在编译器后面，于是省掉那一整套 ——&#160;'
         '但你也就<tspan font-weight="700">再没有「单独指挥某一条 lane」这个动作</tspan>了。'
         '<tspan fill="%s">「那六层为什么是六层」那张，讲的是同一件事的另一半。</tspan></text>'
         % (YB + 80, GY))
p.append('</svg>')
io.open('fig3-3.svg', 'w', encoding='utf-8').write('\n'.join(p))
print('fig3-3 ok', H)
