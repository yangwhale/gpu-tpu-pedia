# -*- coding: utf-8 -*-
"""topic05-anim-cut.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
措辞跟 topic-05.en.html 里各图的 aria-label / 图注一致（卡 → Dev，机器 → machine）。

📌 渲染（在 Courses/ 下）：构建手册/脚本/render.sh \
        tools/manim/topic05-anim-cut-en.py TPSplit WebPages/media/topic05-tpsplit.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

EN = {
    "卡 0": "Dev 0", "卡 1": "Dev 1", "卡 2": "Dev 2", "卡 3": "Dev 3",
    # TPSplit
    "张量并行切一个 MLP：Y ＝ GeLU(X·W1)·W2": "Tensor parallelism splits an MLP: Y = GeLU(X·W1)·W2",
    "竖切一刀": "cut by column",
    "横切一刀": "cut by row",
    "① W1 按列切：每张卡算出中间结果的一半": "① W1 split by columns: each device computes half of the intermediate",
    "H 半": "½ H",                             # 1.1 宽的块，"H half" 顶边
    "② 激活函数逐元素算：各算各的，这一段没有任何通信":
        "② The activation is elementwise: each computes its own, no communication",
    "③ W2 按行切：每张卡只得到 Y 的一个部分和": "③ W2 split by rows: each device gets only a partial sum of Y",
    "部分和": "Y part",                        # 1.1 宽的块，"partial sum" 两行也顶边；字幕里写全 partial sum
    "④ AllReduce：两份部分和相加，两张卡都拿到完整的 Y":
        "④ AllReduce: the two partial sums are added; both devices get the full Y",
    "整个 MLP 只在最后通信一次　——　代价是每一层都有这一次":
        "The whole MLP communicates once, at the end, but every layer pays that once",
    # Pipeline
    "流水线并行：灰色是气泡，每个 stage 都在空等的时间": "Pipeline parallelism: gray is the bubble, time each stage sits idle",
    "先用 4 个 micro-batch：气泡 ÷ 理想计算时间 ＝ (4−1) ÷ 4 ＝ 3/4":
        "Start with 4 micro-batches: bubble ÷ ideal compute = (4−1) ÷ 4 = 3/4",
    "总量不变、切成 8 份：气泡 ＝ (4−1) ÷ 8 ＝ 3/8，灰色真的短了一半":
        "Same total, split into 8: bubble = (4−1) ÷ 8 = 3/8; the gray really halves",
    "气泡只能摊薄、不能消灭：份数越多越省，可每份太小，卡就吃不饱":
        "Amortized, not eliminated: more pieces help, but tiny ones starve the device",
    # ExpertParallel
    "专家并行：token 飞到专家那里，算完再飞回来": "Expert parallelism: tokens fly to the experts, then fly back",
    "每个 token 由路由挑一个专家（真实的 V3 每个 token 挑 8 个）":
        "The router picks one expert per token (real V3 picks 8 per token)",
    "派发（AllToAll）：token 飞到专家所在的卡，在专家门口排队":
        "Dispatch (AllToAll): tokens fly to their expert's device and queue at its door",
    "专家 0 排了 7 个，别的专家只有 1 到 2 个：它算完之前，大家都得等":
        "Expert 0 has 7 waiting, the others only 1 to 2: everyone waits until it finishes",
    "合并（AllToAll）：算完再送回原来的卡": "Combine (AllToAll): results go back to their original devices",
    "发给谁由数据决定，负载天生不均　——　这是专家并行最重的病":
        "Data decides who gets what, so load is inherently uneven: EP's worst problem",
    # RingAttention
    "Ring Attention：Q 不动，KV 沿环传": "Ring Attention: Q stays, KV travels the ring",
    "注意力块（行 ＝ 哪张卡的 Q，列 ＝ 哪段 KV）": "Attention blocks\n(row = whose Q, column = which KV)",
    "每张卡固定一段 Q；每一步算手上这对（Q, KV），同时把 KV 传给下一张":
        "Each device keeps its Q; each step it computes its (Q, KV) and passes KV on",
    "转完一圈：每张卡都跟所有 KV 算过了，自己那一行填满":
        "After one lap: every device has met every KV chunk; its row is full",
    "关键：传下一块的时候正在算这一块，通信藏在计算后面":
        "Key: the next block flies while this one computes, hiding communication",
    # DecodeCP
    "DCP：decode 时 KV 按 token 轮流存到各张卡": "DCP: during decode, KV is stored round-robin by token",
    "只开 TP：笔记切不开，每张卡都存全部 12 个 token 的 KV":
        "TP only: the notes can't be split; every device stores KV for all 12 tokens",
    "12 个 token": "12 tokens",
    "DCP：每生成一个 token，它的 KV 存到第 (token 号 mod 4) 张卡上":
        "DCP: each new token's KV goes to device (token index mod 4)",
    "12 个 token，每张卡只存 3 个的 KV：同样的卡，能装 4 倍的笔记":
        "12 tokens, KV for only 3 per device: the same devices hold 4× the notes",
    "算注意力：新 token 的 Q 发给所有卡，各自在自己那份 KV 上算":
        "Attention: the new token's Q goes to every device; each uses its own KV share",
    "新 Q": "new Q",
    "四份部分结果合并成一份：每层多几次通信，换回 4 倍的 KV 空间":
        "Four partial results merge into one: a few extra collectives buy 4× KV space",
    # PDDisagg
    "PD 分离：prefill 和 decode 拆到两批机器上": "PD disaggregation: prefill and decode on two sets of machines",
    "放在一起": "colocated",
    "prefill 机器": "prefill\nmachines",      # 车道左边的标签，一行会出画面
    "decode 机器": "decode\nmachines",
    "放在一起：大家一步一步 decode，每格出一个字": "Colocated: everyone decodes step by step, one token per cell",
    "新请求的 prefill": "new request's prefill",
    "这 5 步没人出字": "no tokens for these 5 steps",
    "拆开：prefill 在自己的机器上跑，decode 那边一步不停":
        "Split: prefill runs on its own machines; decode never pauses",
    "KV 传过去": "KV handed over",
    "放在一起：decode 被截走 5 步；拆开：一格不断（多用了一批 prefill 机器）":
        "Colocated: decode loses 5 steps; split: no gap (extra prefill machines needed)",
    "代价：多一趟 KV 传输；按我们 v7x 那套的带宽估算约 100 毫秒":
        "The cost: one extra KV transfer, about 100 ms from bandwidth on our v7x setup",
    # MeshMap
    "同样 8 张卡、TP4 × DP2：TP 组摆在哪，差好几倍": "Same 8 devices, TP4 × DP2: TP placement matters several-fold",
    "机器 0": "machine 0", "机器 1": "machine 1",
    "慢线：机器之间": "slow link",               # 两台机器之间只有 1.8 宽的空档
    "摆法一：一个 TP 组就在一台机器里": "Placement 1: each TP group inside one machine",
    "第 1 轮 TP：每组沿自己的环传一格，全在机器里，快线 1 格":
        "TP round 1: each group passes one hop on its ring, all in-machine, fast: 1 cell",
    "第 1 轮完成：1 格": "Round 1 done: 1 cell",
    "后面 7 轮一模一样，快进": "The next 7 rounds are identical: fast-forward",
    "最后 1 次 DP：两个组里对应的卡交换梯度，要过慢线：9 格":
        "Finally 1 DP: matching devices swap gradients across the slow link: 9 cells",
    "摆法二：TP 组横跨两台机器（红箭头是过慢线的那几段）":
        "Placement 2: TP groups span both machines (red arrows cross the slow link)",
    "第 1 轮 TP：环上有两段过慢线，这一轮要等最慢的那段：9 格":
        "TP round 1: two hops cross the slow link; the round waits for the slowest: 9 cells",
    "第 1 轮完成：9 格": "Round 1 done: 9 cells",
    "最后 1 次 DP：两个组里对应的卡交换梯度，这一对在同一台机器里：1 格":
        "Finally 1 DP: matching devices swap gradients, the pair is in one machine: 1 cell",
    "摆法一：这一步用了 1 格": "Placement 1: 1 cell so far",
    "17 格对 73 格：同样的卡，慢 4.3 倍（示意）": "17 cells vs. 73: same devices, 4.3× slower (schematic)",
    # FSDPStep
    "FSDP：每层用之前把别人那几段复制过来，用完就扔": "FSDP: copy in the other slices before each layer, discard after use",
    "前向：每层先 AllGather，把别人那几段复制过来拼成整层，算完扔掉复制件":
        "Forward: AllGather the other slices into a full layer; drop the copies after use",
    "反向：扔掉的权重要再拼一次；算出的梯度 ReduceScatter 给各自的主人":
        "Backward: reassemble dropped weights; ReduceScatter grads to owners",
    "每层三次：前向拼一次，反向再拼一次、散一次": "Three per layer: gather in forward; gather again and scatter in backward",
}
PAT = [
    (r"专家 (\d)", r"Exp \1"),                 # 专家框只比中文「专家 0」宽一点
    (r"第 (\d) 层", r"Layer \1"),
    (r"通信次数：(\d+)", r"Collectives: \1"),
    (r"摆法一：这一步用了 (\d+) 格", r"Placement 1: \1 cells so far"),
    (r"摆法二：这一步用了 (\d+) 格", r"Placement 2: \1 cells so far"),
]
run_en(__file__, EN, PAT, globals())
