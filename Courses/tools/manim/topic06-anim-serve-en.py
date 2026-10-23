# -*- coding: utf-8 -*-
"""topic06-anim-serve.py 的英文版：只换画面里的字，布局与时序原样（见 _lang_en.py）。
只翻英文页用到的五支：ContBatch、SpecStep、PagedKV、PDSplit、BigEP（KVStore / StepMix / KVLoad 只在 L300 中文页）。
措辞跟专题六英文页（块 02 / 03 / 04 / 06 / 07）里各段动画的 aria-label、图注一致：
静态批 / 连续批 → static / continuous batching、混部 → colocated、PD 分离 → PD disaggregation、大 EP → large EP、
卡 → device / Dev、小模型 / 大模型 → small model / big model。

⭐ SpecStep 的例句：中文用古诗「床前明月光，疑是地上霜」，画面每格一个字。英文换成格言
  “If it is to be, it is up to me” 的变体 “So if it is to be it is all up to me” —— 12 个词都只有两三个字母，
  装得进 0.66 宽的格子（32pt 粗体不缩字）；不用标点：逗号、句号按包围盒居中后像「›」「·」，看着像别的符号。
  画面逻辑仍按中文 TARGET / DRAFTS 跑，这里只是逐字换成英文 token，所以同一个汉字必须对应同一个英文词：
  「地」在第 1 步是草稿第 4 格（作废），第 2 步是正确的第 8 个词，两处都译 is —— 第 1 步草稿读作
  “to be all is”（猜错的是第 3 格 all，大模型自己的是 it），第 2 步 “is all up to” 全对，白送的第 5 格是 me。
  两步的模式与中文一致：第 1 步猜对 2 个、第 3 个错、出 3 个；第 2 步猜对 4 个 ＋ 白送 1 个 ＝ 出 5 个。

📌 渲染（在 Courses/ 下）：
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve-en.py ContBatch WebPages/media/topic06-contbatch.en.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve-en.py SpecStep  WebPages/media/topic06-spec.en.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve-en.py PagedKV   WebPages/media/topic06-paged.en.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve-en.py PDSplit   WebPages/media/topic06-pdsplit.en.mp4
   构建手册/脚本/render.sh tools/manim/topic06-anim-serve-en.py BigEP     WebPages/media/topic06-bigep.en.mp4
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _lang_en import run_en  # noqa: E402

EN = {
    # ── ContBatch
    "同样 4 个槽位、同一串请求：两种排法": "Same 4 slots, same queue of requests: two ways to schedule",
    "静态批：凑满一批，等最长的做完": "Static batching: fill a batch, wait for the longest to finish",
    "连续批：每一步都重新排，谁做完谁让位": "Continuous batching: reschedule every step; whoever finishes yields",
    "请求长度是示意": "Illustrative",
    "16 个请求排队，长短不一；每一列是一步 decode，每一步都要搬一趟权重":
        "16 requests, mixed lengths; each column is one decode step, one weight trip",
    "上面：短的早做完了，槽位也得空等最长的那个（灰格＝这一趟白搬）":
        "Top: short ones finish early, but their slots wait for the longest (gray = a wasted trip)",
    "下面：谁做完，下一步就换排队的下一个进来，槽位一格不空":
        "Bottom: when one finishes, the next in line takes its slot the next step; no slot sits idle",
    "同样 15 步：静态批做完 8 个（空转 23 格），连续批做完 14 个（空转 0 格）":
        "Same 15 steps: static finishes 8 (23 idle cells), continuous finishes 14 (0 idle cells)",
    # ── SpecStep
    "投机解码：小模型先猜 4 个，大模型一趟验完": "Speculative decoding: small guesses 4, big checks them in one trip",
    "已写下": "Written",
    "小模型猜": "Small\nguesses",
    "大模型验": "Big\nverifies",
    "用时": "Time",
    "紫＝小模型猜一个（便宜）　灰＝大模型搬一趟权重（贵）":
        "Purple = one small-model guess (cheap)    Gray = one big-model trip through the weights (expensive)",
    "大模型搬一趟权重，同时算出这 5 个位置它自己会写的字":
        "One trip of the big model computes its own token at all 5 positions",
    "一趟": "1 trip",
    "从左往右逐个比：一样就收下": "Compare left to right: matches are kept",
    "第 3 个不一样：换成大模型自己的字，后面的全丢掉":
        "The 3rd differs: replaced by the big model's token, everything after it dropped",
    "？＝前面那个字已经错了，这个位置算了也作废": "? = an earlier token was wrong, so this one is void",
    "4 个全对；第 5 个位置大模型本来就算了，白送 1 个":
        "All 4 match; the big model computed the 5th position anyway, so it comes free",
    "猜对几个就出几个，再加大模型自己的 1 个：每步至少出 1 个":
        "Tokens out = correct guesses + 1 from the big model: at least 1 per step",
    # 例句逐字换成英文 token（见文件头）：So if it is | to be [all→it] is | is all up to | me
    "床": "So", "前": "if", "明": "it", "月": "is",
    "光": "to", "疑": "be", "似": "all", "是": "it",
    "地": "is", "上": "all", "霜": "up", "举": "to", "头": "me",
    # ── PagedKV
    "分页 KV：写满一块才领下一块，用完立刻还": "Paged KV: take a new block only when one is full, return it at once",
    "页表：第几块 → 住在哪": "Page table: block n → location",
    "已结束：块已还回": "Done: blocks returned",
    "KV 显存切成固定大小的块，每块装 4 个 token（示意）":
        "KV memory is cut into fixed-size blocks of 4 tokens each (illustrative)",
    "三个请求进来、边生成边长：写满一块，才去空闲池领一块新的":
        "Three requests grow as they generate; only a full block makes one take a new block",
    "领到哪块算哪块：一个请求的块东一块西一块，页表记住先后":
        "Whatever block comes up: a request's blocks scatter, the page table keeps their order",
    "请求 2 结束：它的 2 块立刻还回空闲池": "Request 2 finishes: its 2 blocks go straight back to the free pool",
    "新请求 4 进来，马上用上刚还回来的那 2 块": "New request 4 arrives and immediately uses those 2 freed blocks",
    "浪费只剩每个请求最后一块的空尾巴（此刻共 8 格）":
        "The only waste left: the empty tail of each request's last block (8 slots now)",
    "对照：按最大长度连续预留，没写到的位置也一直占着":
        "Compare: reserve contiguous space for the max length, and empty slots stay taken",
    "每种颜色一个请求；灰框＝空闲块": "One color per request; gray = free",
    # ── PDSplit
    "PD 分离：prefill 挪走，decode 的字不再被卡住": "PD disaggregation: move prefill out, and decode tokens stop stalling",
    "混部：prefill 和 decode 挤在一台机器上": "Colocated: prefill and decode share one machine",
    "PD 分离：prefill 机只算 prefill，decode 机只出字": "PD disaggregated: the prefill machine only prefills, the decode machine only emits tokens",
    "整机在算": "Machine",
    "新请求": "New req.",
    "prefill 机": "Prefill",
    "decode 机": "Decode",
    "长 prompt 的 prefill": "Long-prompt prefill",
    "下一个字": "Next token",
    "被推迟": "delayed",
    "字距": "Spacing",
    "始终均匀": "stays even",
    "传 KV": "Send KV",
    "一格＝一步 decode 的时间，只表相对长短（示意）": "One cell = one decode step; relative length only (illustrative)",
    "三条请求在 decode：每一步，每条请求出一个字，格子等间距":
        "Three requests in decode: each emits one token per step, evenly spaced",
    "来了一个长 prompt：prefill 一次算完整段的 KV。混部时整台机器被它占住":
        "Long prompt: prefill computes all its KV at once; colocated, it holds the machine",
    "分离时它在 prefill 机上算，算完把 KV 传给 decode 机（传的是 KV，不是权重）":
        "Split, it runs on the prefill machine, then sends the KV (not the weights) to decode",
    "新请求带着 prefill 给的第一个字进 decode 批，之后一步一个字":
        "The new request joins the decode batch with its first token, then one token per step",
    "拆开后 decode 的字距始终均匀；代价是多一批机器、每个请求多传一趟 KV":
        "Split: even token spacing; cost: extra machines, one more KV transfer per request",
    # ── BigEP
    "大 EP 为什么省": "Why large EP saves",
    "一张卡上的专家": "Experts on one GPU",
    "每个专家分到几行": "Rows per expert",
    "每层读多少权重": "Read per layer",
    "路由专家占显存": "Expert memory",
    "32 个路由专家": "32 routed experts",
    "2 个路由专家": "2 routed experts",
    "80 GB 卡": "80 GB",          # 红线下的标签贴着画面右缘，只留数字
    "示意：每卡 32 个请求、每 token 选 8 个专家、负载均匀；EP144 含 32 份冗余专家（共 288 份）":
        "Illustrative: 32 requests per GPU, 8 experts per token, even load; EP144 includes 32 redundant experts (288 total)",
    "同一个 V3，每张卡都带着自己的 32 个请求": "The same V3; every GPU brings its own 32 requests",
    "每个专家分到几行：EP8 只有 8 行，EP144 约 128 行": "Rows per expert: just 8 on EP8, about 128 on EP144",
    "8 行": "8",          # 列头已是 Rows per expert；写成 128 rows 会顶到右边 88 MB 那根条
    "128 行": "128",
    "每层要读的专家权重：1.41 GB 对 88 MB，差 16 倍": "Expert weights read per layer: 1.41 GB vs. 88 MB, 16× apart",
    "读进来的每个字节：一个用 8 次，一个用 128 次": "Each byte read: used 8 times on one, 128 times on the other",
    "每字节用 8 次": "8 uses per byte",
    "每字节用 128 次": "128 uses per byte",
    "光放路由专家：82 GB 放不下，5 GB 还剩一大块装 KV": "Routed experts alone: 82 GB won't fit; at 5 GB, plenty is left for KV",
    "剩下装 KV": "rest → KV",    # 夹在条和 80 GB 红线之间，只有中文那么宽
    "代价：token 每层都要跨机器去一趟、回一趟": "The cost: tokens make a cross-machine round trip in every layer",
    "其余 143 张卡（跨 18 台机器）": "143 other GPUs (18 machines)",
    "专家铺开：每卡读得少，读一次用得多": "Spread the experts: each GPU reads less and uses each read more",
}
PAT = [
    (r"槽 (\d+)", r"Slot \1"),
    (r"做完 (\d+) 个", r"\1 done"),
    (r"已走 (\d+) 步", r"Step \1"),
    (r"第 (\d+) 步：小模型一个接一个猜 4 个字，每个都很快",
     r"Step \1: the small model guesses 4 tokens one after another, each one fast"),
    (r"第 (\d+) 步完成：猜对 (\d+) 个 ＋ 大模型补 1 个 ＝ 出 (\d+) 个字",
     r"Step \1 done: \2 correct guesses + 1 from the big model = \3 tokens out"),
    (r"第 (\d+) 步完成：猜对 (\d+) 个 ＋ 白送 1 个 ＝ 出 (\d+) 个字",
     r"Step \1 done: \2 correct + 1 free = \3 tokens out"),
    (r"块 (\d+)", r"Block \1"),
    (r"请求 (\d+)", r"Req. \1"),
    (r"请求 ([A-Z])", r"Request \1"),
    (r"空闲块：(\d+)", r"Free blocks: \1"),
    (r"第 (\d+) 步", r"Step \1"),
    (r"(\d+) 张卡", r"\1 GPUs"),
]

run_en(__file__, EN, PAT, globals())
