#!/usr/bin/env python3
"""把 xplane 的 trace.json 拆成「通信 / 计算 / 各占多少 / 同步还是异步」。

⚠️ 先看这里：**判断"通信有没有被计算掩盖"不要用本脚本，用 XProf。**
把 `.xplane.pb` 传上去就有官方 trace viewer，能同时看 TensorCore / SparseCore /
Host Offload 多条 lane 的真实并发。本脚本只适合做**单条 lane 上的批量统计**。

原因：`XLA Ops` 这条 lane 上顶层 op 天然首尾相接（实测 40560 个事件里，
16550 处时间交集全是容器包子 op 的纯嵌套，部分交叉为 0）。在这条 lane 上算
「通信 ∩ 计算」恒等于 0，**不管实际有没有重叠**。据此下"通信完全裸露"的结论
是同义反复 —— 本项目 2026-07-31 踩过这个坑。

正确的判法是看 op 名字的语义：TPU 上异步集合通信拆成 `-start` / `-done` 一对，
`-done` 的时长才是没藏完的残余；没有这对拆分的（`all-gather` / `reduce-scatter` /
`all-reduce`）就是同步阻塞。下面的 `--sync-async` 输出做的就是这件事。

用法:
  gcloud storage cp gs://<bucket>/<run>/tensorboard/plugins/profile/*/*.trace.json.gz .
  gunzip -c *.trace.json.gz > t.json
  python3 analyze-trace.py t.json            # 默认看 TPU:0
  python3 analyze-trace.py t.json --dev 3

为什么不能直接把 op 时长相加：trace 里 `while` 这类容器 op 会把内部子 op 的时间
一起算进自己，直接求和会得到 150%+ 的占空比。这里一律用**区间并集**。
"""
import argparse, collections, json, re, sys

COMM = ('all-', 'reduce-scatter', 'collective', 'ragged-all')


def categorize(name):
    base = re.split(r'[.\d]', name)[0] or name
    if base.startswith(COMM):
        return 'comm'
    if base == 'while':
        return 'while'          # 容器，不计入占比
    if base in ('gmm', 'tgmm'):
        return 'moe_gemm'       # MoE 分组矩阵乘
    if base.startswith('splash'):
        return 'attn'
    if base in ('fusion', 'dot', 'custom-call', 'convolution') or 'fusion' in base:
        return 'compute'
    return 'other'


def union(intervals):
    """区间并集总长，去掉嵌套与并发造成的重复计时。"""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s if cur_s is not None else 0)


def intersect(a, b):
    """两组区间的交集总长。

    ⚠️ 在单条顺序 lane（如 XLA Ops）上，不同顶层 op 不可能相交，本函数恒返回 0。
    **不要用它判断"通信藏住了吗"**，见文件头说明。保留仅为跨 lane 分析预留。
    """
    a, b = sorted(a), sorted(b)
    i = j = total = 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('trace_json')
    ap.add_argument('--dev', type=int, default=0, help='看第几个 TPU device')
    args = ap.parse_args()

    data = json.load(open(args.trace_json))
    proc, thread = {}, {}
    for e in data['traceEvents']:
        if e.get('ph') != 'M':
            continue
        if e.get('name') == 'process_name':
            proc[e['pid']] = e['args'].get('name', '')
        elif e.get('name') == 'thread_name':
            thread[(e['pid'], e['tid'])] = e['args'].get('name', '')

    want = f'/device:TPU:{args.dev}'
    events = [e for e in data['traceEvents']
              if e.get('ph') == 'X' and e.get('dur', 0) > 0
              and proc.get(e['pid']) == want
              and thread.get((e['pid'], e.get('tid'))) == 'XLA Ops']
    if not events:
        sys.exit(f'{want} 上没有 XLA Ops 事件 —— 先确认 profiler 真的跑到了稳态步')

    t0 = min(e['ts'] for e in events)
    span = max(e['ts'] + e['dur'] for e in events) - t0

    buckets = collections.defaultdict(list)
    for e in events:
        buckets[categorize(e['name'])].append((e['ts'], e['ts'] + e['dur']))

    naive = sum(e['dur'] for e in events)
    print(f'{want}  {len(events)} 个 op，时间轴跨度 {span/1e6:.3f}s')
    print(f'  朴素求和 {naive/1e6:.3f}s = 跨度的 {naive/span*100:.1f}%'
          f'  ← 超过 100% 就说明有容器/并发，必须用并集\n')

    print(f'  {"类别":<16}{"并集":>9}{"占墙钟":>9}')
    for key in ('comm', 'moe_gemm', 'compute', 'attn', 'other'):
        if key in buckets:
            u = union(buckets[key])
            print(f'  {key:<16}{u/1e6:8.3f}s{u/span*100:8.1f}%')

    # 通信按「同步阻塞 / 异步残余」拆 —— 这才是判断能不能藏的正确维度
    sync_d = collections.defaultdict(float)
    async_start = async_done = 0.0
    for e in events:
        base = re.split(r'[.\d]', e['name'])[0]
        if not base.startswith(COMM):
            continue
        if '-start' in e['name']:
            async_start += e['dur']
        elif '-done' in e['name']:
            async_done += e['dur']
        else:
            sync_d[base] += e['dur']
    sync_total = sum(sync_d.values())
    print(f'\n  通信拆解（判断"能不能藏"看这里，不看时间交集）:')
    for k, v in sorted(sync_d.items(), key=lambda x: -x[1]):
        print(f'    {k:<28} {v/1e6:7.3f}s   同步阻塞')
    print(f'    {"(async -start)":<28} {async_start/1e6:7.3f}s   异步发起')
    print(f'    {"(async -done)":<28} {async_done/1e6:7.3f}s   异步残余（已被部分掩盖）')
    tot = sync_total + async_start + async_done
    if tot:
        print(f'  → 同步阻塞 {sync_total/1e6:.3f}s ({sync_total/tot*100:.0f}%) = 墙钟 {sync_total/span*100:.1f}%'
              f'   异步残余 {async_done/1e6:.3f}s ({async_done/tot*100:.0f}%)')
    print('  跨 lane 的真实并发请用 XProf 看，本脚本只统计单条 XLA Ops lane。')


if __name__ == '__main__':
    main()
