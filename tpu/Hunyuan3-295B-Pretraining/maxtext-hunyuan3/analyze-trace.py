#!/usr/bin/env python3
"""把 xplane 的 trace.json 拆成「通信 / 计算 / 各占多少 / 有没有重叠」。

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
    """两组区间的交集总长 —— 用来回答「通信藏住了吗」。"""
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

    comm = buckets['comm']
    comp = buckets['moe_gemm'] + buckets['compute'] + buckets['attn']
    cu, pu, ov = union(comm), union(comp), intersect(comm, comp)
    print(f'\n  通信 {cu/1e6:.3f}s  计算 {pu/1e6:.3f}s  重叠 {ov/1e6:.3f}s')
    print(f'  → 通信被掩盖 {ov/cu*100 if cu else 0:.1f}%，'
          f'裸露 {(cu-ov)/1e6:.3f}s = 墙钟 {(cu-ov)/span*100:.1f}%')


if __name__ == '__main__':
    main()
