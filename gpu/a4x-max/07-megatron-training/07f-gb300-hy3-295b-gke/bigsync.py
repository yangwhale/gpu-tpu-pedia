#!/usr/bin/env python3
"""大文件的并行分块 GCS 传输。

为什么需要它：
    单进程转换产出的是**一个** 597.7 GB 的 __0_0.distcp。
    GCS SDK 对单个对象只能单流上传/下载（~100-200 MB/s），597 GB 要跑一小时。
    这里把它按字节区间切成 N 份并行传，实测多线程可跑到 1.6 GB/s。

做法：
    上传：每个线程 seek 到自己的区间读取，作为独立对象 xxx.partNNNN 上传。
    下载：预分配目标文件，每个线程下载一个 part 并 seek 写回原偏移。
    不用 GCS compose——分片对象直接对称收发，少一层状态。
幂等：同名同大小跳过。

用法: python bigsync.py up|down
"""
import os, sys, time, concurrent.futures as cf
from google.cloud import storage

ROOT   = "/raid/hy3-megatron"
BIG    = "iter_0000000/__0_0.distcp"
PREFIX = "Hy3-megatron/"
BUCKET = "chrisya-gb300-models"
NPART  = 128
THREADS = 32

mode = sys.argv[1]
c = storage.Client(project="tencent-gcp-taiji-poc")
b = c.bucket(BUCKET)
big_path = os.path.join(ROOT, BIG)
t0 = time.time()


def small_files():
    """除大文件外的所有小文件（.metadata / tokenizer / common.pt …）"""
    out = []
    for r, _, fs in os.walk(ROOT):
        for f in fs:
            p = os.path.join(r, f)
            rel = os.path.relpath(p, ROOT)
            if rel == BIG:
                continue
            out.append((p, PREFIX + rel, os.path.getsize(p)))
    return out


if mode == "up":
    size = os.path.getsize(big_path)
    step = (size + NPART - 1) // NPART
    remote = {x.name: x.size for x in c.list_blobs(b, prefix=PREFIX)}
    print(f"[up] 大文件 {size/1e9:.1f} GB → {NPART} 片 × {step/1e9:.2f} GB", flush=True)

    def up_part(i):
        off = i * step
        n = min(step, size - off)
        if n <= 0:
            return 0
        key = f"{PREFIX}{BIG}.part{i:04d}"
        if remote.get(key) == n:
            return 0
        with open(big_path, "rb") as f:
            f.seek(off)
            data = f.read(n)
        b.blob(key).upload_from_string(data)
        return n

    def up_small(t):
        p, key, sz = t
        if remote.get(key) == sz:
            return 0
        bl = b.blob(key); bl.chunk_size = 32 * 1024 * 1024
        bl.upload_from_filename(p); return sz

    with cf.ThreadPoolExecutor(THREADS) as ex:
        tot = sum(ex.map(up_part, range(NPART))) + sum(ex.map(up_small, small_files()))
    # 记录原始大小，供 down 端预分配与校验
    b.blob(PREFIX + "MANIFEST.txt").upload_from_string(f"{BIG}\n{size}\n{NPART}\n")

else:  # down
    man = b.blob(PREFIX + "MANIFEST.txt").download_as_text().split()
    size, npart = int(man[1]), int(man[2])
    step = (size + npart - 1) // npart
    os.makedirs(os.path.dirname(big_path), exist_ok=True)
    need = not (os.path.exists(big_path) and os.path.getsize(big_path) == size)
    if need:
        with open(big_path, "wb") as f:      # 预分配，稀疏文件
            f.truncate(size)
    print(f"[down] 大文件 {size/1e9:.1f} GB / {npart} 片  需要下载={need}", flush=True)

    def dn_part(i):
        if not need:
            return 0
        off = i * step
        n = min(step, size - off)
        if n <= 0:
            return 0
        data = b.blob(f"{PREFIX}{BIG}.part{i:04d}").download_as_bytes()
        assert len(data) == n, f"part{i} 大小不符 {len(data)} != {n}"
        with open(big_path, "r+b") as f:
            f.seek(off); f.write(data)
        return n

    def dn_small(t):
        _, key, _ = t
        rel = key[len(PREFIX):]
        dst = os.path.join(ROOT, rel)
        bl = b.blob(key)
        bl.reload()
        if os.path.exists(dst) and os.path.getsize(dst) == bl.size:
            return 0
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        bl.download_to_filename(dst); return bl.size

    smalls = [(None, x.name, x.size) for x in c.list_blobs(b, prefix=PREFIX)
              if not x.name.endswith("MANIFEST.txt") and ".distcp.part" not in x.name]
    with cf.ThreadPoolExecutor(THREADS) as ex:
        tot = sum(ex.map(dn_part, range(npart))) + sum(ex.map(dn_small, smalls))

el = time.time() - t0
print(f"[{mode}] 完成 {tot/1e9:.1f} GB / {el:.0f}s / {tot/max(el,1)/1e6:.0f} MB/s", flush=True)
