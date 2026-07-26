#!/usr/bin/env python3
"""大文件并行分块 GCS 下载（v2：流式写盘）。

v1 为什么失败：
    用 download_as_bytes() 把整个 4.67 GB 分片缓冲进内存再写。
    32 路并发分摊带宽后单流只有 ~10 MB/s，一片要 400+ 秒，
    超过 SDK 默认超时就整片重来——网卡一直在收，磁盘却零增长，死循环。

v2 做法：
    download_to_file() 直接流式写进目标文件的对应偏移，不缓冲。
    线程降到 12 让单流带宽更足，超时放宽到 1 小时。
    每片完成后落一个 .done 标记，中断可续传。
"""
import os, sys, time, concurrent.futures as cf
from google.cloud import storage

ROOT   = "/raid/hy3-megatron"
BIG    = "iter_0000000/__0_0.distcp"
PREFIX = "Hy3-megatron/"
THREADS = 12
TIMEOUT = 3600

c = storage.Client(project="tencent-gcp-taiji-poc")
b = c.bucket("chrisya-gb300-models")
big_path = os.path.join(ROOT, BIG)
mark_dir = "/raid/.parts_done"
os.makedirs(mark_dir, exist_ok=True)

man = b.blob(PREFIX + "MANIFEST.txt").download_as_text().split()
size, npart = int(man[1]), int(man[2])
step = (size + npart - 1) // npart
os.makedirs(os.path.dirname(big_path), exist_ok=True)
if not (os.path.exists(big_path) and os.path.getsize(big_path) == size):
    with open(big_path, "wb") as f:
        f.truncate(size)
print(f"[down2] {size/1e9:.1f} GB / {npart} 片 / {THREADS} 线程", flush=True)

done0 = len(os.listdir(mark_dir))
t0 = time.time()

def dn(i):
    mk = os.path.join(mark_dir, f"{i:04d}")
    if os.path.exists(mk):
        return 0
    off = i * step
    n = min(step, size - off)
    if n <= 0:
        return 0
    with open(big_path, "r+b") as f:
        f.seek(off)
        b.blob(f"{PREFIX}{BIG}.part{i:04d}").download_to_file(f, timeout=TIMEOUT)
    open(mk, "w").close()
    d = len(os.listdir(mark_dir))
    print(f"[down2] {d}/{npart} 片  {(time.time()-t0):.0f}s", flush=True)
    return n

with cf.ThreadPoolExecutor(THREADS) as ex:
    tot = sum(ex.map(dn, range(npart)))

# 小文件（.metadata / tokenizer / common.pt …）
for bl in c.list_blobs(b, prefix=PREFIX):
    if ".distcp.part" in bl.name or bl.name.endswith("MANIFEST.txt"):
        continue
    dst = os.path.join(ROOT, bl.name[len(PREFIX):])
    if os.path.exists(dst) and os.path.getsize(dst) == bl.size:
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    bl.download_to_filename(dst, timeout=TIMEOUT)

el = time.time() - t0
print(f"[down2] 完成 {tot/1e9:.1f} GB / {el:.0f}s / {tot/max(el,1)/1e6:.0f} MB/s", flush=True)
