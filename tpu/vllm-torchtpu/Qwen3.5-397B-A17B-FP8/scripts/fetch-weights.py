#!/usr/bin/env python3
"""把 397B FP8 权重从同 region 的 GCS 拉到本地 tmpfs。

为什么不用 gcloud：镜像里没有 gcloud SDK，装一套 SDK 比装一个 client 库慢。
为什么不用 HF：权重已在 gs://chrisya-v7x-us-central1（us-central1，与集群同区），
GCS→GCE 同区带宽远高于 HF 的 ~967 MB/s（那是我上周在 bodaborg 实测的数）。

进度每 15 秒打一次 —— 这是个 6 分钟量级的操作，静默会让人以为卡死。
"""
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.cloud import storage

BUCKET = "chrisya-v7x-us-central1"
PREFIX = "models/qwen3.5-397b-a17b-fp8/weights/"
DEST = sys.argv[1] if len(sys.argv) > 1 else "/work/models/qwen3.5-397b"
WORKERS = int(os.environ.get("DL_WORKERS", "48"))

done_bytes = 0
done_files = 0
lock = threading.Lock()


def fetch(blob_name, size):
    global done_bytes, done_files
    rel = blob_name[len(PREFIX):]
    if not rel:
        return
    out = os.path.join(DEST, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # 断点续传：大小一致就跳过。pod 长期存活，重跑脚本不该重下 378 GiB。
    if os.path.exists(out) and os.path.getsize(out) == size:
        with lock:
            done_bytes += size
            done_files += 1
        return
    # 每个线程用自己的 client，storage.Client 不是线程安全的。
    storage.Client().bucket(BUCKET).blob(blob_name).download_to_filename(out)
    with lock:
        done_bytes += size
        done_files += 1


def main():
    os.makedirs(DEST, exist_ok=True)
    blobs = [(b.name, b.size) for b in storage.Client().list_blobs(BUCKET, prefix=PREFIX)
             if not b.name.endswith("/")]
    total = sum(s for _, s in blobs)
    print(f"[fetch] {len(blobs)} 个对象，共 {total/2**30:.1f} GiB → {DEST}，{WORKERS} 并发", flush=True)

    t0 = time.time()
    stop = threading.Event()

    def ticker():
        while not stop.wait(15):
            el = time.time() - t0
            gb = done_bytes / 2**30
            rate = gb / el if el else 0
            eta = (total / 2**30 - gb) / rate if rate > 0.01 else 0
            print(f"[fetch] {done_files}/{len(blobs)} 文件  {gb:.1f}/{total/2**30:.1f} GiB  "
                  f"{rate*1024:.0f} MiB/s  ETA {eta/60:.1f} 分", flush=True)

    threading.Thread(target=ticker, daemon=True).start()
    errs = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, n, s): n for n, s in blobs}
        for f in as_completed(futs):
            try:
                f.result()
            except Exception as e:
                errs.append((futs[f], repr(e)))
    stop.set()

    el = time.time() - t0
    print(f"[fetch] 完成 {done_files}/{len(blobs)}，{done_bytes/2**30:.1f} GiB，"
          f"耗时 {el/60:.2f} 分，均速 {done_bytes/2**30/el*1024:.0f} MiB/s", flush=True)
    if errs:
        for n, e in errs[:10]:
            print(f"[fetch] 失败 {n}: {e}", flush=True)
        print("WEIGHTS_FAILED", flush=True)
        sys.exit(1)
    # 只有 94 片齐全才算数 —— 少一片会在加载到一半时才炸，那时已经烧掉半小时。
    got = len([f for f in os.listdir(DEST) if f.endswith(".safetensors")])
    if got != 94:
        print(f"WEIGHTS_FAILED: safetensors 分片 {got}/94", flush=True)
        sys.exit(1)
    print("WEIGHTS_OK", flush=True)


if __name__ == "__main__":
    main()
