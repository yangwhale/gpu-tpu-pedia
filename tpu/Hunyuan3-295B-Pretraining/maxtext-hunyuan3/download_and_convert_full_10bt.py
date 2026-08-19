#!/usr/bin/env python3
"""
并发全量下载 FineWeb-Edu 10BT 并转换为 Grain/MaxText 严格兼容的 ArrayRecord 格式：
1. group_size:1,zstd (Grain 强制要求)
2. 内部结构封装为标准 tf.train.Example Protobuf (含 'text' 字段)
"""

import os
import sys
import time
import concurrent.futures
import pyarrow.parquet as pq
from array_record.python import array_record_module
from huggingface_hub import hf_hub_download

REPO_ID = "HuggingFaceFW/fineweb-edu"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data/fineweb-edu-raw")
OUT_DIR = os.path.join(BASE_DIR, "data/fineweb-edu-arrayrecord")

FILES = [f"sample/10BT/{i:03d}_00000.parquet" for i in range(14)]

def encode_varint(n):
    b = bytearray()
    while True:
        towrite = n & 0x7f
        n >>= 7
        if n:
            b.append(towrite | 0x80)
        else:
            b.append(towrite)
            break
    return bytes(b)

def make_tf_example_proto(text: str) -> bytes:
    t_bytes = text.encode('utf-8')
    # BytesList: tag 1 (wire 2) -> 0x0a + len + t_bytes
    bl = b'\x0a' + encode_varint(len(t_bytes)) + t_bytes
    # Feature: tag 1 (wire 2) -> 0x0a + len + bl
    feat = b'\x0a' + encode_varint(len(bl)) + bl
    # FeatureEntry: key (0x0a 0x04 'text') + value (0x12 + len(feat) + feat)
    key_entry = b'\x0a\x04text\x12' + encode_varint(len(feat)) + feat
    # Features: tag 1 (wire 2) -> 0x0a + len + key_entry
    feats = b'\x0a' + encode_varint(len(key_entry)) + key_entry
    # Example: tag 1 (wire 2) -> 0x0a + len + feats
    ex = b'\x0a' + encode_varint(len(feats)) + feats
    return ex

def process_file(idx_and_filename):
    idx, filename = idx_and_filename
    print(f"[{idx}/14] 开始下载: {filename} ...")
    t0 = time.time()
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=filename,
            local_dir=RAW_DIR,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        print(f"[!] 下载失败 {filename}: {e}")
        return False

    print(f"[{idx}/14] 下载完成 ({time.time() - t0:.1f}s)，开始转换为 ArrayRecord (group_size:1) ...")
    t1 = time.time()
    
    parquet_file = pq.ParquetFile(local_path)
    records_per_shard = 50000
    sub_idx = 0
    writer = None
    shard_count = 0
    total_docs = 0

    for batch in parquet_file.iter_batches(batch_size=10000, columns=["text"]):
        texts = batch.column(0).to_pylist()
        for text_val in texts:
            if not text_val:
                continue
            if writer is None or shard_count >= records_per_shard:
                if writer:
                    writer.close()
                shard_filename = f"fineweb_edu_train_part{idx:02d}-{sub_idx:04d}.arrayrecord"
                out_path = os.path.join(OUT_DIR, shard_filename)
                # 显式指定 group_size:1 和 zstd
                writer = array_record_module.ArrayRecordWriter(out_path, "group_size:1,zstd")
                sub_idx += 1
                shard_count = 0

            proto_bytes = make_tf_example_proto(text_val)
            writer.write(proto_bytes)
            shard_count += 1
            total_docs += 1

    if writer:
        writer.close()

    print(f"[{idx}/14] 分片 {idx} 转换完成！包含 {total_docs} 篇标准 tf.train.Example 文档，耗时 {time.time() - t1:.1f}s")
    try:
        os.remove(local_path)
    except Exception:
        pass
    return True

def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # 先清理旧格式文件
    print("[*] 清理旧格式 ArrayRecord 文件...")
    for f in os.listdir(OUT_DIR):
        if f.endswith(".arrayrecord"):
            os.remove(os.path.join(OUT_DIR, f))
            
    print(f"[*] 启动 10BT 全量处理流水线（group_size:1 + tf.train.Example）...")
    start_all = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_file, enumerate(FILES)))
        
    print(f"[+] 全量 10BT 处理完成，成功: {sum(results)}/14，总耗时: {time.time() - start_all:.1f}s")

if __name__ == "__main__":
    main()
