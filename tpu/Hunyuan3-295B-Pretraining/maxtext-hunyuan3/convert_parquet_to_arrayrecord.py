#!/usr/bin/env python3
"""
从 HuggingFace 下载 FineWeb-Edu 分片并高效烘焙为 ArrayRecord 格式。
"""

import os
import sys
import time
import json
import pyarrow.parquet as pq
from array_record.python import array_record_module
from huggingface_hub import hf_hub_download

REPO_ID = "HuggingFaceFW/fineweb-edu"
# sample/10BT 下的 parquet 文件
SAMPLE_FILE = "sample/10BT/000_00000.parquet"
LOCAL_RAW_DIR = "/home/chrisya/gpu-tpu-pedia/tpu/Hunyuan3-295B-Pretraining/data/fineweb-edu-raw"
OUTPUT_DIR = "/home/chrisya/gpu-tpu-pedia/tpu/Hunyuan3-295B-Pretraining/data/fineweb-edu-arrayrecord"

def main():
    os.makedirs(LOCAL_RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"[*] 1. 开始从 Hugging Face 下载 FineWeb-Edu 样例分片: {SAMPLE_FILE} ...")
    start_t = time.time()
    downloaded_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=SAMPLE_FILE,
        local_dir=LOCAL_RAW_DIR,
        local_dir_use_symlinks=False
    )
    print(f"[+] 下载完成: {downloaded_path} (耗时 {time.time() - start_t:.2f}s, 大小: {os.path.getsize(downloaded_path)/1024/1024:.2f} MB)")
    
    print("[*] 2. 开始流式读取 Parquet 并批量转换为 ArrayRecord 格式...")
    t0 = time.time()
    parquet_file = pq.ParquetFile(downloaded_path)
    total_written = 0
    records_per_shard = 50000
    shard_idx = 0
    writer = None
    shard_count = 0

    for batch in parquet_file.iter_batches(batch_size=10000, columns=["text"]):
        texts = batch.column(0).to_pylist()
        for text_val in texts:
            if not text_val:
                continue
            if writer is None or shard_count >= records_per_shard:
                if writer:
                    writer.close()
                out_file = os.path.join(OUTPUT_DIR, f"fineweb_edu_train-{shard_idx:05d}.arrayrecord")
                writer = array_record_module.ArrayRecordWriter(out_file, "zstd")
                shard_idx += 1
                shard_count = 0
                print(f"[*] 正在写入分片: {out_file}")

            doc_data = json.dumps({"text": text_val}, ensure_ascii=False).encode("utf-8")
            writer.write(doc_data)
            shard_count += 1
            total_written += 1

    if writer:
        writer.close()
        
    print(f"[+] 转换完成！总计写入 {total_written} 条文档，耗时: {time.time() - t0:.2f}s")
    
    # 验证生成的 ArrayRecord
    print("[*] 3. 验证 ArrayRecord 随机读取...")
    first_shard = os.path.join(OUTPUT_DIR, "fineweb_edu_train-00000.arrayrecord")
    reader = array_record_module.ArrayRecordReader(first_shard)
    num_records = reader.record_count()
    first_record = reader.read_record(0)
    print(f"[+] 验证通过！第一分片包含 {num_records} 条记录，第一条长度: {len(first_record)} bytes")

if __name__ == "__main__":
    main()
