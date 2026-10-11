# 04-bench：组件验收基准

04 章「组件验收」C1–C3、C5、C6 用的脚本。都在实测集群实测跑过；
阈值在 `thresholds.csv`，判定用 `bench_judge.py`，驱动在上一层 `../04-bench.sh`。

| 文件 | 测什么 | 出处（实测） | 在一个切片上多久 |
|---|---|---|---|
| `c1_chip_bench.py` | 每 host：bf16 / fp8 matmul（单 device 与 8 device 并发）、HBM 读写、JAX `device_put` H2D / D2H。每 host 各打一份 `C1_ROW` | 芯片基准、P7 档实测 | pip 约 40 s + 约 2–3 min |
| `c3_ici_axes.py` | 片内 ICI：按物理坐标排成 (X,Y,Z,C) mesh，分轴 / 组合 / 全片 all-reduce 与 all-gather。只有 JAX process 0 打 `C3_ROW` | 分轴 ICI 实测 | 约 2 min |
| `c2_ici_bench.py` | 一维 mesh 全片集合通信，只做对照；`04-bench.sh` 里的模式名是 `ici1d`。**和 04 卡片 C2（HBM，含在 c1）无关** | 一维 ICI 实测 | 1–2 min |
| `gcsfuse-read.yaml.tmpl` | N 台 host 同时读桶里同一个大文件，每台 R 路并发 pread | gcsfuse 读实测 | 每个 Job < 1 min |
| `bench_judge.py` | 解析以上全部结果行，按 `thresholds.csv` 判 PASS / WARN / FAIL；算力同时给两种占峰值 | — | 秒级 |
| `thresholds.csv` | 基线 + 阈值（阈值全是 [推断]，出处列写了基线来自哪次运行） | — | — |

`c1_chip_bench.py` 的 info 行带上 `libtpu_init_args / libtpu / p_state`（P7 档实测：同一颗芯片单 device bf16
默认档 868、P7 995 TFLOP/s，不带功耗档的算力数没法判）。

## 最短用法
```bash
source ~/runbook-env.sh && cd "$RUNBOOK_DIR/scripts"
./04-bench.sh configmap                       # 一次
./04-bench.sh run 4x4x4 c1c3                  # 生成 → 避手工片 → apply → 等 → 收日志 → 判定 → 删；结果在 $STATE_DIR/04/bench/<名>/
./04-bench.sh run 4x4x4 c1 --p7 --name bc1-444p7   # 功耗档 A/B（可选）
./04-bench.sh gcsfuse --pool <开了 WI 的池> --hosts 16 --readers 32 --file <桶内 ≥10 GB 文件> | kubectl apply -f -   # 默认进 $INSPECT_NS，前提见 04 C6 [未实测]
kubectl -n $INSPECT_NS wait job/gcs16r32 --for=condition=complete --timeout=10m
kubectl -n $INSPECT_NS logs -l rb-bench-job=gcs16r32 --tail=1 --prefix > gcs.log && ./04-bench.sh judge gcs.log
./04-bench.sh cleanup && ./04-teardown-check.sh
```

## 坑（都实测踩过）
1. **reduce-scatter 行作废**：`psum_scatter(tiled)` 接 `jnp.tile` 被 XLA 改写成 1 个 all-reduce（`--hlo` 打开 `C3_HLO=1` 可见），
   按 RS 公式算出的 busbw 恰好是 AR 的一半。验收只看 all-reduce。
2. **结果在 process 0 的 pod 里，不在 worker-0-0**：实测见过 worker-0-2 / 0-3 / 0-7 / 0-29 / 0-51。`collect` 收全部 pod。
3. **HBM 并发数组别用 `(8, n)`**：每片 `(1, n)` 被 TPU tile 补成 8 行，只测到 40%；脚本已用 `(8, n/1024, 1024)`。
4. **H2D 有两个口径**：这里是 JAX `device_put`（sharded 约 43 GB/s/host）；04 §4.1 参考作业的 223.6 GB/s 是 raw DMA 工具。不能互比。
5. **换 libtpu 版本先重测基线**：全部基线只有 jax 0.11.2 / libtpu 0.0.49（`-U --pre` 当天装到的），DCN 基准则钉 0.11.0 / 0.0.48。
6. **8x8x16 一次偶发塌陷**（全片 AR 19 GB/s，z 轴 27）：重跑正常。单次 FAIL 先在同一批 cube 上重跑一次再判。
