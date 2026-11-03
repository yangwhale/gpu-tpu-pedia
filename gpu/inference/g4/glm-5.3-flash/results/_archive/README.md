# 归档：不用于结论的中间结果

| 目录 | 为什么归档 |
|---|---|
| `pd-1p1d-fp8-ucxdefault`、`pd-1p1d-nvfp4-ucxdefault` | PD 实例未设 `UCX_TCP_TX/RX_SEG_SIZE`，KV 传输慢 3 倍；且 1P1D 用 TP8 做 prefill，P:D 配比不对 |
| `pd-2p2d-nvfp4-ucxdefault`、`rate-2p2d-nvfp4-ucxdefault` | 未做 UCX 调优，已由 `../pd-2p2d-nvfp4` 替代（保留其 decode 日志作为调优前的 KV 传输对照） |
| `pd-2p1d-fp8-ucxdefault`、`rate-2p1d-fp8-ucxdefault` | 未做 UCX 调优；FP8 只能 TP8，一个 D 先饱和，非推荐配置 |
| `pd-4p2d-nvfp4-ucxdefault` | 未做 UCX 调优，已由 `../pd-4p2d-nvfp4` 替代 |
| `rate-dp4-nvfp4-coarse` | 速率步长 0.5 req/s 偏粗，已由 `../rate-colocated-nvfp4-2xtp4` 的细步长测量替代 |

未保留的失败尝试（无有效数据）：P=TP4 / D=TP8 混合 TP 的 2P1D（NIXL 握手失败，全部请求 500）；第一次速率扫描（每档同一随机种子，命中前缀缓存）；第一次 4P2D 速率扫描（被操作失误打断）。细节见 `../../JOURNAL.md`。
