# worker — 在 CPU 上跑 AOT

用与生产同一个 tag 的 docker 镜像跑 `train_compile.py`，
挂上探针、开 XLA dump，产出 stdout / HLO / (三期) LLO。

| 文件 | 状态 |
|---|---|
| `probe_codepath.py` | ✅ **已验证可用**（2026-08-16 实测跑出分片规格与 kernel 入参） |
| `run_aot.sh` | 待写：命令转换 + docker 调用 |
| `convert.py` | 待写：`train` → `train_compile` 的参数映射（见 ../README.md §5） |

## probe_codepath.py 输出样例

```
@@@PSP cfg shard_exp_on_fsdp= True num_experts= 192 ep_size= 1 quant= True
@@@PSP wi_kernel_axes= ('embed_moe', None, 'mlp_moe') -> pspec P('fsdp', None, None)
@@@PSP GLOBAL x= (896,4096,4096)  w0= (192,4096,1536)  wo= (192,1536,4096)
@@@PSP KERNEL lhs=(229376,4096) rhs=(3,4096,1536) gs=(192,) wga=[('fsdp',0)] tokamax=False
```

**最后一行就是本工具存在的理由** —— `rhs` 第 0 维是 3 而不是 192，
意味着这份配置只会算本地那 3 个专家。这一行在真机上跑两天才发现，在这里 3 分钟就能看到。

## 注意

- **镜像 tag 必须与生产一致**，编译器版本变了结论不可迁移
- 探针挂钩时注意：同一个 `.py` 会被以 `pkg.*` 和 `src.pkg.*` 两条路径导入成**两个模块对象**，
  两个都要打，否则补丁静默不生效
