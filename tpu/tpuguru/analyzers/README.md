# analyzers — 把 worker 的产物变成结构化结论

每个分析器输出写进 `result.analyses.<name>`，各自带 `version`。
新增维度 = 加一个分析器 + 一个 key，不动 schema。

| 分析器 | 输入 | 产出 | 期 |
|---|---|---|---|
| `memory` | stdout | argument/output/temp/峰值、离上限余量、建议 batch 上限 | v0 |
| `failure` | stdout | 失败分类（见 ../README.md §6A 的五类） | v0 |
| `compile_time` | stdout | HLO_PASSES / BACKEND_PASSES / CODE_GEN / END_TO_END | v0 |
| `codepath` | 探针输出 | kernel 分支、pspec、kernel 入参、量化配置 | v1 |
| `hlo` | XLA dump | 算子数、融合数、集合通信清单（含是否 async / 在不在循环体） | v1 |
| `llo` | Mosaic dump | kernel 内部分块循环、VMEM、有无跨卡原语 | v2 |
