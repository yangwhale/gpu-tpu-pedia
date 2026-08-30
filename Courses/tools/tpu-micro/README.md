# TPU 显微镜 —— 八张图拆开一颗 TPU v7

对标 `../gpu-micro/`，同一套视觉标准、同一套「三级来源」纪律。

## 用法

```bash
python3 build_doc.py out.html --mode internal   # 内部版
python3 build_doc.py out.html --mode public     # 公开版（自带出厂自检）
./render.sh fig_t4_mxu                          # 单图 渲染 → 截图
```

**两个版本由同一份稿子构建。** 内部条目全部包在 `gate.I()` / `gate.IP()` 里：
`I` = 仅内部可见（公开版删除），`IP` = 内外两套说法（公开版替换）。
`--mode public` 会跑 `gate.lint_public()`，命中内部标记就 `exit 1`，不出文件。

## 一条不能省的纪律

每张图都必须走 **渲染 → 截图 → 目视核对 → 修** 的循环。
文字压边框、图例和实际颜色对不上、标签压在箭头上 —— 这些在源码里一个都看不出来，
只有看 PNG 才会暴露。这批图里约 18 个缺陷是这么找出来的。

## 图

| 脚本 | 讲什么 |
|---|---|
| `fig_t1_chip.py` | 封装 → 两个 die → 核 → HBM → 六个 ICI 出口 |
| `fig_t2_core.py` | 一个 TensorCore 拆开；**右边是 GPU 有而这里没有的五样** |
| `fig_t3_hierarchy.py` | 并行层级，**按问题对齐不按名词对齐** |
| `fig_t4_mxu.py` | 脉动阵列快照 + 峰值公式那条推导链 |
| `fig_t5_sparsecore.py` | SparseCore；核间数据流是**单向**的 |
| `fig_t6_datapath.py` | 一个数走完全程；每站只问「谁决定搬」 |
| `fig_t7_vliw.py` | VLIW 一拍多槽与编译期排班（**示意，不是实测 trace**） |
| `fig_t8_pod.py` | 一颗 → 9,216 颗，对数横轴 |
