# maxtext-hunyuan3 — 跑测试用的脚本

**代码不在这里。** 模型实现、配置、以及对 MaxText 的全部改动，
唯一真相在 **[`yangwhale/maxtext` 的 `hunyuan3` 分支](https://github.com/yangwhale/maxtext/tree/hunyuan3)**。

这个目录只放两个脚本：

| 文件 | 作用 |
|---|---|
| [`prep.sh`](prep.sh) | clone 分支 → 6 项自检 → 打包 `src/maxtext` 传 GCS |
| [`run.sh`](run.sh) | 提交 JobSet，pod 里用分支那棵树覆盖容器的 `/deps/src/maxtext` |

```bash
export GCS_STAGE=gs://your-bucket/hy3
export IMAGE=us-docker.pkg.dev/YOUR-PROJECT/gcr.io/YOUR-maxtext-latest:runner

bash prep.sh                                                   # 改了代码才要重跑
PLATFORM=v5p bash run.sh myrun                                 # 或 PLATFORM=v7

# 4 芯片冒烟：必须显式缩规模，否则默认按 64 台 / 256 芯片起
NODES=1 TOPO=2x2x1 PLATFORM=v5p MODEL=hunyuan3-smoke STEPS=8 \
  bash run.sh smoke per_device_batch_size=1 max_target_length=2048
```

## 为什么这里不再放代码

早期这里有一份 `hunyuan3.py` 加一个 `port.py`（把改动打到任意上游 checkout）。
建了分支之后，**同一份改动就同时存在于两个地方**——
这正是 2026-07-28 夜连撞两次的那类 bug：在一处改对了、另一处没跟上，
而所有测试恰好都跑在改对的那一处，所以毫无征兆。

分支建立后，`port.py` 唯一剩下的用途是「跟上游」，而这件事 `git rebase` 做得更好：
**冲突会明确报冲突**，不像字符串锚点匹配那样，锚点漂了只能靠断言去猜。

所以：

- 改代码 → 在分支上提 commit / 开 PR
- 跟上游 → `git rebase upstream/main`
- 跑测试 → `prep.sh` 从分支拉

## 相关文档

- [移植范式（写给外部团队）](../MAXTEXT-PORTING-GUIDE.md)
- [完整实验记录与性能数据](../README.md)
