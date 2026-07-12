# GB300 (A4X Max) 生产运维

GB300 NVL72 集群的生产级运维指南。

> 待基础测试完成后根据实际运维经验补充。

## 与 GB200 的差异

| 维度 | GB200 | GB300 |
|------|-------|-------|
| 主机维护 | Live Migration 可用 | **TERMINATE only** (Bare Metal) |
| Spot/Preemptible | 支持 | **不支持** |
| CPU 监控 | Hypervisor 指标 | 需 **Ops Agent** (OS Reported CPU %) |
| 故障检测 | VM 级别 | Bare Metal 级别 |
| Hugepages | 不需要 | 需要配置 `hugepage_size2m: 4096` |

## 待补充

- [ ] 故障节点排除和替换流程
- [ ] Checkpoint 恢复策略（Bare Metal 无 Live Migration）
- [ ] 节点健康检查脚本
- [ ] GPU ECC 错误监控
- [ ] RDMA 链路监控

## GB200 参考

GB200 生产运维文档: [a4x/10-production-ops/](../../a4x/10-production-ops/)
