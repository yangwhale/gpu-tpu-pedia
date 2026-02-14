# GCP 基础设施

## 项目
- TPU 项目: `cloud-tpu-multipod-dev`
- GPU 项目: `gpu-launchpad-playground`

## Chris 的 MIG (gpu-launchpad-playground)
| MIG 名称 | Zone | GPU 类型 | 备注 |
|-----------|------|----------|------|
| chrisya-b200-mig | us-central1-b | B200 | |
| chrisya-b200-mig-ase1 | asia-southeast1-b | B200 | 曾缺货 ZONE_RESOURCE_POOL_EXHAUSTED |
| chrisya-b200-spot-mig-ase1 | asia-southeast1-b | B200 Spot | GPU 测试主力 MIG |
| chrisya-h200-mig | us-south1-b | H200 | |
| chrisya-a3m-mig | us-east4-a | A3 Mega | |

## TPU 测试机
| Alias | TPU 类型 | HostName | 跳板机 | 项目 | Zone | 备注 |
|-------|---------|----------|--------|------|------|------|
| `t1` / `chrisya-v6e-8` | v6e-8 | 10.146.0.18 | sg-tpu-jmp (IAP) | chris-pgp-host | asia-southeast1-b | 常驻不关机 |

## TPU v7x 预留 (cloud-tpu-multipod-dev)
| 预留名 | Zone | Chips | 在用 | 到期 | 状态 |
|--------|------|-------|------|------|------|
| ghostfish-ue1w5sepvwwdd | us-central1-ai1a | 8,640 | 8,576 | 2026-02-25 | DEGRADED |
| ghostfish-n5yz4l5ckudco | us-central1-c | 8,704 | 8,128 | 2026-02-16 | DEGRADED |
| ghostfish-nyya9waxxi4mt | us-central1-ai1a | 6,080 | 5,948 | 2026-02-25 | DEGRADED |
| ghostfish-wvwl2yfmw8agn | us-central1-ai1a | 1,280 | 1,216 | 2026-02-25 | DEGRADED |
| cloudtpu-20260211010000-500993041 | us-central1-ai1a | 1,280 | 1,028 | 无 | OK |

创建 TPU v7x 测试机时用 `--reservation=预留名` 指定，zone 必须匹配。

## GKE
- 集群: chrisya-v7x-training (TPU 训练用, us-central1)
- 集群: chrisya-gke-a4 (GPU 测试用)

## Web 服务
- CC Pages: https://cc.higcp.com/
- 架构: 公网 → GCP ALB (cc-alb) → hk-jmp:80 (nginx 反代) → 10.8.0.200:80 (本机 nginx)
- hk-jmp: asia-east2-c, tags: allow-hk + lb-health-check
- Web root: `/var/www/cc/`，pages 目录: `/var/www/cc/pages/`
- Health check: `hc-http-80` (HTTP GET `/` on port 80)
- Backend services: `cc-bs`(公开) + `cc-bs-iap`(IAP 保护 /pages/*)
- IAP 授权: `whale@chrisya.altostrat.com` (roles/iap.httpsResourceAccessor)

## SSH
- 密钥: `~/.ssh/google_compute_engine`
- 用 gcloud MCP 工具管理实例
- GPU: `b1` → chrisya-b200-spot-mig-ase1 实例 (10.8.0.32, WireGuard VPN), Spot IP 可能变
- TPU: `t1` → chrisya-v6e-8 (10.146.0.18, 经 sg-tpu-jmp IAP tunnel)
- 跳板机: sg-tpu-jmp (chris-pgp-host, asia-southeast1-b)
- 跳板机: hk-jmp (chris-pgp-host, asia-east2-c, 10.240.32.2, CC Pages 反代)
