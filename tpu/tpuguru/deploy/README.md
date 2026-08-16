# deploy

- `systemd` unit：常驻 + `Restart=always`
- 反代片段：跳板机侧 `uri strip_prefix /tpuguru` + `reverse_proxy <内网IP>:<PORT>`，
  **后端不感知前缀**
- Firestore 索引：`status` / `created_by` / `tags` / `created_at desc` /
  `metrics.peak_hbm_gb` / `metrics.end_to_end_s`
