---
name: web-server-setup
description: 安装配置 nginx 静态文件服务器，用于展示 Claude Code 生成的富内容（表格、图表、报告等）。当用户说"装nginx"、"配web服务"、"setup web server"时触发。
---

# Web Server 配置

本机 nginx 静态文件服务器，用于 serve Claude Code 生成的 HTML 内容。

## 架构

```
Claude Code 生成 HTML → /var/www/cc/ → 本机 nginx :80 → hk-jmp nginx 反代 → GCP ALB → https://cc.higcp.com/
```

## 目录结构

```
/var/www/cc/
├── index.html          # 文件列表首页
├── pages/              # Claude 生成的富内容页面
│   ├── gpu-comparison-20260213.html
│   └── benchmark-results.html
└── assets/             # 静态资源（CSS/JS/图片）
    └── style.css
```

## 安装

```bash
scripts/setup-nginx.sh
```

脚本会：
1. 安装 nginx（如果没装）
2. 创建 `/var/www/cc/` 目录结构
3. 部署 nginx 配置到 `/etc/nginx/sites-available/cc`
4. 启用站点并重启 nginx

## 生成页面

Claude Code 在需要展示复杂内容时：
1. 生成 HTML 文件到 `/var/www/cc/pages/`
2. 文件名格式: `{topic}-{YYYYMMDD-HHmmss}.html`
3. 在 Discord 消息中发送链接: `https://cc.higcp.com/pages/{filename}`

## nginx 配置要点

- 监听 80 端口
- root: `/var/www/cc/`
- autoindex off（目录列表已关闭，防止枚举页面）
- 允许跨域（Discord embed 预览）
- gzip 压缩
- 安全: server_tokens off, X-Content-Type-Options, X-Frame-Options, CSP, 只允许 GET/HEAD, 隐藏文件 404

## 安全模型

不加 IAP，靠 URL 不可猜测性保护内容：
- autoindex 关闭，无法浏览目录
- 只有知道完整文件名才能访问
- 首页和 assets 公开（Discord OG 爬虫需要访问）
