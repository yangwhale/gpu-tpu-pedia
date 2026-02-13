#!/usr/bin/env bash
set -euo pipefail

WEB_ROOT="/var/www/cc"

echo "=== 安装 nginx ==="
if ! command -v nginx &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq nginx
    echo "nginx installed"
else
    echo "nginx already installed: $(nginx -v 2>&1)"
fi

echo "=== 创建目录结构 ==="
sudo mkdir -p "$WEB_ROOT"/{pages,assets}
sudo chown -R "$USER:$USER" "$WEB_ROOT"

# 默认样式
cat > "$WEB_ROOT/assets/style.css" << 'CSS'
:root { --bg: #1a1a2e; --fg: #e0e0e0; --accent: #58acff; --card: #16213e; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, 'Segoe UI', sans-serif; background: var(--bg); color: var(--fg); padding: 2rem; max-width: 960px; margin: 0 auto; line-height: 1.6; }
h1, h2, h3 { color: var(--accent); margin: 1.5rem 0 0.5rem; }
h1 { font-size: 1.8rem; border-bottom: 2px solid var(--accent); padding-bottom: 0.5rem; }
table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
th, td { padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid #2a2a4a; }
th { background: var(--card); color: var(--accent); font-weight: 600; }
tr:hover { background: var(--card); }
pre, code { background: var(--card); border-radius: 4px; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.9rem; }
pre { padding: 1rem; overflow-x: auto; margin: 1rem 0; }
code { padding: 0.15rem 0.4rem; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.card { background: var(--card); border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
.meta { color: #888; font-size: 0.85rem; margin-bottom: 1rem; }
ul, ol { padding-left: 1.5rem; margin: 0.5rem 0; }
li { margin: 0.3rem 0; }
CSS

# 首页
cat > "$WEB_ROOT/index.html" << 'HTML'
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CC Pages</title>
<link rel="stylesheet" href="/assets/style.css">
</head>
<body>
<h1>Claude Code Pages</h1>
<p class="meta">Claude Code 生成的富内容页面</p>
<div class="card">
<p>页面目录: <a href="/pages/">/pages/</a></p>
</div>
</body>
</html>
HTML

echo "=== 配置 nginx ==="
sudo tee /etc/nginx/sites-available/cc > /dev/null << 'NGINX'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;

    root /var/www/cc;
    index index.html;

    # 目录列表（关闭，靠 URL 不可猜测性保护内容）
    autoindex off;

    # gzip 压缩
    gzip on;
    gzip_types text/html text/css application/javascript application/json text/plain;
    gzip_min_length 256;

    # CORS (Discord embed 预览)
    add_header Access-Control-Allow-Origin "*" always;

    # 缓存静态资源
    location /assets/ {
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    location / {
        try_files $uri $uri/ =404;
    }
}
NGINX

# 启用站点
sudo rm -f /etc/nginx/sites-enabled/default 2>/dev/null || true
sudo ln -sf /etc/nginx/sites-available/cc /etc/nginx/sites-enabled/cc

echo "=== 测试配置 ==="
sudo nginx -t

echo "=== 重启 nginx ==="
sudo systemctl enable nginx
sudo systemctl restart nginx

echo ""
echo "✅ nginx 已配置完成"
echo "   Web root: $WEB_ROOT"
echo "   URL: http://$(hostname -I | awk '{print $1}')/"
