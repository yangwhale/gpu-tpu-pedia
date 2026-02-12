#!/bin/bash
# 通过 Discord Bot 发送消息到 Discord 频道
# 用法: send-to-discord.sh "报告内容" [标题] [图片路径或URL]
# 或通过 stdin: echo "内容" | send-to-discord.sh "" "标题" [图片]

BOT_TOKEN="${DISCORD_BOT_TOKEN:-YOUR_TOKEN_HERE}"
CHANNEL_ID="${DISCORD_CHANNEL_ID:-1471088850712531055}"
API_BASE="https://discord.com/api/v10"

CONTENT="$1"
TITLE="${2:-Claude Code 任务报告}"
IMAGE="$3"

# 如果内容为空，从 stdin 读取
if [ -z "$CONTENT" ]; then
    CONTENT=$(cat)
fi

if [ -z "$CONTENT" ]; then
    echo "错误：没有内容可发送"
    exit 1
fi

# Discord embed 描述限制 4096 字符，总消息限制 6000 字符
# 超长时分多条发送
send_message() {
    local payload="$1"
    local file="$2"

    if [ -n "$file" ] && [ -f "$file" ]; then
        # 带本地文件上传
        RESPONSE=$(curl -s -X POST \
            -H "Authorization: Bot ${BOT_TOKEN}" \
            -F "payload_json=${payload}" \
            -F "files[0]=@${file}" \
            "${API_BASE}/channels/${CHANNEL_ID}/messages")
    else
        RESPONSE=$(curl -s -X POST \
            -H "Authorization: Bot ${BOT_TOKEN}" \
            -H "Content-Type: application/json" \
            -d "$payload" \
            "${API_BASE}/channels/${CHANNEL_ID}/messages")
    fi

    # 检查结果
    if echo "$RESPONSE" | grep -q '"id"'; then
        return 0
    else
        ERROR_MSG=$(echo "$RESPONSE" | grep -o '"message":"[^"]*"' | head -1)
        echo "❌ 发送失败: ${ERROR_MSG:-$RESPONSE}" >&2
        return 1
    fi
}

# 构建 embed 消息
build_embed_payload() {
    local title="$1"
    local description="$2"
    local image_url="$3"

    if [ -n "$image_url" ]; then
        # 网络图片通过 embed image 嵌入
        jq -n \
            --arg title "$title" \
            --arg desc "$description" \
            --arg img "$image_url" \
            '{
                "embeds": [{
                    "title": $title,
                    "description": $desc,
                    "color": 5814783,
                    "image": {"url": $img},
                    "footer": {"text": "Claude Code"}
                }]
            }'
    else
        jq -n \
            --arg title "$title" \
            --arg desc "$description" \
            '{
                "embeds": [{
                    "title": $title,
                    "description": $desc,
                    "color": 5814783,
                    "footer": {"text": "Claude Code"}
                }]
            }'
    fi
}

# 判断图片是 URL 还是本地文件
IMAGE_URL=""
IMAGE_FILE=""
if [ -n "$IMAGE" ]; then
    if [[ "$IMAGE" =~ ^https?:// ]]; then
        IMAGE_URL="$IMAGE"
    elif [ -f "$IMAGE" ]; then
        IMAGE_FILE="$IMAGE"
    else
        echo "⚠️  图片不存在: $IMAGE" >&2
    fi
fi

# 分割长内容（embed description 限制 4096 字符）
MAX_LEN=4000
SENT=0

if [ ${#CONTENT} -le $MAX_LEN ]; then
    # 内容不超长，一次发送
    if [ -n "$IMAGE_FILE" ]; then
        # 本地文件：用 attachment:// 引用，保留原始文件名
        FNAME=$(basename "$IMAGE_FILE")
        PAYLOAD=$(jq -n \
            --arg title "$TITLE" \
            --arg desc "$CONTENT" \
            --arg att "attachment://${FNAME}" \
            '{
                "embeds": [{
                    "title": $title,
                    "description": $desc,
                    "color": 5814783,
                    "image": {"url": $att},
                    "footer": {"text": "Claude Code"}
                }]
            }')
        send_message "$PAYLOAD" "$IMAGE_FILE"
    else
        PAYLOAD=$(build_embed_payload "$TITLE" "$CONTENT" "$IMAGE_URL")
        send_message "$PAYLOAD"
    fi
    SENT=$?
else
    # 内容超长，分片发送
    FIRST=true
    SENT=0
    while [ ${#CONTENT} -gt 0 ]; do
        if [ ${#CONTENT} -le $MAX_LEN ]; then
            CHUNK="$CONTENT"
            CONTENT=""
        else
            # 找最近的换行分割
            CHUNK="${CONTENT:0:$MAX_LEN}"
            LAST_NL=$(echo "$CHUNK" | grep -b -o $'\n' | tail -1 | cut -d: -f1)
            if [ -n "$LAST_NL" ] && [ "$LAST_NL" -gt 2000 ]; then
                CHUNK="${CONTENT:0:$LAST_NL}"
                CONTENT="${CONTENT:$LAST_NL}"
            else
                CONTENT="${CONTENT:$MAX_LEN}"
            fi
        fi

        if [ "$FIRST" = true ]; then
            # 第一条带标题和图片
            if [ -n "$IMAGE_FILE" ]; then
                FNAME=$(basename "$IMAGE_FILE")
                PAYLOAD=$(jq -n \
                    --arg title "$TITLE" \
                    --arg desc "$CHUNK" \
                    --arg att "attachment://${FNAME}" \
                    '{
                        "embeds": [{
                            "title": $title,
                            "description": $desc,
                            "color": 5814783,
                            "image": {"url": $att},
                            "footer": {"text": "Claude Code"}
                        }]
                    }')
                send_message "$PAYLOAD" "$IMAGE_FILE" || SENT=1
            else
                PAYLOAD=$(build_embed_payload "$TITLE" "$CHUNK" "$IMAGE_URL")
                send_message "$PAYLOAD" || SENT=1
            fi
            FIRST=false
        else
            # 后续条只有内容
            PAYLOAD=$(build_embed_payload "$TITLE (续)" "$CHUNK" "")
            send_message "$PAYLOAD" || SENT=1
        fi

        [ ${#CONTENT} -gt 0 ] && sleep 0.5
    done
fi

if [ $SENT -eq 0 ]; then
    echo "✅ 报告已发送到 Discord"
else
    exit 1
fi
