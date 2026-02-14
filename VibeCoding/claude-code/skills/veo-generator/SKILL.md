---
name: veo-generator
description: Generate videos using Google Veo 3.1 on Vertex AI. Use when the user says "生成视频", "做个视频", "generate video", "帮我生成一段视频", "create a video", "视频生成", "text to video", "文生视频", or when you need to create video content.
---

# Veo 3.1 Video Generator

Generate videos from text prompts using Veo 3.1 on Vertex AI, save to CC Pages, and send to Discord.

## Usage

Call the generation script directly:

```bash
~/.claude/scripts/veo-generate.sh "your prompt here"
```

### Options

```bash
# Basic generation (1 video, 16:9, 8s, fast model, 720p)
~/.claude/scripts/veo-generate.sh "a cat playing with a laser pointer"

# Portrait video for mobile
~/.claude/scripts/veo-generate.sh "smartphone app demo" --aspect 9:16

# Shorter clip
~/.claude/scripts/veo-generate.sh "ocean waves at sunset" --duration 6

# Higher quality model (slower, more expensive)
~/.claude/scripts/veo-generate.sh "cinematic drone shot over mountains" --model standard

# 1080p resolution
~/.claude/scripts/veo-generate.sh "product showcase" --resolution 1080p

# Multiple videos (1-4)
~/.claude/scripts/veo-generate.sh "abstract art animation" --count 2

# Negative prompt (avoid certain content)
~/.claude/scripts/veo-generate.sh "peaceful park scene" --negative "people, crowds, text"

# Disable prompt rewriter (use exact prompt)
~/.claude/scripts/veo-generate.sh "minimal geometric shapes" --no-rewrite

# Custom output filename
~/.claude/scripts/veo-generate.sh "logo animation" --output brand-intro

# Longer timeout for complex videos
~/.claude/scripts/veo-generate.sh "detailed cityscape timelapse" --timeout 600

# Combine options
~/.claude/scripts/veo-generate.sh "B200 GPU rack with blinking LEDs, cinematic" --aspect 16:9 --model standard --resolution 1080p --duration 8
```

### Output

- Videos saved to `/var/www/cc/assets/veo/` as MP4
- Returns the public URL: `https://cc.higcp.com/assets/veo/{filename}.mp4`
- When called from Discord context, send the URL using `send-to-discord.sh --plain`

### Workflow for Discord

```bash
# 1. Generate video
URL=$(~/.claude/scripts/veo-generate.sh "your prompt" --aspect 16:9)

# 2. Send to Discord
~/.claude/scripts/send-to-discord.sh --plain "$URL"
```

## Important: Async Operation

Unlike Imagen (synchronous), Veo uses **long-running operations**:
1. Submit request → get operation ID
2. Poll `fetchPredictOperation` every N seconds
3. When `done: true`, extract video bytes or GCS URI

The script handles all of this automatically. Default poll interval is 10s, timeout is 300s (5 min). For complex prompts, use `--timeout 600`.

**Progress output goes to stderr**, so `URL=$(...veo-generate.sh...)` captures only the final URL.

## Models

- **`veo-3.1-fast-generate-preview`** (fast, default) — Faster generation, good quality
- **`veo-3.1-generate-preview`** (standard) — Best quality, slower

### Older Models (available but not default)

- `veo-3.0-generate-001` / `veo-3.0-fast-generate-001` — Veo 3 GA
- `veo-2.0-generate-001` — Veo 2 (no 1080p support)

## Parameters

- **Aspect Ratio**: `16:9` (landscape, default), `9:16` (portrait)
- **Duration**: 4, 6, or 8 seconds (default: 8)
- **Resolution**: `720p` (default), `1080p` (Veo 3+ only)
- **Sample Count**: 1-4 videos per request
- **Negative Prompt**: describe content to avoid
- **Enhance Prompt**: Gemini-powered prompt rewriting (on by default)
- **Person Generation**: `allow_adult` (default), `disallow`

## Prompt Tips

- Be specific: "A golden retriever running through autumn leaves in slow motion, cinematic lighting" > "dog running"
- Specify camera movement: "dolly zoom", "tracking shot", "aerial drone shot", "timelapse"
- Specify style: "cinematic", "documentary", "anime", "watercolor", "photorealistic"
- Supports Chinese prompts natively
- Use `--negative` to exclude unwanted elements

## Prerequisites

- gcloud CLI with valid credentials
- Vertex AI API enabled on the GCP project
- `/var/www/cc/` served by nginx (CC Pages)

## Files

```
~/.claude/scripts/
└── veo-generate.sh      # Video generation script

/var/www/cc/assets/veo/  # Generated videos (web-accessible)
```
