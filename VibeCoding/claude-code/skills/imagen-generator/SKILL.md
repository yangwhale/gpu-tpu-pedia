---
name: imagen-generator
description: Generate images using Google Imagen 4 on Vertex AI. Use when the user says "生成图片", "画一张", "generate image", "帮我画", "生成一张图", "create an image", "图片生成", or when you need to create visual content to explain concepts.
---

# Imagen 4 Image Generator

Generate images from text prompts using Imagen 4 on Vertex AI, save to CC Pages, and send to Discord.

## Usage

Call the generation script directly:

```bash
~/.claude/scripts/imagen-generate.sh "your prompt here"
```

### Options

```bash
# Basic generation (1 image, 1:1 aspect ratio, fast model)
~/.claude/scripts/imagen-generate.sh "a cute cat sitting on a GPU server"

# Custom aspect ratio: 1:1, 3:4, 4:3, 16:9, 9:16
~/.claude/scripts/imagen-generate.sh "TPU pod in a datacenter" --aspect 16:9

# Multiple images (1-4)
~/.claude/scripts/imagen-generate.sh "neural network visualization" --count 2

# Model tier: fast (default, $0.02), standard ($0.04), ultra ($0.06)
~/.claude/scripts/imagen-generate.sh "detailed architecture diagram" --model standard

# High resolution (2K/4K, only standard and ultra models)
~/.claude/scripts/imagen-generate.sh "landscape photo" --model standard --resolution 2048x2048

# Custom output filename
~/.claude/scripts/imagen-generate.sh "logo design" --output my-logo

# Disable prompt rewriter (use exact prompt)
~/.claude/scripts/imagen-generate.sh "minimal line drawing" --no-rewrite

# Combine options
~/.claude/scripts/imagen-generate.sh "B200 GPU rack in datacenter, photorealistic" --aspect 16:9 --model standard --count 2
```

### Output

- Images saved to `/var/www/cc/assets/imagen/` as PNG
- Returns the public URL: `https://cc.higcp.com/assets/imagen/{filename}.png`
- When called from Discord context, send the URL using `send-to-discord.sh --plain`

### Workflow for Discord

```bash
# 1. Generate image
URL=$(~/.claude/scripts/imagen-generate.sh "your prompt" --aspect 16:9)

# 2. Send to Discord
~/.claude/scripts/send-to-discord.sh --plain "$URL"
```

Or just run the script — it prints the URL which you can share.

## Models

- **`imagen-4.0-fast-generate-001`** (default) — Fast, $0.02/image, 1K resolution, 150 QPM
- **`imagen-4.0-generate-001`** (standard) — Best quality, $0.04/image, up to 4K resolution, 75 QPM
- **`imagen-4.0-ultra-generate-001`** (ultra) — Highest quality, $0.06/image, up to 4K resolution, 30 QPM

## Supported Aspect Ratios

All models: `1:1`, `3:4`, `4:3`, `16:9`, `9:16`

## Supported Resolutions

- **fast**: 1024x1024, 896x1280, 1280x896, 768x1408, 1408x768
- **standard/ultra**: Above + 2048x2048, 1792x2560, 2560x1792, 1536x2816, 2816x1536

## Prompt Tips

- Be specific and descriptive: "a red sports car on a mountain road at sunset, photorealistic" > "car"
- Imagen 4 has a built-in prompt rewriter that enhances your prompt (enabled by default)
- Supports Chinese prompts natively (simplified & traditional)
- For technical diagrams, add style keywords: "technical illustration", "blueprint style", "infographic"
- To disable prompt enhancement, use `--no-rewrite`

## Prerequisites

- `google-genai` Python SDK (already installed for STT)
- GCP project with Vertex AI API enabled
- Application Default Credentials configured

## Files

```
~/.claude/scripts/
└── imagen-generate.sh    # Image generation script

/var/www/cc/assets/imagen/ # Generated images (web-accessible)
```
