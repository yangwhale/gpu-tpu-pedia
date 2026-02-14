#!/usr/bin/env bash
# Imagen 4 — Text-to-image generation via Vertex AI REST API
#
# Usage:
#   imagen-generate.sh "prompt" [options]
#
# Options:
#   --aspect RATIO    Aspect ratio: 1:1 (default), 3:4, 4:3, 16:9, 9:16
#   --count N         Number of images to generate: 1-4 (default: 1)
#   --model TIER      Model tier: fast (default), standard, ultra
#   --output NAME     Custom output filename (without extension)
#   --no-rewrite      Disable Imagen's built-in prompt rewriter
#   --resolution WxH  Output resolution (standard/ultra only), e.g. 2048x2048
#
# Output:
#   Prints public URL(s) of generated image(s) to stdout.
#   Images saved to /var/www/cc/assets/imagen/
#
# Prerequisites:
#   - gcloud CLI with valid credentials
#   - Vertex AI API enabled on the GCP project
#   - /var/www/cc/ served by nginx (CC Pages)

set -euo pipefail

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
PROJECT="${GOOGLE_CLOUD_PROJECT:-gpu-launchpad-playground}"
REGION="us-central1"
OUTPUT_DIR="/var/www/cc/assets/imagen"
BASE_URL="https://cc.higcp.com/assets/imagen"

# ──────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────
PROMPT=""
ASPECT="1:1"
COUNT=1
MODEL_TIER="fast"
OUTPUT_NAME=""
ENHANCE="true"
RESOLUTION=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --aspect)     ASPECT="$2"; shift 2 ;;
        --count)      COUNT="$2"; shift 2 ;;
        --model)      MODEL_TIER="$2"; shift 2 ;;
        --output)     OUTPUT_NAME="$2"; shift 2 ;;
        --no-rewrite) ENHANCE="false"; shift ;;
        --resolution) RESOLUTION="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/{ s/^# //; s/^#//; p }' "$0"
            exit 0
            ;;
        -*)  echo "Error: Unknown option: $1" >&2; exit 1 ;;
        *)   PROMPT="$1"; shift ;;
    esac
done

if [[ -z "$PROMPT" ]]; then
    echo "Error: No prompt provided." >&2
    echo "Usage: imagen-generate.sh \"prompt\" [--aspect 16:9] [--count N] [--model fast|standard|ultra]" >&2
    exit 1
fi

# ──────────────────────────────────────────────
# Resolve model ID
# ──────────────────────────────────────────────
case "$MODEL_TIER" in
    fast)     MODEL_ID="imagen-4.0-fast-generate-001" ;;
    standard) MODEL_ID="imagen-4.0-generate-001" ;;
    ultra)    MODEL_ID="imagen-4.0-ultra-generate-001" ;;
    *)
        echo "Error: Invalid model tier '$MODEL_TIER'. Use: fast, standard, ultra" >&2
        exit 1
        ;;
esac

# ──────────────────────────────────────────────
# Build JSON request body (via Python for safe escaping)
# ──────────────────────────────────────────────
REQUEST_BODY=$(python3 -c "
import json, sys

prompt = sys.argv[1]
params = {
    'sampleCount': int(sys.argv[2]),
    'aspectRatio': sys.argv[3],
    'enhancePrompt': sys.argv[4] == 'true',
}

resolution = sys.argv[5]
if resolution:
    w = resolution.split('x')[0]
    params['outputOptions'] = {'mimeType': 'image/png'}
    params['sampleImageSize'] = w

print(json.dumps({
    'instances': [{'prompt': prompt}],
    'parameters': params,
}))
" "$PROMPT" "$COUNT" "$ASPECT" "$ENHANCE" "$RESOLUTION")

# ──────────────────────────────────────────────
# Call Vertex AI predict API
# ──────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

ENDPOINT="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${REGION}/publishers/google/models/${MODEL_ID}:predict"
TOKEN=$(gcloud auth print-access-token)

RESPONSE=$(curl -s -X POST "$ENDPOINT" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d "$REQUEST_BODY")

# ──────────────────────────────────────────────
# Check for API errors
# ──────────────────────────────────────────────
ERROR=$(python3 -c "
import json, sys
try:
    r = json.loads(sys.stdin.read())
    if 'error' in r:
        print(r['error'].get('message', str(r['error'])))
    elif 'predictions' not in r:
        print('No predictions in response: ' + json.dumps(r)[:300])
except Exception as e:
    print(f'Failed to parse response: {e}')
" <<< "$RESPONSE" 2>/dev/null || true)

if [[ -n "$ERROR" ]]; then
    echo "Error: $ERROR" >&2
    exit 1
fi

# ──────────────────────────────────────────────
# Save images and print URLs
# ──────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
if [[ -z "$OUTPUT_NAME" ]]; then
    # Sanitize prompt: keep ASCII alphanumeric + hyphen, truncate to 30 chars
    OUTPUT_NAME=$(echo "$PROMPT" | tr -cs 'a-zA-Z0-9' '-' | head -c 30 | sed 's/-$//')
    OUTPUT_NAME="${OUTPUT_NAME}-${TIMESTAMP}"
fi

python3 -c "
import json, sys, base64, os

response = json.loads(sys.stdin.read())
predictions = response.get('predictions', [])
output_dir = sys.argv[1]
output_name = sys.argv[2]
base_url = sys.argv[3]

for i, pred in enumerate(predictions):
    img_bytes = base64.b64decode(pred['bytesBase64Encoded'])
    suffix = f'-{i+1}' if len(predictions) > 1 else ''
    filename = f'{output_name}{suffix}.png'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'wb') as f:
        f.write(img_bytes)
    print(f'{base_url}/{filename}')
" "$OUTPUT_DIR" "$OUTPUT_NAME" "$BASE_URL" <<< "$RESPONSE"
