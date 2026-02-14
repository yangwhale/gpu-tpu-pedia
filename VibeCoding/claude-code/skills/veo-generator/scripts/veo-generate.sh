#!/usr/bin/env bash
# Veo 3.1 — Text-to-video generation via Vertex AI REST API
#
# Usage:
#   veo-generate.sh "prompt" [options]
#
# Options:
#   --aspect RATIO      Aspect ratio: 16:9 (default), 9:16
#   --count N           Number of videos to generate: 1-4 (default: 1)
#   --model TIER        Model tier: fast (default), standard
#   --duration N        Video duration in seconds: 4, 6, or 8 (default: 8)
#   --resolution RES    Video resolution: 720p (default), 1080p
#   --negative TEXT     Negative prompt (content to avoid)
#   --output NAME       Custom output filename (without extension)
#   --no-rewrite        Disable built-in prompt rewriter
#   --poll-interval N   Polling interval in seconds (default: 10)
#   --timeout N         Max wait time in seconds (default: 300)
#
# Output:
#   Prints public URL(s) of generated video(s) to stdout.
#   Videos saved to /var/www/cc/assets/veo/
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
OUTPUT_DIR="/var/www/cc/assets/veo"
BASE_URL="https://cc.higcp.com/assets/veo"

# ──────────────────────────────────────────────
# Parse arguments
# ──────────────────────────────────────────────
PROMPT=""
ASPECT="16:9"
COUNT=1
MODEL_TIER="fast"
DURATION=8
RESOLUTION="720p"
NEGATIVE=""
OUTPUT_NAME=""
ENHANCE="true"
POLL_INTERVAL=10
TIMEOUT=300

while [[ $# -gt 0 ]]; do
    case "$1" in
        --aspect)        ASPECT="$2"; shift 2 ;;
        --count)         COUNT="$2"; shift 2 ;;
        --model)         MODEL_TIER="$2"; shift 2 ;;
        --duration)      DURATION="$2"; shift 2 ;;
        --resolution)    RESOLUTION="$2"; shift 2 ;;
        --negative)      NEGATIVE="$2"; shift 2 ;;
        --output)        OUTPUT_NAME="$2"; shift 2 ;;
        --no-rewrite)    ENHANCE="false"; shift ;;
        --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
        --timeout)       TIMEOUT="$2"; shift 2 ;;
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
    echo "Usage: veo-generate.sh \"prompt\" [--aspect 9:16] [--duration 8] [--model fast|standard]" >&2
    exit 1
fi

# ──────────────────────────────────────────────
# Resolve model ID
# ──────────────────────────────────────────────
case "$MODEL_TIER" in
    fast)     MODEL_ID="veo-3.1-fast-generate-preview" ;;
    standard) MODEL_ID="veo-3.1-generate-preview" ;;
    *)
        echo "Error: Invalid model tier '$MODEL_TIER'. Use: fast, standard" >&2
        exit 1
        ;;
esac

# ──────────────────────────────────────────────
# Build JSON request body
# ──────────────────────────────────────────────
REQUEST_BODY=$(python3 -c "
import json, sys

prompt = sys.argv[1]
params = {
    'sampleCount': int(sys.argv[2]),
    'aspectRatio': sys.argv[3],
    'durationSeconds': int(sys.argv[4]),
    'resolution': sys.argv[5],
    'enhancePrompt': sys.argv[6] == 'true',
    'personGeneration': 'allow_adult',
}

negative = sys.argv[7]
if negative:
    params['negativePrompt'] = negative

print(json.dumps({
    'instances': [{'prompt': prompt}],
    'parameters': params,
}))
" "$PROMPT" "$COUNT" "$ASPECT" "$DURATION" "$RESOLUTION" "$ENHANCE" "$NEGATIVE")

# ──────────────────────────────────────────────
# Submit video generation (long-running operation)
# ──────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"

ENDPOINT="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${REGION}/publishers/google/models/${MODEL_ID}:predictLongRunning"
TOKEN=$(gcloud auth print-access-token)

echo "Submitting video generation request..." >&2
SUBMIT_RESPONSE=$(curl -s -X POST "$ENDPOINT" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d "$REQUEST_BODY")

# Extract operation name
OPERATION_NAME=$(python3 -c "
import json, sys
try:
    r = json.loads(sys.stdin.read())
    if 'error' in r:
        print('ERROR:' + r['error'].get('message', str(r['error'])), file=sys.stderr)
        sys.exit(1)
    name = r.get('name', '')
    if not name:
        print('ERROR:No operation name in response: ' + json.dumps(r)[:300], file=sys.stderr)
        sys.exit(1)
    print(name)
except Exception as e:
    print(f'ERROR:Failed to parse response: {e}', file=sys.stderr)
    sys.exit(1)
" <<< "$SUBMIT_RESPONSE")

echo "Operation: $OPERATION_NAME" >&2
echo "Polling for completion (interval: ${POLL_INTERVAL}s, timeout: ${TIMEOUT}s)..." >&2

# ──────────────────────────────────────────────
# Poll for completion
# ──────────────────────────────────────────────
FETCH_ENDPOINT="https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT}/locations/${REGION}/publishers/google/models/${MODEL_ID}:fetchPredictOperation"
ELAPSED=0

while true; do
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "Error: Timeout after ${TIMEOUT}s waiting for video generation." >&2
        echo "Operation: $OPERATION_NAME" >&2
        exit 1
    fi

    sleep "$POLL_INTERVAL"
    ELAPSED=$((ELAPSED + POLL_INTERVAL))

    # Refresh token if needed (long operations may outlive token)
    TOKEN=$(gcloud auth print-access-token)

    POLL_RESPONSE=$(curl -s -X POST "$FETCH_ENDPOINT" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json; charset=utf-8" \
        -d "{\"operationName\": \"$OPERATION_NAME\"}")

    STATUS=$(python3 -c "
import json, sys
try:
    r = json.loads(sys.stdin.read())
    done = r.get('done', False)
    if done:
        if 'error' in r:
            print('ERROR:' + str(r['error']))
        else:
            print('DONE')
    else:
        print('PENDING')
except Exception as e:
    print(f'ERROR:{e}')
" <<< "$POLL_RESPONSE")

    case "$STATUS" in
        DONE)
            echo "Video generation complete! (${ELAPSED}s)" >&2
            break
            ;;
        PENDING)
            echo "  Still generating... (${ELAPSED}s)" >&2
            ;;
        ERROR:*)
            echo "Error: ${STATUS#ERROR:}" >&2
            exit 1
            ;;
    esac
done

# ──────────────────────────────────────────────
# Save videos and print URLs
# ──────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
if [[ -z "$OUTPUT_NAME" ]]; then
    OUTPUT_NAME=$(echo "$PROMPT" | tr -cs 'a-zA-Z0-9' '-' | head -c 30 | sed 's/-$//')
    OUTPUT_NAME="${OUTPUT_NAME}-${TIMESTAMP}"
fi

python3 -c "
import json, sys, base64, os, subprocess

response = json.loads(sys.stdin.read())
output_dir = sys.argv[1]
output_name = sys.argv[2]
base_url = sys.argv[3]

# Traverse nested response structure
resp = response.get('response', response)
# Possible keys: 'results', 'videos', 'predictions'
results = resp.get('results', [])
videos = resp.get('videos', [])
predictions = resp.get('predictions', [])

saved = 0

# Format A: results[].content — base64 encoded video bytes
if results:
    for i, res in enumerate(results):
        content = res.get('content') or res.get('bytesBase64Encoded', '')
        if not content:
            print(f'Warning: result {i} has no video content, keys: {list(res.keys())}', file=sys.stderr)
            continue
        video_bytes = base64.b64decode(content)
        suffix = f'-{i+1}' if len(results) > 1 else ''
        filename = f'{output_name}{suffix}.mp4'
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(video_bytes)
        print(f'{base_url}/{filename}')
        saved += 1

# Format B: predictions[].bytesBase64Encoded — same as Imagen style
elif predictions:
    for i, pred in enumerate(predictions):
        content = pred.get('bytesBase64Encoded', '')
        if not content:
            continue
        video_bytes = base64.b64decode(content)
        suffix = f'-{i+1}' if len(predictions) > 1 else ''
        filename = f'{output_name}{suffix}.mp4'
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(video_bytes)
        print(f'{base_url}/{filename}')
        saved += 1

# Format C: videos[] — may contain bytesBase64Encoded or gcsUri
elif videos:
    for i, vid in enumerate(videos):
        suffix = f'-{i+1}' if len(videos) > 1 else ''
        filename = f'{output_name}{suffix}.mp4'
        filepath = os.path.join(output_dir, filename)

        # Try base64 first (no storageUri was specified)
        b64 = vid.get('bytesBase64Encoded', '')
        if b64:
            video_bytes = base64.b64decode(b64)
            with open(filepath, 'wb') as f:
                f.write(video_bytes)
            print(f'{base_url}/{filename}')
            saved += 1
            continue

        # Try GCS URI
        gcs_uri = vid.get('gcsUri') or vid.get('uri', '')
        if gcs_uri:
            subprocess.run(['gsutil', 'cp', gcs_uri, filepath], capture_output=True)
            print(f'{base_url}/{filename}')
            saved += 1
            continue

        print(f'Warning: video {i} has no content, keys: {list(vid.keys())}', file=sys.stderr)

if saved == 0:
    filtered = resp.get('raiMediaFilteredCount', 0)
    if filtered:
        print(f'Warning: {filtered} video(s) filtered by safety checks.', file=sys.stderr)
    else:
        # Debug: dump response structure
        print('Error: No videos found in response.', file=sys.stderr)
        print('Response keys: ' + str(list(resp.keys())), file=sys.stderr)
        for k in ['results', 'videos', 'predictions']:
            items = resp.get(k, [])
            if items:
                print(f'  {k}[0] keys: {list(items[0].keys()) if isinstance(items[0], dict) else type(items[0])}', file=sys.stderr)
    sys.exit(1)
" "$OUTPUT_DIR" "$OUTPUT_NAME" "$BASE_URL" <<< "$POLL_RESPONSE"
