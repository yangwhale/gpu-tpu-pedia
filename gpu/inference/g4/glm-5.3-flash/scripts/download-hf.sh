#!/bin/bash
# Download a HuggingFace repo to /data/<name> with an isolated uv venv (DLVM image has no pip).
# Usage: REPO=RedHatAI/GLM-5.3-Flash-NVFP4 DEST=/data/GLM-5.3-Flash-NVFP4 bash download-hf.sh
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x $HOME/hfenv/bin/python ] || { uv venv -q $HOME/hfenv && uv pip install -q --python $HOME/hfenv/bin/python "huggingface_hub[hf_xet]"; }
sudo mkdir -p /data && sudo chown $USER /data
HF_XET_HIGH_PERFORMANCE=1 $HOME/hfenv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', local_dir='$DEST', max_workers=32)"
du -sh $DEST
