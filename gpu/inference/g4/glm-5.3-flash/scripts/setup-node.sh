#!/bin/bash
# Prepare one G4 node: Docker + NVIDIA toolkit, pull vLLM image, copy weights. No server start.
# Usage: [MODEL_URI=gs://<bucket>/models/zai-org/GLM-5.3-Flash] [IMAGE=vllm/vllm-openai@sha256:...] bash setup-node.sh
set -x
IMAGE=${IMAGE:-vllm/vllm-openai@sha256:88cc2118dab845fec1ff759c2fc8afa517cf8a668162ed8683b2f1b781372f57}
W=$HOME/g4glm; mkdir -p $W
echo "=== STEP install-docker $(date -u +%FT%TZ)"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit || true   # DLVM ships a held 1.19.x; a version-mismatch error here is harmless
  sudo apt-get install -y iperf3
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi
sudo docker pull $IMAGE > $W/pull.log 2>&1 &
echo "=== STEP copy-weights $(date -u +%FT%TZ)"
sudo mkdir -p /data && sudo chown $USER /data
if [ -n "$MODEL_URI" ]; then
  [ -f /data/GLM-5.3-Flash/model.safetensors.index.json ] || gcloud storage cp -r "$MODEL_URI" /data/ || { echo "DOWNLOAD FAILED"; exit 1; }
else
  echo "MODEL_URI not set: skip weights (use download-hf.sh)"
fi
wait
echo "=== STEP SETUP_DONE $(date -u +%FT%TZ)"
