#!/bin/bash
# GLM-5.3-Flash on G4 (8x RTX PRO 6000) setup + bench. Log: ~/g4glm/run.log
set -x
W=$HOME/g4glm; mkdir -p $W; cd $W
step(){ echo "=== STEP $* $(date -u +%FT%TZ)"; }
step install-docker
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi
sudo docker pull vllm/vllm-openai:nightly > $W/pull.log 2>&1 &
PULL=$!
step download-model
python3 -m pip install -q --break-system-packages -U "huggingface_hub[hf_xet]" 2>&1 | tail -2
sudo mkdir -p /data && sudo chown $USER /data
HF_XET_HIGH_PERFORMANCE=1 python3 -c "from huggingface_hub import snapshot_download as s; s('zai-org/GLM-5.3-Flash', local_dir='/data/GLM-5.3-Flash', max_workers=32)" || { echo "DOWNLOAD FAILED"; exit 1; }
du -sh /data/GLM-5.3-Flash
wait $PULL; tail -3 $W/pull.log
sudo docker run --rm --gpus all --entrypoint python3 vllm/vllm-openai:nightly -c "import vllm;print('VLLM_VERSION',vllm.__version__)"
step start-server
sudo docker rm -f glm >/dev/null 2>&1
sudo docker run -d --name glm --gpus all --shm-size=64g --ipc=host --network host \
  -v /data/GLM-5.3-Flash:/model vllm/vllm-openai:nightly \
  --model /model --served-model-name glm --tensor-parallel-size 8 \
  --max-model-len 131072 --max-num-seqs 256 --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 --tool-call-parser glm47 --reasoning-parser glm45 --port 8000
for i in $(seq 1 180); do
  curl -sf localhost:8000/v1/models >/dev/null && break
  sudo docker ps -q -f name=glm | grep -q . || { echo "SERVER DIED"; sudo docker logs glm 2>&1 | tail -80 > $W/server-fail.log; exit 2; }
  sleep 20
done
sudo docker logs glm > $W/server.log 2>&1
curl -sf localhost:8000/v1/models >/dev/null || { echo "SERVER TIMEOUT"; exit 3; }
step smoke
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"glm","messages":[{"role":"user","content":"用一句话介绍香港。"}],"max_tokens":200}' | tee $W/smoke.json
step bench
for cfg in "1024 512 1" "1024 512 4" "1024 512 16" "1024 512 64" "8192 512 1" "8192 512 16" "32768 256 1"; do
  set -- $cfg
  n=$(( $3 * 4 )); [ $n -lt 8 ] && n=8
  sudo docker exec glm vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --model glm --tokenizer /model \
    --dataset-name random --random-input-len $1 --random-output-len $2 --max-concurrency $3 --num-prompts $n \
    --ignore-eos --percentile-metrics ttft,tpot,itl,e2el 2>&1 | tail -40 > $W/bench-in$1-out$2-c$3.txt
  grep -E "Output token throughput|Mean TTFT|Mean TPOT|Successful" $W/bench-in$1-out$2-c$3.txt
done
sudo docker logs glm > $W/server.log 2>&1
step ALL_DONE
