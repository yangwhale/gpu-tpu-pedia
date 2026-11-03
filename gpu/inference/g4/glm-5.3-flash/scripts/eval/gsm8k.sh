#!/bin/bash
# GSM8K accuracy against any running OpenAI-compatible endpoint (single node :8000 or PD proxy :8000).
# Same settings for every precision so FP8 vs NVFP4 differences come from the weights only.
# Usage: TAG=fp8 TOKENIZER=/data/GLM-5.3-Flash bash gsm8k.sh
set -euo pipefail
TAG=${TAG:?fp8 or nvfp4}; TOKENIZER=${TOKENIZER:-/data/GLM-5.3-Flash}
BASE=${BASE:-http://127.0.0.1:8000}; OUT=${OUT:-$HOME/g4glm/eval-$TAG}; mkdir -p $OUT
export PATH=$HOME/.local/bin:$PATH
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x $HOME/evalenv/bin/lm_eval ] || { uv venv -q $HOME/evalenv --python 3.12 && uv pip install -q --python $HOME/evalenv/bin/python "lm_eval[api]==0.4.13" transformers; }
# 5-shot GSM8K, full test set (1,319 questions), greedy decoding, raw completions API
$HOME/evalenv/bin/lm_eval --model local-completions \
  --model_args model=glm,base_url=$BASE/v1/completions,num_concurrent=${CONC:-64},max_retries=3,tokenized_requests=False,tokenizer=$TOKENIZER \
  --tasks gsm8k --num_fewshot 5 --batch_size 1 \
  --gen_kwargs temperature=0 \
  --seed 1234 --output_path $OUT --log_samples 2>&1 | tee $OUT/gsm8k.log
