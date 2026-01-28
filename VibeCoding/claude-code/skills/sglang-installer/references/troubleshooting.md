# SGLang Troubleshooting Guide

This document provides comprehensive troubleshooting for common SGLang installation and runtime issues.

## Quick Diagnostic

Run the diagnostic script first:

```bash
python3 scripts/diagnose.py
```

This will check all common issues and suggest fixes.

## Import Errors

### libcudnn.so.9: cannot open shared object file

**Symptom:**
```python
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

**Cause:** cuDNN library not installed or not in LD_LIBRARY_PATH.

**Fix:**
```bash
# Install cuDNN
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps

# Set LD_LIBRARY_PATH
source scripts/setup_env.sh
```

### libcusparseLt.so.0: cannot open shared object file

**Symptom:**
```python
ImportError: libcusparseLt.so.0: cannot open shared object file: No such file or directory
```

**Cause:** cuSPARSELt library not installed or not in LD_LIBRARY_PATH.

**Fix:**
```bash
pip install nvidia-cusparselt-cu12
source scripts/setup_env.sh
```

### libnccl.so.2: cannot open shared object file

**Symptom:**
```python
ImportError: libnccl.so.2: cannot open shared object file
```

**Fix:**
```bash
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
source scripts/setup_env.sh
```

### No module named 'sglang'

**Fix:**
```bash
cd /sgl-workspace/sglang
pip install -e "python[blackwell]" --extra-index-url https://download.pytorch.org/whl/cu129
```

### No module named 'sgl_kernel'

**Fix:**
```bash
pip install sgl-kernel==0.3.21
```

## Server Startup Errors

### AssertionError: assert self.total_num_heads % tp_size == 0

**Symptom:**
```
AssertionError: assert self.total_num_heads % tp_size == 0
```

**Cause:** Tensor parallelism size (--tp) is not compatible with the model's attention head count.

**Diagnosis:**
```bash
# Check model's attention head count
python3 -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('MODEL_NAME'); print(f'Heads: {c.num_attention_heads}')"
```

**Fix:** Use a TP value that divides the head count evenly. See `references/version_matrix.md` for common models.

Example for Qwen2.5-7B (28 heads):
```bash
# Valid: tp=1, 2, 4, 7, 14, 28
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --tp 4  # OK
python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct --tp 8  # FAIL
```

### NCCL timeout or initialization failure

**Symptom:**
```
NCCL error: unhandled system error
# or
NCCL timeout after X seconds
```

**Possible causes:**
1. NCCL version mismatch
2. Network configuration issues
3. GPU memory exhaustion

**Fix:**
```bash
# Update NCCL
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps

# Increase timeout (if network is slow)
export NCCL_TIMEOUT=1800
```

### CUDA out of memory

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Fix options:**
1. Reduce batch size
2. Increase tensor parallelism (--tp)
3. Reduce context length (--context-length)
4. Use memory fraction (--mem-fraction-static 0.8)

```bash
python3 -m sglang.launch_server \
    --model-path MODEL \
    --tp 8 \
    --mem-fraction-static 0.8 \
    --context-length 8192
```

### Server starts but port not listening

**Diagnosis:**
```bash
# Check if port is already in use
ss -tlnp | grep 30000

# Check server process
ps aux | grep sglang

# Check server logs
tail -100 /tmp/sglang_server.log
```

**Fix:** Kill existing process or use different port:
```bash
pkill -f "sglang.launch_server"
python3 -m sglang.launch_server --port 30001 ...
```

## Environment Issues

### CUDA_HOME not set

**Symptom:**
```
FileNotFoundError: nvcc not found
# or
CUDA_HOME environment variable is not set
```

**Fix:**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### nvcc not in PATH

**Diagnosis:**
```bash
which nvcc
# Should return: /usr/local/cuda/bin/nvcc
```

**Fix:**
```bash
export PATH=/usr/local/cuda/bin:$PATH
```

### LD_LIBRARY_PATH not configured

**Symptom:** Various library not found errors even after installing.

**Fix:** Run the setup script:
```bash
source scripts/setup_env.sh
```

Or add to your shell profile (~/.bashrc or ~/.zshrc):
```bash
# Add nvidia pip package libs to LD_LIBRARY_PATH
for d in /usr/local/lib/python3.12/dist-packages/nvidia/*/lib ~/.local/lib/python3.12/site-packages/nvidia/*/lib; do
    [ -d "$d" ] && export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done
```

## Model Loading Issues

### Model download fails

**Symptom:**
```
OSError: We couldn't connect to huggingface.co
# or
401 Unauthorized
```

**Fix:**
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HF_TOKEN=your_token_here
```

### Model weights not found in cache

**Symptom:**
```
Local HF snapshot has no files matching ['*.safetensors', '*.bin']
```

**Cause:** Model cache is incomplete.

**Fix:**
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface/hub/models--MODEL_NAME
python3 -m sglang.launch_server --model-path MODEL_NAME ...
```

### Tokenizer errors

**Symptom:**
```
ValueError: Tokenizer not found
# or
AutoTokenizer.from_pretrained() failed
```

**Fix:**
```bash
# Add trust-remote-code for custom models
python3 -m sglang.launch_server --trust-remote-code ...
```

## Performance Issues

### Slow first request (JIT compilation)

**Symptom:** First request takes several minutes.

**Cause:** CUDA graph capture and JIT compilation.

**Mitigation:**
1. This is normal behavior - subsequent requests will be fast
2. Consider using `--skip-server-warmup=false` (default)
3. Download flashinfer cubin cache: `python3 -m flashinfer --download-cubin`

### Low throughput

**Diagnosis:**
```bash
# Check GPU utilization
nvidia-smi -l 1

# Check server metrics
curl http://localhost:30000/metrics
```

**Optimizations:**
1. Increase batch size with `--max-running-requests`
2. Tune `--mem-fraction-static`
3. Enable `--enable-mixed-chunk`

## Debugging Commands

```bash
# Check GPU status
nvidia-smi

# Check CUDA version
nvcc --version

# Check PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# Check SGLang version
python3 -c "import sglang; print(sglang.__version__)"

# Check sgl_kernel
python3 -c "import sgl_kernel; print('OK')"

# Check all nvidia pip packages
pip list | grep nvidia

# Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep nvidia

# Test server health
curl http://localhost:30000/health

# Test generation
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hi"}]}'
```

## Getting Help

If issues persist:

1. Run diagnostic: `python3 scripts/diagnose.py`
2. Check SGLang GitHub issues: https://github.com/sgl-project/sglang/issues
3. Check SGLang Discord for community help
4. Include the following when reporting:
   - Output of `python3 scripts/diagnose.py`
   - CUDA version (`nvcc --version`)
   - GPU model (`nvidia-smi -L`)
   - Full error traceback
