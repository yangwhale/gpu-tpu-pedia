# vLLM Troubleshooting Guide

## Library Loading Errors

### libcudnn.so.9 not found

**Symptom:**
```
ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory
```

**Cause:** cuDNN library not installed or not in LD_LIBRARY_PATH.

**Solution:**
```bash
# 1. Install cuDNN
pip install nvidia-cudnn-cu12==9.16.0.29 --force-reinstall --no-deps

# 2. Set LD_LIBRARY_PATH
source /vllm-workspace/vllm-env.sh
# Or run scripts/setup_env.sh
```

### libcusparseLt.so.0 not found

**Symptom:**
```
ImportError: libcusparseLt.so.0: cannot open shared object file: No such file or directory
```

**Cause:** cuSPARSELt library not installed or not in LD_LIBRARY_PATH.

**Solution:**
```bash
# 1. Install cuSPARSELt
pip install nvidia-cusparselt-cu12 --force-reinstall --no-deps

# 2. Set LD_LIBRARY_PATH
source /vllm-workspace/vllm-env.sh
```

### libnccl.so errors

**Symptom:**
```
ImportError: libnccl.so.2: cannot open shared object file
```
or NCCL timeout errors.

**Solution:**
```bash
pip install nvidia-nccl-cu12==2.28.3 --force-reinstall --no-deps
```

## FlashInfer Errors

### Version Mismatch

**Symptom:**
```
ModuleNotFoundError: No module named 'flashinfer.jit.cubin_loader'
```
or
```
FLASHINFER_CUBIN_DIR: _get_cubin_dir() failed
```

**Cause:** `flashinfer-python` and `flashinfer-cubin` versions don't match.

**Diagnosis:**
```bash
pip list | grep flashinfer
# flashinfer-cubin    0.6.2
# flashinfer-python   0.5.3  <- Version mismatch!
```

**Solution:**
```bash
# Install matching versions
pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3 --force-reinstall
```

### FlashInfer Import Error

**Symptom:**
```
ERROR: WorkerProc failed to start.
File "vllm/v1/attention/backends/flashinfer.py", line 10, in <module>
    from flashinfer import (
```

**Cause:** FlashInfer installation corrupted or version mismatch.

**Solution:**
```bash
# Reinstall both packages
pip uninstall flashinfer-python flashinfer-cubin -y
pip install flashinfer-python==0.5.3 flashinfer-cubin==0.5.3
```

## Model Loading Errors

### Tensor Parallelism Error

**Symptom:**
```
AssertionError: assert self.total_num_heads % tp_size == 0
```

**Cause:** TP size doesn't divide the model's attention head count evenly.

**Solution:**
```bash
# Find the model's attention head count
python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('MODEL_NAME')
print(f'Attention heads: {c.num_attention_heads}')
"
```

Then choose a TP value that divides the head count. Example for Qwen2.5-7B (28 heads):
- Valid: 1, 2, 4, 7, 14
- Invalid: 3, 5, 6, 8

### Out of Memory (OOM)

**Symptom:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**
1. **Increase TP size** - Distribute model across more GPUs
2. **Use quantization** - Add `--quantization awq` or `--quantization gptq`
3. **Reduce max_model_len** - Add `--max-model-len 4096`
4. **Enable CPU offload** - Add `--cpu-offload-gb 10`

### Model Download Stuck

**Symptom:** Model download hangs or is very slow.

**Solutions:**
1. **Use LSSD for caching:**
```bash
export HF_HOME=/lssd/huggingface
```

2. **Pre-download the model:**
```bash
huggingface-cli download MODEL_NAME
```

## Server Startup Issues

### Port Already in Use

**Symptom:**
```
OSError: [Errno 98] Address already in use
```

**Solution:**
```bash
# Find and kill the process using the port
lsof -i :8000
kill -9 <PID>

# Or use a different port
vllm serve MODEL --port 8001
```

### Server Starts but No Response

**Symptom:** Server logs show startup but curl returns nothing.

**Diagnosis:**
```bash
# Check if server is listening
ss -tlnp | grep 8000

# Check server health
curl -v http://localhost:8000/health
```

**Solution:** Wait for model loading to complete. Large models can take several minutes.

### Zombie Processes

**Symptom:** vLLM processes show as `<defunct>` or `Z` state.

**Solution:**
```bash
# Kill all vLLM processes
pkill -9 -f "vllm"

# Wait and restart
sleep 5
vllm serve MODEL ...
```

## LSSD Issues

### LSSD Not Mounted

**Symptom:** `/lssd` doesn't exist or isn't mounted.

**Solution:** Use the lssd-mounter skill:
```bash
/lssd-mounter
```

### HF_HOME Not Set

**Symptom:** Models downloading to home directory instead of LSSD.

**Solution:**
```bash
# Set HF_HOME
export HF_HOME=/lssd/huggingface

# Add to bashrc for persistence
echo 'export HF_HOME=/lssd/huggingface' >> ~/.bashrc
```

## DeepEP Issues (MoE Models)

### DeepEP Not Installed

**Symptom:** MoE model (DeepSeek-V3, DeepSeek-R1) fails to load.

**Solution:** Use the deepep-installer skill:
```bash
/deepep-installer
```

### DeepEP Import Error

**Symptom:**
```
ModuleNotFoundError: No module named 'deep_ep'
```

**Solution:**
```bash
# Check installation paths
ls /opt/deepep/

# Set PYTHONPATH if needed
export PYTHONPATH=/opt/deepep/DeepEP:$PYTHONPATH
```

## Diagnostic Commands

### Quick Health Check

```bash
# 1. Check PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

# 2. Check vLLM import
python3 -c "import vllm; print(vllm.__version__)"

# 3. Check FlashInfer versions
pip list | grep flashinfer

# 4. Check NVIDIA libs
pip list | grep nvidia

# 5. Check LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH | tr ':' '\n' | grep nvidia | wc -l
```

### Full Diagnostic

```bash
python3 scripts/diagnose.py
```

### Check GPU Usage

```bash
nvidia-smi
watch -n 1 nvidia-smi
```

### Check Server Logs

```bash
# If running in background
tail -f /tmp/vllm.log

# If running in foreground with nohup
tail -f nohup.out
```
