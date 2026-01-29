# Ray Cluster Troubleshooting Guide

## Common Issues and Solutions

### 1. Connection Timeout

**Symptoms:**
```
ConnectionError: Could not connect to Ray cluster at 10.8.0.79:6379
```

**Solutions:**
```bash
# Check if head node is running
ray status

# Check firewall rules
sudo ufw status
sudo ufw allow 6379/tcp
sudo ufw allow 8265/tcp

# Check if port is listening
ss -tlnp | grep 6379

# Test network connectivity
ping 10.8.0.79
nc -zv 10.8.0.79 6379
```

### 2. GPU Not Detected

**Symptoms:**
```
ray.cluster_resources() shows GPU: 0
```

**Solutions:**
```bash
# Verify CUDA installation
nvidia-smi
nvcc --version

# Check if Ray was started with --num-gpus
ray stop --force
ray start --head --num-gpus=8

# Verify CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES  # If set incorrectly
```

### 3. Worker Node Won't Join

**Symptoms:**
```
Worker node starts but doesn't appear in ray.nodes()
```

**Solutions:**
```bash
# Verify head IP is correct
ping ${HEAD_IP}

# Check if head node is running
ssh ${HEAD_IP} "ray status"

# Check worker logs
tail -f /tmp/ray/session_latest/logs/raylet.out

# Restart with explicit IP
ray stop --force
ray start --address="${HEAD_IP}:6379" --node-ip-address=$(hostname -I | awk '{print $1}')
```

### 4. Tasks Not Distributed Across Nodes

**Symptoms:**
All tasks execute on a single node despite multiple nodes being available.

**Solutions:**
```python
# Check node status
import ray
ray.init(address="auto")
for node in ray.nodes():
    print(f"Node: {node['NodeManagerAddress']}, Alive: {node['Alive']}")

# Force task placement
@ray.remote(resources={"node:10.8.0.80": 1})
def run_on_specific_node():
    pass
```

### 5. Out of Memory Errors

**Symptoms:**
```
RayOutOfMemoryError: More than 95% of the memory is used
```

**Solutions:**
```bash
# Start with memory limits
ray start --head --object-store-memory=50000000000  # 50GB

# Monitor memory usage
ray memory

# Force garbage collection in code
import gc
gc.collect()
```

### 6. NCCL Errors in Multi-GPU Tasks

**Symptoms:**
```
NCCL error: unhandled system error
```

**Solutions:**
```bash
# Set NCCL environment variables
export NCCL_SOCKET_IFNAME=enp0s19
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # If not using InfiniBand

# For NVLink-enabled systems
export NCCL_MNNVL_ENABLE=1
export NCCL_CUMEM_ENABLE=1
```

### 7. Dashboard Not Accessible

**Symptoms:**
- Cannot access http://HEAD_IP:8265
- Dashboard log shows "http server disabled"
- Log shows "Module cannot be loaded... No module named 'aiohttp_cors'"

**Solutions:**

```bash
# 1. Check if dashboard dependencies are installed
pip3 show aiohttp_cors opentelemetry-sdk

# 2. If missing, install ray[default] for full dashboard support
pip3 install --user --break-system-packages 'ray[default]'

# 3. Restart Ray cluster after installing dependencies
ray stop --force
ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265

# 4. Verify dashboard is listening
ss -tlnp | grep 8265
curl http://localhost:8265/

# 5. Check dashboard logs for errors
tail -f /tmp/ray/session_latest/logs/dashboard.log
```

**Accessing Dashboard from remote machine (via SSH tunnel):**

```bash
# On your local machine, create SSH tunnel through IAP
gcloud compute ssh INSTANCE_NAME --zone=ZONE --tunnel-through-iap -- -L 8265:localhost:8265

# Then open http://localhost:8265 in browser
```

### 8. Ray Processes Not Cleaning Up

**Symptoms:**
Old Ray processes persist after `ray stop`

**Solutions:**
```bash
# Force stop all Ray processes
ray stop --force

# Kill remaining processes manually
pkill -9 -f raylet
pkill -9 -f plasma
pkill -9 -f gcs_server

# Clean up shared memory
rm -rf /dev/shm/plasma*
rm -rf /tmp/ray/*
```

## Diagnostic Commands

```bash
# Full cluster status
ray status

# Node information
python3 -c "import ray; ray.init(address='auto'); print(ray.nodes())"

# Resource availability
python3 -c "import ray; ray.init(address='auto'); print(ray.cluster_resources())"

# Check logs
tail -f /tmp/ray/session_latest/logs/raylet.out
tail -f /tmp/ray/session_latest/logs/gcs_server.out

# Memory usage
ray memory

# Check running tasks
ray summary tasks
```

## Environment Variables Reference

| Variable | Description | Example |
|----------|-------------|---------|
| `RAY_ADDRESS` | Default cluster address | `ray://10.8.0.79:6379` |
| `RAY_OBJECT_STORE_MEMORY` | Object store size in bytes | `50000000000` |
| `RAY_memory_monitor_refresh_ms` | Memory check interval | `250` |
| `CUDA_VISIBLE_DEVICES` | GPU visibility | `0,1,2,3,4,5,6,7` |
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL | `enp0s19` |
| `NCCL_DEBUG` | NCCL debug level | `INFO` |
