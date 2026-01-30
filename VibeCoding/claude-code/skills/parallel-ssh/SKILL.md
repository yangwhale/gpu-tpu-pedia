---
name: parallel-ssh
description: Parallel SSH operations across multiple hosts using google_compute_engine key. Use this skill when the user provides a list of host IPs and wants to execute commands in parallel across them, such as starting prefill/decode nodes for distributed inference.
license: MIT
---

# Parallel SSH

This skill enables parallel SSH operations across multiple hosts for distributed workloads like SGLang/vLLM prefill-decode disaggregation.

## When to Use This Skill

- User provides a list of host IPs (internal or external)
- User wants to run commands in parallel across multiple machines
- User needs to start distributed services (prefill nodes, decode nodes)
- User wants to collect logs from multiple machines simultaneously
- User needs to monitor GPU status across a cluster

## SSH Configuration

### Default Key File
```
~/.ssh/google_compute_engine
```

### Step 0: Ensure SSH Key Exists (IMPORTANT)
Before running any SSH commands, ALWAYS check if the key file exists. If not, generate it using gcloud:

```bash
# Check if key exists
if [ ! -f ~/.ssh/google_compute_engine ]; then
    echo "SSH key not found, generating via gcloud..."
    gcloud compute config-ssh --quiet
fi
```

This command:
1. Generates `~/.ssh/google_compute_engine` (private key) and `~/.ssh/google_compute_engine.pub` (public key)
2. Uploads the public key to the GCP project metadata
3. Configures SSH aliases for all instances in the project

### SSH Command Format
```bash
ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=accept-new <IP> "<command>"
```

The `-o StrictHostKeyChecking=accept-new` option automatically accepts new host keys (safe for first connection).

## Parallel Execution Pattern

### Step 1: Parse Host List
When user provides hosts like:
- "10.8.0.81 10.8.0.82 10.8.0.83"
- "10.8.0.81, 10.8.0.82, 10.8.0.83"
- Listed in a file

Parse into individual IPs.

### Step 2: Launch Background Tasks
For each host, launch SSH command with `run_in_background: true`:

```bash
ssh -i ~/.ssh/google_compute_engine -o StrictHostKeyChecking=accept-new 10.8.0.81 "command"
```

IMPORTANT: Launch ALL tasks in a SINGLE message with multiple Bash tool calls to achieve true parallelism.

### Step 3: Collect Output
Use `TaskOutput` to wait for completion, or `Read` to check progress:
- Output files: `/tmp/claude-*/tasks/<task_id>.output`

## Common Commands

### GPU Status Check
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

### System Status
```bash
vmstat 2 15
```

### Service Status
```bash
systemctl status <service> || pgrep -a <process>
```

### Start SGLang Prefill Node
```bash
source /opt/deepep/unified-env.sh && \
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --disaggregation-mode prefill \
    --tp-size 8 \
    --port 30000 \
    --host 0.0.0.0 \
    ...
```

### Start SGLang Decode Node
```bash
source /opt/deepep/unified-env.sh && \
python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --disaggregation-mode decode \
    --tp-size 8 \
    --port 30001 \
    --host 0.0.0.0 \
    ...
```

## Example Workflows

### Example 1: Check GPU Status Across Cluster
User: "Check GPU status on 10.8.0.81 10.8.0.82 10.8.0.83"

Action: Launch 3 parallel SSH tasks with `nvidia-smi` command, collect and summarize results.

### Example 2: Start Distributed Inference
User: "Start prefill on 10.8.0.81, decode on 10.8.0.82 and 10.8.0.83"

Action:
1. SSH to 10.8.0.81: Start prefill node
2. SSH to 10.8.0.82: Start decode node 1
3. SSH to 10.8.0.83: Start decode node 2
(All launched in parallel)

### Example 3: Kill Services Across Cluster
User: "Kill all python processes on all machines"

Action: Launch parallel `pkill -f python` commands on all hosts.

### Example 4: Collect Logs
User: "Get last 100 lines of sglang logs from all machines"

Action: Launch parallel `tail -100 /path/to/logs` commands.

## Host Key Management

If SSH fails with "Host key verification failed":

```bash
# Add host key to known_hosts
ssh-keyscan -H <IP> >> ~/.ssh/known_hosts

# Or use StrictHostKeyChecking=accept-new (recommended)
ssh -o StrictHostKeyChecking=accept-new ...
```

## Best Practices

1. **Always use background mode** for long-running commands
2. **Launch all tasks in single message** for true parallelism
3. **Collect output progressively** for long tasks
4. **Summarize results in table format** for easy comparison
5. **Handle failures gracefully** - report which hosts failed

## Output Format

When reporting results, use table format:

| Host | Status | Key Metrics |
|------|--------|-------------|
| 10.8.0.81 | OK | GPU 0-7: 95% util |
| 10.8.0.82 | OK | GPU 0-7: 80% util |
| 10.8.0.83 | FAILED | SSH timeout |

## Environment Variables for Distributed Training

Common environment variables to set on each node:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=enp0s19
export GLOO_SOCKET_IFNAME=enp0s19
export MASTER_ADDR=<prefill_node_ip>
export MASTER_PORT=29500
```

## Troubleshooting

### SSH Key Not Found
If you see "Identity file not accessible" or "No such file or directory":
```bash
# Generate SSH key using gcloud (recommended for GCE)
gcloud compute config-ssh --quiet

# This creates ~/.ssh/google_compute_engine and uploads public key to project metadata
```

### SSH Connection Refused
- Check if SSH service is running on target
- Verify network connectivity: `ping <IP>`

### Permission Denied (publickey)
1. First, check if key exists:
   ```bash
   ls -la ~/.ssh/google_compute_engine
   ```
2. If not exists, generate it:
   ```bash
   gcloud compute config-ssh --quiet
   ```
3. If exists but still fails, verify key is uploaded to project:
   ```bash
   gcloud compute project-info describe --format="value(commonInstanceMetadata.items.filter(key:ssh-keys))"
   ```

### Command Timeout
- Use `nohup` for long-running processes
- Use `screen` or `tmux` for persistent sessions
- Increase Bash timeout parameter
