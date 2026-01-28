#!/bin/bash
# DeepEP Intranode Test Script
# Tests DeepEP on a single node with multiple GPUs

set -e

# Configuration
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export NVSHMEM_HOME=${NVSHMEM_HOME:-/opt/deepep/nvshmem}
export GDRCOPY_HOME=${GDRCOPY_HOME:-/opt/deepep/gdrcopy}
export LD_LIBRARY_PATH=${NVSHMEM_HOME}/lib:${GDRCOPY_HOME}/lib:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check prerequisites
log_info "Checking prerequisites..."

# Check GPU count
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$GPU_COUNT" -lt 2 ]; then
    log_error "Need at least 2 GPUs for intranode test, found $GPU_COUNT"
    exit 1
fi
log_info "Found $GPU_COUNT GPUs"

# Check DeepEP
python3 -c "import deep_ep" 2>/dev/null || {
    log_error "DeepEP not installed. Run install-deepep.sh first"
    exit 1
}
log_info "DeepEP module OK"

# Check gdrdrv
if ! lsmod | grep -q gdrdrv; then
    log_warn "gdrdrv module not loaded, attempting to load..."
    sudo modprobe gdrdrv || log_warn "Failed to load gdrdrv"
fi

# Number of GPUs to use (default: all available, max 8)
NUM_GPUS=${1:-$GPU_COUNT}
if [ "$NUM_GPUS" -gt 8 ]; then
    NUM_GPUS=8
fi
log_info "Testing with $NUM_GPUS GPUs"

# Create test script
TEST_SCRIPT=$(mktemp /tmp/deepep_test_XXXXXX.py)
cat > "$TEST_SCRIPT" << 'PYTHON_EOF'
import argparse
import os
import sys
import time
import torch
import torch.distributed as dist

# Add DeepEP to path if needed
try:
    import deep_ep
except ImportError:
    sys.path.insert(0, '/tmp/deepep_build')
    import deep_ep

def init_dist(local_rank: int, num_local_ranks: int):
    """Initialize distributed process group"""
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')
    port = int(os.getenv('MASTER_PORT', '29500'))
    
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{ip}:{port}',
        world_size=num_local_ranks,
        rank=local_rank,
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(local_rank)
    
    return local_rank, num_local_ranks, dist.new_group(list(range(num_local_ranks)))


def test_buffer_creation(rank, num_ranks, group):
    """Test basic DeepEP buffer creation"""
    print(f"[Rank {rank}] Testing buffer creation...", flush=True)
    
    try:
        buffer = deep_ep.Buffer(
            group,
            int(512 * 1024 * 1024),  # 512MB buffer
            0,  # No RDMA bytes for intranode
            low_latency_mode=False,
            num_qps_per_rank=1,
            explicitly_destroy=True,
            allow_mnnvl=False,
            use_fabric=False
        )
        print(f"[Rank {rank}] Buffer created successfully!", flush=True)
        return buffer
    except Exception as e:
        print(f"[Rank {rank}] Buffer creation failed: {e}", flush=True)
        raise


def test_dispatch_layout(rank, num_ranks, buffer, num_tokens=1024, num_experts=256, num_topk=8):
    """Test dispatch layout computation"""
    print(f"[Rank {rank}] Testing dispatch layout...", flush=True)
    
    # Create random top-k indices
    torch.manual_seed(42 + rank)
    topk_idx = torch.randint(0, num_experts, (num_tokens, num_topk), 
                             dtype=deep_ep.topk_idx_t, device='cuda')
    
    # Get dispatch layout
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    
    print(f"[Rank {rank}] Layout: tokens_per_rank shape={num_tokens_per_rank.shape}, "
          f"tokens_per_expert shape={num_tokens_per_expert.shape}", flush=True)
    
    return True


def test_dispatch_combine(rank, num_ranks, buffer, group, num_tokens=512, hidden=1024, num_experts=64, num_topk=4):
    """Test dispatch and combine operations"""
    print(f"[Rank {rank}] Testing dispatch/combine with {num_tokens} tokens, hidden={hidden}...", flush=True)
    
    torch.manual_seed(42 + rank)
    
    # Create input data
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device='cuda')
    
    # Create top-k indices
    scores = torch.randn(num_tokens, num_experts, dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=False)[1]
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)
    
    # Get dispatch layout
    num_tokens_per_rank, _, num_tokens_per_expert, is_token_in_rank, _ = \
        buffer.get_dispatch_layout(topk_idx, num_experts)
    
    # Create config
    config = deep_ep.Config(24, 8, 256)  # num_sms, nvl_chunk, nvl_buffer
    
    # Test dispatch
    start = time.time()
    recv_x, recv_topk_idx, recv_topk_weights, recv_num_tokens_per_expert, handle, event = \
        buffer.dispatch(
            x=x,
            num_tokens_per_rank=num_tokens_per_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            config=config
        )
    torch.cuda.synchronize()
    dispatch_time = (time.time() - start) * 1000
    
    print(f"[Rank {rank}] Dispatch: recv_x shape={recv_x.shape}, time={dispatch_time:.2f}ms", flush=True)
    
    # Test combine
    start = time.time()
    combined_x, _ = buffer.combine(x=recv_x, handle=handle, config=config)
    torch.cuda.synchronize()
    combine_time = (time.time() - start) * 1000
    
    print(f"[Rank {rank}] Combine: combined_x shape={combined_x.shape}, time={combine_time:.2f}ms", flush=True)
    
    return True


def run_test(local_rank: int, num_local_ranks: int):
    """Main test function"""
    try:
        rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
        
        if rank == 0:
            print("=" * 60, flush=True)
            print("DeepEP Intranode Test", flush=True)
            print(f"Testing with {num_ranks} GPUs", flush=True)
            print("=" * 60, flush=True)
        
        dist.barrier()
        
        # Test 1: Buffer creation
        buffer = test_buffer_creation(rank, num_ranks, group)
        dist.barrier()
        
        if rank == 0:
            print("\n[PASS] Buffer creation test passed!\n", flush=True)
        
        # Test 2: Dispatch layout
        test_dispatch_layout(rank, num_ranks, buffer)
        dist.barrier()
        
        if rank == 0:
            print("\n[PASS] Dispatch layout test passed!\n", flush=True)
        
        # Test 3: Dispatch and combine
        # Use smaller sizes for quick test
        test_dispatch_combine(rank, num_ranks, buffer, group, 
                             num_tokens=512, hidden=1024, num_experts=64, num_topk=4)
        dist.barrier()
        
        if rank == 0:
            print("\n[PASS] Dispatch/Combine test passed!\n", flush=True)
            print("=" * 60, flush=True)
            print("All tests passed! DeepEP is working correctly.", flush=True)
            print("=" * 60, flush=True)
        
        # Cleanup
        buffer.destroy()
        dist.barrier()
        dist.destroy_process_group()
        
    except Exception as e:
        print(f"[Rank {local_rank}] ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-gpus', type=int, default=2, help='Number of GPUs')
    args = parser.parse_args()
    
    torch.multiprocessing.spawn(run_test, args=(args.num_gpus,), nprocs=args.num_gpus)
PYTHON_EOF

# Run the test
log_info "Running DeepEP intranode test..."
echo ""

cd /tmp/deepep_build
python3 "$TEST_SCRIPT" --num-gpus "$NUM_GPUS"
EXIT_CODE=$?

# Cleanup
rm -f "$TEST_SCRIPT"

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    log_info "Test completed successfully!"
else
    echo ""
    log_error "Test failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
