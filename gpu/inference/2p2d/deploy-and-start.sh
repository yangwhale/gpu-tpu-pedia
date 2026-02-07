#!/bin/bash
#
# Deploy and Start 2P+2D SGLang with DeepEP
#
# Architecture:
#   Prefill: b7 (node 0) + b8 (node 1) -> TP=16, DeepEP enabled
#   Decode:  b9 (node 0) + b10 (node 1) -> TP=16, DeepEP enabled
#
# Usage:
#   ./deploy-and-start.sh [deploy|start-prefill|start-decode|start-all|stop-all]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Node configuration
PREFILL_NODES=("b7" "b8")
DECODE_NODES=("b9" "b10")
PREFILL_IPS=("10.8.0.12" "10.8.0.17")
DECODE_IPS=("10.8.0.15" "10.8.0.19")

deploy_scripts() {
    echo "Deploying scripts to all nodes..."

    for node in "${PREFILL_NODES[@]}" "${DECODE_NODES[@]}"; do
        echo "  -> $node"
        ssh "$node" "mkdir -p /tmp/2p2d"
        scp -q "$SCRIPT_DIR"/*.sh "$node:/tmp/2p2d/"
        ssh "$node" "chmod +x /tmp/2p2d/*.sh"
    done

    echo "Deploy complete!"
}

start_prefill() {
    echo "Starting Prefill nodes..."
    echo "  Node 0 (b7): Prefill master"
    ssh b7 "cd /tmp/2p2d && nohup ./start-prefill-node0.sh > /tmp/prefill_node0.log 2>&1 &"

    sleep 3

    echo "  Node 1 (b8): Prefill worker"
    ssh b8 "cd /tmp/2p2d && nohup ./start-prefill-node1.sh > /tmp/prefill_node1.log 2>&1 &"

    echo ""
    echo "Prefill nodes starting. Check logs:"
    echo "  ssh b7 'tail -f /tmp/prefill_node0.log'"
    echo "  ssh b8 'tail -f /tmp/prefill_node1.log'"
    echo ""
    echo "Wait for 'Server is ready' before starting Decode nodes!"
}

start_decode() {
    echo "Starting Decode nodes..."
    echo "  Node 0 (b9): Decode master"
    ssh b9 "cd /tmp/2p2d && nohup ./start-decode-node0.sh > /tmp/decode_node0.log 2>&1 &"

    sleep 3

    echo "  Node 1 (b10): Decode worker"
    ssh b10 "cd /tmp/2p2d && nohup ./start-decode-node1.sh > /tmp/decode_node1.log 2>&1 &"

    echo ""
    echo "Decode nodes starting. Check logs:"
    echo "  ssh b9 'tail -f /tmp/decode_node0.log'"
    echo "  ssh b10 'tail -f /tmp/decode_node1.log'"
}

start_router() {
    echo "Starting Router on b7..."
    ssh b7 "nohup python3 -m sglang_router.launch_router \
        --pd-disaggregation \
        --prefill http://10.8.0.12:30000 \
        --decode http://10.8.0.15:30000 \
        --host 0.0.0.0 \
        --port 8000 \
        > /tmp/router.log 2>&1 &"

    echo "Router starting on b7:8000"
    echo "Check log: ssh b7 'tail -f /tmp/router.log'"
}

stop_all() {
    echo "Stopping all SGLang processes..."

    for node in "${PREFILL_NODES[@]}" "${DECODE_NODES[@]}"; do
        echo "  -> $node"
        ssh "$node" "pkill -f 'sglang.launch_server' || true"
        ssh "$node" "pkill -f 'sglang_router' || true"
    done

    echo "All processes stopped."
}

check_status() {
    echo "Checking SGLang status on all nodes..."
    echo ""

    for node in "${PREFILL_NODES[@]}" "${DECODE_NODES[@]}"; do
        echo "=== $node ==="
        ssh "$node" "pgrep -a -f 'sglang' | head -3 || echo 'No SGLang processes'"
        echo ""
    done
}

case "${1:-}" in
    deploy)
        deploy_scripts
        ;;
    start-prefill)
        start_prefill
        ;;
    start-decode)
        start_decode
        ;;
    start-router)
        start_router
        ;;
    start-all)
        start_prefill
        echo ""
        echo "Waiting 120s for Prefill to initialize..."
        sleep 120
        start_decode
        echo ""
        echo "Waiting 60s for Decode to initialize..."
        sleep 60
        start_router
        ;;
    stop-all)
        stop_all
        ;;
    status)
        check_status
        ;;
    *)
        echo "Usage: $0 {deploy|start-prefill|start-decode|start-router|start-all|stop-all|status}"
        echo ""
        echo "Commands:"
        echo "  deploy        - Deploy scripts to all nodes"
        echo "  start-prefill - Start Prefill nodes (b7, b8)"
        echo "  start-decode  - Start Decode nodes (b9, b10)"
        echo "  start-router  - Start Router on b7"
        echo "  start-all     - Start everything in sequence"
        echo "  stop-all      - Stop all SGLang processes"
        echo "  status        - Check process status"
        echo ""
        echo "Recommended order:"
        echo "  1. ./deploy-and-start.sh deploy"
        echo "  2. ./deploy-and-start.sh start-prefill"
        echo "  3. Wait for 'Server is ready' in Prefill logs"
        echo "  4. ./deploy-and-start.sh start-decode"
        echo "  5. Wait for 'Server is ready' in Decode logs"
        echo "  6. ./deploy-and-start.sh start-router"
        exit 1
        ;;
esac
