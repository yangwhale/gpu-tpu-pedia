#!/bin/bash
# ============================================================================
# xpk-lustre-setup.sh — XPK + Managed Lustre End-to-End Setup Script
#
# Usage:
#   ./xpk-lustre-setup.sh <command>
#
# Commands:
#   info        Show Lustre instance details
#   create      Create GKE cluster with Lustre CSI driver
#   network     Setup VPC peering and firewall (skip if already done)
#   manifest    Generate PV/PVC manifest from Lustre instance info
#   attach      Attach Lustre storage to cluster
#   test        Run test workload to verify Lustre mount
#   all         Run: create → manifest → attach → test
#   cleanup     Delete test workload, detach storage, delete cluster
#
# Verified: 2026-03-04, GKE 1.33.5, xpk 1.3.0
# ============================================================================

set -euo pipefail

# ========================= CONFIGURATION ====================================
# Modify these variables to match your environment

PROJECT_ID="${PROJECT_ID:?ERROR: Set PROJECT_ID}"
LOCATION="${LOCATION:?ERROR: Set LOCATION (e.g. us-central2-b)}"
CLUSTER_NAME="${CLUSTER_NAME:?ERROR: Set CLUSTER_NAME}"
LUSTRE_INSTANCE="${LUSTRE_INSTANCE:?ERROR: Set LUSTRE_INSTANCE}"

# Network (must match Lustre instance's network)
NETWORK_NAME="${NETWORK_NAME:?ERROR: Set NETWORK_NAME}"
SUBNET_NAME="${SUBNET_NAME:?ERROR: Set SUBNET_NAME}"

# Cluster settings
GKE_VERSION="${GKE_VERSION:-1.33.5-gke.2469000}"
DEVICE_TYPE="${DEVICE_TYPE:-n2-standard-32-1}"       # CPU for testing; use tpu-type for training
CAPACITY_TYPE="${CAPACITY_TYPE:-on-demand}"            # on-demand | spot | reservation=NAME
NUM_SLICES="${NUM_SLICES:-1}"

# Lustre mount settings
MOUNT_POINT="${MOUNT_POINT:-/lustre-data}"
MANIFEST_DIR="${MANIFEST_DIR:-$(cd "$(dirname "$0")" && pwd)}"
MANIFEST_FILE="${MANIFEST_DIR}/lustre-manifest.yaml"

# Workload settings
TEST_WORKLOAD_NAME="${TEST_WORKLOAD_NAME:-lustre-mount-test}"
DOCKER_IMAGE="${DOCKER_IMAGE:-ubuntu:22.04}"

# ========================= HELPERS ==========================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[✗]${NC} $*" >&2; }
info() { echo -e "${BLUE}[→]${NC} $*"; }
step() { echo -e "\n${BLUE}━━━ Step: $* ━━━${NC}"; }

# Fix gLinux context_aware cert issue
fix_gcloud_env() {
  unset CLOUDSDK_CONTEXT_AWARE_CERTIFICATE_CONFIG_FILE_PATH 2>/dev/null || true
  unset CLOUDSDK_CONTEXT_AWARE_USE_CLIENT_CERTIFICATE 2>/dev/null || true
  unset CLOUDSDK_CONTEXT_AWARE_USE_ECP_HTTP_PROXY 2>/dev/null || true
}

# Build capacity type flag for xpk
get_capacity_flag() {
  case "${CAPACITY_TYPE}" in
    on-demand) echo "--on-demand" ;;
    spot)      echo "--spot" ;;
    reservation=*) echo "--reservation=${CAPACITY_TYPE#reservation=}" ;;
    *) err "Unknown CAPACITY_TYPE: ${CAPACITY_TYPE}"; exit 1 ;;
  esac
}

# Get region from zone (e.g., asia-southeast1-b → asia-southeast1)
get_region() {
  echo "${LOCATION%-*}"
}

# ========================= COMMANDS =========================================

cmd_info() {
  step "Lustre Instance Info"
  fix_gcloud_env

  info "Fetching details for ${LUSTRE_INSTANCE} in ${LOCATION}..."
  local details
  details=$(gcloud lustre instances describe "${LUSTRE_INSTANCE}" \
    --location="${LOCATION}" \
    --project="${PROJECT_ID}" \
    --format=json 2>&1)

  local ip fs capacity throughput gke_support network state
  ip=$(echo "$details" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['mountPoint'].split('@')[0])")
  fs=$(echo "$details" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['mountPoint'].split(':/')[-1])")
  capacity=$(echo "$details" | python3 -c "import sys,json; print(json.load(sys.stdin)['capacityGib'])")
  throughput=$(echo "$details" | python3 -c "import sys,json; print(json.load(sys.stdin)['perUnitStorageThroughput'])")
  gke_support=$(echo "$details" | python3 -c "import sys,json; print(json.load(sys.stdin).get('gkeSupportEnabled', False))")
  network=$(echo "$details" | python3 -c "import sys,json; print(json.load(sys.stdin)['network'].split('/')[-1])")
  state=$(echo "$details" | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")

  echo ""
  echo "  Instance:       ${LUSTRE_INSTANCE}"
  echo "  State:          ${state}"
  echo "  IP:             ${ip}"
  echo "  Filesystem:     ${fs}"
  echo "  Capacity:       ${capacity} GiB ($(( ${capacity} / 1000 )) TiB)"
  echo "  Throughput:     ${throughput} MiB/s per TiB"
  echo "  Network:        ${network}"
  echo "  GKE Support:    ${gke_support}"
  echo ""

  if [[ "${gke_support}" == "True" ]]; then
    warn "gkeSupportEnabled=true → --enable-legacy-lustre-port is REQUIRED"
  fi
}

cmd_create() {
  step "Create GKE Cluster: ${CLUSTER_NAME}"
  fix_gcloud_env

  # Check if cluster already exists
  local existing
  existing=$(gcloud container clusters list \
    --project="${PROJECT_ID}" \
    --filter="name=${CLUSTER_NAME}" \
    --format="value(name)" 2>/dev/null || true)

  if [[ -n "${existing}" ]]; then
    warn "Cluster ${CLUSTER_NAME} already exists. Running xpk to ensure node pool..."
  else
    info "Creating new cluster..."
  fi

  # Check if Lustre instance uses legacy port
  local legacy_flag=""
  local gke_support
  gke_support=$(gcloud lustre instances describe "${LUSTRE_INSTANCE}" \
    --location="${LOCATION}" \
    --project="${PROJECT_ID}" \
    --format="value(gkeSupportEnabled)" 2>/dev/null || echo "")

  if [[ "${gke_support}" == "True" ]]; then
    legacy_flag="--enable-legacy-lustre-port"
    info "Lustre has gkeSupportEnabled=true, adding --enable-legacy-lustre-port"
  fi

  local capacity_flag
  capacity_flag=$(get_capacity_flag)

  info "Running xpk cluster create..."
  info "  Project:     ${PROJECT_ID}"
  info "  Zone:        ${LOCATION}"
  info "  Device:      ${DEVICE_TYPE}"
  info "  GKE Version: ${GKE_VERSION}"
  info "  Network:     ${NETWORK_NAME} / ${SUBNET_NAME}"
  info "  Capacity:    ${CAPACITY_TYPE}"
  echo ""

  xpk cluster create \
    --cluster "${CLUSTER_NAME}" \
    --device-type="${DEVICE_TYPE}" \
    --num-slices="${NUM_SLICES}" \
    --zone="${LOCATION}" \
    --project="${PROJECT_ID}" \
    --gke-version="${GKE_VERSION}" \
    ${capacity_flag} \
    --custom-cluster-arguments="--network=${NETWORK_NAME} --subnetwork=${SUBNET_NAME} --release-channel=None" \
    --enable-lustre-csi-driver \
    ${legacy_flag}

  log "Cluster ${CLUSTER_NAME} is ready"

  # Verify Lustre CSI driver
  info "Verifying Lustre CSI driver..."
  local csi_enabled
  csi_enabled=$(gcloud container clusters describe "${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --location="$(get_region)" \
    --format="value(addonsConfig.lustreCsiDriverConfig.enabled)" 2>/dev/null)

  if [[ "${csi_enabled}" == "True" ]]; then
    log "Lustre CSI driver: enabled"
  else
    err "Lustre CSI driver not enabled!"
    exit 1
  fi
}

cmd_network() {
  step "Network Setup (VPC Peering & Firewall)"
  fix_gcloud_env

  local ip_range_name="lustre-peering-range"
  local fw_rule_name="allow-lustre-internal"

  # Check if IP range already exists
  info "Checking VPC peering IP range..."
  if gcloud compute addresses describe "${ip_range_name}" --global --project="${PROJECT_ID}" &>/dev/null; then
    warn "IP range ${ip_range_name} already exists, skipping creation"
  else
    info "Creating IP range for VPC peering..."
    gcloud compute addresses create "${ip_range_name}" \
      --global \
      --purpose=VPC_PEERING \
      --prefix-length=20 \
      --network="${NETWORK_NAME}" \
      --project="${PROJECT_ID}"
    log "IP range created"
  fi

  # Get CIDR range
  local cidr_range
  cidr_range=$(gcloud compute addresses describe "${ip_range_name}" \
    --global \
    --format="value[separator=/](address, prefixLength)" \
    --project="${PROJECT_ID}")
  info "CIDR range: ${cidr_range}"

  # Check if firewall rule exists
  info "Checking firewall rules..."
  if gcloud compute firewall-rules describe "${fw_rule_name}" --project="${PROJECT_ID}" &>/dev/null; then
    warn "Firewall rule ${fw_rule_name} already exists, skipping"
  else
    info "Creating firewall rule for Lustre ports (988, 6988)..."
    gcloud compute firewall-rules create "${fw_rule_name}" \
      --allow=tcp:988,tcp:6988 \
      --network="${NETWORK_NAME}" \
      --source-ranges="${cidr_range}" \
      --project="${PROJECT_ID}"
    log "Firewall rule created"
  fi

  # VPC peering
  info "Establishing VPC peering (idempotent)..."
  gcloud services vpc-peerings connect \
    --network="${NETWORK_NAME}" \
    --ranges="${ip_range_name}" \
    --service=servicenetworking.googleapis.com \
    --project="${PROJECT_ID}" || true

  log "Network setup complete"
}

cmd_manifest() {
  step "Generate PV/PVC Manifest"
  fix_gcloud_env

  # Fetch Lustre instance details
  info "Fetching Lustre instance details..."
  local details
  details=$(gcloud lustre instances describe "${LUSTRE_INSTANCE}" \
    --location="${LOCATION}" \
    --project="${PROJECT_ID}" \
    --format=json)

  local ip fs capacity
  ip=$(echo "$details" | python3 -c "import sys,json; print(json.load(sys.stdin)['mountPoint'].split('@')[0])")
  fs=$(echo "$details" | python3 -c "import sys,json; print(json.load(sys.stdin)['mountPoint'].split(':/')[-1])")
  capacity=$(echo "$details" | python3 -c "import sys,json; print(json.load(sys.stdin)['capacityGib'])")

  info "Lustre IP: ${ip}, Filesystem: ${fs}, Capacity: ${capacity}Gi"
  info "Writing manifest to: ${MANIFEST_FILE}"

  cat > "${MANIFEST_FILE}" <<YAML
apiVersion: v1
kind: PersistentVolume
metadata:
  name: xpk-lustre-pv
spec:
  storageClassName: ""
  capacity:
    storage: ${capacity}Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  volumeMode: Filesystem
  claimRef:
    namespace: default
    name: xpk-lustre-pvc
  csi:
    driver: lustre.csi.storage.gke.io
    volumeHandle: "projects/${PROJECT_ID}/locations/${LOCATION}/instances/${LUSTRE_INSTANCE}"
    volumeAttributes:
      ip: "${ip}"
      filesystem: "${fs}"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: xpk-lustre-pvc
  namespace: default
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: ""
  volumeName: xpk-lustre-pv
  resources:
    requests:
      storage: ${capacity}Gi
YAML

  log "Manifest generated: ${MANIFEST_FILE}"
  echo ""
  cat "${MANIFEST_FILE}"
}

cmd_attach() {
  step "Attach Lustre Storage to Cluster"
  fix_gcloud_env

  if [[ ! -f "${MANIFEST_FILE}" ]]; then
    err "Manifest file not found: ${MANIFEST_FILE}"
    err "Run './xpk-lustre-setup.sh manifest' first"
    exit 1
  fi

  info "Attaching ${LUSTRE_INSTANCE} to ${CLUSTER_NAME}..."
  info "  Mount point: ${MOUNT_POINT}"
  info "  Auto-mount:  true"

  xpk storage attach "${LUSTRE_INSTANCE}" \
    --cluster="${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${LOCATION}" \
    --type=lustre \
    --mount-point="${MOUNT_POINT}" \
    --readonly=false \
    --auto-mount=true \
    --manifest="${MANIFEST_FILE}"

  # Verify PV/PVC binding
  info "Waiting for PV/PVC binding..."
  sleep 5

  local pv_status pvc_status
  pv_status=$(kubectl get pv xpk-lustre-pv -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")
  pvc_status=$(kubectl get pvc xpk-lustre-pvc -n default -o jsonpath='{.status.phase}' 2>/dev/null || echo "NotFound")

  if [[ "${pv_status}" == "Bound" && "${pvc_status}" == "Bound" ]]; then
    log "PV/PVC both Bound — storage ready"
    kubectl get pv,pvc -n default
  else
    err "PV status: ${pv_status}, PVC status: ${pvc_status}"
    err "Check: kubectl describe pv xpk-lustre-pv; kubectl describe pvc xpk-lustre-pvc"
    exit 1
  fi
}

cmd_test() {
  step "Run Test Workload: ${TEST_WORKLOAD_NAME}"
  fix_gcloud_env

  local capacity_flag
  capacity_flag=$(get_capacity_flag)

  # Delete existing test workload if present
  info "Checking for existing test workload..."
  xpk workload delete \
    --workload "${TEST_WORKLOAD_NAME}" \
    --cluster="${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${LOCATION}" 2>/dev/null || true

  info "Creating test workload..."
  xpk workload create \
    --workload "${TEST_WORKLOAD_NAME}" \
    --cluster="${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${LOCATION}" \
    --device-type="${DEVICE_TYPE}" \
    --num-slices=1 \
    ${capacity_flag} \
    --skip-validation \
    --docker-image="${DOCKER_IMAGE}" \
    --command="echo '=== XPK Lustre Mount Test ===' && \
echo '--- 1. Mount Check ---' && \
df -h ${MOUNT_POINT} && \
echo '--- 2. Write Test ---' && \
echo \"hello from xpk lustre test at \$(date)\" > ${MOUNT_POINT}/xpk-test-\$(hostname).txt && \
cat ${MOUNT_POINT}/xpk-test-\$(hostname).txt && \
echo '--- 3. List Files ---' && \
ls -la ${MOUNT_POINT}/ | head -20 && \
echo '--- 4. IO Performance (1GB sequential write) ---' && \
dd if=/dev/zero of=${MOUNT_POINT}/perf-test-\$(hostname).bin bs=1M count=1024 2>&1 && \
echo '--- 5. Cleanup ---' && \
rm -f ${MOUNT_POINT}/perf-test-\$(hostname).bin ${MOUNT_POINT}/xpk-test-\$(hostname).txt && \
echo '=== All Tests Passed ==='"

  # Wait for pod to complete
  info "Waiting for test pod to complete..."
  local pod_name=""
  local max_wait=180
  local waited=0

  while [[ ${waited} -lt ${max_wait} ]]; do
    pod_name=$(kubectl get pods \
      -l "jobset.sigs.k8s.io/jobset-name=${TEST_WORKLOAD_NAME}" \
      -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

    if [[ -n "${pod_name}" ]]; then
      local phase
      phase=$(kubectl get pod "${pod_name}" -o jsonpath='{.status.phase}' 2>/dev/null || echo "")
      if [[ "${phase}" == "Succeeded" ]]; then
        log "Pod ${pod_name} completed successfully"
        break
      elif [[ "${phase}" == "Failed" ]]; then
        err "Pod ${pod_name} failed!"
        kubectl logs "${pod_name}" 2>/dev/null || true
        exit 1
      fi
    fi

    sleep 5
    waited=$((waited + 5))
    printf "\r  Waiting... %ds / %ds" "${waited}" "${max_wait}"
  done
  echo ""

  if [[ ${waited} -ge ${max_wait} ]]; then
    err "Timeout waiting for test pod (${max_wait}s)"
    kubectl get pods -l "jobset.sigs.k8s.io/jobset-name=${TEST_WORKLOAD_NAME}" 2>/dev/null || true
    exit 1
  fi

  # Show logs
  step "Test Results"
  kubectl logs "${pod_name}" 2>/dev/null

  # Parse write speed
  local speed
  speed=$(kubectl logs "${pod_name}" 2>/dev/null | grep "copied" | grep -oP '[\d.]+ [GMKT]B/s' || echo "N/A")
  echo ""
  log "Write throughput: ${speed}"
}

cmd_all() {
  echo -e "${BLUE}╔═══════════════════════════════════════════════╗${NC}"
  echo -e "${BLUE}║  XPK + Managed Lustre Full Setup              ║${NC}"
  echo -e "${BLUE}║  Project:  ${PROJECT_ID}              ║${NC}"
  echo -e "${BLUE}║  Cluster:  ${CLUSTER_NAME}                    ║${NC}"
  echo -e "${BLUE}║  Lustre:   ${LUSTRE_INSTANCE}         ║${NC}"
  echo -e "${BLUE}╚═══════════════════════════════════════════════╝${NC}"
  echo ""

  cmd_create
  cmd_manifest
  cmd_attach
  cmd_test

  echo ""
  log "All steps completed successfully!"
  echo ""
  echo "  Cluster console: https://console.cloud.google.com/kubernetes/clusters/details/$(get_region)/${CLUSTER_NAME}/details?project=${PROJECT_ID}"
  echo "  To cleanup:      ./xpk-lustre-setup.sh cleanup"
}

cmd_cleanup() {
  step "Cleanup"
  fix_gcloud_env

  local capacity_flag
  capacity_flag=$(get_capacity_flag)

  # Delete test workload
  info "Deleting test workload..."
  xpk workload delete \
    --workload "${TEST_WORKLOAD_NAME}" \
    --cluster="${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${LOCATION}" 2>/dev/null || warn "No test workload to delete"

  # Detach storage (note: storage detach does NOT take --type)
  info "Detaching Lustre storage..."
  xpk storage detach "${LUSTRE_INSTANCE}" \
    --cluster="${CLUSTER_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${LOCATION}" 2>/dev/null || warn "No storage to detach"

  # Delete cluster
  echo ""
  read -p "Delete cluster ${CLUSTER_NAME}? This cannot be undone. [y/N] " -n 1 -r
  echo ""
  if [[ $REPLY =~ ^[Yy]$ ]]; then
    info "Deleting cluster ${CLUSTER_NAME}..."
    xpk cluster delete \
      --cluster="${CLUSTER_NAME}" \
      --project="${PROJECT_ID}" \
      --zone="${LOCATION}"
    log "Cluster deleted"
  else
    warn "Cluster ${CLUSTER_NAME} kept running (costs $$$)"
  fi

  log "Cleanup complete"
}

cmd_help() {
  echo "Usage: $0 <command>"
  echo ""
  echo "Commands:"
  echo "  info        Show Lustre instance details"
  echo "  create      Create GKE cluster with Lustre CSI driver"
  echo "  network     Setup VPC peering and firewall (if not done)"
  echo "  manifest    Generate PV/PVC manifest from Lustre instance"
  echo "  attach      Attach Lustre storage to cluster"
  echo "  test        Run test workload to verify mount"
  echo "  all         Full setup: create → manifest → attach → test"
  echo "  cleanup     Delete workload, detach storage, delete cluster"
  echo ""
  echo "Required environment variables:"
  echo "  PROJECT_ID       GCP project ID"
  echo "  LOCATION         Zone (e.g. us-central2-b, asia-southeast1-b)"
  echo "  CLUSTER_NAME     GKE cluster name to create"
  echo "  LUSTRE_INSTANCE  Managed Lustre instance name"
  echo "  NETWORK_NAME     VPC network name (must match Lustre)"
  echo "  SUBNET_NAME      Subnet name within the VPC"
  echo ""
  echo "Optional environment variables:"
  echo "  DEVICE_TYPE      (default: n2-standard-32-1) CPU for testing; use tpu-type for training"
  echo "  CAPACITY_TYPE    (default: on-demand) [on-demand|spot|reservation=NAME]"
  echo "  GKE_VERSION      (default: 1.33.5-gke.2469000)"
  echo "  MOUNT_POINT      (default: /lustre-data)"
  echo "  NUM_SLICES       (default: 1)"
  echo ""
  echo "Examples:"
  echo "  # Source your env file, then run"
  echo "  source env.sh && ./xpk-lustre-setup.sh all"
  echo ""
  echo "  # Or inline"
  echo "  PROJECT_ID=my-project LOCATION=us-central2-b CLUSTER_NAME=test-cluster \\"
  echo "    LUSTRE_INSTANCE=my-lustre NETWORK_NAME=my-net SUBNET_NAME=my-subnet \\"
  echo "    ./xpk-lustre-setup.sh all"
  echo ""
  echo "  # Individual steps"
  echo "  source env.sh && ./xpk-lustre-setup.sh create"
  echo "  source env.sh && ./xpk-lustre-setup.sh manifest"
  echo "  source env.sh && ./xpk-lustre-setup.sh attach"
  echo "  source env.sh && ./xpk-lustre-setup.sh test"
  echo ""
  echo "  # Cleanup"
  echo "  source env.sh && ./xpk-lustre-setup.sh cleanup"
}

# ========================= MAIN =============================================

case "${1:-help}" in
  info)     cmd_info ;;
  create)   cmd_create ;;
  network)  cmd_network ;;
  manifest) cmd_manifest ;;
  attach)   cmd_attach ;;
  test)     cmd_test ;;
  all)      cmd_all ;;
  cleanup)  cmd_cleanup ;;
  help|-h|--help) cmd_help ;;
  *)
    err "Unknown command: $1"
    cmd_help
    exit 1
    ;;
esac
