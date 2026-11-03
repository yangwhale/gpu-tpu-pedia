#!/bin/bash
# Create one G4 Spot VM: g4-standard-384 = 8x NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB each).
# Usage: PROJECT=<your-project> ZONE=us-central1-b ./create-vm.sh
set -euo pipefail
: "${PROJECT:?set PROJECT}"; ZONE=${ZONE:-us-central1-b}; NAME=${NAME:-g4-glm53}
gcloud compute instances create "$NAME" --project "$PROJECT" --zone "$ZONE" \
  --machine-type g4-standard-384 \
  --provisioning-model SPOT --instance-termination-action STOP \
  --maintenance-policy TERMINATE \
  --image-family common-cu129-ubuntu-2404-nvidia-580 \
  --image-project deeplearning-platform-release \
  --boot-disk-size 2000GB --boot-disk-type hyperdisk-balanced \
  --scopes cloud-platform \
  --metadata install-nvidia-driver=True
# On-demand instead of Spot: drop --provisioning-model / --instance-termination-action.
# Using a specific reservation: add --reservation-affinity=specific --reservation=<name>
#   (the reservation must be for g4-standard-384; a g4-standard-48 reservation is 1 GPU/VM and cannot host this model).
