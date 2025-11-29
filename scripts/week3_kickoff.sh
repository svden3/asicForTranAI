#!/bin/bash
# Week 3 Kickoff Script - Run this Monday morning (Dec 1)
# Automates vast.ai setup and initial GPU verification

set -e  # Exit on error

echo "=================================================="
echo "Week 3: GPU Demo Kickoff"
echo "Target: AMD MI210 on vast.ai"
echo "=================================================="
echo ""

# Step 1: Install vast.ai CLI
echo "[1/5] Installing vast.ai CLI..."
pip3 install vastai --quiet
echo "✓ vast.ai CLI installed"
echo ""

# Step 2: Login to vast.ai
echo "[2/5] Logging in to vast.ai..."
echo "Visit: https://vast.ai/console/account/"
echo "Copy your API key and paste it here:"
read -sp "API Key: " VASTAI_API_KEY
echo ""
vastai set api-key "$VASTAI_API_KEY"
echo "✓ Logged in successfully"
echo ""

# Step 3: Search for MI210 instances
echo "[3/5] Searching for AMD MI210 instances..."
vastai search offers 'gpu_name=MI210 num_gpus=1 rocm_version>=6.0 disk_space>=100' \
  --order price_per_gpu_hour

echo ""
echo "Top 5 recommendations (sorted by price):"
echo ""

# Step 4: Interactive instance selection
echo "[4/5] Select an instance to rent:"
echo "Enter the OFFER_ID from the list above:"
read -p "OFFER_ID: " OFFER_ID

echo ""
echo "Renting instance $OFFER_ID..."
vastai create instance "$OFFER_ID" \
  --image rocm/pytorch:rocm6.0_ubuntu22.04_py3.10_pytorch_2.1.1 \
  --disk 100 \
  --ssh \
  --env '-e TZ=America/Los_Angeles'

echo "✓ Instance created!"
echo ""

# Step 5: Get connection details
echo "[5/5] Getting SSH connection details..."
sleep 5  # Wait for instance to initialize

INSTANCE_ID=$(vastai show instances --raw | jq -r '.[0].id')
SSH_HOST=$(vastai show instance "$INSTANCE_ID" --raw | jq -r '.ssh_host')
SSH_PORT=$(vastai show instance "$INSTANCE_ID" --raw | jq -r '.ssh_port')

echo ""
echo "=================================================="
echo "✓ GPU Instance Ready!"
echo "=================================================="
echo ""
echo "Connection Details:"
echo "  SSH: ssh -p $SSH_PORT root@$SSH_HOST"
echo "  Instance ID: $INSTANCE_ID"
echo ""
echo "Next steps:"
echo "  1. SSH into the instance"
echo "  2. Run: rocm-smi --showproductname"
echo "  3. Verify PyTorch: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "  4. Clone repo: git clone https://github.com/[username]/asicForTranAI.git"
echo "  5. Run HIP kernel benchmark"
echo ""
echo "Estimated cost: $1.50/hour (stop instance when done!)"
echo "Stop command: vastai stop instance $INSTANCE_ID"
echo "=================================================="

# Save connection details for later
cat > ~/.vastai_connection <<EOF
INSTANCE_ID=$INSTANCE_ID
SSH_HOST=$SSH_HOST
SSH_PORT=$SSH_PORT
EOF

echo "Connection details saved to: ~/.vastai_connection"
