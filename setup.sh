#!/bin/bash
# One-click environment setup for scMomer (Python 3.9, CUDA)
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# For CUDA 11.8, change CUDA_VERSION below to "cu118".

set -e

CUDA_VERSION="cu121"
TORCH_VERSION="2.1.2"
TORCHVISION_VERSION="0.16.2"

echo "=== scMomer environment setup ==="
echo "CUDA target: ${CUDA_VERSION}"
echo ""

# Step 1: Install PyTorch with CUDA (must come from PyTorch index, not PyPI)
echo "[1/2] Installing PyTorch ${TORCH_VERSION} (${CUDA_VERSION}) ..."
pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
    --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Step 2: Install the rest from requirements.txt
echo ""
echo "[2/2] Installing remaining packages ..."
pip install -r requirements.txt

echo ""
echo "=== Done. Verify with: python -c 'import torch; print(torch.cuda.is_available())' ==="
