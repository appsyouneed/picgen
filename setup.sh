#!/bin/bash

# =============================================================
# Combined Setup Script - Image Editor (GitHub Source)
# Run this from your local machine / Termux after cloning:
#   git clone https://github.com/appsyouneed/imageeditor.git
#   cd imageeditor && bash setup.sh
# =============================================================

set -e  # Exit immediately if any command fails

echo "============================================="
echo "  Image Editor - Full Setup"
echo "============================================="

# -----------------------------------------------------------
# STEP 1: Update system & install core dependencies
# -----------------------------------------------------------
echo ""
echo "[1/8] Updating system and installing core dependencies..."
apt update && apt install -y python3-pip git-lfs python3.12-venv

# -----------------------------------------------------------
# STEP 2: Install core AI libraries globally
# -----------------------------------------------------------
echo ""
echo "[2/8] Installing core AI libraries..."
python3 -m pip install gradio diffusers transformers accelerate safetensors \
    --break-system-packages --ignore-installed

# -----------------------------------------------------------
# STEP 3: Install app requirements from the cloned repo
# -----------------------------------------------------------
echo ""
echo "[3/8] Installing app Python dependencies..."
if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt --break-system-packages
else
    echo "  WARNING: requirements.txt not found in current directory, skipping."
fi

# -----------------------------------------------------------
# STEP 4: Install Hugging Face CLI
# -----------------------------------------------------------
echo ""
echo "[4/8] Installing Hugging Face CLI..."
curl -LsSf https://hf.co/cli/install.sh | bash -s -- --force

HF_BIN="/root/.local/bin/hf"

# -----------------------------------------------------------
# STEP 5: Create local model directories
# -----------------------------------------------------------
echo ""
echo "[5/8] Creating local model directories..."
mkdir -p /models && chmod 777 /models
mkdir -p /models/Qwen-Image-Edit-2511
mkdir -p /models/rapid-aio/v23

# -----------------------------------------------------------
# STEP 6: Download models from Hugging Face
# -----------------------------------------------------------
echo ""
echo "[6/8] Downloading models (this may take a while)..."

echo "  -> Downloading Base Model..."
$HF_BIN download Qwen/Qwen-Image-Edit-2511 \
    --local-dir /models/Qwen-Image-Edit-2511

echo "  -> Downloading NSFW Weights (v23)..."
$HF_BIN download Phr00t/Qwen-Image-Edit-Rapid-AIO \
    --include "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" \
    --local-dir /models/rapid-aio



# -----------------------------------------------------------
# STEP 7: Verify GPU accessibility
# -----------------------------------------------------------
echo ""
echo "[7/8] Verifying GPU accessibility..."
python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'Total VRAM: {total_vram:.1f}GB')
else:
    print('GPU Device: None')
"

# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -----------------------------------------------------------
# STEP 8: Export environment variables
# -----------------------------------------------------------
echo ""
echo "[8/8] Setting environment variables..."
export BASE_MODEL_PATH="/models/Qwen-Image-Edit-2511"
export NSFW_WEIGHTS_PATH="/models/rapid-aio/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors"
# Persist them to ~/.bashrc so they survive reboots
grep -qxF 'export BASE_MODEL_PATH="/models/Qwen-Image-Edit-2511"' ~/.bashrc || \
    echo 'export BASE_MODEL_PATH="/models/Qwen-Image-Edit-2511"' >> ~/.bashrc

grep -qxF 'export NSFW_WEIGHTS_PATH="/models/rapid-aio/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors"' ~/.bashrc || \
    echo 'export NSFW_WEIGHTS_PATH="/models/rapid-aio/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors"' >> ~/.bashrc

grep -qxF 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' ~/.bashrc || \
    echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc

echo ""
echo "============================================="
echo "  Setup Complete!"
echo "  BASE_MODEL_PATH     = $BASE_MODEL_PATH"
echo "  NSFW_WEIGHTS_PATH   = $NSFW_WEIGHTS_PATH"
echo "============================================="
echo ""
echo "  Press ENTER to start the server..."
read -r

# Source ~/.bashrc so environment variables are live in this session
# shellcheck disable=SC1090
source ~/.bashrc

echo "Starting server..."
python3 app.py --share
