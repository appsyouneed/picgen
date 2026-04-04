#!/bin/bash

# =============================================================
# Combined Setup Script - picgen (GitHub Source)
# Run this from your local machine / Termux after cloning:
#   git clone https://github.com/appsyouneed/picgen.git
#   cd picgen && bash setup.sh
# =============================================================

echo "============================================="
echo "  picgen - Full Setup"
echo "============================================="

# -----------------------------------------------------------
# STEP -1: Clean up all previous installations and caches
# -----------------------------------------------------------
echo ""
echo "[-1/8] Cleaning up previous installations and caches..."
rm -rf venv
rm -rf models
rm -rf /models
rm -rf ~/.cache/huggingface
rm -rf /tmp/*
rm -rf /var/tmp/*
echo "  Cleanup complete."

# -----------------------------------------------------------
# STEP 0: Find fastest PyPI mirror
# -----------------------------------------------------------
echo ""
echo "[0/8] Testing PyPI mirrors for fastest connection..."

MIRRORS=(
    "https://pypi.org/simple"
    "https://mirrors.aliyun.com/pypi/simple"
    "https://pypi.tuna.tsinghua.edu.cn/simple"
    "https://pypi.mirrors.ustc.edu.cn/simple"
)

FASTEST_MIRROR=""
FASTEST_TIME=999999

for mirror in "${MIRRORS[@]}"; do
    echo -n "  Testing $mirror ... "
    TIME=$(timeout 10 curl -o /dev/null -s -w '%{time_total}' --connect-timeout 10 --max-time 10 "$mirror" 2>/dev/null || echo "999")
    echo "${TIME}s"
    if (( $(echo "$TIME < $FASTEST_TIME" | bc -l) )); then
        FASTEST_TIME=$TIME
        FASTEST_MIRROR=$mirror
    fi
done

echo ""
echo "  Selected fastest mirror: $FASTEST_MIRROR (${FASTEST_TIME}s)"
PIP_INDEX="--index-url $FASTEST_MIRROR"

# -----------------------------------------------------------
# STEP 1: Update system & install core dependencies
# -----------------------------------------------------------
echo ""
echo "[1/8] Updating system and installing core dependencies..."
apt update && apt install -y python3-pip git-lfs python3.12-venv bc

# -----------------------------------------------------------
# STEP 1.5: Create virtual environment
# -----------------------------------------------------------
echo ""
echo "[1.5/8] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created."
else
    echo "  Virtual environment already exists."
fi

# Activate virtual environment
source venv/bin/activate
echo "  Virtual environment activated."

# -----------------------------------------------------------
# STEP 2: Install core AI libraries
# -----------------------------------------------------------
echo ""
echo "[2/8] Installing core AI libraries..."
pip install gradio $PIP_INDEX

# -----------------------------------------------------------
# STEP 3: Install all requirements from requirements.txt
# -----------------------------------------------------------
echo ""
echo "[3/8] Installing app Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt $PIP_INDEX
else
    echo "  WARNING: requirements.txt not found in current directory, skipping."
fi

# -----------------------------------------------------------
# STEP 4: Install Hugging Face CLI
# -----------------------------------------------------------
echo ""
echo "[4/8] Installing Hugging Face CLI..."
if [ ! -f "/root/.local/bin/hf" ]; then
    curl -LsSf https://hf.co/cli/install.sh | bash -s -- --force
else
    echo "  Hugging Face CLI already installed, skipping."
fi

HF_BIN="/root/.local/bin/hf"

# -----------------------------------------------------------
# STEP 5: Create local model directories
# -----------------------------------------------------------
echo ""
echo "[5/8] Creating local model directories..."
mkdir -p models && chmod 777 models
mkdir -p models/Qwen-Image-Edit-2511
mkdir -p models/rapid-aio/v23

# -----------------------------------------------------------
# STEP 6: Download models from Hugging Face
# -----------------------------------------------------------
echo ""
echo "[6/8] Downloading models (this may take a while)..."

if [ -f "models/Qwen-Image-Edit-2511/model.safetensors" ] || [ -f "models/Qwen-Image-Edit-2511/pytorch_model.bin" ]; then
    echo "  -> Base Model already downloaded, skipping."
else
    echo "  -> Downloading Base Model..."
    $HF_BIN download Qwen/Qwen-Image-Edit-2511 \
        --local-dir models/Qwen-Image-Edit-2511 \
        --resume-download || {
        echo "ERROR: Download failed. Check disk space and network."
        exit 1
    }
fi

if [ -f "models/rapid-aio/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" ]; then
    echo "  -> NSFW Weights already downloaded, skipping."
else
    echo "  -> Downloading NSFW Weights (v23)..."
    $HF_BIN download Phr00t/Qwen-Image-Edit-Rapid-AIO \
        --include "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" \
        --local-dir models/rapid-aio \
        --resume-download || {
        echo "ERROR: Download failed. Check disk space and network."
        exit 1
    }
fi



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
export BASE_MODEL_PATH="models/Qwen-Image-Edit-2511"
export NSFW_WEIGHTS_PATH="models/rapid-aio/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors"
# Persist them to ~/.bashrc so they survive reboots
grep -qxF 'export BASE_MODEL_PATH="models/Qwen-Image-Edit-2511"' ~/.bashrc || \
    echo 'export BASE_MODEL_PATH="models/Qwen-Image-Edit-2511"' >> ~/.bashrc

grep -qxF 'export NSFW_WEIGHTS_PATH="models/rapid-aio/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors"' ~/.bashrc || \
    echo 'export NSFW_WEIGHTS_PATH="models/rapid-aio/v23/Qwen-Rapid-AIO-NSFW-v23.safetensors"' >> ~/.bashrc

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
source venv/bin/activate
python3 app.py --share
