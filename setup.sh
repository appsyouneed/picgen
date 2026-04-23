#!/bin/bash
set -e

echo "=== Picgen Image Editor VPS Setup ==="

UBUNTU_VER=$(lsb_release -rs)
if (( $(echo "$UBUNTU_VER < 24" | bc -l) )); then
    echo "Ubuntu $UBUNTU_VER detected: upgrading pip first..."
    pip install --upgrade pip
fi

echo "Installing system dependencies..."
apt-get update && apt-get install -y python3-pip python3-venv python3.10-venv ffmpeg wget git git-lfs bc curl

echo "Creating cache directory..."
mkdir -p /root/.cache/huggingface

echo "Creating Python virtual environment..."
rm -rf /root/picgen/venv
python3 -m venv /root/picgen/venv
source /root/picgen/venv/bin/activate

# --- CUDA 12.4 Toolkit (if not already installed) ---
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Installing CUDA 12.4 toolkit..."
    UBUNTU_VERSION=$(lsb_release -rs | tr -d '.')
    CUDA_DEB="cuda-keyring_1.1-1_all.deb"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/${CUDA_DEB}"
    dpkg -i "$CUDA_DEB"
    apt-get update
    apt-get install -y cuda-toolkit-12-4
    rm -f "$CUDA_DEB"
    export PATH=/usr/local/cuda-12.4/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
    echo 'export PATH=/usr/local/cuda-12.4/bin:$PATH' >> /root/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH' >> /root/.bashrc
    echo "CUDA 12.4 installed."
else
    echo "CUDA already installed: $(nvcc --version | head -1)"
fi

# Upgrade pip only if needed
if python3 -m pip install --upgrade pip --dry-run 2>&1 | grep -q "Would install"; then
    echo "Upgrading pip..."
    python3 -m pip install --upgrade pip
else
    echo "pip already up to date, skipping."
fi

echo "Installing PyTorch with CUDA 12.4 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --ignore-installed

echo "Installing Python dependencies..."
pip install -r requirements.txt --ignore-installed

echo "Installing Hugging Face CLI..."
pip install "huggingface_hub[cli]>=1.5.0"

echo "Fixing pyOpenSSL compatibility..."
python3 -c "from OpenSSL import SSL" 2>/dev/null || pip install --upgrade pyopenssl

echo "Creating local model directories..."
mkdir -p models/Qwen-Image-Edit-2511
mkdir -p models/rapid-aio/v23
chmod -R 777 models

echo "=== Model Download ==="
echo "Models will be downloaded automatically on first run."
echo "To pre-download models now, run:"
echo "  huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511"
echo "  huggingface-cli download Phr00t/Qwen-Image-Edit-Rapid-AIO --include 'v23/Qwen-Rapid-AIO-NSFW-v23.safetensors' --local-dir models/rapid-aio"

echo "Verifying GPU accessibility..."
python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'Total VRAM: {total_vram:.1f}GB')
else:
    print('GPU Device: None (CPU mode)')
"

echo "=== Setup Complete ==="
echo ""
echo "Setting up systemd service..."
cp picgen.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable picgen
systemctl start picgen

echo ""
echo "Service commands:"
echo "  systemctl start picgen   - Start picgen"
echo "  systemctl stop picgen    - Stop picgen"
echo "  systemctl status picgen  - Check status"
echo "  systemctl restart picgen - Restart picgen"
echo ""
echo "View live output:"
echo "  tail -f /root/picgen/picgen.log"
echo ""
echo "To run manually: python3 app.py"
echo "The app will be accessible at: http://0.0.0.0:7860"
