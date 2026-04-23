#!/bin/bash
set -e

echo "=== Picgen Image Editor VPS Setup ==="

UBUNTU_VER=$(lsb_release -rs)
if (( $(echo "$UBUNTU_VER < 24" | bc -l) )); then
    echo "Ubuntu $UBUNTU_VER detected: upgrading pip first..."
    pip3 install --upgrade pip
fi

echo "Creating cache directory..."
mkdir -p /root/.cache/huggingface

echo "Installing system dependencies..."
apt-get update && apt-get install -y python3-pip python3-venv ffmpeg wget git git-lfs bc curl

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
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --break-system-packages --ignore-installed

echo "Installing Python dependencies..."
pip3 install -r requirements.txt --break-system-packages --ignore-installed

echo "Installing Hugging Face CLI..."
pip3 install huggingface_hub[cli] --break-system-packages

echo "Fixing pyOpenSSL compatibility..."
python3 -c "from OpenSSL import SSL" 2>/dev/null || pip3 install --upgrade pyopenssl --break-system-packages

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
echo "To run manually: python3 app.py"
echo "The app will be accessible at: http://0.0.0.0:7860"
