#!/bin/bash
set -e

echo "=== Picgen Image Editor VPS Setup ==="

echo "Creating cache directory..."
mkdir -p /root/.cache/huggingface

echo "Installing system dependencies..."
apt-get update && apt-get install -y python3-pip ffmpeg wget git git-lfs bc

echo "Installing PyTorch with CUDA 12.4 support..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --break-system-packages --ignore-installed

echo "Installing Python dependencies..."
pip3 install -r requirements.txt --break-system-packages --ignore-installed

echo "Installing Hugging Face CLI..."
if [ ! -f "/root/.local/bin/hf" ]; then
    curl -LsSf https://hf.co/cli/install.sh | bash -s -- --force
else
    echo "Hugging Face CLI already installed, skipping..."
fi

echo "Creating local model directories..."
mkdir -p models/Qwen-Image-Edit-2511
mkdir -p models/rapid-aio/v23
chmod -R 777 models

echo "=== Model Download ==="
echo "Models will be downloaded automatically on first run."
echo "To pre-download models now, run:"
echo "  /root/.local/bin/hf download Qwen/Qwen-Image-Edit-2511 --local-dir models/Qwen-Image-Edit-2511"
echo "  /root/.local/bin/hf download Phr00t/Qwen-Image-Edit-Rapid-AIO --include 'v23/Qwen-Rapid-AIO-NSFW-v23.safetensors' --local-dir models/rapid-aio"

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
