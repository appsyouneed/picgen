#!/bin/bash
set -e

echo "=== Picgen Image Editor VPS Setup ==="

echo "Creating cache directory..."
mkdir -p /root/.cache/huggingface

echo "Installing system dependencies..."
apt-get update && apt-get install -y python3-pip python3-venv ffmpeg wget git git-lfs bc

echo "Installing PyTorch (auto-detect system CUDA)..."
pip3 install torch torchvision --break-system-packages --ignore-installed

echo "Installing Python dependencies..."
pip3 install -r requirements.txt --break-system-packages --ignore-installed

echo "Installing Hugging Face CLI..."
pip3 install huggingface_hub[cli] --break-system-packages

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
