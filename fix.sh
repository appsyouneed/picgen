#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

echo "=== Fixing Picgen Installation ==="

cd "$SCRIPT_DIR"

echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "Reinstalling dependencies into venv..."
source "$SCRIPT_DIR/venv/bin/activate"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --force-reinstall

echo "Rewriting service file..."
cat > /etc/systemd/system/picgen.service <<EOF
[Unit]
Description=Picgen Image Editor Application
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$SCRIPT_DIR
Environment="PYTHONUNBUFFERED=1"
Environment="HF_HOME=/root/.cache/huggingface"
Environment="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
Environment="CUDA_LAUNCH_BLOCKING=0"
ExecStart=$SCRIPT_DIR/venv/bin/python3 $SCRIPT_DIR/app.py
Restart=always
RestartSec=10
StandardOutput=append:$SCRIPT_DIR/picgen.log
StandardError=append:$SCRIPT_DIR/picgen.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload

echo "=== Fix Complete ==="
echo "Run: $SCRIPT_DIR/venv/bin/python3 $SCRIPT_DIR/app.py"
