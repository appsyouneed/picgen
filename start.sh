#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

# Regenerate service file with correct paths before installing
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
systemctl enable picgen.service
systemctl restart picgen.service

echo "✓ picgen service started!"
echo ""
echo "Check status: systemctl status picgen"
echo "View logs: tail -f $SCRIPT_DIR/picgen.log"
