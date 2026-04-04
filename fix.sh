#!/bin/bash
set -e

echo "=== Fixing Picgen Installation ==="

cd /root/picgen

echo "Clearing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

echo "Reinstalling dependencies..."
pip3 install -r requirements.txt --break-system-packages --ignore-installed --force-reinstall

echo "Copying service file..."
cp picgen.service /etc/systemd/system/
systemctl daemon-reload

echo "=== Fix Complete ==="
echo "Run: python3 app.py"
