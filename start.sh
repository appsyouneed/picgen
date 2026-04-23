#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

cp "$SCRIPT_DIR/picgen.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable picgen.service
systemctl restart picgen.service

echo "✓ picgen service started!"
echo ""
echo "Check status: systemctl status picgen"
echo "View logs: tail -f $SCRIPT_DIR/picgen.log"
