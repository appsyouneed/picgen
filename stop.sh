#!/bin/bash

if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

systemctl stop picgen.service
systemctl disable picgen.service

echo "✓ picgen service stopped!"
