#!/usr/bin/env bash
# Force-kill picgen app.py and the Cloudflare tunnel (SIGKILL), regardless of
# how they were launched. Use "bash run.sh stop" for a clean shutdown; use
# this if stop doesn't work or a process is stuck.

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$APP_DIR/app.pid"
CLOUDFLARED_PID_FILE="$APP_DIR/cloudflared.pid"
KILLED=0

# ── 1. Force-kill app.py by PID file ──────────────────────────────────────
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[kill] Force-killing app.py PID $PID..."
        kill -9 "$PID" && KILLED=1
    else
        echo "[kill] PID $PID in app.pid is already gone — removing stale file."
    fi
    rm -f "$PID_FILE"
fi

# ── 2. Catch any remaining app.py processes not tracked by PID file ───────
# Scoped to the picgen folder so it cannot kill a newgen app.py running
# elsewhere on the same machine.
while IFS= read -r PID; do
    echo "[kill] Force-killing app.py PID $PID..."
    kill -9 "$PID" 2>/dev/null && KILLED=1
done < <(pgrep -f "picgen/app\.py" 2>/dev/null)

# ── 3. Force-kill Cloudflare tunnel by PID file ───────────────────────────
if [[ -f "$CLOUDFLARED_PID_FILE" ]]; then
    CF_PID=$(cat "$CLOUDFLARED_PID_FILE")
    if kill -0 "$CF_PID" 2>/dev/null; then
        echo "[kill] Force-killing Cloudflare tunnel PID $CF_PID..."
        kill -9 "$CF_PID" && KILLED=1
    else
        echo "[kill] Cloudflare tunnel PID $CF_PID already gone — removing stale file."
    fi
    rm -f "$CLOUDFLARED_PID_FILE"
fi

# ── 4. Catch any remaining cloudflared processes for port 7860 ────────────
pkill -9 -f "cloudflared.*7860" 2>/dev/null && KILLED=1 || true

if [[ $KILLED -eq 1 ]]; then
    echo "[kill] Done."
else
    echo "[kill] Nothing was running."
fi
