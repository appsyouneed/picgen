#!/usr/bin/env bash
# ============================================================
#  PicGen VPS Launcher  --  run this on the VPS
#
#  Usage:
#    bash run.sh              # start the app (picgen-only build)
#    bash run.sh stop         # kill it
#    bash run.sh status       # show PID
#    bash run.sh logs         # tail live log
#    bash run.sh restart      # stop then start
#    bash run.sh url          # print current HTTPS URL
#
#  Cloudflare Tunnel is set up automatically on first run.
#  Every start prints the HTTPS URL to both the terminal and picgen.log.
# ============================================================

APP_DIR="$(cd "$(dirname "$0")" && pwd)"
APP="$APP_DIR/app.py"
LOG="$APP_DIR/picgen.log"
PID_FILE="$APP_DIR/app.pid"
CLOUDFLARED_PID_FILE="$APP_DIR/cloudflared.pid"
CLOUDFLARE_URL_FILE="$APP_DIR/cloudflare-url.txt"
APP_VENV="$APP_DIR/.app-venv"
CLOUDFLARED_BIN="/usr/local/bin/cloudflared"
CLOUDFLARED_LOG="$APP_DIR/cloudflared.log"

# ---------------------------------------------------------------------------
# Python: prefer isolated venv, fall back to system python3
# ---------------------------------------------------------------------------
if [ -f "$APP_VENV/bin/python" ]; then
    PYTHON="$APP_VENV/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi

# ---------------------------------------------------------------------------
# Cloudflare Tunnel bootstrap
# Installs cloudflared once, starts it alongside the app on every run.
# ---------------------------------------------------------------------------
ensure_cloudflared() {
    # ── 1. Install binary if missing ──────────────────────────────────────
    if [ ! -f "$CLOUDFLARED_BIN" ]; then
        echo "[cloudflare] cloudflared not found — downloading..."
        ARCH="$(uname -m)"
        case "$ARCH" in
            x86_64)  CF_ARCH="amd64" ;;
            aarch64) CF_ARCH="arm64" ;;
            armv7l)  CF_ARCH="arm"   ;;
            *)       CF_ARCH="amd64" ;;
        esac
        CF_URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CF_ARCH}"
        if command -v curl &>/dev/null; then
            curl -fsSL "$CF_URL" -o "$CLOUDFLARED_BIN"
        elif command -v wget &>/dev/null; then
            wget -q "$CF_URL" -O "$CLOUDFLARED_BIN"
        else
            echo "[cloudflare] ERROR: neither curl nor wget found — cannot download cloudflared."
            return 1
        fi
        chmod +x "$CLOUDFLARED_BIN"
        echo "[cloudflare] cloudflared installed at $CLOUDFLARED_BIN"
    else
        echo "[cloudflare] cloudflared already installed — skipping download."
    fi

    # ── 2. Kill any stale cloudflared process from a previous run ─────────
    if [ -f "$CLOUDFLARED_PID_FILE" ]; then
        OLD_PID=$(cat "$CLOUDFLARED_PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            kill "$OLD_PID" 2>/dev/null
        fi
        rm -f "$CLOUDFLARED_PID_FILE"
    fi
    # Belt-and-suspenders: kill any orphaned cloudflared tunnels for this port
    pkill -f "cloudflared.*7860" 2>/dev/null || true
    sleep 1

    # ── 3. Start tunnel — quick URL mode (no Cloudflare account needed) ───
    rm -f "$CLOUDFLARE_URL_FILE" "$CLOUDFLARED_LOG"
    nohup setsid "$CLOUDFLARED_BIN" tunnel --url http://localhost:7860 \
        > "$CLOUDFLARED_LOG" 2>&1 &
    echo $! > "$CLOUDFLARED_PID_FILE"
    echo "[cloudflare] Tunnel started (PID $!), waiting for URL..."

    # ── 4. Wait up to 30 s for the trycloudflare.com URL to appear ────────
    local CF_URL=""
    for i in $(seq 1 30); do
        sleep 1
        CF_URL=$(grep -oE 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$CLOUDFLARED_LOG" 2>/dev/null | head -1)
        if [ -n "$CF_URL" ]; then
            break
        fi
    done

    if [ -n "$CF_URL" ]; then
        echo "$CF_URL" > "$CLOUDFLARE_URL_FILE"
        echo ""
        echo "╔══════════════════════════════════════════════════════╗"
        echo "║  SECURE ACCESS URL (HTTPS, encrypted, share-safe)   ║"
        echo "║                                                      ║"
        printf  "║  %-52s  ║\n" "$CF_URL"
        echo "║                                                      ║"
        echo "║  Open this in your browser. Changes each restart.   ║"
        echo "╚══════════════════════════════════════════════════════╝"
        echo ""
        # Also write it into the main app log so grep/logs subcommand shows it
        echo "[cloudflare] HTTPS URL: $CF_URL" >> "$LOG"
    else
        echo "[cloudflare] WARNING: tunnel started but URL not detected within 30s."
        echo "[cloudflare] Check $CLOUDFLARED_LOG for details."
    fi
}

stop_cloudflared() {
    if [ -f "$CLOUDFLARED_PID_FILE" ]; then
        PID=$(cat "$CLOUDFLARED_PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[cloudflare] Stopping tunnel (PID $PID)..."
            kill "$PID"
        fi
        rm -f "$CLOUDFLARED_PID_FILE"
    fi
    pkill -f "cloudflared.*7860" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
is_running() {
    [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

do_stop() {
    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "[run] Stopping PID $PID..."
            kill "$PID"
        fi
        rm -f "$PID_FILE"
    fi
    pkill -f "picgen/app\.py" 2>/dev/null || true
    stop_cloudflared
    echo "[run] Stopped."
}

do_start() {
    if is_running; then
        echo "[run] Already running (PID $(cat "$PID_FILE")). Run:  bash run.sh restart"
        # Show current tunnel URL if available
        if [ -f "$CLOUDFLARE_URL_FILE" ]; then
            echo "[run] Current HTTPS URL: $(cat "$CLOUDFLARE_URL_FILE")"
        fi
        exit 0
    fi
    [[ -f "$APP" ]] || { echo "[run] ERROR: $APP not found."; exit 1; }

    echo "[run] Starting app.py in picgen mode..."
    echo "[run] Log  -> $LOG"

    # Start Cloudflare Tunnel first so the URL is ready by the time the app loads
    ensure_cloudflared

    # nohup + setsid: process survives SSH disconnect and terminal close
    nohup setsid "$PYTHON" "$APP" > "$LOG" 2>&1 &
    echo $! > "$PID_FILE"

    echo "[run] Waiting for Gradio startup..."
    for i in $(seq 1 30); do
        sleep 2
        if grep -q "Running on local URL" "$LOG" 2>/dev/null; then
            echo "[run] Started (PID $(cat "$PID_FILE")). Gradio is up."
            echo ""
            # Re-print the URL prominently after app is confirmed up
            if [ -f "$CLOUDFLARE_URL_FILE" ]; then
                CF_URL="$(cat "$CLOUDFLARE_URL_FILE")"
                echo "  ► Open in browser: $CF_URL"
                echo ""
            fi
            echo "[run] ── Live log (Ctrl+C to detach, app keeps running) ──"
            tail -n 80 -f "$LOG"
            return
        fi
        if ! is_running; then
            echo "[run] ERROR: process died. Last log:"
            tail -20 "$LOG"
            exit 1
        fi
    done
    echo "[run] Still starting (taking longer than usual)."
    if [ -f "$CLOUDFLARE_URL_FILE" ]; then
        echo "  ► Open in browser: $(cat "$CLOUDFLARE_URL_FILE")"
    fi
    echo ""
    echo "[run] ── Live log (Ctrl+C to detach, app keeps running) ──"
    tail -n 80 -f "$LOG"
}

# ---------------------------------------------------------------------------
# Parse command
# (This is the picgen-only build: there is no vidgen mode. The original
# newgen run.sh accepted "picgen" as a mode flag — keep accepting it so old
# muscle memory / scripts keep working, but it is a no-op here.)
# ---------------------------------------------------------------------------
CMD="${1:-start}"
MODE="${2:-}"

case "${CMD,,}" in
    picgen|-picgen|--picgen)
        CMD="start"
        ;;
esac

case "$CMD" in
    start)   do_start ;;
    stop)    do_stop ;;
    restart) do_stop; sleep 2; do_start ;;
    status)
        if is_running; then
            echo "[run] App running (PID $(cat "$PID_FILE"))"
        else
            echo "[run] App not running."
        fi
        if [ -f "$CLOUDFLARED_PID_FILE" ] && kill -0 "$(cat "$CLOUDFLARED_PID_FILE")" 2>/dev/null; then
            echo "[run] Cloudflare tunnel running (PID $(cat "$CLOUDFLARED_PID_FILE"))"
            [ -f "$CLOUDFLARE_URL_FILE" ] && echo "[run] HTTPS URL: $(cat "$CLOUDFLARE_URL_FILE")"
        else
            echo "[run] Cloudflare tunnel not running."
        fi
        ;;
    logs)
        echo "[run] Tailing $LOG  (Ctrl+C to stop)..."
        tail -f "$LOG"
        ;;
    url)
        if [ -f "$CLOUDFLARE_URL_FILE" ]; then
            echo "$(cat "$CLOUDFLARE_URL_FILE")"
        else
            echo "[run] No URL on file — is the app running?"
        fi
        ;;
    *)
        echo "Usage: bash run.sh [start|stop|restart|status|logs|url]"
        echo "       bash run.sh picgen          # same as start (picgen is the only mode)"
        echo "       bash run.sh restart         # restart the app"
        echo "       bash run.sh url             # print current HTTPS URL"
        exit 1
        ;;
esac
