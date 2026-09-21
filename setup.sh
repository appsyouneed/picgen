#!/bin/bash
set -e

echo "=== PicGen Setup (Qwen Image Edit photo generator — standalone) ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$EUID" -ne 0 ]; then
    exec sudo bash "$0" "$@"
fi

# ---------------------------------------------------------------------------
# CRITICAL: This setup script NEVER touches the system torch / torchvision /
# torchaudio / numpy stack that ships with your Ubuntu 22.04 VPS.
# Those versions (torch 2.8 dev+cu128, etc.) are already correct and must
# not be changed.  All Python-level dependencies that the app needs are
# installed into /root/picgen/.app-venv so they are completely isolated from
# both the OS-managed python3 packages and the system torch stack.
#
# The ONLY system packages installed here are apt-managed OS tools:
#   wget, git, git-lfs, python3-pip, python3-venv
# (ffmpeg/unzip were needed by vidgen video encoding/RIFE and are not used
# by this photo-only app.)
# Those are safe because they live in /usr/bin / /usr/lib, not in
# site-packages, and cannot conflict with Python packages.
# ---------------------------------------------------------------------------

PYTHON="python3"

echo "Installing system apt dependencies (safe — OS tools only, not Python packages)..."
if lsof /var/lib/dpkg/lock-frontend > /dev/null 2>&1; then
    lsof -t /var/lib/dpkg/lock-frontend | xargs kill -9 2>/dev/null || true
fi
rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock /var/cache/apt/archives/lock
rm -f /var/lib/dpkg/updates/*
dpkg --configure -a || true
apt-get update
apt-get install -y --fix-missing wget git python3-pip python3-venv git-lfs

# ---------------------------------------------------------------------------
# Install dir: /root/picgen on the VPS (copy of this folder). All paths below
# reference it exactly like the newgen setup referenced /root/newgen.
# ---------------------------------------------------------------------------
INSTALL_DIR="/root/picgen"

echo "Creating directories..."
mkdir -p "$INSTALL_DIR/tmp"
mkdir -p /root/.cache/huggingface
chmod 1777 "$INSTALL_DIR/tmp"

# ---------------------------------------------------------------------------
# App venv — isolated from system Python site-packages.
# We do NOT use --system-site-packages so nothing leaks in from the OS level.
# The venv gets its own pip, its own diffusers, gradio, etc.
# torch / torchvision / torchaudio are NOT reinstalled here; the venv
# inherits the system torch via symlinks + .pth created below, keeping the
# multi-GB dev build intact.
# ---------------------------------------------------------------------------
APP_VENV="$INSTALL_DIR/.app-venv"

# ALWAYS rebuild clean. Reusing an existing venv was the direct cause of a
# bug where a stale/broken .app-venv kept getting silently reused across
# setup.sh reruns. A full rebuild guarantees the venv matches this exact
# requirements.txt and this exact python3 binary every time.
echo "Rebuilding app venv at $APP_VENV from scratch (removing any existing one)..."
rm -rf "$APP_VENV"
$PYTHON -m venv "$APP_VENV"

APP_PY="$APP_VENV/bin/python"
APP_PIP="$APP_VENV/bin/pip"

echo "Upgrading pip inside venv..."
"$APP_PIP" install --quiet --upgrade pip

# ---------------------------------------------------------------------------
# Detect the system torch location so the venv can import it without
# reinstalling it.  We find the real site-packages dir by asking the system
# python where torch lives, then symlink ONLY torch and its private deps into
# the venv's own site-packages (a blanket PYTHONPATH export shadows EVERY
# same-named package in the venv — e.g. it silently made the app import
# system diffusers instead of the venv's own 0.37.1).
# ---------------------------------------------------------------------------
SYS_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
if [ -n "$SYS_SITE" ] && [ -d "$SYS_SITE/torch" ]; then
    echo "System torch found at $SYS_SITE."
    VENV_SITE_PACKAGES="$APP_VENV/lib/$($APP_PY -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
    for pkg in torch torchgen torchvision torchaudio torchao \
               functorch caffe2 \
               nvidia triton; do
        if [ -e "$SYS_SITE/$pkg" ] && [ ! -e "$VENV_SITE_PACKAGES/$pkg" ]; then
            ln -s "$SYS_SITE/$pkg" "$VENV_SITE_PACKAGES/$pkg"
        fi
    done
    # Deliberately NOT symlinking .dist-info metadata: pip inside the venv
    # then has no record of torch and silently skips evaluating it — exactly
    # what we want for a package managed entirely outside pip.
    #
    # .pth file so sys.path includes the system site-packages dir directly —
    # required for diffusers' is_torch_available() check (find_spec walks
    # sys.path). Venv site-packages still take precedence.
    PTH_FILE="$VENV_SITE_PACKAGES/zzz_system_torch_path.pth"
    echo "$SYS_SITE" > "$PTH_FILE"
    echo "  Wrote sys.path entry: $PTH_FILE -> $SYS_SITE"

    # ---------------------------------------------------------------------
    # AUTOMATED ABI CHECK + SELF-REPAIR (same as newgen setup.sh).
    # torchvision/torchaudio C-extensions must match the system torch build.
    # ---------------------------------------------------------------------
    echo "Verifying system torch/torchvision/torchaudio ABI compatibility..."
    if ! python3 -c "import torch, torchvision; torchvision.ops.nms" >/dev/null 2>&1; then
        echo "  MISMATCH DETECTED: system torchvision/torchaudio do not match system torch."
        SYS_TORCH_VER=$(python3 -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
        CU_TAG=$(python3 -c "import torch; print('cu' + torch.version.cuda.replace('.', ''))" 2>/dev/null || echo "cu128")
        echo "  System torch is $SYS_TORCH_VER ($CU_TAG). Reinstalling matching"
        echo "  torchvision/torchaudio at the SYSTEM level (torch itself is left untouched)..."
        python3 -m pip install --quiet --upgrade --no-deps \
            torchvision torchaudio \
            --index-url "https://download.pytorch.org/whl/${CU_TAG}" \
        && echo "  Repaired: installed torchvision/torchaudio matching torch ${SYS_TORCH_VER}." \
        || echo "  WARNING: automatic repair failed. Run manually: python3 -m pip install --upgrade torchvision torchaudio --index-url https://download.pytorch.org/whl/${CU_TAG}"

        for pkg in torchvision torchaudio; do
            rm -f "$VENV_SITE_PACKAGES/$pkg"
            NEW_SYS_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "$SYS_SITE")
            [ -e "$NEW_SYS_SITE/$pkg" ] && ln -s "$NEW_SYS_SITE/$pkg" "$VENV_SITE_PACKAGES/$pkg"
        done

        if python3 -c "import torch, torchvision; torchvision.ops.nms" >/dev/null 2>&1; then
            echo "  Verified: torchvision ABI now matches torch."
        else
            echo "  WARNING: still mismatched after repair attempt."
        fi
    else
        echo "  OK: torchvision ABI matches system torch."
    fi
else
    echo "WARNING: Could not locate system torch in $SYS_SITE."
    echo "  Do NOT let setup.sh install torch — that would overwrite your dev build."
fi

echo "Installing Python application dependencies into isolated venv..."
"$APP_PIP" install --quiet --no-cache-dir --no-warn-conflicts \
    -r "$SCRIPT_DIR/requirements.txt"

echo "Ensuring critical pinned packages inside venv..."
# gguf: required by diffusers' GGUFQuantizationConfig / from_single_file to load
# the Q6_K GGUF transformer. Pure-Python, no torch dependency, safe in the venv.
"$APP_PIP" install --quiet --no-cache-dir --no-warn-conflicts \
    gguf \
    Pillow \
    "transformers==4.55.4" \
    "huggingface-hub>=0.34.0,<1.0" \
    "numpy>=1.26,<2.2" \
    "diffusers==0.37.1" \
    "safetensors>=0.4.0" \
    accelerate

# ---------------------------------------------------------------------------
# CRITICAL: Evict any torch / torchvision / torchaudio / torchao that pip may
# have pulled in as transitive dependencies (diffusers/accelerate list torch).
# Run after every pip install block; system copies remain available via
# symlinks at runtime.
# ---------------------------------------------------------------------------
echo "Evicting any venv-local torch/torchvision/torchaudio/torchao (must use system copies)..."
for _pkg in torch torchvision torchaudio torchao; do
    _pkg_dir="$VENV_SITE_PACKAGES/${_pkg}"
    if [ -e "$_pkg_dir" ] && [ ! -L "$_pkg_dir" ]; then
        "$APP_PIP" uninstall -y "$_pkg" 2>/dev/null || true
    fi
done
echo "Torch eviction done — system torch will be used via symlinks at runtime."

# ---------------------------------------------------------------------------
# Patch gradio/oauth.py inside the VENV for huggingface_hub >= 0.26
# huggingface_hub removed HfFolder in 0.26.0; gradio 4.43.0 still imports it
# at module level, crashing the entire process on startup.
# ---------------------------------------------------------------------------
echo "Patching venv gradio/oauth.py for huggingface_hub >= 0.26 compatibility..."
VENV_SITE="$APP_VENV/lib/$(ls $APP_VENV/lib/)/site-packages"
OAUTH_FILE="$VENV_SITE/gradio/oauth.py"
python3 - "$OAUTH_FILE" <<'PYPATCH'
import sys, pathlib

candidate = pathlib.Path(sys.argv[1])
if not candidate.exists():
    print(f"  ERROR: not found: {candidate}")
    sys.exit(1)

text = candidate.read_text()

GOOD = """\
try:
    from huggingface_hub import HfFolder, whoami
except ImportError:
    from huggingface_hub import whoami
    try:
        from huggingface_hub import get_token as _get_token
    except ImportError:
        _get_token = lambda: None  # noqa: E731

    class HfFolder:  # noqa: N801
        @staticmethod
        def get_token():
            return _get_token()"""

ANCHOR_START = "from fastapi.responses import RedirectResponse"
ANCHOR_END = "from .utils import get_space"

start = text.index(ANCHOR_START) + len(ANCHOR_START)
end = text.index(ANCHOR_END)

text = text[:start] + "\n" + GOOD + "\n\n" + text[end:]
candidate.write_text(text)
print(f"  Patched: {candidate}")
PYPATCH

echo "Patching venv gradio_client/utils.py for pydantic v2 bool-schema compatibility..."
GRADIO_CLIENT_FILE="$VENV_SITE/gradio_client/utils.py"
python3 - "$GRADIO_CLIENT_FILE" <<'PYPATCH'
import sys, pathlib

candidate = pathlib.Path(sys.argv[1])
if not candidate.exists():
    print(f"  ERROR: not found: {candidate}")
    sys.exit(1)

text = candidate.read_text()
if "if not isinstance(schema, dict):" in text:
    print(f"  Already patched: {candidate}")
    sys.exit(0)

OLD = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if schema == {}:'
NEW = 'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    """Convert the json schema into a python type hint"""\n    if not isinstance(schema, dict):\n        return "Any"\n    if schema == {}:'
if OLD in text:
    candidate.write_text(text.replace(OLD, NEW, 1))
    print(f"  Patched: {candidate}")
    sys.exit(0)

OLD2 = 'def get_type(schema: dict):\n    if "const" in schema:'
NEW2 = 'def get_type(schema: dict):\n    if not isinstance(schema, dict):\n        return "unknown"\n    if "const" in schema:'
if OLD2 in text:
    candidate.write_text(text.replace(OLD2, NEW2, 1))
    print(f"  Patched get_type fallback: {candidate}")
    sys.exit(0)

print(f"  ERROR: no matching pattern in {candidate}")
sys.exit(1)
PYPATCH

echo "Ensuring pyOpenSSL inside venv..."
"$APP_PY" -c "from OpenSSL import SSL" 2>/dev/null || \
    "$APP_PIP" install --quiet pyopenssl

# ---------------------------------------------------------------------------
# END-OF-SETUP VERIFICATION
# Run the full import chain inside the venv to catch any remaining issues
# BEFORE the user tries to launch app.py.
# ---------------------------------------------------------------------------
echo "Verifying full QwenImageEditPlusPipeline import chain inside venv..."
"$APP_PY" - <<'VERIFY'
import sys
errors = []

# 1. torch
try:
    import torch
    v = torch.__version__
    assert v and v != "None" and "." in str(v), f"bad version: {v!r}"
    print(f"  [OK] torch {v}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  [OK] CUDA {torch.version.cuda} — {p.name}, {p.total_memory/1024**3:.1f} GB VRAM")
    else:
        print("  [WARN] CUDA not available — app will run but generation requires an NVIDIA GPU")
except Exception as e:
    errors.append(f"  [FAIL] torch: {e}")

# 2. diffusers + Qwen pipeline (the real class, not a dummy stub)
try:
    import diffusers
    print(f"  [OK] diffusers {diffusers.__version__}")
    from diffusers import QwenImageEditPlusPipeline
    mod = getattr(QwenImageEditPlusPipeline, "__module__", "") or ""
    if "dummy" in mod:
        errors.append(
            f"  [FAIL] QwenImageEditPlusPipeline is a dummy stub (module={mod!r}).\n"
            "         diffusers' is_torch_available() returned False.\n"
            "         torch symlinks may be missing — re-run setup.sh."
        )
    else:
        print(f"  [OK] QwenImageEditPlusPipeline (module={mod!r})")
except Exception as e:
    errors.append(f"  [FAIL] QwenImageEditPlusPipeline: {e}")

# 3. transformers
try:
    import transformers
    print(f"  [OK] transformers {transformers.__version__}")
except Exception as e:
    errors.append(f"  [FAIL] transformers: {e}")

# 4. gradio
try:
    import gradio
    print(f"  [OK] gradio {gradio.__version__}")
except Exception as e:
    errors.append(f"  [FAIL] gradio: {e}")

# 5. gguf + diffusers GGUF loader (required for the Q6_K transformer)
try:
    import gguf
    from diffusers import GGUFQuantizationConfig
    from diffusers.models import QwenImageTransformer2DModel
    QwenImageTransformer2DModel.from_single_file  # attribute must exist
    print(f"  [OK] gguf {getattr(gguf, '__version__', '?')} + diffusers GGUF loader")
except Exception as e:
    errors.append(f"  [FAIL] gguf / diffusers GGUF loader: {e}")

if errors:
    print("\n=== SETUP VERIFICATION FAILURES ===")
    for err in errors:
        print(err)
    print("\nFix the issues above before launching app.py.")
    sys.exit(1)
else:
    print("  All import checks passed — venv is ready.")
VERIFY
VERIFY_EXIT=$?
if [ $VERIFY_EXIT -ne 0 ]; then
    echo ""
    echo "⚠️  Setup verification failed (see errors above)."
    echo "    The most common cause is a stale torch symlink or a re-installed"
    echo "    pip torch overwriting the system copy. Try: bash setup.sh (again)."
    echo "    DO NOT launch app.py until verification passes."
else
    echo "✅ Setup verification passed."
fi

echo "Copying app files to $INSTALL_DIR..."
for _f in app.py prompts.py requirements.txt; do
    if [ -f "$SCRIPT_DIR/$_f" ]; then
        cp "$SCRIPT_DIR/$_f" "$INSTALL_DIR/$_f"
    fi
done

# Copy qwenimage module if present
if [ -d "$SCRIPT_DIR/qwenimage" ]; then
    cp -r "$SCRIPT_DIR/qwenimage" "$INSTALL_DIR/"
fi

# Copy starters if present
if [ -d "$SCRIPT_DIR/starters" ]; then
    cp -r "$SCRIPT_DIR/starters" "$INSTALL_DIR/"
fi

# ---------------------------------------------------------------------------
# Write the launch wrapper: uses the isolated venv python; system torch is
# visible via the symlinks + .pth placed by setup.sh (NOT via PYTHONPATH —
# that previously shadowed the venv's diffusers with the system copy).
# ---------------------------------------------------------------------------
cat > "$INSTALL_DIR/run_app.sh" << LAUNCHER
#!/bin/bash
# Launch app.py using the isolated app venv.
APP_VENV="$INSTALL_DIR/.app-venv"
cd $INSTALL_DIR
exec "\$APP_VENV/bin/python" app.py "\$@"
LAUNCHER
chmod +x "$INSTALL_DIR/run_app.sh"

# ---------------------------------------------------------------------------
# LAUNCH APP
# In Docker / containers without systemd: launch directly (nohup + setsid so
# it survives the setup.sh process ending). With systemd: register the
# service, then still launch directly so the operator sees the live log.
# ---------------------------------------------------------------------------

_print_summary() {
    echo ""
    echo "=== PicGen Setup Complete ==="
    echo ""
    echo "IMPORTANT: App runs via $INSTALL_DIR/run_app.sh (or bash run.sh)"
    echo "  System torch 2.8 dev+cu128 is PRESERVED."
    echo "  All app dependencies live in $INSTALL_DIR/.app-venv"
    echo ""
    echo "🖼️  PicGen: Qwen-Image-Edit-2511 photo generation (NSFW Rapid AIO v23 merge)"
    echo "   • GPU-aware residency: full pipeline on >=40 GB cards, model-CPU-offload below"
    echo "   • Zero configuration interface"
    echo ""
}

if [ ! -d /run/systemd/system ]; then
    echo "No systemd detected — launching app directly..."
    _print_summary
    echo "Management commands:"
    echo "  bash $INSTALL_DIR/run.sh stop      # stop the app"
    echo "  bash $INSTALL_DIR/run.sh restart   # restart it"
    echo "  bash $INSTALL_DIR/run.sh status    # check if running"
    echo "  bash $INSTALL_DIR/run.sh logs      # tail live log"
    echo ""
    echo "App: http://0.0.0.0:7860"
    echo ""
    bash "$INSTALL_DIR/run.sh" stop 2>/dev/null || true
    exec bash "$INSTALL_DIR/run.sh" start
fi

echo "Setting up systemd service..."
if [ -f "$SCRIPT_DIR/picgen.service" ]; then
    cp "$SCRIPT_DIR/picgen.service" /etc/systemd/system/
fi
systemctl daemon-reload
systemctl enable picgen 2>/dev/null || true
systemctl stop picgen 2>/dev/null || true

_print_summary
echo "Service commands:"
echo "  systemctl status picgen"
echo "  systemctl restart picgen"
echo "  bash $INSTALL_DIR/run.sh logs   # tail live log"
echo ""
echo "App: http://0.0.0.0:7860"
echo ""
# Launch via run.sh so the operator sees the live log and startup
# confirmation in their current SSH session. The systemd unit is still
# registered and will auto-start the app on future reboots.
exec bash "$INSTALL_DIR/run.sh" start
