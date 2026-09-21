# -*- coding: utf-8 -*-
import os
import shutil
import subprocess
import sys
import random
import math
import tempfile
import warnings
import logging
import time
import gc
import uuid
import threading
import json
import base64
import hashlib
import contextlib
import functools
import queue as _queue
import os
from pathlib import Path
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torchao").setLevel(logging.ERROR)
# The local qwenimage pipeline logs via diffusers' logger under its own module
# name, so silence it too (it emits the noisy "classifier-free guidance is not
# enabled since true_cfg_scale <= 1" / "negative_prompt is passed but ..."
# warnings on every generation).
logging.getLogger("qwenimage").setLevel(logging.ERROR)
logging.getLogger("qwenimage.pipeline_qwenimage_edit_plus").setLevel(logging.ERROR)

class _TorchaoFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage().lower()
        if "torchao" in msg:
            return False
        # Drop the classifier-free guidance / negative_prompt noise from the
        # Qwen pipeline regardless of which logger emits it.
        if "classifier-free guidance is not enabled" in msg:
            return False
        if "negative_prompt is passed but classifier-free guidance" in msg:
            return False
        return True

logging.getLogger().addFilter(_TorchaoFilter())



# ---------------------------------------------------------------------------
# STARTUP MODE
#
# This is the standalone PICGEN build (Qwen Image Edit photo app). The newgen
# app's combined vidgen/picgen startup-mode parser is intentionally NOT used
# here: there is no vidgen tab, no Wan pipeline, no MMAudio, no RIFE, and no
# lip-sync in this app -- only the Photo Editor.
# ---------------------------------------------------------------------------
STARTUP_MODE = "picgen"


# One process-wide lock owns model residency, pipeline mutation, inference, RIFE,
# and CUDA cleanup.  RLock permits helpers to compose without deadlocking.
_gpu_op_lock = threading.RLock()


def _gpu_serialized(fn):
    """Serialize a complete GPU operation, including activation and cleanup."""
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        with _gpu_op_lock:
            return fn(*args, **kwargs)
    return wrapped

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs("/dev/shm/picgen", exist_ok=True)
os.makedirs(os.path.join(SCRIPT_DIR, "tmp", "gradio"), exist_ok=True)

# ---------------------------------------------------------------------------

# EARLY FILE PATCHES — runs before any gradio import.
# sysconfig.get_path("purelib") always returns the site-packages of the
# running interpreter, so this works correctly inside a venv.
# Both patches are idempotent — safe to run on every startup.
# ---------------------------------------------------------------------------
def _patch_gradio_oauth_early():
    import sysconfig, pathlib
    f = pathlib.Path(sysconfig.get_path("purelib")) / "gradio" / "oauth.py"
    if not f.exists():
        return
    t = f.read_text()
    A = "from fastapi.responses import RedirectResponse\n"
    B = "from .utils import get_space"
    if A not in t or B not in t:
        return
    s, e = t.index(A) + len(A), t.index(B)
    good = (
        "try:\n"
        "    from huggingface_hub import HfFolder, whoami\n"
        "except ImportError:\n"
        "    from huggingface_hub import whoami\n"
        "    try:\n"
        "        from huggingface_hub import get_token as _get_token\n"
        "    except ImportError:\n"
        "        _get_token = lambda: None  # noqa: E731\n"
        "\n"
        "    class HfFolder:  # noqa: N801\n"
        "        @staticmethod\n"
        "        def get_token():\n"
        "            return _get_token()\n"
        "\n"
    )
    patched = t[:s] + good + t[e:]
    if patched != t:
        f.write_text(patched)
        print("[EarlyPatch] gradio/oauth.py fixed")
    else:
        print("[EarlyPatch] gradio/oauth.py already clean")

def _patch_gradio_client_early():
    import sysconfig, pathlib
    f = pathlib.Path(sysconfig.get_path("purelib")) / "gradio_client" / "utils.py"
    if not f.exists():
        return
    t = f.read_text()
    if "if not isinstance(schema, dict):" in t:
        print("[EarlyPatch] gradio_client/utils.py already clean")
        return
    for old, new in [
        (
            'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    \"\"\"Convert the json schema into a python type hint\"\"\"\n    if schema == {}:',
            'def _json_schema_to_python_type(schema: Any, defs) -> str:\n    \"\"\"Convert the json schema into a python type hint\"\"\"\n    if not isinstance(schema, dict):\n        return \"Any\"\n    if schema == {}:',
        ),
        (
            'def get_type(schema: dict):\n    if \"const\" in schema:',
            'def get_type(schema: dict):\n    if not isinstance(schema, dict):\n        return \"unknown\"\n    if \"const\" in schema:',
        ),
    ]:
        if old in t:
            f.write_text(t.replace(old, new, 1))
            print("[EarlyPatch] gradio_client/utils.py fixed")
            return

_patch_gradio_oauth_early()
_patch_gradio_client_early()
# ---------------------------------------------------------------------------


try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _aesgcm_check
    del _aesgcm_check
except ImportError:
    print("[SelfHeal] cryptography missing — installing into current venv...")
    _r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "cryptography"],
        capture_output=True, text=True,
    )
    if _r.returncode != 0:
        raise RuntimeError(
            f"cryptography install failed:\n{_r.stderr.strip()}\n"
            "Run setup.sh to rebuild the app venv."
        )
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _aesgcm_check
    del _aesgcm_check
    print("[SelfHeal] cryptography installed OK")

# ---------------------------------------------------------------------------
# Version sanity checks — READ ONLY, never pip-install into the running env.
#
# All runtime dependencies (gradio, diffusers, transformers, numpy, etc.) are
# installed into the isolated app venv by setup.sh.  The self-heal functions
# that used to pip-install packages at runtime have been removed because they
# modified whichever Python environment launched the app — on a system where
# torch 2.8 dev+cu128 is already installed, that caused irreversible version
# downgrades.
#
# If a version check fails below the app exits immediately with a clear
# message so the operator can fix the venv, rather than silently degrading.
# ---------------------------------------------------------------------------

def _self_heal_patch_gradio_oauth():
    """Reapply the HfFolder shim into gradio/oauth.py if needed.
    Patches the file in-place (no pip, no env modification).
    Safe to call multiple times — is a no-op if already patched.

    Uses gradio.__file__ to locate the actual loaded gradio package so this
    works correctly inside a venv (site.getsitepackages() returns system paths
    when called from venv python, causing patches to land in the wrong place).
    """
    try:
        import gradio as _gradio
        candidate = Path(_gradio.__file__).parent / "oauth.py"
        if not candidate.exists():
            print(f"[Patch] gradio/oauth.py not found at {candidate}")
            return
        text = candidate.read_text()

        # Strip any previously botched double-try patch by replacing everything
        # between the RedirectResponse import and 'from .utils import get_space'
        # so repeated runs always produce a clean result.
        ANCHOR_START = "from fastapi.responses import RedirectResponse"
        ANCHOR_END = "from .utils import get_space"
        if ANCHOR_START not in text or ANCHOR_END not in text:
            print(f"[Patch] gradio/oauth.py: anchor lines not found, skipping")
            return

        # Check if already cleanly patched (has try/except but no double-try)
        between_start = text.index(ANCHOR_START) + len(ANCHOR_START)
        between_end = text.index(ANCHOR_END)
        between = text[between_start:between_end]
        if "except ImportError" in between and "    try:\n    from" not in between and "try:\n    try:" not in between:
            print(f"[Patch] gradio/oauth.py already correctly patched")
            return

        GOOD = (
            "\ntry:\n"
            "    from huggingface_hub import HfFolder, whoami\n"
            "except ImportError:\n"
            "    from huggingface_hub import whoami\n"
            "    try:\n"
            "        from huggingface_hub import get_token as _get_token\n"
            "    except ImportError:\n"
            "        _get_token = lambda: None  # noqa: E731\n\n"
            "    class HfFolder:  # noqa: N801\n"
            "        @staticmethod\n"
            "        def get_token():\n"
            "            return _get_token()\n\n"
        )
        text = text[:between_start] + GOOD + text[between_end:]
        candidate.write_text(text)
        print(f"[Patch] gradio/oauth.py patched at {candidate}")
    except Exception as _e:
        print(f"[Patch] gradio oauth patch failed (non-fatal): {_e}")


def _self_heal_patch_gradio_client_utils():
    """Reapply the pydantic v2 bool-schema guard into gradio_client/utils.py.
    Patches the file in-place (no pip, no env modification).
    Safe to call multiple times — is a no-op if already patched.

    Uses gradio_client.__file__ to locate the actual loaded package so this
    works correctly inside a venv.
    """
    try:
        import gradio_client as _gradio_client
        candidate = Path(_gradio_client.__file__).parent / "utils.py"
        if not candidate.exists():
            print(f"[Patch] gradio_client/utils.py not found at {candidate}")
            return
        text = candidate.read_text()
        if "if not isinstance(schema, dict):" in text:
            print(f"[Patch] gradio_client/utils.py already correctly patched")
            return
        OLD = (
            'def _json_schema_to_python_type(schema: Any, defs) -> str:\n'
            '    """Convert the json schema into a python type hint"""\n'
            '    if schema == {}:'
        )
        NEW = (
            'def _json_schema_to_python_type(schema: Any, defs) -> str:\n'
            '    """Convert the json schema into a python type hint"""\n'
            '    if not isinstance(schema, dict):\n'
            '        return "Any"\n'
            '    if schema == {}:'
        )
        if OLD in text:
            candidate.write_text(text.replace(OLD, NEW, 1))
            print(f"[Patch] gradio_client/utils.py patched at {candidate}")
            return
        OLD2 = 'def get_type(schema: dict):\n    if "const" in schema:'
        NEW2 = (
            'def get_type(schema: dict):\n'
            '    if not isinstance(schema, dict):\n'
            '        return "unknown"\n'
            '    if "const" in schema:'
        )
        if OLD2 in text:
            candidate.write_text(text.replace(OLD2, NEW2, 1))
            print(f"[Patch] gradio_client/utils.py (get_type fallback) patched at {candidate}")
            return
        print(f"[Patch] gradio_client/utils.py: no matching pattern found, skipping")
    except Exception as _e:
        print(f"[Patch] gradio_client utils patch failed (non-fatal): {_e}")


def _self_heal_gradio_version():
    """Check gradio version and apply file patches. NEVER pip-installs.
    If the wrong version is found, prints a clear error — the operator must
    rebuild the app venv via setup.sh rather than having the app silently
    modify the Python environment at runtime."""
    _pinned_gradio        = "4.43.0"
    _pinned_gradio_client = "1.3.0"
    try:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        try:
            _cur_gradio = _pkg_version("gradio")
        except PackageNotFoundError:
            _cur_gradio = None
        if _cur_gradio != _pinned_gradio:
            print(
                f"[VersionCheck] WARNING: gradio {_cur_gradio} found, expected {_pinned_gradio}.\n"
                f"  Run setup.sh to rebuild the app venv with the correct versions.\n"
                f"  The app will attempt to continue but may crash due to API mismatches."
            )
        else:
            print(f"[VersionCheck] gradio {_cur_gradio} ?")
        # Always reapply file patches — they are idempotent and touch only
        # the gradio/gradio_client source files inside the venv, not packages.
        _self_heal_patch_gradio_oauth()
        _self_heal_patch_gradio_client_utils()
    except Exception as _e:
        print(f"[VersionCheck] gradio version check failed (non-fatal): {_e}")


def _self_heal_transformers_version():
    """Check transformers version. NEVER pip-installs.
    Qwen-Image-Edit-2511 requires transformers>=4.52.0 for the nested
    text_config composite format. If the wrong version is found, prints
    a clear error pointing to setup.sh."""
    _min_transformers = (4, 52, 0)

    def _parse(v):
        parts = []
        for p in (v or "0").split(".")[:3]:
            num = ""
            for ch in p:
                if ch.isdigit(): num += ch
                else: break
            parts.append(int(num) if num else 0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)

    try:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        try:
            _cur = _pkg_version("transformers")
        except PackageNotFoundError:
            _cur = None

        if _cur is None or _parse(_cur) < _min_transformers:
            _want = ".".join(str(x) for x in _min_transformers)
            print(
                f"[VersionCheck] WARNING: transformers {_cur} found, need >={_want}.\n"
                f"  Run setup.sh to rebuild the app venv with the correct versions."
            )
        else:
            print(f"[VersionCheck] transformers {_cur} ?")
    except Exception as _e:
        print(f"[VersionCheck] transformers version check failed (non-fatal): {_e}")


def _self_heal_torch():
    """Verify torch is installed and has a valid __version__ string.
    NEVER reinstalls torch — the system torch 2.8 dev+cu128 must be preserved.
    Raises RuntimeError if torch is absent or broken so the user gets a clear
    message rather than a confusing downstream crash."""
    try:
        import subprocess as _sp
        _check = _sp.run(
            [sys.executable, "-c",
             "import torch; v=torch.__version__; "
             "assert v and v != 'None' and '.' in str(v), f'bad version: {v!r}'"],
            capture_output=True, text=True,
        )
        if _check.returncode == 0:
            print(f"[VersionCheck] torch ?")
            return
        raise RuntimeError(
            f"torch sanity check failed: {_check.stderr.strip()[:300]}\n"
            "The system torch install may be broken. Do NOT reinstall torch via "
            "pip — that would overwrite the dev build. Check your CUDA/Python env."
        )
    except RuntimeError:
        raise
    except Exception as _e:
        print(f"[VersionCheck] torch check failed (non-fatal): {_e}")


def _self_heal_diffusers_version():
    """Check diffusers version. NEVER pip-installs.
    QwenImageEditPlusPipeline requires diffusers>=0.28.0 (added in that release).
    If the wrong version is found, prints a clear error pointing to setup.sh."""

    def _parse(v):
        parts = []
        for p in (v or "0").split(".")[:3]:
            num = ""
            for ch in p:
                if ch.isdigit(): num += ch
                else: break
            parts.append(int(num) if num else 0)
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)

    try:
        from importlib.metadata import version as _pkg_version, PackageNotFoundError
        try:
            _cur = _pkg_version("diffusers")
        except PackageNotFoundError:
            _cur = None

        if _cur is None or _parse(_cur) < (0, 34, 0):
            print(
                f"[VersionCheck] WARNING: diffusers {_cur} found, need >=0.34.0.\n"
                f"  The Qwen pipeline import may fail.\n"
                f"  Run setup.sh to rebuild the app venv with the correct versions."
            )
        else:
            print(f"[VersionCheck] diffusers {_cur} ?")
    except Exception as _e:
        print(f"[VersionCheck] diffusers version check failed (non-fatal): {_e}")


_self_heal_torch()              # first: transformers crashes if torch.__version__ is None
_self_heal_diffusers_version()
_self_heal_transformers_version()
# Gradio file patches run here — BEFORE gradio is imported — by importing
# gradio temporarily just for __file__ resolution, patching, then letting the
# real import below pick up the corrected files.
_self_heal_gradio_version()     # patches oauth.py and gradio_client/utils.py in-place
_self_heal_diffusers_version()  # re-check after transformers may have rolled deps

# PRE-IMPORT DIFFUSERS BACKEND CACHE RESET
# diffusers evaluates is_torch_available() at module import time and caches
# the result.  If something imported diffusers before torch was on sys.path
# (e.g. a transitive import during the gradio patch functions above), the
# cache records "torch not available" and every subsequent pipeline class
# resolves to a dummy stub — causing the misleading "PyTorch library not
# found" error even though torch is perfectly installed.
#
# We reset the cache here, AFTER confirming torch is importable (_self_heal_torch
# would have raised if it weren't), and BEFORE the real 'import diffusers'
# below, so diffusers evaluates is_torch_available() fresh.
# ---------------------------------------------------------------------------
try:
    if "diffusers" in sys.modules or "diffusers.utils.import_utils" in sys.modules:
        _diu = sys.modules.get("diffusers.utils.import_utils")
        if _diu is not None:
            # Force the module-level bool back to True.
            if hasattr(_diu, "_torch_available"):
                _diu._torch_available = True
            # Clear any lru_cache wrapping is_torch_available.
            _fn = getattr(_diu, "is_torch_available", None)
            if _fn is not None and hasattr(_fn, "cache_clear"):
                _fn.cache_clear()
            # Also reset the transformers-availability flag since transformers
            # has the same caching pattern and is checked alongside torch.
            if hasattr(_diu, "_transformers_available"):
                _diu._transformers_available = True
            _fn2 = getattr(_diu, "is_transformers_available", None)
            if _fn2 is not None and hasattr(_fn2, "cache_clear"):
                _fn2.cache_clear()
            print("[StartupFix] diffusers backend availability cache reset.")
except Exception as _cache_reset_e:
    print(f"[StartupFix] diffusers cache reset (non-fatal): {_cache_reset_e}")

os.environ.update({
    "TMPDIR": "/dev/shm/picgen",
    "TEMP": "/dev/shm/picgen",
    "TMP": "/dev/shm/picgen",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "ABSL_MIN_LOG_LEVEL": "3",
    "GRPC_VERBOSITY": "ERROR",
    "TOKENIZERS_PARALLELISM": "true",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,backend:cudaMallocAsync",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_EXPERIMENTAL_WARNING": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
    "HF_HOME": "/root/.cache/huggingface",
    "CUDA_LAUNCH_BLOCKING": "0",
    "OMP_NUM_THREADS": "8",
    # Suppress pip noise globally for all subprocess pip calls:
    # root-user warning, version-check nag, deprecation notices, conflict chatter.
    "PIP_ROOT_USER_ACTION": "ignore",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PIP_NO_WARN_CONFLICTS": "1",
    "PYTHONWARNINGS": "ignore::DeprecationWarning,ignore::UserWarning,ignore::FutureWarning",
    # Force onnxruntime to use CPU if CUDA libs missing (for rembg background removal)
    "ORT_FORCE_CPU": "1",
})


import numpy as np  # (import cv2 removed: picgen uses no OpenCV features)
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True

logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

import warnings as _warnings
# torch/transformers/diffusers install their own warnings filters during
# import, which can override the blanket warnings.filterwarnings("ignore")
# set at the top of this file. This specific message ("Unable to import
# `torchao` Tensor objects...") is emitted via warnings.warn (not logging),
# so the _TorchaoFilter logging.Filter above never catches it. Re-assert
# suppression here, after the heavy imports, targeting it by message text.
_warnings.filterwarnings("ignore", message=r".*torchao.*Tensor objects.*")

from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file


import gradio as gr

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

sys.path.insert(0, SCRIPT_DIR)



_media_store: dict[str, tuple[bytes, str]] = {}   # key -> (data, filename)
_media_store_lock = threading.Lock()



import secrets as _secrets
from cryptography.hazmat.primitives.ciphers.aead import AESGCM as _AESGCM
from cryptography.hazmat.primitives import hashes as _hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF as _HKDF

_log_queue: "_queue.Queue[str]" = _queue.Queue(maxsize=2000)

_builtin_print = print

def print(*args, **kwargs):  # noqa: A001
    """Drop-in print replacement: writes to stdout AND pushes to _log_queue."""
    sep = kwargs.get("sep", " ")
    line = sep.join(str(a) for a in args)
    _builtin_print(*args, **kwargs)
    try:
        _log_queue.put_nowait(line)
    except Exception:
        pass


def _derive_media_key(browser_secret_hex: str) -> bytes:
    """Derive a 32-byte AES-256-GCM key from the browser's localStorage secret.

    Uses HKDF-SHA256 with a fixed salt and info so the same secret always
    produces the same key — but the key is never transmitted, only the secret.
    """
    try:
        secret_bytes = bytes.fromhex(browser_secret_hex)
    except ValueError:
        return _secrets.token_bytes(32)
    hkdf = _HKDF(
        algorithm=_hashes.SHA256(),
        length=32,
        salt=b"newgen-media-v1",
        info=b"aes-256-gcm-media",
    )
    return hkdf.derive(secret_bytes)


def _encrypt_bytes(data: bytes, key_bytes: bytes) -> bytes:
    """Encrypt data with AES-256-GCM. Returns 12-byte nonce + ciphertext."""
    nonce = _secrets.token_bytes(12)
    aesgcm = _AESGCM(key_bytes)
    ciphertext = aesgcm.encrypt(nonce, data, None)
    return nonce + ciphertext


def _derive_log_key(browser_secret_hex: str) -> bytes:
    """Derive a 32-byte AES-256-GCM key for log line encryption."""
    try:
        secret_bytes = bytes.fromhex(browser_secret_hex)
    except ValueError:
        return _secrets.token_bytes(32)
    hkdf = _HKDF(
        algorithm=_hashes.SHA256(),
        length=32,
        salt=b"newgen-logs-v1",
        info=b"aes-256-gcm-logs",
    )
    return hkdf.derive(secret_bytes)


def _encrypt_log_line(line: str, key_bytes: bytes) -> str:
    """Encrypt a log line and return base64(nonce+ciphertext)."""
    nonce = _secrets.token_bytes(12)
    aesgcm = _AESGCM(key_bytes)
    ct = aesgcm.encrypt(nonce, line.encode("utf-8"), None)
    return base64.b64encode(nonce + ct).decode()


def _media_store_put(data: bytes, filename: str) -> str:
    """Store bytes in RAM, return the /media/ URL to serve them."""
    key = uuid.uuid4().hex
    with _media_store_lock:
        _media_store[key] = (data, filename)
    return f"/media/{key}/{filename}"


def _media_store_get(key: str):
    """Return (bytes, filename) or None."""
    with _media_store_lock:
        return _media_store.get(key)


def _media_store_release(url_or_key: str):
    """Remove one entry by /media/<key>/... URL or bare key."""
    if not url_or_key:
        return
    key = url_or_key.split("/")[2] if url_or_key.startswith("/media/") else url_or_key
    with _media_store_lock:
        _media_store.pop(key, None)


def _media_store_release_prefix(prefix: str):
    """Remove all entries whose filename starts with prefix."""
    with _media_store_lock:
        remove = [k for k, (_, fn) in _media_store.items() if fn.startswith(prefix)]
        for k in remove:
            del _media_store[k]


def _media_name(kind: str, extension: str, index: int = None) -> str:
    """Build a collision-free download filename (no path, no folder)."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    token = uuid.uuid4().hex[:8]
    suffix = f"_{index:02d}" if index is not None else ""
    ext = extension if extension.startswith(".") else f".{extension}"
    return f"{kind}_{stamp}_{token}{suffix}{ext}"


from prompts import (
    solo_prompts_dict, couple_man_unseen_prompts_dict, couple_man_seen_prompts_dict,
    multiple_women_prompts_dict, multiple_man_unseen_prompts_dict, multiple_man_seen_prompts_dict, multistep_prompts_dict,
    update_solo_prompt, update_couple_man_unseen_prompt, update_couple_man_seen_prompt,
    update_multiple_women_prompt, update_multiple_man_unseen_prompt, update_multiple_man_seen_prompt, update_multistep_prompt,
)

FORCE_DUAL_RESIDENT = False
AGGRESSIVE_OPTIMIZATION = True

_gpu_count = torch.cuda.device_count()
if _gpu_count < 1:
    raise RuntimeError("No CUDA device visible  this app requires a GPU.")

# ---------------------------------------------------------------------------
# GPU PROFILE — detected once at startup, used throughout for adaptive
# behaviour (VRAM budgets, wheel index selection, offload strategy, etc.)
#
# _GPU_VRAM_GB    : total VRAM of the primary WAN device in GiB (float)
# _GPU_CC         : (major, minor) compute capability tuple, e.g. (12, 0)
# _GPU_NAME       : human-readable device name string
# _GPU_HIGH_VRAM  : True when VRAM >= 40 GB  (Blackwell 6000 Pro = 95 GB)
#                   False for consumer cards  (RTX 5090 = 32 GB)
# _GPU_SM_STR     : semicolon-separated arch list for TORCH_CUDA_ARCH_LIST
#                   — includes the actual device arch, capped to what mmcv
#                   supports at build time (sm_120 / Blackwell is NOT in
#                   older mmcv sources so we keep it out of that var).
# ---------------------------------------------------------------------------
def _detect_gpu_profile(device_idx: int = 0) -> dict:
    """Return a dict with VRAM, compute capability, name, and derived flags."""
    try:
        props = torch.cuda.get_device_properties(device_idx)
        vram_gb = props.total_memory / (1024 ** 3)
        cc = (props.major, props.minor)
        name = props.name
    except Exception:
        vram_gb = 0.0
        cc = (0, 0)
        name = "unknown"

    high_vram = vram_gb >= 40.0

    # Build a TORCH_CUDA_ARCH_LIST that covers common arches up to and
    # including this device, but never exceeds what mmcv's cpp_extension.py
    # knows about (it doesn't recognise sm_120 / Blackwell yet).
    # RTX 5090  ? sm_89  (Ada Lovelace / compute 8.9)
    # Blackwell ? sm_120 (compute 12.0) — excluded from mmcv arch list,
    #             but included separately where PyTorch itself needs it.
    _base_arches = ["8.0", "8.6", "8.9", "9.0"]
    # Cap to arches <= this device's compute capability for the mmcv build,
    # but never include 12.x (not in mmcv's table).
    cc_float = cc[0] + cc[1] / 10.0
    mmcv_arches = [a for a in _base_arches if float(a) <= min(cc_float, 9.0)]
    if not mmcv_arches:
        mmcv_arches = ["8.0"]
    sm_str = ";".join(mmcv_arches)

    # PyTorch wheel index: cu128 for Blackwell (sm_120), cu130 for sm_89
    # (Ada / RTX 5090) and anything else modern.
    if cc[0] >= 12:
        torch_index = "https://download.pytorch.org/whl/cu128"
    else:
        torch_index = "https://download.pytorch.org/whl/cu130"

    profile = {
        "vram_gb": vram_gb,
        "cc": cc,
        "name": name,
        "high_vram": high_vram,
        "sm_str": sm_str,
        "torch_index": torch_index,
    }
    print(
        f"[GPUProfile] {name} | VRAM {vram_gb:.1f} GB | "
        f"sm_{cc[0]}{cc[1]} | high_vram={high_vram} | "
        f"torch_index={torch_index} | mmcv_arches={sm_str}"
    )
    return profile

GPU_PROFILE = _detect_gpu_profile(0)

GPU_VRAM_GB   = GPU_PROFILE["vram_gb"]
GPU_CC        = GPU_PROFILE["cc"]
GPU_NAME      = GPU_PROFILE["name"]
GPU_HIGH_VRAM = GPU_PROFILE["high_vram"]

# Single device serves the whole picgen app (Qwen pipeline only).
PIC_DEVICE = "cuda:0"
PIC_QUEUE_ID = "gpu"
device = torch.device(PIC_DEVICE)


PICGEN_MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
BASE_MODEL_LOCAL_PATH = os.path.join(PICGEN_MODELS_DIR, "Qwen-Image-Edit-2511")
NSFW_WEIGHTS_LOCAL_PATH = os.path.join(PICGEN_MODELS_DIR, "rapid-aio", "v23", "Qwen-Rapid-AIO-NSFW-v23.safetensors")

# ---------------------------------------------------------------------------
# Q6_K GGUF transformer (the NSFW-merged Rapid-AIO model, prequantized).
#
# This REPLACES the old "load base transformer + merge NSFW .safetensors + cast
# to fp8" path. The GGUF is a single ~17 GB file whose weights stay in a low-
# memory dtype and are dynamically dequantized to bf16 during each forward pass
# (diffusers' GGUF support), so the transformer's resident footprint is small
# enough to sit fully on a 24 GB RTX 4090 with the VAE, while the text encoder
# streams in once per generation. No fp8 hang, no torchao, no 38 GB bf16.
#
# Downloaded once to models/rapid-aio-gguf/v23/ on first run; reused after.
# ---------------------------------------------------------------------------
GGUF_REPO_ID = "Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF"
GGUF_FILENAME = "v23/Qwen-Rapid-NSFW-v23_Q6_K.gguf"
GGUF_LOCAL_DIR = os.path.join(PICGEN_MODELS_DIR, "rapid-aio-gguf")
GGUF_LOCAL_PATH = os.path.join(GGUF_LOCAL_DIR, "v23", "Qwen-Rapid-NSFW-v23_Q6_K.gguf")

# ---------------------------------------------------------------------------
# OOM-safe pipeline move helpers
#
# Moving a 16-18 GB transformer to GPU in one shot while the CUDA caching
# allocator still holds fragmented pages from the NSFW weight merge causes
# OOM on 32 GB cards (RTX 5090).  Moving one component at a time with a
# full GC + empty_cache + synchronize between each component keeps the peak
# allocation at one component at a time, eliminating the spike.
#
# These helpers are used at startup (picgen mode) and before each storage
# clear so the move path is equally safe.
# ---------------------------------------------------------------------------
def _safe_move_to_device(pipe, target_device):
    """Move a pipeline to target_device one component at a time.

    Each component move is followed by gc.collect() + empty_cache() +
    synchronize() so the CUDA caching allocator never sees a spike larger
    than the single largest component.  This prevents OOM on 32 GB cards
    (RTX 5090) where the transformer alone is ~16-18 GB and leftover
    allocator fragmentation can push the peak over the limit.
    """
    components = []
    if hasattr(pipe, 'transformer'):
        components.append(('transformer', pipe.transformer))
    if hasattr(pipe, 'text_encoder'):
        components.append(('text_encoder', pipe.text_encoder))
    if hasattr(pipe, 'vae'):
        components.append(('vae', pipe.vae))
    # Any additional sub-models (transformer_2, image_encoder, etc.)
    for attr in ('transformer_2', 'image_encoder', 'image_projection'):
        if hasattr(pipe, attr) and getattr(pipe, attr) is not None:
            components.append((attr, getattr(pipe, attr)))

    for name, component in components:
        # Preflight admission: before moving a component to GPU, verify the
        # device has enough free VRAM for its parameter+buffer bytes plus a
        # safety headroom for the transfer, activations, and allocator
        # fragmentation. Raising here (BEFORE the .to() call) turns an
        # unavoidable low-level torch.OutOfMemoryError deep inside the move
        # into a deterministic, actionable error while the previous pipeline
        # can still be rolled back by the caller (_activate_model).
        if target_device != "cpu":
            try:
                comp_bytes = 0
                for p in component.parameters():
                    comp_bytes += p.numel() * p.element_size()
                for b in component.buffers():
                    comp_bytes += b.numel() * b.element_size()
            except Exception:
                comp_bytes = 0
            # Headroom: transfer needs the source copy briefly resident too,
            # plus activation/workspace + allocator slack. 1.5x the component
            # plus a 1 GB floor is a conservative, quality-neutral guard.
            required = int(comp_bytes * 1.5) + (1024 ** 3)
            try:
                free_bytes = torch.cuda.mem_get_info()[0]
            except Exception:
                free_bytes = None
            if free_bytes is not None and free_bytes < required:
                raise torch.cuda.OutOfMemoryError(
                    f"Insufficient VRAM on {target_device} to move '{name}': "
                    f"need ~{required / 1024**3:.1f} GB "
                    f"(component ~{comp_bytes / 1024**3:.1f} GB + headroom), "
                    f"only {free_bytes / 1024**3:.1f} GB free. "
                    f"Aborting move before OOM so the previous model can be restored."
                )
        print(f"    [{target_device}] moving {name}...")
        component.to(target_device)
        gc.collect()
        torch.cuda.empty_cache()
        if target_device != "cpu":
            torch.cuda.synchronize()
            free_gb = torch.cuda.mem_get_info()[0] / 1024**3
            print(f"    [{target_device}] {name} moved — {free_gb:.1f} GB free")


def _safe_offload_to_cpu(pipe, label="pipeline"):
    """Move a pipeline to CPU and fully reclaim VRAM."""
    print(f"    Offloading {label} to CPU...")
    pipe.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_gb = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"    {label} offloaded — {free_gb:.1f} GB VRAM free")


# ---------------------------------------------------------------------------
# fp8 transformer materialization
#
# The Qwen-Image-Edit-2511 transformer is ~38 GB in bf16 — it does NOT fit on
# a 32 GB card (RTX 5090).  But the NSFW checkpoint that is merged into it
# (Qwen-Rapid-AIO-NSFW-v23) stores the transformer weights in fp8 (F8_E4M3):
# that is the model author's intended runtime precision, and at fp8 the
# transformer is ~19 GB — which fits with room for activations.
#
# The load path copies the fp8 checkpoint values into the bf16 module, which
# keeps the module bf16-sized (38 GB) and causes the OOM.  Casting the
# transformer's Linear weights to float8_e4m3fn AFTER the merge drops resident
# size to ~19 GB.  This is NOT a quality reduction — it is the precision the
# NSFW weights were trained/quantized at.  VAE and text encoder stay bf16
# (small, and need bf16 for numerical stability).
#
# Guarded by NEWGEN_QWEN_FP8 (default "1"); set to "0" to keep bf16 on cards
# that have >=40 GB (e.g. the Blackwell 6000 Pro), where full bf16 fits and is
# marginally faster.
# ---------------------------------------------------------------------------
_QWEN_FP8_ENABLED = os.environ.get("NEWGEN_QWEN_FP8", "1") == "1"


def _cast_transformer_to_fp8(pipe):
    """Cast the transformer's large Linear weights to float8_e4m3fn.

    Only weights (2D Linear params) are cast; biases, norms, and non-Linear
    params stay bf16 so the module remains numerically stable. Returns the
    resident byte size after the cast for logging. Safe no-op if fp8 is
    unavailable or disabled.
    """
    if not _QWEN_FP8_ENABLED:
        return None
    if not hasattr(torch, "float8_e4m3fn"):
        print("    [fp8] torch has no float8_e4m3fn — keeping bf16 (may not fit <40 GB cards).")
        return None
    fp8 = torch.float8_e4m3fn
    transformer = getattr(pipe, "transformer", None)
    if transformer is None:
        return None
    cast_count = 0
    for module in transformer.modules():
        w = getattr(module, "weight", None)
        # Only cast 2D Linear-style weight matrices — the bulk of the size.
        if isinstance(module, torch.nn.Linear) and w is not None and w.dim() == 2:
            if w.dtype != fp8:
                module.weight = torch.nn.Parameter(
                    w.data.to(fp8), requires_grad=False
                )
                cast_count += 1
    gc.collect()
    nbytes = 0
    for p in transformer.parameters():
        nbytes += p.numel() * p.element_size()
    for b in transformer.buffers():
        nbytes += b.numel() * b.element_size()
    print(f"    [fp8] cast {cast_count} Linear weights to float8_e4m3fn — "
          f"transformer now ~{nbytes / 1024**3:.1f} GB resident")
    return nbytes


# ---------------------------------------------------------------------------
# TEMPORARY VRAM PROBE  (enable with NEWGEN_VRAM_PROBE=1)
#
# Reports the exact resident byte size of every pipeline component WITHOUT
# moving anything to the GPU, so it can never OOM. For each component it prints:
#   - the component's own size (params + buffers), by dtype
#   - the running cumulative total if all components so far were co-resident
# Then it prints the card's total VRAM and a verdict on what fits.
#
# This is a diagnostic aid — it does not change how the app loads models.
# Remove the probe block (and this helper) once sizing is understood.
# ---------------------------------------------------------------------------
def _probe_pipeline_vram(pipe, label="pipeline", exit_after=True):
    """Print exact per-component + cumulative sizes; optionally exit."""
    def _bytes_and_dtypes(module):
        total = 0
        dtypes = {}
        for p in module.parameters():
            n = p.numel() * p.element_size()
            total += n
            k = str(p.dtype)
            dtypes[k] = dtypes.get(k, 0) + n
        for b in module.buffers():
            n = b.numel() * b.element_size()
            total += n
            k = str(b.dtype)
            dtypes[k] = dtypes.get(k, 0) + n
        return total, dtypes

    order = []
    for attr in ("transformer", "transformer_2", "text_encoder",
                 "text_encoder_2", "vae", "image_encoder", "image_projection"):
        comp = getattr(pipe, attr, None)
        if comp is not None and hasattr(comp, "parameters"):
            order.append((attr, comp))

    try:
        total_vram = torch.cuda.get_device_properties(0).total_memory
    except Exception:
        total_vram = 0

    print("=" * 70)
    print(f"[VRAM PROBE] {label} — measured resident sizes (no GPU load):")
    print("=" * 70)
    cumulative = 0
    for name, comp in order:
        nbytes, dtypes = _bytes_and_dtypes(comp)
        cumulative += nbytes
        dt_str = ", ".join(f"{k.replace('torch.','')}: {v/1024**3:.2f}GB"
                            for k, v in sorted(dtypes.items()))
        print(f"  {name:16s} = {nbytes/1024**3:7.2f} GB   [{dt_str}]")
        print(f"  {'':16s}   cumulative if co-resident: {cumulative/1024**3:7.2f} GB")
    print("-" * 70)
    print(f"  ALL COMPONENTS CO-RESIDENT: {cumulative/1024**3:.2f} GB")
    if total_vram:
        print(f"  CARD TOTAL VRAM:            {total_vram/1024**3:.2f} GB")
        headroom = (total_vram - cumulative) / 1024**3
        if headroom >= 4:
            print(f"  VERDICT: fits fully resident with {headroom:.1f} GB free for activations.")
        elif headroom >= 0:
            print(f"  VERDICT: barely fits ({headroom:.1f} GB free) — likely OOM under activations.")
        else:
            print(f"  VERDICT: does NOT fit fully resident (short by {-headroom:.1f} GB). "
                  f"Needs per-component offload; largest single component must fit alone.")
        # Largest single component (the minimum peak for offload-based execution)
        if order:
            biggest = max(order, key=lambda x: _bytes_and_dtypes(x[1])[0])
            bname, bcomp = biggest
            bbytes = _bytes_and_dtypes(bcomp)[0]
            print(f"  Largest single component: {bname} = {bbytes/1024**3:.2f} GB "
                  f"({'fits alone' if bbytes < total_vram else 'too big even alone'}).")
    print("=" * 70)
    if exit_after:
        print("[VRAM PROBE] exit_after=True — stopping before any GPU load. "
              "Set NEWGEN_VRAM_PROBE=0 to run normally.")
        os._exit(0)


_VRAM_PROBE = os.environ.get("NEWGEN_VRAM_PROBE", "0") == "1"

# ---------------------------------------------------------------------------
# model-CPU-offload management (shared by picgen AND vidgen)
#
# Measured on the RTX 5090 (31.4 GB usable):
#   Qwen picgen fully resident = 34.7 GB  -> does NOT fit
#   Wan vidgen  fully resident > 32 GB    -> does NOT fit (transformer 26.7 GB
#                                            + transformer_2 + VAE + text enc)
#
# diffusers' enable_model_cpu_offload() keeps every component on CPU and moves
# each to GPU ONLY while it executes, then back. Peak VRAM = the largest single
# component + activations, not the co-resident total. Crucially, the active
# denoising transformer stays resident for the WHOLE denoise loop, so there is
# no per-step transfer penalty — steady-state speed is unchanged. Precision is
# unchanged (fp8 transformer for Qwen, bf16 elsewhere), so no quality/prompt
# adherence loss.
#
# On a >=40 GB card (Blackwell 6000 Pro, etc.) everything fits, so we skip
# offload entirely and keep the pipeline FULLY resident for maximum speed.
#
# _offload_state[label] records whether hooks are installed for that pipeline
# so we can cleanly tear them down before the OTHER model takes the GPU.
# ---------------------------------------------------------------------------
_offload_state = {"pic": False}
FULL_RESIDENCY_VRAM_GB = 40.0   # cards >= this hold everything; below use offload



def _device_total_vram_gb(dev):
    try:
        idx = torch.device(dev).index or 0
        return torch.cuda.get_device_properties(idx).total_memory / 1024**3
    except Exception:
        return 0.0


def _enable_offload(pipe, device, label="pic"):
    """Place a pipeline on GPU: full residency on big cards, else model-CPU-offload.

    label is 'pic'. Returns nothing; records state in _offload_state.
    """
    total_vram = _device_total_vram_gb(device)
    # Big card: keep everything resident, fastest path, no hooks.
    if total_vram >= FULL_RESIDENCY_VRAM_GB:
        _safe_move_to_device(pipe, device)
        _offload_state[label] = False
        print(f"    [{label}] full residency on {device} (card {total_vram:.0f} GB).")
        return
    # Small card: stream components on demand.
    try:
        pipe.to("cpu")
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # picgen (Qwen): model-CPU-offload. The Q6_K GGUF transformer (~13-15 GB
    #   resident, weights stay quantized) plus image-edit activations fit easily
    #   on a 24 GB card, so whole-component offload keeps the transformer
    #   resident for the WHOLE denoise loop (fast, no per-step transfer penalty).
    #   Only the ~15.5 GB text encoder swaps to GPU once to encode the prompt,
    #   then back to CPU. On >=40 GB cards the big-card branch above keeps
    #   everything resident instead.
    pipe.enable_model_cpu_offload(device=device)
    _offload_state[label] = True
    print(f"    [{label}] model-CPU-offload enabled on {device} "
          f"(card {total_vram:.0f} GB — GGUF transformer resident during denoise, "
          f"text encoder streams once).")


def _disable_offload(pipe, label):
    """Remove model-CPU-offload hooks (if any) and return the pipeline to CPU.

    Frees VRAM before storage clears so stale hooks can't fire.
    """
    if not _offload_state.get(label):
        # No hooks installed (big card / full residency) — plain CPU move.
        _safe_offload_to_cpu(pipe, label.title())
        return
    try:
        from accelerate.hooks import remove_hook_from_module
        for attr in ("transformer", "transformer_2", "text_encoder",
                     "text_encoder_2", "vae", "image_encoder"):
            comp = getattr(pipe, attr, None)
            if comp is not None and hasattr(comp, "parameters"):
                try:
                    remove_hook_from_module(comp, recurse=True)
                except Exception:
                    pass
    except Exception as exc:
        print(f"    [{label}] hook removal fallback ({exc})")
    for _attr in ("_all_hooks", "_offload_gpu_id", "_offload_device"):
        if hasattr(pipe, _attr):
            try:
                setattr(pipe, _attr, [] if _attr == "_all_hooks" else None)
            except Exception:
                pass
    try:
        pipe.to("cpu")
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    _offload_state[label] = False
    print(f"    [{label}] model-CPU-offload disabled — pipeline returned to CPU.")


# Backwards-compatible thin wrappers used by the picgen startup path.
def _enable_pic_offload(pipe, device=None):
    _enable_offload(pipe, device or PIC_DEVICE, "pic")


def _disable_pic_offload(pipe):
    _disable_offload(pipe, "pic")



print(" PICGEN MODE: Loading Qwen to GPU first for immediate use...")
torch.cuda.set_device(PIC_DEVICE)

print(" AGGRESSIVE LOADING: Qwen Image Edit pipeline...")
start_qwen = time.time()

# ---------------------------------------------------------------------------
# STEP 1: download the Q6_K GGUF transformer once (or reuse if already present).
# ---------------------------------------------------------------------------
if not os.path.exists(GGUF_LOCAL_PATH):
    print(f"Downloading Q6_K GGUF transformer to {GGUF_LOCAL_PATH} (first run only)...", flush=True)
    os.makedirs(os.path.dirname(GGUF_LOCAL_PATH), exist_ok=True)
    _gguf_dl = hf_hub_download(
        repo_id=GGUF_REPO_ID,
        filename=GGUF_FILENAME,
        local_dir=GGUF_LOCAL_DIR,
    )
    # hf_hub_download returns the resolved path; make GGUF_LOCAL_PATH point at it
    # in case the repo layout differs from our expected subfolder.
    if os.path.abspath(_gguf_dl) != os.path.abspath(GGUF_LOCAL_PATH):
        GGUF_LOCAL_PATH = _gguf_dl
    print(f"  GGUF downloaded: {GGUF_LOCAL_PATH}", flush=True)
else:
    print(f"Q6_K GGUF already present: {GGUF_LOCAL_PATH}", flush=True)

# ---------------------------------------------------------------------------
# STEP 2: load the base Edit-Plus pipeline for its VAE / text encoder /
# tokenizer / processor / scheduler (these are NOT in the GGUF, which contains
# only the transformer). The base transformer weights are loaded here but will
# be immediately replaced by the GGUF transformer below, so this is just to get
# the correct surrounding components and config.
# ---------------------------------------------------------------------------
model_index_path = os.path.join(BASE_MODEL_LOCAL_PATH, "model_index.json")
if not os.path.exists(model_index_path):
    print(f"Downloading Qwen base pipeline (VAE/text-encoder/etc.) to {BASE_MODEL_LOCAL_PATH}...", flush=True)
    os.makedirs(PICGEN_MODELS_DIR, exist_ok=True)
    pic_pipe = QwenImageEditPlusPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit-2511",
        torch_dtype=torch.bfloat16,
        cache_dir=BASE_MODEL_LOCAL_PATH,
        use_safetensors=True,
    )
else:
    pic_pipe = QwenImageEditPlusPipeline.from_pretrained(
        BASE_MODEL_LOCAL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=True,
    )

# ---------------------------------------------------------------------------
# STEP 3: load the GGUF as the transformer and swap it into the pipeline.
# GGUF weights stay in a low-memory dtype and are dequantized to bf16 during
# each forward pass, so there is no fp8 hang and the resident footprint fits a
# 24 GB card. This replaces the old NSFW .safetensors merge + fp8 cast entirely.
# ---------------------------------------------------------------------------
print("Loading Q6_K GGUF transformer (dynamic dequant to bf16)...", flush=True)
from diffusers import GGUFQuantizationConfig
from diffusers.models import QwenImageTransformer2DModel as _QwenTransformer

# Free the base transformer we just loaded before building the GGUF one so we
# never hold two transformers in memory at once.
try:
    _old_transformer = pic_pipe.transformer
    pic_pipe.transformer = None
    del _old_transformer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
except Exception:
    pass

# config points diffusers at the transformer's architecture config so it can
# build the module before loading the GGUF weights into it. Using the HF repo
# id (resolved from the local cache when already downloaded) avoids ambiguity
# with the cache_dir snapshot layout of BASE_MODEL_LOCAL_PATH.
_gguf_transformer = _QwenTransformer.from_single_file(
    GGUF_LOCAL_PATH,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    config="Qwen/Qwen-Image-Edit-2511",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
pic_pipe.transformer = _gguf_transformer
print("  GGUF transformer loaded and attached to pipeline.", flush=True)

gc.collect()
torch.cuda.empty_cache()
torch.cuda.synchronize()

pic_pipe.vae.enable_tiling()
pic_pipe.vae.enable_slicing()

# TEMPORARY: measure exact per-component VRAM needs without loading to GPU.
if _VRAM_PROBE:
    _probe_pipeline_vram(pic_pipe, label="Qwen picgen pipeline", exit_after=True)

# Residency with the Q6_K GGUF transformer:
#   GGUF transformer resident ~13-15 GB (weights stay quantized, dequant to
#     bf16 per forward) + VAE ~0.2 GB + text encoder ~15.5 GB bf16
#     ≈ 29-31 GB co-resident. That FITS on >=40 GB cards but NOT on a 24 GB
#     RTX 4090 — the Qwen2.5-VL text encoder alone is the reason.
#
#   >=40 GB cards: keep everything resident, no hooks, fastest.
#   24-32 GB cards (RTX 4090 / 3090 / 5090): use diffusers model-CPU-offload.
#     Unlike the old bf16/fp8 path, the GGUF transformer is small enough to stay
#     GPU-resident for the ENTIRE denoise loop, so there is NO per-step transfer
#     penalty. Only the 15.5 GB text encoder swaps in once to encode the prompt
#     (and that result is cached across repeat prompts). This is the fast path
#     on a 24 GB card without also quantizing the text encoder.
if _device_total_vram_gb(PIC_DEVICE) >= FULL_RESIDENCY_VRAM_GB:
    _safe_move_to_device(pic_pipe, PIC_DEVICE)
    _offload_state["pic"] = False
    print(f"    [pic] full residency on {PIC_DEVICE} "
          f"({_device_total_vram_gb(PIC_DEVICE):.0f} GB card) -- no offload.")
else:
    _enable_pic_offload(pic_pipe)


qwen_time = time.time() - start_qwen
print(f" QWEN READY in {qwen_time:.1f}s - Picgen functional!")
_active_model = "pic"


def _activate_model(target):
    """Ensure the Qwen (picgen) pipeline is the single GPU-resident model.

    (picgen build) There is only one model in this process, so activation
    reduces to: place the Qwen pipeline on PIC_DEVICE via the shared offload
    logic -- full residency on >=40 GB cards, model-CPU-offload below that.
    The RLock still serialises activation against inference and cleanup so
    behaviour matches the newgen app exactly.
    """
    global _active_model
    with _gpu_op_lock:
        if target != "pic":
            return
        if _active_model == "pic" and pic_pipe is not None:
            return
        if pic_pipe is None:
            raise RuntimeError("Qwen pipeline has not finished loading")
        _enable_offload(pic_pipe, PIC_DEVICE, "pic")
        _active_model = "pic"


def activate_pic():
    """Ensure Qwen alone is fully resident and ready."""
    started = time.time()
    _activate_model("pic")
    if time.time() - started > 3:
        print(f" Qwen active in {time.time() - started:.1f}s")



PICGEN_MAX_SEED = np.iinfo(np.int32).max

_decode_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="b64decode")

_picgen_cache = {
    "vae_latents": {},      # keyed by image hash
    "prompt_embeds": {},    # keyed by (prompt, neg_prompt, images_hash)
}
_picgen_cache_lock = threading.Lock()
MAX_CACHE_ENTRIES = 20  # Keep last 20 to avoid memory bloat


def _hash_images(images):
    """Create a stable hash from a list of PIL images."""
    hasher = hashlib.sha256()
    for img in images:
        hasher.update(f"{img.size}".encode())
        img_array = np.array(img.resize((64, 64), Image.LANCZOS))
        hasher.update(img_array.tobytes())
    return hasher.hexdigest()


def _cpu_detach_tree(value):
    if torch.is_tensor(value):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {k: _cpu_detach_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_detach_tree(v) for v in value)
    if isinstance(value, list):
        return [_cpu_detach_tree(v) for v in value]
    return value


def _device_tree(value, target):
    if torch.is_tensor(value):
        return value.to(target)
    if isinstance(value, dict):
        return {k: _device_tree(v, target) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_device_tree(v, target) for v in value)
    if isinstance(value, list):
        return [_device_tree(v, target) for v in value]
    return value


def _clear_picgen_cache():
    with _picgen_cache_lock:
        for cache in _picgen_cache.values():
            cache.clear()


def _get_cached_vae_latents(images):
    """Return a device copy while retaining cache data on CPU only."""
    img_hash = _hash_images(images)
    with _picgen_cache_lock:
        value = _picgen_cache["vae_latents"].get(img_hash)
    return _device_tree(value, PIC_DEVICE) if value is not None else None


def _cache_vae_latents(images, latents):
    img_hash = _hash_images(images)
    with _picgen_cache_lock:
        cache = _picgen_cache["vae_latents"]
        if len(cache) >= MAX_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[img_hash] = _cpu_detach_tree(latents)


def _get_cached_prompt_embeds(prompt, negative_prompt, images, num_images_per_prompt):
    img_hash = _hash_images(images)
    key = (prompt, negative_prompt or "", img_hash, num_images_per_prompt)
    with _picgen_cache_lock:
        value = _picgen_cache["prompt_embeds"].get(key)
    return _device_tree(value, PIC_DEVICE) if value is not None else None


def _cache_prompt_embeds(prompt, negative_prompt, images, num_images_per_prompt, embeds_data):
    img_hash = _hash_images(images)
    key = (prompt, negative_prompt or "", img_hash, num_images_per_prompt)
    with _picgen_cache_lock:
        cache = _picgen_cache["prompt_embeds"]
        if len(cache) >= MAX_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = _cpu_detach_tree(embeds_data)


def _find_starter_path(starter_num: int):
    """Find a starter image file by number, trying .jpg / .png / .webp."""
    for ext in (".jpg", ".png", ".webp"):
        p = os.path.join(SCRIPT_DIR, f"starters/start{starter_num}{ext}")
        if os.path.exists(p):
            return p, ext
    return None, None


def add_starter_image(starter_num):
    path, ext = _find_starter_path(starter_num)
    if path is None:
        return ""
    mime = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"data:{mime};base64,{b64}"


def _build_starter_grid_html():
    """Read starters/start1.jpg .. start10.jpg from disk, base64-encode each,
    and return a self-contained HTML string with thumbnails + click buttons.

    Called ONCE at UI build time and used as the initial value of the starter
    gr.HTML component. It is intentionally NOT wired to demo.load: a queued
    Python demo.load handler interferes with gr.Progress(track_tqdm=True)
    progress streaming on subsequent picgen generations. Building the grid
    inline at startup keeps the queue clean while still showing the thumbnails
    immediately when the app loads."""
    starters_dir = Path(SCRIPT_DIR) / "starters"
    cards = []
    for n in range(1, 11):
        p = starters_dir / f"start{n}.jpg"
        if p.exists():
            data = p.read_bytes()
            b64 = base64.b64encode(data).decode()
            img_tag = (
                f'<img class="starter-thumb" '
                f'src="data:image/jpeg;base64,{b64}" '
                f'alt="{n}" />'
            )
        else:
            img_tag = f'<div class="starter-thumb-placeholder">{n}</div>'
        # Button onclick: inline JS reads the pre-embedded b64 from the img src
        # and passes it straight to __addImage — no fetch, no route, no async.
        if p.exists():
            onclick = (
                f"(function(){{"
                f"var img=this.closest('.starter-card').querySelector('img');"
                f"if(img&&window.__addImage)window.__addImage(img.src,'starter{n}.jpg');"
                f"}}).call(this)"
            )
        else:
            onclick = ""
        btn_attrs = f'class="starter-btn" onclick="{onclick}"' if onclick else 'class="starter-btn" disabled style="opacity:.35;cursor:default;"'
        cards.append(
            f'<div class="starter-card">'
            f'{img_tag}'
            f'<button {btn_attrs}>{n}</button>'
            f'</div>'
        )
    grid_items = "".join(cards)
    return f"""
    <style>
    .starter-card{{display:flex;flex-direction:column;align-items:center;gap:0;
      border:1px solid var(--border-color-primary);border-radius:6px;overflow:hidden;
      background:var(--background-fill-secondary);transition:border-color .15s;min-width:0;}}
    .starter-card:hover{{border-color:var(--color-accent);}}
    .starter-thumb{{width:100%;aspect-ratio:1;object-fit:cover;display:block;}}
    .starter-thumb-placeholder{{width:100%;aspect-ratio:1;display:flex;align-items:center;
      justify-content:center;font-size:13px;font-weight:600;
      color:var(--body-text-color-subdued);background:var(--background-fill-primary);}}
    .starter-btn{{width:100%;border:none;border-top:1px solid var(--border-color-primary);
      background:var(--background-fill-secondary);color:var(--body-text-color);
      font-size:12px;font-weight:700;padding:4px 0;cursor:pointer;text-align:center;
      line-height:1.4;}}
    .starter-btn:hover:not([disabled]){{background:var(--color-accent);color:#fff;}}
    #starter-grid-container .prose{{margin:0!important;padding:0!important;}}
    #starter-grid{{display:grid;grid-template-columns:repeat(10,1fr);gap:4px;width:100%;margin-bottom:8px;}}
    </style>
    <div id="starter-grid">{grid_items}</div>
    """


def _decode_single_b64(b64_str):
    """Decode a single base64 image string to PIL (used by thread pool)."""
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        if b64_str.startswith("data:image"):
            _, data = b64_str.split(",", 1)
        else:
            data = b64_str
        return Image.open(BytesIO(base64.b64decode(data))).convert("RGB")
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None


def b64_to_pil_list(b64_json_str):
    """Decode base64 JSON array to PIL images using thread pool for speedup."""
    if not b64_json_str or b64_json_str.strip() in ("", "[]"):
        return []
    try:
        b64_list = json.loads(b64_json_str)
    except Exception:
        return []
    
    if len(b64_list) == 1:
        img = _decode_single_b64(b64_list[0])
        return [img] if img is not None else []
    
    try:
        from concurrent.futures import as_completed
        futures = {_decode_executor.submit(_decode_single_b64, b64_str): idx 
                   for idx, b64_str in enumerate(b64_list)}
        pil_images = [None] * len(b64_list)
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            if result is not None:
                pil_images[idx] = result
        return [img for img in pil_images if img is not None]
    except Exception as e:
        print(f"Thread pool decode failed, falling back to sequential: {e}")
        pil_images = []
        for b64_str in b64_list:
            img = _decode_single_b64(b64_str)
            if img is not None:
                pil_images.append(img)
        return pil_images



def _full_gpu_cleanup(offload_pipelines=True):
    """Release caches and, optionally, all pipeline VRAM. (picgen build)"""
    global _active_model
    with _gpu_op_lock:
        _clear_picgen_cache()
        if offload_pipelines:
            if pic_pipe is not None:
                try:
                    _disable_offload(pic_pipe, "pic")
                except Exception as exc:
                    print(f"Cleanup warning (Qwen): {exc}")
            _active_model = None
        gc.collect()
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()


def _do_clear_storage():
    """
    Module-level storage clear helper called by infer_with_preclear() and the
    automatic post-generation clear. (picgen build: no vidgen player/sequence
    files to handle.)

    'Clearing storage' means:
      1. Releasing all transient _media_store entries (picgen images) so their
         RAM is freed.
      2. Cleaning up Gradio's session upload dir (tmp/gradio/) for uploaded
         input-image temp files, respecting the protection system so files
         still needed by a running generation are kept.

    Returns the count of released/deleted items.
    """
    import shutil as _shutil

    count = 0

    import glob as _glob
    _tmp = os.path.join(SCRIPT_DIR, "tmp", "gradio")
    import time as _time
    _now = _time.time()
    for _pat in (
        _tmp + "/picgen_*.png",
    ):
        for _f in _glob.glob(_pat):
            try:
                import os as _oss
                if _now - _oss.path.getmtime(_f) > 60:
                    _oss.unlink(_f); count += 1
            except Exception:
                pass
    _media_store_release_prefix("picgen_")

    for gradio_dir in [
        Path(SCRIPT_DIR) / "tmp" / "gradio",
    ]:
        if gradio_dir.exists():
            for item in gradio_dir.iterdir():
                if item.name == "vibe_edit_history":
                    continue
                if _is_protected(item):
                    continue
                try:
                    if item.is_dir():
                        _shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)
                    count += 1
                except Exception:
                    pass
            break

    return count


def infer_with_preclear(
    images_b64_json,
    prompt,
    negative_prompt=" ",
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    height=None,
    width=None,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generator wrapper around infer(), structured exactly like vidgen's
    _dispatch_generate / generate_video (both of which are generators that yield
    output updates and show step progress correctly).

    1. Yields an immediate output update that CLEARS the result gallery, so the
       previously generated photos are removed and cannot paint over the
       progress overlay.
    2. Clears old storage before generation.
    3. Yields the final result once infer() completes.

    The gallery clear MUST yield `[]` (empty list) for pic_result. The earlier
    `gr.update(value=None)` did NOT clear a gr.Gallery, which is why the old
    images stayed on screen and hid the progress.

    Progress does not depend on this function being a plain function: the
    per-step display is driven by the explicit callback_on_step_end installed in
    infer(), mirroring vidgen's animate_frame _step_cb. vidgen proves generators
    display progress fine.
    """
    # Clear the result gallery immediately: [] empties pic_result, gr.update()
    # leaves the seed slider untouched, "" drops stale download URLs.
    yield [], gr.update(), ""

    # Drive the progress object immediately so the overlay appears right away,
    # before denoising starts (model activation + image decode take a moment).
    if callable(progress):
        try:
            progress(0.0, desc="Preparing")
        except Exception:
            pass

    try:
        n = _do_clear_storage()
        print(f"[picgen pre-clear] cleared {n} item(s)")
    except Exception as _e:
        print(f"[picgen pre-clear] storage clear failed (non-fatal): {_e}")

    try:
        filepaths, seed_out, urls_json = infer(
            images_b64_json, prompt, negative_prompt, seed, randomize_seed,
            true_guidance_scale, num_inference_steps, height, width,
            num_images_per_prompt, progress,
        )
        yield filepaths, seed_out, urls_json
    except gr.Error:
        # Re-raise gr.Error cleanly — Gradio shows the message in the UI.
        # The exception is intentional (e.g. "no images uploaded") so we
        # don't need the full Python traceback in the terminal.
        raise
    except Exception as _infer_e:
        raise gr.Error(f"Generation failed: {_infer_e}") from None


@_gpu_serialized
def infer(
    images_b64_json,
    prompt,
    negative_prompt=" ",
    seed=42,
    randomize_seed=False,
    true_guidance_scale=1.0,
    num_inference_steps=4,
    height=None,
    width=None,
    num_images_per_prompt=1,
    progress=gr.Progress(track_tqdm=True),
):
    if randomize_seed:
        seed = random.randint(0, PICGEN_MAX_SEED)

    _t_enter = time.time()
    activate_pic()
    _t_active = time.time()

    torch.cuda.set_device(PIC_DEVICE)
    generator = torch.Generator(device=PIC_DEVICE).manual_seed(seed)
    pil_images = b64_to_pil_list(images_b64_json)
    if not pil_images:
        raise gr.Error("Please upload at least one image.") from None
    _t_decoded = time.time()

    if height == 256 and width == 256:
        height, width = None, None

    print(f"Seed: {seed} | Steps: {num_inference_steps}")
    print(f"  input images: {[im.size for im in pil_images]}")
    
    cached_embeds = _get_cached_prompt_embeds(prompt, negative_prompt, pil_images, num_images_per_prompt)
    cache_status = "cached" if cached_embeds is not None else "computing"
    
    print(f"  timing: activate {_t_active - _t_enter:.2f}s, "
          f"decode {_t_decoded - _t_active:.2f}s, embeds: {cache_status} "
          f"(active model: {_active_model})")
    _t_pipe = time.time()

    original_encode_prompt = pic_pipe.encode_prompt
    original_prepare_latents = pic_pipe.prepare_latents
    
    encode_called = [False]
    prepare_called = [False]
    
    def cached_encode_prompt(*args, **kwargs):
        encode_called[0] = True
        if cached_embeds is not None:
            return cached_embeds["prompt_embeds"], cached_embeds["prompt_embeds_mask"]
        result = original_encode_prompt(*args, **kwargs)
        embeds_data = {
            "prompt_embeds": result[0],
            "prompt_embeds_mask": result[1]
        }
        _cache_prompt_embeds(prompt, negative_prompt, pil_images, num_images_per_prompt, embeds_data)
        return result
    
    def cached_prepare_latents(images, *args, **kwargs):
        prepare_called[0] = True
        result = original_prepare_latents(images, *args, **kwargs)
        if images is not None and result[1] is not None:
            pass
        return result
    
    pic_pipe.encode_prompt = cached_encode_prompt
    pic_pipe.prepare_latents = cached_prepare_latents

    # Per-step progress: drive the gr.Progress object EXPLICITLY from a
    # pipeline step callback (same pattern as vidgen's animate_frame _step_cb).
    # Relying on gr.Progress(track_tqdm=True) to auto-hook the pipeline's
    # internal tqdm is unreliable through this nested blocking-generator call
    # shape and does NOT stream the per-step overlay to the output gallery.
    # Driving progress(...) directly each step pushes updates to the queue
    # during the blocking pipeline call, which is what actually renders the
    # "Step x/N" overlay in real time. Works for 1 or many input/output images
    # (num_inference_steps is the same denoising loop regardless).
    _pic_total_steps = max(1, int(num_inference_steps))

    def _pic_step_cb(_pipe, step_index, _timestep, cb_kwargs):
        if callable(progress):
            try:
                progress(
                    (step_index + 1) / _pic_total_steps,
                    desc=f"Step {step_index + 1}/{_pic_total_steps}",
                )
            except Exception:
                pass
        return cb_kwargs

    _pic_call_kwargs = dict(
        image=pil_images if pil_images else None,
        prompt=prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        true_cfg_scale=true_guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        callback_on_step_end=_pic_step_cb,
    )

    try:
        with torch.cuda.device(PIC_DEVICE):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                try:
                    image = pic_pipe(**_pic_call_kwargs).images
                except TypeError as _cb_e:
                    # Older pipeline without callback_on_step_end support:
                    # retry without the callback rather than failing.
                    if "callback_on_step_end" in str(_cb_e):
                        _pic_call_kwargs.pop("callback_on_step_end", None)
                        image = pic_pipe(**_pic_call_kwargs).images
                    else:
                        raise
    finally:
        pic_pipe.encode_prompt = original_encode_prompt
        pic_pipe.prepare_latents = original_prepare_latents

    print(f"  pipeline call took {time.time() - _t_pipe:.2f}s")

    import os as _os
    _shm_dir = os.path.join(SCRIPT_DIR, "tmp", "gradio")
    _os.makedirs(_shm_dir, exist_ok=True)
    if not _os.path.isdir(_shm_dir):
        _shm_dir = _os.path.join(_os.environ.get("TMPDIR", "/tmp"), "picgen_out")
        _os.makedirs(_shm_dir, exist_ok=True)
    multiple = len(image) > 1
    filepaths = []
    for i, img in enumerate(image, start=1):
        filename = _media_name("picgen", ".png", index=i if multiple else None)
        fpath = _os.path.join(_shm_dir, filename)
        img.save(fpath, format="PNG")
        filepaths.append(fpath)
    print(f"  saved: {[_os.path.basename(p) for p in filepaths]}")
    urls_json = json.dumps(filepaths)

    return filepaths, seed, urls_json



_protected_image_paths = set()
_protected_paths_lock = threading.Lock()

_generation_active_paths = set()
_generation_active_lock = threading.Lock()


def _generation_protect(path):
    """Pin a path as in-use by a running generation."""
    if not path:
        return
    try:
        p = str(path)
        with _generation_active_lock:
            _generation_active_paths.add(p)
        _protect_path(p)          # also in the general set
    except Exception:
        pass


def _generation_release(path):
    """Release a path from the generation-active set (generation done/error)."""
    if not path:
        return
    try:
        p = str(path)
        with _generation_active_lock:
            _generation_active_paths.discard(p)
    except Exception:
        pass


def _is_generation_active(path) -> bool:
    """True if path is currently held by a running generation."""
    if not path:
        return False
    try:
        p = str(path)
        with _generation_active_lock:
            return p in _generation_active_paths
    except Exception:
        return False


_protected_image_filenames = set()
_protected_filenames_lock = threading.Lock()


def _protect_filename(path):
    """Register the basename of path as protected from clear_storage()."""
    if not path:
        return
    try:
        name = os.path.basename(str(path))
        if name:
            with _protected_filenames_lock:
                _protected_image_filenames.add(name)
    except Exception:
        pass


def _unprotect_filename(path):
    """Remove the basename of path from the protected-filenames set."""
    if not path:
        return
    try:
        name = os.path.basename(str(path))
        if name:
            with _protected_filenames_lock:
                _protected_image_filenames.discard(name)
    except Exception:
        pass


def _is_filename_protected(item_path) -> bool:
    """True if the item's basename is in the protected-filenames set."""
    try:
        name = os.path.basename(str(item_path))
        with _protected_filenames_lock:
            return name in _protected_image_filenames
    except Exception:
        return False


def _protect_path(path):
    """Mark a filesystem path as protected from clear_storage()."""
    if not path:
        return
    try:
        p = str(path)
    except Exception:
        return
    with _protected_paths_lock:
        _protected_image_paths.add(p)


def _unprotect_path(path):
    """Remove a path from the protected set (safe no-op if absent)."""
    if not path:
        return
    try:
        p = str(path)
    except Exception:
        return
    with _protected_paths_lock:
        _protected_image_paths.discard(p)


def _is_protected(item_path) -> bool:
    """True if item_path is, or contains, a currently-protected input image.

    Two layers of protection:
    1. Full-path set: _protected_image_paths (and legacy _current_input_image_path).
    2. Filename set: _protected_image_filenames — any item whose *basename*
       matches a currently-live first-frame or last-frame filename is skipped,
       even if the full path lookup misses (e.g. Gradio moved/renamed the file).
    """
    if _is_filename_protected(item_path):
        return True

    item_path = Path(item_path)
    with _protected_paths_lock:
        protected_now = set(_protected_image_paths)
    if _current_input_image_path:
        protected_now.add(_current_input_image_path)
    for p in protected_now:
        if not p:
            continue
        if p.startswith("/media/"):
            continue
        protected_path = Path(p)
        if item_path == protected_path:
            return True
        if item_path.is_dir():
            try:
                protected_path.relative_to(item_path)
                return True
            except ValueError:
                pass
    return False


_current_input_image_path = None




gallery_js = r"""
() => {
// Keep the image array alive across re-inits so images added before a
// Gradio DOM refresh are not lost.
if (!window.__picgenImages) window.__picgenImages = [];
if (!window.__picgenSelected) window.__picgenSelected = { idx: -1 };

function init() {
    const galleryGrid  = document.getElementById('image-gallery-grid');
    const dropZone     = document.getElementById('gallery-drop-zone');
    const uploadPrompt = document.getElementById('upload-prompt');
    const uploadClick  = document.getElementById('upload-click-area');
    const fileInput    = document.getElementById('custom-file-input');
    const btnUpload    = document.getElementById('tb-upload');
    const btnRemove    = document.getElementById('tb-remove');
    const btnClear     = document.getElementById('tb-clear');
    if (!galleryGrid || !fileInput || !dropZone) { setTimeout(init, 250); return; }

    // If the grid is already connected and has our listener marker, skip.
    // But if the element was replaced (e.g. Gradio re-rendered the tab),
    // we must re-attach all listeners to the new DOM nodes.
    if (galleryGrid._picgenListenerAttached && galleryGrid.isConnected) return;
    galleryGrid._picgenListenerAttached = true;

    let images = window.__picgenImages;
    let selectedIdx = window.__picgenSelected.idx;
    window.__uploadedImages = images;

    function syncToGradio() {
        if (window.__picgenSuppressSync) return;
        window.__uploadedImages = images;
        window.__picgenImages = images;
        window.__picgenSelected.idx = selectedIdx;
        const b64Array = images.map(img => img.b64);
        const container = document.getElementById('hidden-images-b64');
        if (!container) return;
        container.querySelectorAll('input,textarea').forEach(el => {
            const proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            const ns = Object.getOwnPropertyDescriptor(proto, 'value');
            if (ns && ns.set) {
                ns.set.call(el, JSON.stringify(b64Array));
                el.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                el.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            }
        });
    }

    function addImage(b64, name) {
        images.push({id: Date.now() + Math.random(), b64: b64, name: name});
        renderGallery(); syncToGradio();
    }
    window.__addImage = addImage;

    // Replace all images with a single new one (used by outpaint popup)
    function replaceImages(b64, name) {
        images.length = 0;
        selectedIdx = -1;
        images.push({id: Date.now() + Math.random(), b64: b64, name: name});
        window.__uploadedImages = images;
        window.__picgenImages = images;
        renderGallery(); syncToGradio();
    }
    window.__replaceImages = replaceImages;

    function removeImage(idx) {
        images.splice(idx, 1);
        if (selectedIdx === idx) selectedIdx = -1;
        else if (selectedIdx > idx) selectedIdx--;
        renderGallery(); syncToGradio();
    }

    function clearAll() {
        images.length = 0;
        window.__uploadedImages = images;
        window.__picgenImages = images;
        selectedIdx = -1;
        renderGallery(); syncToGradio();
    }
    window.__clearAll = clearAll;

    function setSelected(idx) {
        selectedIdx = idx;
        window.__picgenSelected.idx = selectedIdx;
        // Re-render to reflect selection highlight without full DOM rebuild
        galleryGrid.querySelectorAll('.gallery-thumb').forEach((el) => {
            const i = parseInt(el.dataset.idx);
            if (i === selectedIdx) el.classList.add('selected');
            else el.classList.remove('selected');
        });
    }

    function renderGallery() {
        if (images.length === 0) {
            galleryGrid.innerHTML = ''; galleryGrid.style.display = 'none';
            if (uploadPrompt) uploadPrompt.style.display = '';
            return;
        }
        if (uploadPrompt) uploadPrompt.style.display = 'none';
        galleryGrid.style.display = 'grid';
        let html = '';
        images.forEach((img, i) => {
            const sel = i === selectedIdx ? ' selected' : '';
            html += '<div class="gallery-thumb' + sel + '" draggable="true" data-idx="' + i + '">'
                  + '<img src="' + img.b64 + '" alt="' + (img.name||'image') + '">'
                  + '<span class="thumb-badge">#' + (i+1) + '</span>'
                  + '<button class="thumb-remove" data-remove="' + i + '" '
                  + 'style="position:absolute;top:4px;right:4px;z-index:100;'
                  + 'width:22px;height:22px;border-radius:50%;background:rgba(0,0,0,0.8);'
                  + 'color:#fff;border:1px solid rgba(255,255,255,0.5);cursor:pointer;'
                  + 'display:flex;align-items:center;justify-content:center;font-size:11px;'
                  + 'pointer-events:auto;">\u2715</button>'
                  + '</div>';
        });
        html += '<div class="gallery-add-card" id="gallery-add-card" style="pointer-events:auto;">'
              + '<span class="add-icon">+</span><span class="add-text">Add</span></div>';
        galleryGrid.innerHTML = html;
    }

    function showLightbox(b64) {
        let lb = document.getElementById('picgen-lightbox');
        if (!lb) {
            lb = document.createElement('div');
            lb.id = 'picgen-lightbox';
            lb.style.cssText = 'display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.92);z-index:9999;align-items:center;justify-content:center;cursor:zoom-out;';
            lb.innerHTML = '<div style="position:relative;max-width:92vw;max-height:92vh;display:flex;align-items:center;justify-content:center;">'
                + '<img id="picgen-lb-img" style="max-width:92vw;max-height:92vh;width:auto;height:auto;border-radius:6px;display:block;object-fit:contain;image-rendering:auto;box-shadow:0 8px 40px rgba(0,0,0,0.7);">'
                + '<button id="picgen-lb-close" style="position:fixed;top:16px;right:20px;width:36px;height:36px;border-radius:50%;background:rgba(0,0,0,0.7);color:#fff;border:1px solid rgba(255,255,255,0.3);cursor:pointer;font-size:20px;line-height:1;display:flex;align-items:center;justify-content:center;z-index:10000;">\u00d7</button>'
                + '</div>';
            document.body.appendChild(lb);
            lb.addEventListener('click', () => { lb.style.display = 'none'; });
            document.getElementById('picgen-lb-close').addEventListener('click', (e) => { e.stopPropagation(); lb.style.display = 'none'; });
            document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && lb.style.display === 'flex') lb.style.display = 'none'; });
        }
        document.getElementById('picgen-lb-img').src = b64;
        lb.style.display = 'flex';
    }

    function processFiles(files) {
        const maxSize = window.__picgenMaxSize || 512;
        Array.from(files).forEach(file => {
            if (!file.type.startsWith('image/')) return;
            const reader = new FileReader();
            reader.onload = (e) => {
                // Keep the ORIGINAL full-resolution image on import. Downscaling
                // to the selected Quality now happens only at Generate time (see
                // the _pic_infer_js prehook), so the same imported image can be
                // generated at any quality without reimporting or refreshing.
                addImage(e.target.result, file.name);
            };
            reader.readAsDataURL(file);
        });
    }

    // -- Quality selector buttons ----------------------------------------------
    if (!window.__picgenMaxSize) window.__picgenMaxSize = 512;
    document.querySelectorAll('.tb-quality').forEach(btn => {
        btn.addEventListener('click', () => {
            window.__picgenMaxSize = parseInt(btn.dataset.size, 10);
            document.querySelectorAll('.tb-quality').forEach(b => b.classList.remove('tb-quality-active'));
            btn.classList.add('tb-quality-active');
        });
    });

    fileInput.addEventListener('change', (e) => { processFiles(e.target.files); e.target.value = ''; });
    if (uploadClick) uploadClick.addEventListener('click', () => fileInput.click());
    if (btnUpload)   btnUpload.addEventListener('click',   () => fileInput.click());

    // -- Single capture-phase delegated listener on the grid ------------------
    // Fires before any bubble-phase handlers on children. Handles X remove,
    // Add card, and thumb selection. Using capture ensures the dropZone's
    // bubble-phase click handler (which opens the file picker) never receives
    // events that originated inside the grid.
    galleryGrid.addEventListener('click', (e) => {
        // X button — remove by index
        const removeBtn = e.target.closest('.thumb-remove');
        if (removeBtn) {
            e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
            removeImage(parseInt(removeBtn.dataset.remove, 10));
            return;
        }
        // Add card — open file picker
        if (e.target.closest('.gallery-add-card')) {
            e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
            fileInput.click();
            return;
        }
        // Thumb body — toggle selection (single click) / lightbox (dblclick)
        const thumb = e.target.closest('.gallery-thumb');
        if (thumb) {
            e.preventDefault(); e.stopPropagation(); e.stopImmediatePropagation();
            const idx = parseInt(thumb.dataset.idx, 10);
            if (!isNaN(idx)) {
                if (selectedIdx === idx) {
                    // Second click on same thumb opens lightbox
                    if (images[idx]) showLightbox(images[idx].b64);
                } else {
                    setSelected(idx);
                }
            }
            return;
        }
    }, true); // capture phase

    // Drop-zone background click opens file picker — but only when the click
    // did NOT originate inside the grid (grid events are fully stopped above).
    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('#image-gallery-grid')) return;
        if (e.target.closest('#upload-click-area')) return;
        fileInput.click();
    });

    // -- Drag-to-reorder (bubble phase — separate from click capture above) ----
    let _dragSrcIdx = -1;
    galleryGrid.addEventListener('dragstart', (e) => {
        const thumb = e.target.closest('.gallery-thumb');
        if (!thumb) return;
        _dragSrcIdx = parseInt(thumb.dataset.idx, 10);
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', String(_dragSrcIdx));
        // Slight delay so the browser snapshot doesn't show the highlight
        setTimeout(() => thumb.classList.add('dragging'), 0);
    });
    galleryGrid.addEventListener('dragend', (e) => {
        _dragSrcIdx = -1;
        galleryGrid.querySelectorAll('.gallery-thumb').forEach(t => {
            t.classList.remove('drag-over', 'dragging');
        });
    });
    galleryGrid.addEventListener('dragover', (e) => {
        const thumb = e.target.closest('.gallery-thumb');
        if (!thumb) return;
        const overIdx = parseInt(thumb.dataset.idx, 10);
        if (overIdx === _dragSrcIdx) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        galleryGrid.querySelectorAll('.gallery-thumb').forEach(t => t.classList.remove('drag-over'));
        thumb.classList.add('drag-over');
    });
    galleryGrid.addEventListener('dragleave', (e) => {
        const thumb = e.target.closest('.gallery-thumb');
        if (thumb) thumb.classList.remove('drag-over');
    });
    galleryGrid.addEventListener('drop', (e) => {
        const thumb = e.target.closest('.gallery-thumb');
        if (!thumb) return;
        e.preventDefault();
        const destIdx = parseInt(thumb.dataset.idx, 10);
        if (_dragSrcIdx < 0 || destIdx === _dragSrcIdx) return;
        // Move image from srcIdx to destIdx
        const moved = images.splice(_dragSrcIdx, 1)[0];
        images.splice(destIdx, 0, moved);
        // Keep selection tracking consistent
        if (selectedIdx === _dragSrcIdx) selectedIdx = destIdx;
        else if (_dragSrcIdx < selectedIdx && destIdx >= selectedIdx) selectedIdx--;
        else if (_dragSrcIdx > selectedIdx && destIdx <= selectedIdx) selectedIdx++;
        window.__picgenSelected.idx = selectedIdx;
        renderGallery();
        syncToGradio();
    });

    if (btnRemove) btnRemove.addEventListener('click', () => { if (selectedIdx >= 0) removeImage(selectedIdx); });
    if (btnClear)  btnClear.addEventListener( 'click', clearAll);
    dropZone.addEventListener('dragover',  (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); });
    dropZone.addEventListener('drop',      (e) => { e.preventDefault(); dropZone.classList.remove('drag-over'); if (e.dataTransfer.files.length) processFiles(e.dataTransfer.files); });
    // Render any images that were already in the persistent store before this re-init
    renderGallery();
    if (images.length > 0) syncToGradio();
}
init();
// Re-run init on DOM mutations in case Gradio remounts the picgen tab's HTML
let _picgenObserverTimer = null;
const _picgenObserver = new MutationObserver(() => {
    // Debounce: only act after mutations settle (50ms quiet period)
    clearTimeout(_picgenObserverTimer);
    _picgenObserverTimer = setTimeout(() => {
        const g = document.getElementById('image-gallery-grid');
        if (g && !g._picgenListenerAttached) init();
    }, 50);
});
_picgenObserver.observe(document.body, { childList: true, subtree: true });
}
"""


css = """
body, .gradio-container { margin: 0 !important; padding: 0 !important; max-width: 100% !important; }
#col-container { margin: 0 !important; max-width: 100% !important; padding: 0 !important; }
.contain { padding: 0 !important; }
#preset-row { display: flex !important; align-items: center !important; gap: 8px !important; }
#preset-row > * { flex: 1 !important; }
#preset-row button { flex: 0 0 auto !important; min-width: 80px !important; }
#preset-row input[type="text"] { pointer-events: none !important; user-select: none !important; }
.hidden-input { display: none !important; height: 0 !important; overflow: hidden !important; margin: 0 !important; padding: 0 !important; }
#gallery-drop-zone { position: relative; min-height: 320px; overflow: auto; border: 1px solid var(--border-color-primary); border-radius: 8px; margin-bottom: 8px; }
#gallery-drop-zone.drag-over { outline: 2px solid var(--color-accent); outline-offset: -2px; }
.upload-prompt-modern { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); z-index: 1; }
/* The upload prompt is a centered absolute overlay. Only its inner click area
   should capture clicks; the rest must let clicks through to the thumbnails
   and their remove/add buttons underneath. */
.upload-prompt-modern { pointer-events: none; }
.upload-prompt-modern #upload-click-area, #upload-click-area { pointer-events: auto; }
/* The thumbnail grid must sit ABOVE the absolute upload prompt so its X and
   Add buttons are always clickable (previously they shared/were under the
   prompt's stacking context and got blocked). */
#image-gallery-grid { position: relative; z-index: 5; }
.gallery-thumb, .gallery-add-card { position: relative; z-index: 5; }
.upload-click-area { display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; padding: 36px 52px; border: 2px dashed var(--border-color-primary); border-radius: 16px; transition: all .2s ease; gap: 8px; }
.upload-click-area:hover { border-color: var(--color-accent); transform: scale(1.03); }
.upload-click-area svg { width: 64px; height: 64px; }
.upload-main-text { font-size: 14px; font-weight: 500; margin-top: 4px; }
.upload-sub-text { font-size: 12px; color: var(--body-text-color-subdued); }
.image-gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 10px; padding: 12px; align-content: start; }
/* Picgen tool buttons: three side by side, spaced, together full width. */
#picgen-tool-row { display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important; gap: 10px !important; width: 100% !important; margin-bottom: 8px !important; }
#picgen-tool-row > * { flex: 1 1 0 !important; min-width: 0 !important; }
#merge-bg-radio .wrap { display: flex !important; flex-wrap: nowrap !important; flex-direction: row !important; gap: 4px 8px !important; align-items: center !important; }
#merge-bg-radio .wrap label { white-space: nowrap !important; flex: 1 1 0 !important; font-size: 12px !important; }
/* Custom-mode results: 4 large landscape images side by side, full width.
   Keep this minimal — let Gradio render the gallery natively (heavy DOM
   overrides collapsed the images into blank space). We only force full width
   and let images size to their container. */
#merge-custom-gallery { width: 100% !important; }
#merge-custom-gallery img { object-fit: contain !important; }
.gallery-thumb { position: relative; aspect-ratio: 1; border-radius: 8px; overflow: visible; cursor: grab; border: 2px solid var(--border-color-primary); transition: border-color .2s ease, box-shadow .2s ease; background: var(--background-fill-primary); display: flex; align-items: center; justify-content: center; }
.gallery-thumb > img { border-radius: 6px; overflow: hidden; }
.gallery-thumb:hover { border-color: var(--color-accent); }
.gallery-thumb.selected { border-color: var(--color-accent) !important; box-shadow: 0 0 0 3px rgba(var(--color-accent-soft), .3); }
.gallery-thumb.dragging { opacity: 0.4; cursor: grabbing; }
.gallery-thumb.drag-over { border-color: #fff !important; box-shadow: 0 0 0 3px rgba(255,255,255,0.5) !important; }
.gallery-thumb img { width: 100%; height: 100%; object-fit: contain; display: block !important; }
.thumb-badge { position: absolute; top: 5px; left: 5px; background: var(--color-accent); color: #fff; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: 600; display: block; text-align: left; }
.thumb-remove { position: absolute; top: 5px; right: 5px; width: 24px; height: 24px; background: rgba(0,0,0,.75); color: #fff; border: 1px solid rgba(255,255,255,.35); border-radius: 50%; cursor: pointer; display: flex !important; align-items: center; justify-content: center; font-size: 12px; transition: background .15s; line-height: 1; z-index: 50; padding: 0; pointer-events: auto !important; }
.gallery-thumb:hover .thumb-remove { display: flex !important; }
.thumb-remove:hover { background: #e53e3e !important; }
.gallery-add-card { aspect-ratio: 1; border-radius: 8px; border: 2px dashed var(--border-color-primary); display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: pointer; transition: all .2s ease; gap: 4px; }
.gallery-add-card:hover { border-color: var(--color-accent); }
.gallery-add-card .add-icon { font-size: 26px; font-weight: 300; }
.gallery-add-card .add-text { font-size: 12px; font-weight: 500; }
.uploader-toolbar { display: flex; gap: 6px; align-items: center; margin-bottom: 6px; flex-wrap: wrap; }
.quality-toolbar { display: flex; gap: 5px; align-items: center; margin-bottom: 6px; flex-wrap: wrap; }
.quality-label { font-size: 11px; color: var(--body-text-color-subdued); white-space: nowrap; margin-right: 2px; }
.tb-btn { display: inline-flex; align-items: center; gap: 5px; padding: 5px 12px; border: 1px solid var(--border-color-primary); border-radius: 6px; background: var(--background-fill-secondary); cursor: pointer; font-size: 12px; font-weight: 500; transition: all .15s; }
.tb-btn:hover { border-color: var(--color-accent); }
.tb-quality { font-size: 11px; padding: 4px 9px; }
.tb-quality-active { border-color: var(--color-accent) !important; background: var(--color-accent) !important; color: #fff !important; }
/* Show Media = off: hide actual media pixels but keep every container/upload-zone intact.
   Uses visibility:hidden so the container box stays; only the rendered image/video disappears. */
body.hide-media #picgen-result-gallery img,
body.hide-media #picgen-result-gallery video { visibility: hidden !important; }
/* Make picgen prompt textareas manually resizable */
#col-container textarea { resize: vertical !important; min-height: 60px !important; touch-action: pan-y !important; }
/* Center labels, info/description text, and dropdown option text on all components.
   Scoped away from #gallery-drop-zone so .thumb-badge/.thumb-remove are not affected. */
.gradio-container label > span:first-child,
.gradio-container .label-wrap > span,
.gradio-container legend > span { text-align: center !important; width: 100% !important; }
.gradio-container .info { text-align: center !important; }
.gradio-container select option { text-align: center !important; }
.gradio-container .wrap-inner input,
.gradio-container .secondary-wrap span { text-align: center !important; }
/* Center the Clear Storage button and Show Media checkbox at the top */
#top-bar-row { justify-content: center !important; }
#top-bar-row button { min-width: 140px !important; }
#show-media-row { justify-content: center !important; }
#show-media-row > * { flex: 0 0 auto !important; }
/* Tab bar: single tab, centered */
.gradio-container .tab-nav { display: flex !important; width: 100% !important; }
.gradio-container .tab-nav button { text-align: center !important; justify-content: center !important; }
/* Center all section title / heading text (Markdown headings used as
   section titles) on the Photo Editor tab. */
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4,
.gradio-container .prose h5,
.gradio-container .prose h6 { text-align: center !important; width: 100% !important; }
/* Accordion (expandable section) header bars: keep the title text centered
   but pin the expand/collapse arrow icon fully to the right edge of the
   clickable title bar. */
.gradio-container .label-wrap { display: flex !important; align-items: center !important; justify-content: center !important; width: 100% !important; position: relative !important; }
.gradio-container .label-wrap > span { flex: 1 1 auto !important; text-align: center !important; }
.gradio-container .label-wrap svg { flex: 0 0 auto !important; margin-left: auto !important; position: relative !important; right: 0 !important; }
/* NOTE: Deliberately NO custom CSS targeting the picgen result gallery's
   progress/status overlay. An earlier attempt styled
   `#picgen-result-gallery .wrap`, but in a gr.Gallery `.wrap` is ALSO the
   gallery's own content wrapper — forcing it to position:absolute/inset:0 with
   an opaque background turned the image container into a solid overlay and the
   generated photos stopped showing, while also overriding Gradio's native
   centered progress layout with badly aligned text. Gradio's default status
   tracker styling is correct on its own; leave it alone. */

/* NOTE: No toast/notification hiding rules here on purpose.
   Attempts to hide Gradio's "press ESC to exit full screen" and download
   toasts required selectors broad enough to also match Gradio's loading /
   progress status element, which silently hid the per-step generation
   progress overlay on the output components. Those toasts are accepted as
   visible so the generation progress display is never at risk. */
"""



with gr.Blocks(css=css) as demo:
    with gr.Row(elem_id="top-bar-row"):
        clear_storage_btn = gr.Button("Clear Storage", variant="secondary", size="sm")
    clear_storage_status = gr.Textbox(visible=False, label="")

    def protect_current_inputs():
        """Explicit pre-clear protection step for the automatic storage clear.

        (picgen build) The picgen tab keeps its inputs in RAM (base64 inside
        the hidden textbox), so there are no on-disk input files to protect
        before the automatic storage clear. Kept as a chain step so the
        generate wiring stays identical to the newgen app.
        """
        return None

    def clear_storage():
        """Release stored media without disturbing warm model residency."""
        cleaned = _do_clear_storage()
        return gr.update(visible=True, value=f"? Cleared {cleaned} upload(s).")

    def clear_storage_full():
        """Explicit user cleanup also releases every GPU resident object."""
        cleaned = _do_clear_storage()
        _full_gpu_cleanup(offload_pipelines=True)
        return gr.update(visible=True, value=f"? Cleared {cleaned} upload(s) and GPU caches.")




    clear_storage_btn.click(
        fn=clear_storage_full,
        inputs=[],
        outputs=[clear_storage_status],
    )

    with gr.Row(elem_id="show-media-row"):
        show_media_cb = gr.Checkbox(
            label="Show Media",
            value=True,
            info="When checked, displays input images and generated output on screen.",
        )


    with gr.Tabs():

        with gr.Tab("Photo Editor", id="picgen"):
            with gr.Column(elem_id="col-container"):

                starter_grid_html = gr.HTML(
                    value=_build_starter_grid_html(),
                    elem_id="starter-grid-container",
                )
                starter_b64_output = gr.Textbox(value="", visible=False, elem_id="starter-b64-output")

                with gr.Row():
                    with gr.Column(scale=1):
                        hidden_images_b64 = gr.Textbox(
                            value="[]", elem_id="hidden-images-b64",
                            elem_classes="hidden-input", container=False, visible=False,
                        )
                        gr.HTML("""
                        <div class="uploader-toolbar">
                            <button id="tb-upload" class="tb-btn">Upload</button>
                            <button id="tb-remove" class="tb-btn">Remove Selected</button>
                            <button id="tb-clear" class="tb-btn">Clear All</button>
                        </div>
                        <div class="quality-toolbar">
                            <span class="quality-label">Quality:</span>
                            <button class="tb-btn tb-quality" data-size="256">256 ? Fastest</button>
                            <button class="tb-btn tb-quality tb-quality-active" data-size="512">512 ? Default</button>
                            <button class="tb-btn tb-quality" data-size="1024">1024 ? Better</button>
                            <button class="tb-btn tb-quality" data-size="1920">1920 ?? High</button>
                            <button class="tb-btn tb-quality" data-size="2560">2560 ? Max</button>
                        </div>
                        <div id="gallery-drop-zone">
                            <div id="upload-prompt" class="upload-prompt-modern">
                                <div id="upload-click-area" class="upload-click-area">
                                    <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <rect x="6" y="10" width="52" height="44" rx="5" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3"/>
                                        <polygon points="10,50 24,32 34,42 44,28 54,50" fill="rgba(128,128,128,0.15)" stroke="currentColor" stroke-width="1.5"/>
                                        <circle cx="22" cy="24" r="5" fill="rgba(128,128,128,0.2)" stroke="currentColor" stroke-width="1.5"/>
                                    </svg>
                                    <span class="upload-main-text">Click or drag images here</span>
                                    <span class="upload-sub-text">Supports multiple images</span>
                                </div>
                            </div>
                            <input id="custom-file-input" type="file" accept="image/*" multiple style="display:none;" />
                            <div id="image-gallery-grid" class="image-gallery-grid" style="display:none;"></div>
                        </div>
                        """)
                        # Three tools, side by side, together spanning the full
                        # width of the Generate button below.
                        with gr.Row(elem_id="picgen-tool-row"):
                            pic_complete_body_btn = gr.Button(
                                "Multi-Tool", variant="secondary", scale=1,
                                elem_id="pic-complete-body-btn",
                            )
                            pic_add_l2r_btn = gr.Button(
                                "Add Left To Right", variant="secondary", scale=1,
                            )
                            pic_add_r2l_btn = gr.Button(
                                "Add Right To Left", variant="secondary", scale=1,
                            )
                        pic_run_button_top = gr.Button("Generate", variant="primary", size="lg")
                        pic_num_images = gr.Slider(
                            label="Number of images", minimum=1, maximum=4, step=1, value=1,
                        )
                        pic_prompt = gr.Textbox(
                            label="Prompt", show_label=True,
                            placeholder="describe the edit instruction",
                            lines=3, max_lines=20,
                        )
                        pic_negative_prompt = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="censored, mosaic, blurred, clothed, soft, partial",
                            value="", lines=2, max_lines=10,
                        )

                    with gr.Column(scale=1):
                        pic_result = gr.Gallery(
                            label="Result",
                            show_label=False,
                            type="filepath",
                            interactive=False,
                            columns=2,
                            elem_id="picgen-result-gallery",
                            # Serve full-quality PNG instead of Gradio's default
                            # webp. Without this, right-click > "open image in
                            # new tab" hands the browser a .webp it downloads
                            # rather than displaying inline full size.
                            format="png",
                        )
                        use_output_btn = gr.Button("Use as input", variant="secondary", size="sm")
                        pic_download_btn = gr.Button("Download All currently outputted Images", variant="secondary", size="sm")
                        picgen_urls = gr.Textbox(visible=False, value="", elem_id="picgen-urls")

                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                label="Solo", choices=list(solo_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown2 = gr.Dropdown(
                                label="Couple (Unseen)", choices=list(couple_man_unseen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown3 = gr.Dropdown(
                                label="Couple (Seen)", choices=list(couple_man_seen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown4 = gr.Dropdown(
                                label="Multi Women", choices=list(multiple_women_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )

                        with gr.Row():
                            preset_dropdown5 = gr.Dropdown(
                                label="Multi (Unseen)", choices=list(multiple_man_unseen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown6 = gr.Dropdown(
                                label="Multi (Seen)", choices=list(multiple_man_seen_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )
                            preset_dropdown7 = gr.Dropdown(
                                label="Multi-Step", choices=list(multistep_prompts_dict.keys()),
                                value=None, interactive=True, scale=1,
                            )

                        pic_run_button = gr.Button("Generate", variant="primary", size="lg")

                with gr.Accordion("Advanced Settings", open=False):
                    pic_seed = gr.Slider(label="Seed", minimum=0, maximum=PICGEN_MAX_SEED, step=1, value=0)
                    pic_randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    with gr.Row():
                        pic_guidance = gr.Slider(label="True guidance scale", minimum=1.0, maximum=10.0, step=0.1, value=1.0)
                        pic_steps = gr.Slider(label="Number of inference steps", minimum=1, maximum=40, step=1, value=4)
                        pic_height = gr.Slider(label="Height", minimum=256, maximum=2048, step=8, value=None)
                        pic_width = gr.Slider(label="Width", minimum=256, maximum=2048, step=8, value=None)


                preset_dropdown.change(fn=update_solo_prompt, inputs=[preset_dropdown], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown2.change(fn=update_couple_man_unseen_prompt, inputs=[preset_dropdown2], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown3.change(fn=update_couple_man_seen_prompt, inputs=[preset_dropdown3], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown4.change(fn=update_multiple_women_prompt, inputs=[preset_dropdown4], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown5.change(fn=update_multiple_man_unseen_prompt, inputs=[preset_dropdown5], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown6.change(fn=update_multiple_man_seen_prompt, inputs=[preset_dropdown6], outputs=[pic_prompt], scroll_to_output=False)
                preset_dropdown7.change(fn=update_multistep_prompt, inputs=[preset_dropdown7], outputs=[pic_prompt], scroll_to_output=False)

                _pic_infer_inputs = [
                    hidden_images_b64, pic_prompt, pic_negative_prompt,
                    pic_seed, pic_randomize_seed, pic_guidance, pic_steps,
                    pic_height, pic_width, pic_num_images,
                ]
                # JS pre-hook: pull the current images from window.__uploadedImages
                # into args[0], then return the FULL args array unchanged.
                # Returning all inputs is the correct Gradio 4.x contract (the
                # earlier "needed: 10, got: 0" error came from a prehook that
                # returned the wrong number of elements — this one returns all).
                # This prehook is REQUIRED: on a generator event it is what makes
                # Gradio establish the streaming connection with gr.Progress
                # (track_tqdm) attached, so the per-step progress overlay renders
                # on the output gallery in real time. Without it, generation still
                # works but the step progress does not stream.
                # JS pre-hook. Besides collecting the uploaded input images, it
                # ALSO clicks the result gallery's built-in clear/close ("X")
                # button so any previously generated photos are wiped from the
                # output box the instant Generate is pressed. Yielding [] from
                # the server did NOT visually clear the gr.Gallery in this Gradio
                # build (old thumbnails stayed painted and hid the progress
                # overlay), so we clear it client-side here, exactly like the
                # user clicking the X themselves.
                _pic_infer_js = """
async (...args) => {
    try {
        const gal = document.getElementById('picgen-result-gallery');
        if (gal) {
            // Gradio renders the clear control as a button with aria-label
            // "Clear" or "Close" (an X icon) in the gallery toolbar. Click
            // whichever exists to empty the output box before generating.
            let btn = gal.querySelector('button[aria-label="Clear"]')
                   || gal.querySelector('button[aria-label="Close"]')
                   || gal.querySelector('button[title="Clear"]')
                   || gal.querySelector('button[title="Close"]');
            if (btn) { btn.click(); }
        }
    } catch (e) { console.warn('picgen clear-on-generate failed', e); }

    // Downscale each ORIGINAL imported image to the currently-selected Quality
    // (window.__picgenMaxSize) at Generate time. This is why images are stored
    // at full resolution on import: changing the Quality button and clicking
    // Generate again re-derives the payload from the originals — no reimport,
    // no page refresh. If a downscale fails for any reason we fall back to the
    // original base64 so a generation is never blocked.
    const maxSize = window.__picgenMaxSize || 512;

    function resizeOne(b64) {
        return new Promise((resolve) => {
            try {
                const img = new Image();
                img.onload = () => {
                    try {
                        let w = img.naturalWidth || img.width;
                        let h = img.naturalHeight || img.height;
                        if (!w || !h) { resolve(b64); return; }
                        // Only downscale; never upscale past the original size.
                        const longest = Math.max(w, h);
                        const scale = longest > maxSize ? (maxSize / longest) : 1;
                        const nw = Math.max(1, Math.round(w * scale));
                        const nh = Math.max(1, Math.round(h * scale));
                        if (scale === 1) { resolve(b64); return; }
                        const canvas = document.createElement('canvas');
                        canvas.width = nw; canvas.height = nh;
                        canvas.getContext('2d').drawImage(img, 0, 0, nw, nh);
                        resolve(canvas.toDataURL('image/jpeg', 0.95));
                    } catch (e) { resolve(b64); }
                };
                img.onerror = () => resolve(b64);
                img.src = b64;
            } catch (e) { resolve(b64); }
        });
    }

    const imgs = window.__uploadedImages || [];
    let resized;
    try {
        resized = await Promise.all(imgs.map(i => resizeOne(i.b64)));
    } catch (e) {
        resized = imgs.map(i => i.b64);
    }
    const b64 = JSON.stringify(resized);
    args[0] = b64;
    return args;
}
"""

                def _wire_picgen_btn(trigger):
                    # IMPORTANT: point the event at the PLAIN `infer` function,
                    # NOT the `infer_with_preclear` generator. Wrapping infer in
                    # a generator (which yields [] first) put the gr.Gallery into
                    # generator-streaming mode, which replaced the real-time
                    # "Step x/N" overlay with the generic Gradio loading spinner.
                    # A plain function + gr.Progress(track_tqdm=True) is what lets
                    # Gradio hook the diffusion pipeline's internal tqdm and paint
                    # the per-step progress overlay INSIDE the output gallery, the
                    # way the working backup (backups/1) did it.
                    #
                    # Clearing of previously generated images is handled entirely
                    # client-side by the JS pre-hook (_pic_infer_js), which clicks
                    # the gallery's X/clear button the instant Generate is pressed,
                    # so we no longer need the server-side yield [] to clear.
                    #
                    # concurrency_id / concurrency_limit restore the dedicated
                    # picgen queue the backup used, which track_tqdm streaming
                    # relies on.
                    trigger(
                        fn=infer,
                        inputs=_pic_infer_inputs,
                        js=_pic_infer_js,
                        outputs=[pic_result, pic_seed, picgen_urls],
                        show_progress="full",
                        concurrency_id=PIC_QUEUE_ID,
                        concurrency_limit=1,
                    ).then(
                        fn=lambda: __import__('time').sleep(2),
                        inputs=[],
                        outputs=[],
                    ).then(
                        fn=protect_current_inputs,
                        inputs=[],
                        outputs=None,
                    ).then(
                        fn=clear_storage,
                        inputs=[],
                        outputs=[clear_storage_status],
                    )

                _wire_picgen_btn(pic_run_button.click)
                _wire_picgen_btn(pic_run_button_top.click)
                _wire_picgen_btn(pic_prompt.submit)

                # Tool buttons (Completed Body / Add L->R / Add R->L). These are
                # just PRESET PROMPTS: each button drops its preset text into the
                # prompt box, then runs the normal Qwen generate flow (same as
                # clicking Generate). No special pipeline.
                PICGEN_TOOL_PRESETS = {
                    "complete_body": (
                        "outpaint to only fill in what is missing only in the white part of the photo."
                    ),
                    "add_l2r": (
                        "Perfectly crop the person alone from image 1 to image 2 to add them naturally into only the scene of the second image without editing or regenerating it at all. "
                        "Fit them to be next to them in only 2nd's background. "
                        "Except you outpaint to only fill in what is missing of the parts of the body of the person from the first photo so their full body is fully visible with visibility underneath and everywhere around their body also. "
                        "They're same size as the other person. Everything else is the same. Only environment of the 2nd photo!!! "
                        "The man's penis remains unchanged."
                    ),
                    "add_r2l": (
                        "Perfectly crop the person alone from image 2 to image 1 to add them naturally into only the scene of the first image without editing or regenerating it at all. "
                        "Fit them to be next to them in only 1st's background. "
                        "Except you outpaint to only fill in what is missing of the parts of the body of the person from the second photo so their full body is fully visible with visibility underneath and everywhere around their body also. "
                        "They're same size as the other person. Everything else is the same. Only environment of the 1st photo!!! "
                        "The man's penis remains unchanged."
                    ),
                }

                def _set_prompt(text):
                    """Return the preset text to load into the prompt box."""
                    return text

                def _wire_picgen_tool_preset(button, preset_key):
                    # 1) set the preset prompt into the prompt box, then
                    # 2) run the exact same generate flow as the Generate button.
                    button.click(
                        fn=lambda k=preset_key: PICGEN_TOOL_PRESETS[k],
                        inputs=[],
                        outputs=[pic_prompt],
                    ).then(
                        fn=infer,
                        inputs=_pic_infer_inputs,
                        js=_pic_infer_js,
                        outputs=[pic_result, pic_seed, picgen_urls],
                        show_progress="full",
                        concurrency_id=PIC_QUEUE_ID,
                        concurrency_limit=1,
                    ).then(
                        fn=lambda: __import__('time').sleep(2),
                        inputs=[],
                        outputs=[],
                    ).then(
                        fn=protect_current_inputs,
                        inputs=[],
                        outputs=None,
                    ).then(
                        fn=clear_storage,
                        inputs=[],
                        outputs=[clear_storage_status],
                    )

                _wire_picgen_tool_preset(pic_add_l2r_btn, "add_l2r")
                _wire_picgen_tool_preset(pic_add_r2l_btn, "add_r2l")

                # Completed Body button opens the outpaint popup via JS only.
                # The popup handles everything client-side: padding canvas,
                # replacing the gallery image, setting the prompt, and clicking
                # Generate. No server round-trip until Generate is pressed.
                pic_complete_body_btn.click(
                    fn=None,
                    inputs=[],
                    outputs=[],
                    js="() => { if (window.__openOutpaintPopup) window.__openOutpaintPopup(); }",
                )


                def output_to_b64(output_images):
                    """Convert gallery items to base64 JPEG list for the input gallery.
                    Gallery type="filepath": items are file path strings or dicts with 'name'/'path'."""
                    if not output_images:
                        return "[]"
                    b64_list = []
                    for item in output_images:
                        try:
                            if isinstance(item, dict):
                                fpath = item.get("name") or item.get("path") or item.get("url") or ""
                            elif isinstance(item, (list, tuple)):
                                fpath = item[0] if item else ""
                            else:
                                fpath = str(item)
                            if not fpath or not os.path.exists(fpath):
                                continue
                            img = Image.open(fpath).convert("RGB")
                            max_size = 512
                            if img.width > img.height:
                                img = img.resize((max_size, int(img.height * max_size / img.width)), Image.LANCZOS)
                            else:
                                img = img.resize((int(img.width * max_size / img.height), max_size), Image.LANCZOS)
                            buf = BytesIO()
                            img.save(buf, format="JPEG", quality=95)
                            b64_list.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
                        except Exception as e:
                            print(f"output_to_b64 skipped an item: {e}")
                            continue
                    return json.dumps(b64_list)

                use_output_btn.click(fn=output_to_b64, inputs=[pic_result], outputs=[hidden_images_b64])

                pic_download_btn.click(
                    fn=None,
                    inputs=[],
                    outputs=[],
                    js="""
                    async () => {
                        // Download every image currently shown in the result
                        // gallery. We read the actual <img> src values from the
                        // DOM (works for Gradio /file= URLs, blob: and data:
                        // URLs, and encrypted /media/ URLs alike) rather than a
                        // separate URL list that could be out of sync.
                        const gal = document.getElementById('picgen-result-gallery');
                        if (!gal) { alert('No images to download — generate images first.'); return; }
                        const imgEls = Array.from(gal.querySelectorAll('img'));
                        // De-duplicate by src (galleries can render thumb + preview).
                        const seen = new Set();
                        const srcs = [];
                        imgEls.forEach(im => {
                            const s = im.currentSrc || im.src;
                            if (s && !seen.has(s)) { seen.add(s); srcs.push(s); }
                        });
                        if (srcs.length === 0) { alert('No images to download — generate images first.'); return; }

                        const _f = window.__ngOrigFetch || window.fetch;
                        for (let i = 0; i < srcs.length; i++) {
                            const src = srcs[i];
                            let filename = 'picgen_' + (i + 1) + '.png';
                            try {
                                const clean = src.split('?')[0].split('#')[0];
                                const last = clean.split('/').pop();
                                if (last && last.indexOf('.') !== -1) filename = decodeURIComponent(last);
                            } catch (e) {}
                            try {
                                const headers = {};
                                const isMedia = src.indexOf('/media/') !== -1;
                                if (isMedia && window.__ngSecretHex) headers['X-NG-Secret'] = window.__ngSecretHex;
                                const resp = await _f(src, { headers });
                                if (!resp.ok) { console.warn('[PicDL] fetch failed', src, resp.status); continue; }
                                let plainBytes;
                                const encrypted = resp.headers.get('X-NG-Encrypted');
                                if (encrypted === '1' && window.__ngKey) {
                                    const buf = await resp.arrayBuffer();
                                    try {
                                        plainBytes = await crypto.subtle.decrypt(
                                            { name: 'AES-GCM', iv: buf.slice(0, 12) },
                                            window.__ngKey, buf.slice(12)
                                        );
                                    } catch (e) { console.warn('[PicDL] decrypt:', e); continue; }
                                } else {
                                    plainBytes = await resp.arrayBuffer();
                                }
                                const ext = (filename.split('.').pop() || 'png').toLowerCase();
                                const mime = ext === 'png' ? 'image/png'
                                           : (ext === 'webp' ? 'image/webp' : 'image/jpeg');
                                const blob = new Blob([plainBytes], { type: mime });
                                const blobUrl = URL.createObjectURL(blob);
                                const a = document.createElement('a');
                                a.href = blobUrl;
                                a.download = filename;
                                document.body.appendChild(a);
                                a.click();
                                document.body.removeChild(a);
                                setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);
                                // Small stagger so the browser accepts multiple downloads.
                                await new Promise(r => setTimeout(r, 350));
                            } catch (e) { console.warn('[PicDL] failed:', src, e); }
                        }
                    }
                    """,
                )
                
                hidden_images_b64.change(
                    fn=None, inputs=[hidden_images_b64], outputs=None,
                    js="""(b64List) => {
                        if (!b64List || !window.__addImage) return;
                        // Guard: syncToGradio() dispatches input+change on this
                        // same textbox, which would re-trigger this handler and
                        // loop forever. Skip if the change originated from us.
                        if (window.__picgenSuppressSync) return;
                        try {
                            const incoming = JSON.parse(b64List);
                            if (!Array.isArray(incoming) || incoming.length === 0) return;
                            // Skip if gallery already contains exactly these images
                            const cur = (window.__picgenImages || []).map(i => i.b64);
                            if (incoming.length === cur.length && incoming.every((b,i) => b === cur[i])) return;
                            // Add the incoming images alongside any that are already there.
                            // Do NOT clear first — this handler is triggered by "Use as input"
                            // which should append, not replace.
                            window.__picgenSuppressSync = true;
                            try {
                                incoming.forEach((b64, idx) => {
                                    window.__addImage(b64, `output_${idx + 1}.png`);
                                });
                            } finally {
                                window.__picgenSuppressSync = false;
                            }
                        } catch (e) {
                            window.__picgenSuppressSync = false;
                            console.error('Failed to parse output images:', e);
                        }
                    }""",
                )

                starter_b64_output.change(
                    fn=None, inputs=[starter_b64_output], outputs=None,
                    js="(b64) => { if (b64 && window.__addImage) window.__addImage(b64, 'starter.jpg'); }",
                )




    demo.load(fn=None, js=gallery_js)
    _picgen_dl_intercept_js = """
() => {
    async function fetchAndDownload(mediaUrl) {
        if (!mediaUrl || !mediaUrl.startsWith('/media/')) return false;
        const filename = mediaUrl.split('/').pop().split('?')[0] || 'picgen.png';

        // Wait up to 5s for encryption key
        let waited = 0;
        while (!window.__ngSecretHex && waited < 5000) {
            await new Promise(r => setTimeout(r, 100));
            waited += 100;
        }

        try {
            const headers = {};
            if (window.__ngSecretHex) headers['X-NG-Secret'] = window.__ngSecretHex;
            const _f = window.__ngOrigFetch || window.fetch;
            const resp = await _f(mediaUrl, { headers });
            if (!resp.ok) return false;
            const encrypted = resp.headers.get('X-NG-Encrypted');
            let plainBytes;
            if (encrypted === '1' && window.__ngKey) {
                const buf = await resp.arrayBuffer();
                plainBytes = await crypto.subtle.decrypt(
                    { name: 'AES-GCM', iv: buf.slice(0, 12) },
                    window.__ngKey,
                    buf.slice(12)
                );
            } else {
                plainBytes = await resp.arrayBuffer();
            }
            const ext = filename.split('.').pop().toLowerCase();
            const mime = ext === 'png' ? 'image/png' : ext === 'webp' ? 'image/webp' : 'image/jpeg';
            const blob = new Blob([plainBytes], { type: mime });
            const blobUrl = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = blobUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);
            return true;
        } catch(e) {
            console.warn('[PicGalleryDL] fetch/decrypt failed:', e);
            return false;
        }
    }

    function getMediaUrls() {
        // Read the hidden picgen_urls textbox value
        const el = document.getElementById('picgen-urls');
        if (!el) return [];
        const ta = el.querySelector('textarea') || el.querySelector('input');
        if (!ta) return [];
        try { return JSON.parse(ta.value || '[]'); } catch(e) { return []; }
    }

    function patchGalleryDownloadButtons(gallery) {
        // Find all download anchor/button elements in the gallery.
        // Gradio renders them as <a download> or buttons with a download SVG.
        // We intercept the click event and prevent the default broken /file= request.
        const items = gallery.querySelectorAll('.thumbnail-item, [class*="thumbnail"]');
        items.forEach((item, idx) => {
            // Find the download button inside this item
            const dlBtn = item.querySelector('a[download], button[aria-label*="ownload"], a[aria-label*="ownload"]');
            if (!dlBtn || dlBtn.dataset.ngPatched) return;
            dlBtn.dataset.ngPatched = '1';
            dlBtn.addEventListener('click', async (e) => {
                const urls = getMediaUrls();
                if (urls.length === 0) return; // no urls, let default happen
                e.preventDefault();
                e.stopPropagation();
                const mediaUrl = urls[idx];
                if (mediaUrl) {
                    await fetchAndDownload(mediaUrl);
                }
            }, true);
        });
    }

    function watchPicgenGallery() {
        const gallery = document.getElementById('picgen-result-gallery');
        if (!gallery) { setTimeout(watchPicgenGallery, 500); return; }
        // Watch for gallery content changes (new images rendered after generation)
        const obs = new MutationObserver(() => patchGalleryDownloadButtons(gallery));
        obs.observe(gallery, { childList: true, subtree: true });
        patchGalleryDownloadButtons(gallery);
    }
    setTimeout(watchPicgenGallery, 1000);
}
"""
    demo.load(fn=None, js=_picgen_dl_intercept_js)
    # NOTE: The starter grid is built ONCE at UI construction time via
    # _build_starter_grid_html() as the initial value of starter_grid_html.
    # It is deliberately NOT wired to demo.load here — a queued Python
    # demo.load handler disrupts gr.Progress(track_tqdm=True) progress
    # streaming on subsequent picgen generations. Building it inline keeps the
    # event queue clean while still showing thumbnails at startup.


    _encryption_init_js = """
() => {
    // ---- Key derivation (runs once per page load) -------------------------
    async function initEncryptionKey() {
        if (window.__ngKey) return;  // already initialised

        // Retrieve or generate the 32-byte device secret stored in localStorage.
        // This secret never leaves the browser — the server never sees the key,
        // only the secret from which it derives the matching key server-side.
        let secretHex = localStorage.getItem('__ngSecret__');
        if (!secretHex || secretHex.length !== 64) {
            const raw = new Uint8Array(32);
            crypto.getRandomValues(raw);
            secretHex = Array.from(raw).map(b => b.toString(16).padStart(2,'0')).join('');
            localStorage.setItem('__ngSecret__', secretHex);
        }
        window.__ngSecretHex = secretHex;

        // Import the raw secret as HKDF base key material
        const secretBytes = new Uint8Array(secretHex.match(/../g).map(h => parseInt(h,16)));
        const baseKey = await crypto.subtle.importKey(
            'raw', secretBytes, { name: 'HKDF' }, false, ['deriveKey']
        );

        // Derive AES-256-GCM media key (non-extractable, memory only)
        const enc = new TextEncoder();
        window.__ngKey = await crypto.subtle.deriveKey(
            { name: 'HKDF', hash: 'SHA-256', salt: enc.encode('newgen-media-v1'), info: enc.encode('aes-256-gcm-media') },
            baseKey,
            { name: 'AES-GCM', length: 256 },
            false,
            ['decrypt']
        );

        // Derive AES-256-GCM log key (non-extractable, memory only)
        window.__ngLogKey = await crypto.subtle.deriveKey(
            { name: 'HKDF', hash: 'SHA-256', salt: enc.encode('newgen-logs-v1'), info: enc.encode('aes-256-gcm-logs') },
            baseKey,
            { name: 'AES-GCM', length: 256 },
            false,
            ['decrypt']
        );
    }

    // ---- /media/ fetch interceptor ----------------------------------------
    // Wraps the global fetch so every request to /media/ automatically:
    //   1. Adds the X-NG-Secret header
    //   2. Decrypts the AES-256-GCM response before returning it
    // This is transparent to all other code (video player, gallery, downloads).
    const _origFetch = window.fetch;
    window.__ngOrigFetch = _origFetch;  // saved so video player can bypass interceptor
    window.fetch = async function(input, init) {
        const url = (typeof input === 'string') ? input : (input instanceof Request ? input.url : String(input));
        const isMedia = url.includes('/media/');

        if (!isMedia || !window.__ngSecretHex) {
            return _origFetch(input, init);
        }

        // Inject the secret header
        const headers = new Headers((init && init.headers) ? init.headers : {});
        headers.set('X-NG-Secret', window.__ngSecretHex);
        const newInit = Object.assign({}, init || {}, { headers });

        const response = await _origFetch(input, newInit);
        if (!response.ok) return response;

        const encrypted = response.headers.get('X-NG-Encrypted');
        if (encrypted !== '1' || !window.__ngKey) return response;

        // Decrypt: response body = 12-byte nonce + ciphertext
        const buf = await response.arrayBuffer();
        const nonce = buf.slice(0, 12);
        const ciphertext = buf.slice(12);
        let plaintext;
        try {
            plaintext = await crypto.subtle.decrypt(
                { name: 'AES-GCM', iv: nonce },
                window.__ngKey,
                ciphertext
            );
        } catch(e) {
            console.warn('[NG-Enc] media decrypt failed:', e);
            return new Response(new Uint8Array(0), { status: 200 });
        }

        const origType = response.headers.get('X-NG-Original-Type') || 'application/octet-stream';
        const origName = response.headers.get('X-NG-Filename') || '';
        const respHeaders = new Headers();
        respHeaders.set('Content-Type', origType);
        respHeaders.set('Content-Length', String(plaintext.byteLength));
        if (origName) respHeaders.set('Content-Disposition', 'inline; filename="' + origName + '"');
        return new Response(plaintext, { status: 200, headers: respHeaders });
    };

    // ---- SSE log panel -------------------------------------------------------
    function connectLogStream() {
        if (!window.__ngLogKey || !window.__ngSecretHex) {
            setTimeout(connectLogStream, 1000);
            return;
        }
        const es = new EventSource('/logs/stream?_=' + Date.now());

        // We can't add custom headers to EventSource — use a modified URL approach:
        // close EventSource and switch to fetch-based SSE with the secret header.
        es.close();

        async function fetchSSE() {
            const dec = new TextDecoder();
            try {
                const resp = await _origFetch('/logs/stream', {
                    headers: { 'X-NG-Secret': window.__ngSecretHex },
                });
                if (!resp.body) return;
                const reader = resp.body.getReader();
                let buf = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buf += dec.decode(value, { stream: true });
                    const lines = buf.split('\\n');
                    buf = lines.pop();
                    for (const raw of lines) {
                        const line = raw.trim();
                        if (!line || line.startsWith(':')) continue;  // keepalive
                        const b64 = line.replace(/^data:\\s*/, '');
                        if (!b64) continue;
                        try {
                            const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
                            const nonce = bytes.slice(0, 12);
                            const ct    = bytes.slice(12);
                            const ptBuf = await crypto.subtle.decrypt(
                                { name: 'AES-GCM', iv: nonce },
                                window.__ngLogKey,
                                ct
                            );
                            const text = new TextDecoder().decode(ptBuf);
                            const panel = document.getElementById('ng-log-panel');
                            if (panel) {
                                const line_el = document.createElement('div');
                                line_el.textContent = text;
                                panel.appendChild(line_el);
                                // Keep last 500 lines
                                while (panel.children.length > 500) panel.removeChild(panel.firstChild);
                                panel.scrollTop = panel.scrollHeight;
                            }
                        } catch(e) { /* decrypt error on keepalive or bad frame */ }
                    }
                }
            } catch(e) {
                // Reconnect after 2s on any stream error
                setTimeout(fetchSSE, 2000);
            }
        }
        fetchSSE();
    }

    // Initialise key then start log stream
    initEncryptionKey().then(() => {
        connectLogStream();
    });
}
"""
    demo.load(fn=None, js=_encryption_init_js)

    # -- Multi-Tool popup ------------------------------------------------------
    # Opened when the user clicks "Multi-Tool". Shows the current first gallery
    # image (or an upload zone if none). Edge drag-handles let the user extend
    # (outward = white space added) or crop (inward = image trimmed) on any
    # side. Buttons: Upload (import image), Update Input (push to gallery, no
    # generation), Reset, Generate (sets prompt + triggers normal Generate).
    _outpaint_popup_js = r"""
() => {
window.__openOutpaintPopup = function() {
    // -- constants ------------------------------------------------------------
    const PROMPT = 'outpaint to only fill in what is missing only in the white part of the photo.';
    const HANDLE_SIZE = 28;   // px — draggable edge strip width
    const MIN_PAD = -4000;    // negative = crop inward
    const MAX_PAD = 1200;     // positive = add white space outward

    // -- state ----------------------------------------------------------------
    let srcB64 = null;        // original image base64
    let padTop = 0, padRight = 0, padBottom = 0, padLeft = 0;
    let dragging = null;
    // Index of the gallery image currently shown in the popup (cycles on swap)
    let _galleryIdx = 0;
    // Track whether the loaded image came from the popup's own upload
    // (true) or was pulled from the gallery (false). Determines whether
    // Update/Generate adds to the gallery or replaces.
    let _fromPopupUpload = false;

    // -- overlay --------------------------------------------------------------
    let overlay = document.getElementById('ng-outpaint-overlay');
    if (overlay) {
        const imgs = window.__picgenImages || [];
        // Restore last-viewed gallery index; clamp in case images were removed
        _galleryIdx = Math.min(overlay._galleryIdx || 0, Math.max(0, imgs.length - 1));
        const fresh = imgs.length > 0 ? imgs[_galleryIdx].b64 : null;
        if (fresh && overlay._loadSrc) {
            _fromPopupUpload = false;
            overlay._loadSrc(fresh);
        } else if (!fresh && overlay._render) {
            overlay._render();
        }
        // Show/hide swap button depending on current image count
        if (overlay._swapBtn) {
            overlay._swapBtn.style.display = imgs.length > 1 ? '' : 'none';
            overlay._swapBtn.textContent = '\u21c4 Swap (' + (imgs.length > 0 ? (_galleryIdx + 1) + '/' + imgs.length : '0') + ')';
        }
        overlay.style.display = 'flex';
        return;
    }

    overlay = document.createElement('div');
    overlay.id = 'ng-outpaint-overlay';
    overlay.style.cssText = [
        'position:fixed;top:0;left:0;width:100%;height:100%;',
        'background:rgba(0,0,0,0.85);z-index:20000;',
        'display:flex;flex-direction:column;align-items:center;justify-content:center;',
        'gap:12px;'
    ].join('');
    document.body.appendChild(overlay);

    // Title + close
    const header = document.createElement('div');
    header.style.cssText = 'display:flex;align-items:center;gap:16px;width:min(92vw,860px);';
    header.innerHTML = '<span style="color:#fff;font-size:15px;font-weight:600;flex:1;">'
        + 'Multi-Tool \u2014 drag edges outward to add space, inward to crop</span>'
        + '<button id="ng-op-close" style="width:32px;height:32px;border-radius:50%;background:rgba(255,255,255,0.15);'
        + 'color:#fff;border:1px solid rgba(255,255,255,0.3);cursor:pointer;font-size:18px;'
        + 'display:flex;align-items:center;justify-content:center;">\u00d7</button>';
    overlay.appendChild(header);

    // Upload button row — always visible so user can import at any time
    const uploadRow = document.createElement('div');
    uploadRow.style.cssText = 'display:flex;align-items:center;gap:10px;width:min(92vw,860px);flex-wrap:wrap;';
    uploadRow.innerHTML = '<button id="ng-op-upload-btn" style="padding:6px 16px;border-radius:6px;'
        + 'background:rgba(255,255,255,0.12);color:#fff;border:1px solid rgba(255,255,255,0.35);'
        + 'cursor:pointer;font-size:12px;font-weight:500;">&#128193; Import image</button>'
        + '<button id="ng-op-swap" style="padding:6px 14px;border-radius:6px;display:none;'
        + 'background:rgba(255,200,80,0.18);color:#ffd966;border:1px solid rgba(255,200,80,0.5);'
        + 'cursor:pointer;font-size:12px;font-weight:600;">\u21c4 Swap (1/1)</button>'
        + '<span id="ng-op-upload-hint" style="font-size:11px;color:rgba(255,255,255,0.45);">'
        + 'Imports into the popup only \u2014 use Update Input or Generate to push to the gallery</span>'
        + '<input id="ng-op-file" type="file" accept="image/*" style="display:none;">';
    overlay.appendChild(uploadRow);

    // Upload drop zone — only shown when no image is loaded yet
    const uploadZone = document.createElement('div');
    uploadZone.id = 'ng-op-dropzone';
    uploadZone.style.cssText = 'width:min(92vw,860px);height:280px;border:2px dashed rgba(255,255,255,0.3);'
        + 'border-radius:10px;display:flex;flex-direction:column;align-items:center;justify-content:center;'
        + 'gap:8px;cursor:pointer;color:rgba(255,255,255,0.6);font-size:13px;';
    uploadZone.innerHTML = '<svg width="48" height="48" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2">'
        + '<rect x="6" y="10" width="52" height="44" rx="5" stroke-dasharray="4 3"/>'
        + '<polygon points="10,50 24,32 34,42 44,28 54,50" fill="rgba(255,255,255,0.08)"/>'
        + '<circle cx="22" cy="24" r="5" fill="rgba(255,255,255,0.08)"/></svg>'
        + '<span>Click here or drag an image to load it</span>';
    overlay.appendChild(uploadZone);

    // Canvas wrapper — shown once an image is loaded
    const canvasWrap = document.createElement('div');
    canvasWrap.id = 'ng-op-wrap';
    canvasWrap.style.cssText = 'position:relative;display:none;touch-action:none;user-select:none;';
    overlay.appendChild(canvasWrap);

    const cvs = document.createElement('canvas');
    cvs.id = 'ng-op-canvas';
    cvs.style.cssText = 'display:block;border:1px solid rgba(255,255,255,0.2);border-radius:4px;';
    canvasWrap.appendChild(cvs);

    // Pad labels per edge
    const labels = {};
    ['top','right','bottom','left'].forEach(edge => {
        const lbl = document.createElement('div');
        lbl.id = 'ng-op-lbl-' + edge;
        lbl.style.cssText = 'position:absolute;background:rgba(0,0,0,0.7);color:#fff;'
            + 'font-size:11px;padding:2px 6px;border-radius:3px;pointer-events:none;white-space:nowrap;';
        canvasWrap.appendChild(lbl);
        labels[edge] = lbl;
    });

    // Drag handles — 4 edges
    const handles = {};
    ['top','right','bottom','left'].forEach(edge => {
        const h = document.createElement('div');
        h.dataset.edge = edge;
        h.style.cssText = 'position:absolute;cursor:'
            + (edge === 'top' || edge === 'bottom' ? 'ns-resize' : 'ew-resize')
            + ';z-index:10;';
        canvasWrap.appendChild(h);
        handles[edge] = h;
    });

    // Button row: Reset | Update Input | Generate
    const btnRow = document.createElement('div');
    btnRow.style.cssText = 'display:flex;gap:10px;flex-wrap:wrap;justify-content:center;';
    btnRow.innerHTML = ''
        + '<button id="ng-op-reset" style="padding:7px 18px;border-radius:6px;'
        + 'background:rgba(255,255,255,0.1);color:#fff;border:1px solid rgba(255,255,255,0.3);'
        + 'cursor:pointer;font-size:13px;">Reset</button>'
        + '<button id="ng-op-update" style="padding:7px 20px;border-radius:6px;'
        + 'background:rgba(80,160,255,0.25);color:#fff;border:1px solid rgba(80,160,255,0.6);'
        + 'cursor:pointer;font-size:13px;font-weight:600;">Update Input</button>'
        + '<button id="ng-op-generate" style="padding:7px 24px;border-radius:6px;'
        + 'background:#7c5cbf;color:#fff;border:none;cursor:pointer;font-size:14px;font-weight:600;">'
        + 'Generate (4 images)</button>';
    overlay.appendChild(btnRow);

    // -- helpers ---------------------------------------------------------------
    function _clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

    function _positionHandles(dw, dh) {
        const strip = HANDLE_SIZE;
        handles.top.style.cssText    += `;left:0;top:0;width:${dw}px;height:${strip}px;`;
        handles.bottom.style.cssText += `;left:0;bottom:0;width:${dw}px;height:${strip}px;`;
        handles.left.style.cssText   += `;top:0;left:0;width:${strip}px;height:${dh}px;`;
        handles.right.style.cssText  += `;top:0;right:0;width:${strip}px;height:${dh}px;`;
        labels.top.style.cssText     += `;left:50%;top:${strip+2}px;transform:translateX(-50%);`;
        labels.bottom.style.cssText  += `;left:50%;bottom:${strip+2}px;transform:translateX(-50%);`;
        labels.left.style.cssText    += `;top:50%;left:${strip+2}px;transform:translateY(-50%);`;
        labels.right.style.cssText   += `;top:50%;right:${strip+2}px;transform:translateY(-50%);`;
    }

    function _labelText(px) {
        if (px === 0) return '';
        return (px > 0 ? '+' : '') + px + 'px';
    }

    function _updateLabels() {
        labels.top.textContent    = _labelText(padTop);
        labels.right.textContent  = _labelText(padRight);
        labels.bottom.textContent = _labelText(padBottom);
        labels.left.textContent   = _labelText(padLeft);
    }

    function _render() {
        if (!srcB64) return;
        const img = new Image();
        img.onload = () => {
            const origW = img.naturalWidth;
            const origH = img.naturalHeight;

            // Effective canvas includes padding (positive) and excludes crop (negative)
            const totalW = Math.max(1, origW + padLeft + padRight);
            const totalH = Math.max(1, origH + padTop  + padBottom);

            const maxCssW = Math.floor(window.innerWidth  * 0.82);
            const maxCssH = Math.floor(window.innerHeight * 0.62);
            const scale   = Math.min(maxCssW / totalW, maxCssH / totalH, 1);
            const cssW    = Math.round(totalW * scale);
            const cssH    = Math.round(totalH * scale);

            cvs.width  = totalW;
            cvs.height = totalH;
            cvs.style.width  = cssW + 'px';
            cvs.style.height = cssH + 'px';

            const ctx = cvs.getContext('2d');
            // White background for padded areas
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, totalW, totalH);
            // Draw image offset by left/top padding (negative = image shifted left/up = crop)
            ctx.drawImage(img, padLeft, padTop, origW, origH);

            // Guide lines for padding (blue) and crop borders (red)
            // Top
            if (padTop !== 0) {
                ctx.strokeStyle = padTop > 0 ? 'rgba(100,150,255,0.5)' : 'rgba(255,80,80,0.7)';
                ctx.lineWidth = 2; ctx.setLineDash(padTop < 0 ? [6,3] : []);
                ctx.beginPath(); ctx.moveTo(0, padTop); ctx.lineTo(totalW, padTop); ctx.stroke();
                ctx.setLineDash([]);
            }
            // Bottom
            if (padBottom !== 0) {
                ctx.strokeStyle = padBottom > 0 ? 'rgba(100,150,255,0.5)' : 'rgba(255,80,80,0.7)';
                ctx.lineWidth = 2; ctx.setLineDash(padBottom < 0 ? [6,3] : []);
                ctx.beginPath(); ctx.moveTo(0, origH+padTop); ctx.lineTo(totalW, origH+padTop); ctx.stroke();
                ctx.setLineDash([]);
            }
            // Left
            if (padLeft !== 0) {
                ctx.strokeStyle = padLeft > 0 ? 'rgba(100,150,255,0.5)' : 'rgba(255,80,80,0.7)';
                ctx.lineWidth = 2; ctx.setLineDash(padLeft < 0 ? [6,3] : []);
                ctx.beginPath(); ctx.moveTo(padLeft, 0); ctx.lineTo(padLeft, totalH); ctx.stroke();
                ctx.setLineDash([]);
            }
            // Right
            if (padRight !== 0) {
                ctx.strokeStyle = padRight > 0 ? 'rgba(100,150,255,0.5)' : 'rgba(255,80,80,0.7)';
                ctx.lineWidth = 2; ctx.setLineDash(padRight < 0 ? [6,3] : []);
                ctx.beginPath(); ctx.moveTo(origW+padLeft, 0); ctx.lineTo(origW+padLeft, totalH); ctx.stroke();
                ctx.setLineDash([]);
            }

            canvasWrap.style.width  = cssW + 'px';
            canvasWrap.style.height = cssH + 'px';
            uploadZone.style.display  = 'none';
            canvasWrap.style.display  = 'block';

            _positionHandles(cssW, cssH);
            _updateLabels();
        };
        img.src = srcB64;
    }

    function _loadSrc(b64) {
        srcB64 = b64;
        padTop = padRight = padBottom = padLeft = 0;
        // Persist the current gallery index on the overlay element so the
        // re-open branch can restore it next time the popup is opened.
        if (overlay) overlay._galleryIdx = _galleryIdx;
        _render();
    }
    overlay._loadSrc = _loadSrc;
    overlay._render  = _render;

    // -- swap button wiring ----------------------------------------------------
    const swapBtn = document.getElementById('ng-op-swap');
    overlay._swapBtn = swapBtn;
    // Show button only when more than one gallery image exists
    const _imgs0 = window.__picgenImages || [];
    if (_imgs0.length > 1) {
        swapBtn.style.display = '';
        swapBtn.textContent = '\u21c4 Swap (1/' + _imgs0.length + ')';
    }
    swapBtn.addEventListener('click', () => {
        const imgs = window.__picgenImages || [];
        if (imgs.length < 2) return;
        _galleryIdx = (_galleryIdx + 1) % imgs.length;
        overlay._galleryIdx = _galleryIdx;
        _fromPopupUpload = false;
        _loadSrc(imgs[_galleryIdx].b64);
        swapBtn.textContent = '\u21c4 Swap (' + (_galleryIdx + 1) + '/' + imgs.length + ')';
    });

    // -- drag logic ------------------------------------------------------------
    function _onPointerDown(e) {
        const edge = e.currentTarget.dataset.edge;
        if (!edge || !srcB64) return;
        e.preventDefault();
        dragging = { edge, startX: e.clientX, startY: e.clientY,
                     startPad: { top:padTop, right:padRight, bottom:padBottom, left:padLeft } };
        window.addEventListener('pointermove', _onPointerMove);
        window.addEventListener('pointerup',   _onPointerUp);
    }

    function _onPointerMove(e) {
        if (!dragging) return;
        const dx = e.clientX - dragging.startX;
        const dy = e.clientY - dragging.startY;
        const img = new Image();
        img.src = srcB64;
        const origW = cvs.width  - dragging.startPad.left - dragging.startPad.right;
        const origH = cvs.height - dragging.startPad.top  - dragging.startPad.bottom;
        const totalW = origW + dragging.startPad.left + dragging.startPad.right;
        const totalH = origH + dragging.startPad.top  + dragging.startPad.bottom;
        const maxCssW = Math.floor(window.innerWidth  * 0.82);
        const maxCssH = Math.floor(window.innerHeight * 0.62);
        const scale   = Math.min(maxCssW / totalW, maxCssH / totalH, 1);
        switch (dragging.edge) {
            case 'top':    padTop    = _clamp(dragging.startPad.top    - Math.round(dy / scale), MIN_PAD, MAX_PAD); break;
            case 'bottom': padBottom = _clamp(dragging.startPad.bottom + Math.round(dy / scale), MIN_PAD, MAX_PAD); break;
            case 'left':   padLeft   = _clamp(dragging.startPad.left   - Math.round(dx / scale), MIN_PAD, MAX_PAD); break;
            case 'right':  padRight  = _clamp(dragging.startPad.right  + Math.round(dx / scale), MIN_PAD, MAX_PAD); break;
        }
        _render();
    }

    function _onPointerUp() {
        if (dragging) _recentlyDragged = true;
        dragging = null;
        window.removeEventListener('pointermove', _onPointerMove);
        window.removeEventListener('pointerup',   _onPointerUp);
        setTimeout(() => { _recentlyDragged = false; }, 400);
    }

    Object.values(handles).forEach(h => h.addEventListener('pointerdown', _onPointerDown));

    // -- file import -----------------------------------------------------------
    function _handleFile(file) {
        if (!file || !file.type.startsWith('image/')) return;
        const reader = new FileReader();
        reader.onload = (ev) => {
            _fromPopupUpload = true;
            _loadSrc(ev.target.result);
        };
        reader.readAsDataURL(file);
    }

    const opFile = document.getElementById('ng-op-file');
    document.getElementById('ng-op-upload-btn').addEventListener('click', () => opFile.click());
    opFile.addEventListener('change', (e) => { _handleFile(e.target.files[0]); e.target.value = ''; });
    uploadZone.addEventListener('click', () => opFile.click());
    uploadZone.addEventListener('dragover',  (e) => { e.preventDefault(); uploadZone.style.borderColor='#7c5cbf'; });
    uploadZone.addEventListener('dragleave', () => { uploadZone.style.borderColor='rgba(255,255,255,0.3)'; });
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.style.borderColor='rgba(255,255,255,0.3)';
        _handleFile(e.dataTransfer.files[0]);
    });

    // -- export canvas to b64 -------------------------------------------------
    function _exportCanvas() {
        return cvs.toDataURL('image/jpeg', 0.95);
    }

    // -- push to gallery (shared by Update Input and Generate) -----------------
    // If the image came from the popup's own upload, ADD it alongside existing images.
    // If it came from the gallery, REPLACE only that specific slot (_galleryIdx)
    // so all other images are preserved.
    function _pushToGallery(exportB64, name) {
        window.__picgenSuppressSync = true;
        try {
            if (_fromPopupUpload) {
                // Fresh import — add without disturbing existing images
                if (window.__addImage) window.__addImage(exportB64, name);
            } else {
                // Editing an existing gallery slot — swap only that index
                const imgs = window.__picgenImages;
                if (imgs && imgs.length > 0) {
                    const idx = Math.min(_galleryIdx, imgs.length - 1);
                    // Take a snapshot, replace the target slot, then rebuild gallery
                    const snapshot = imgs.map((img, i) =>
                        i === idx ? { id: Date.now() + Math.random(), b64: exportB64, name: name } : img
                    );
                    // Rebuild: clear then re-add in order (triggers proper render + sync)
                    imgs.length = 0;
                    if (window.__clearAll) window.__clearAll();
                    snapshot.forEach(img => { if (window.__addImage) window.__addImage(img.b64, img.name); });
                } else {
                    // Gallery is empty — just add
                    if (window.__addImage) window.__addImage(exportB64, name);
                }
            }
        } finally {
            window.__picgenSuppressSync = false;
            const _imgs = window.__picgenImages || [];
            const _container = document.getElementById('hidden-images-b64');
            if (_container) {
                _container.querySelectorAll('input,textarea').forEach(el => {
                    const _proto = el.tagName === 'TEXTAREA' ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
                    const _ns = Object.getOwnPropertyDescriptor(_proto, 'value');
                    if (_ns && _ns.set) {
                        _ns.set.call(el, JSON.stringify(_imgs.map(i => i.b64)));
                        el.dispatchEvent(new Event('input', {bubbles:true, composed:true}));
                    }
                });
            }
        }
    }

    // -- close -----------------------------------------------------------------
    let _recentlyDragged = false;
    function _close() { overlay.style.display = 'none'; _recentlyDragged = false; }
    document.getElementById('ng-op-close').addEventListener('click', _close);
    overlay.addEventListener('click', (e) => {
        if (_recentlyDragged) { e.stopImmediatePropagation(); _recentlyDragged = false; }
    });

    // -- reset -----------------------------------------------------------------
    document.getElementById('ng-op-reset').addEventListener('click', () => {
        padTop = padRight = padBottom = padLeft = 0;
        _render();
    });

    // -- update input (no generation) -----------------------------------------
    document.getElementById('ng-op-update').addEventListener('click', () => {
        if (!srcB64) { alert('Load an image first.'); return; }
        _pushToGallery(_exportCanvas(), 'edited_input.jpg');
        _close();
    });

    // -- generate -------------------------------------------------------------
    document.getElementById('ng-op-generate').addEventListener('click', () => {
        if (!srcB64) { alert('Load an image first.'); return; }
        _pushToGallery(_exportCanvas(), 'outpaint_input.jpg');

        // Write prompt into the visible Prompt textarea
        const picgenCol = document.getElementById('col-container');
        let promptTA = null;
        if (picgenCol) {
            for (const ta of Array.from(picgenCol.querySelectorAll('textarea'))) {
                const block = ta.closest('.block') || ta.parentElement;
                const lbl = block ? block.querySelector('label span, .label-wrap span') : null;
                if (lbl && lbl.textContent.trim().toLowerCase() === 'prompt') { promptTA = ta; break; }
            }
            if (!promptTA) {
                const all = Array.from(picgenCol.querySelectorAll('textarea'));
                promptTA = all.find(ta => {
                    const b = ta.closest('.block') || ta.parentElement;
                    const l = b ? (b.querySelector('label span, .label-wrap span') || {}).textContent || '' : '';
                    return !l.toLowerCase().includes('negative');
                }) || all[0];
            }
        }
        if (promptTA) {
            const ns = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
            if (ns && ns.set) {
                ns.set.call(promptTA, PROMPT);
                promptTA.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                promptTA.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
            }
        }

        // Set Number of images slider to 4
        if (picgenCol) {
            picgenCol.querySelectorAll('input[type="range"]').forEach(sl => {
                const b = sl.closest('.block');
                const l = b ? b.querySelector('label span') : null;
                if (l && l.textContent.includes('Number of images')) {
                    const ns = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value');
                    if (ns && ns.set) {
                        ns.set.call(sl, '4');
                        sl.dispatchEvent(new Event('input',  {bubbles:true, composed:true}));
                        sl.dispatchEvent(new Event('change', {bubbles:true, composed:true}));
                    }
                }
            });
        }

        _close();
        setTimeout(() => {
            const genBtn = Array.from(document.querySelectorAll('button')).find(b =>
                b.textContent.trim() === 'Generate' &&
                b.id !== 'ng-op-generate' &&
                b.closest('#col-container')
            );
            if (genBtn) genBtn.click();
        }, 150);
    });

    // -- populate from gallery on first open -----------------------------------
    const existing = window.__picgenImages;
    if (existing && existing.length > 0) {
        _galleryIdx = 0;
        overlay._galleryIdx = 0;
        _fromPopupUpload = false;
        _loadSrc(existing[0].b64);
    }
};
}
"""
    demo.load(fn=None, js=_outpaint_popup_js)



from fastapi.responses import Response as _FastAPIResponse
from fastapi import Request as _FastAPIRequest


@demo.app.get("/starters/{num}")
async def _serve_starter(num: int):
    """Serve a starter image by number (tries .jpg, .png, .webp).
    Used by the thumbnail <img> elements in the picgen starter grid."""
    path, ext = _find_starter_path(num)
    if path is None:
        return _FastAPIResponse(status_code=404, content=b"not found")
    mime = {"jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext.lstrip("."), "image/jpeg")
    with open(path, "rb") as f:
        data = f.read()
    return _FastAPIResponse(
        content=data,
        media_type=mime,
        headers={"Cache-Control": "public, max-age=86400"},
    )

@demo.app.get("/media/{key}/{filename}")
async def _stream_media(key: str, filename: str, request: _FastAPIRequest):
    entry = _media_store_get(key)
    if entry is None:
        return _FastAPIResponse(status_code=404, content=b"not found")
    data, stored_filename = entry
    ext = stored_filename.rsplit(".", 1)[-1].lower() if "." in stored_filename else ""
    media_type_map = {
        "mp4": "video/mp4",
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webm": "video/webm",
    }
    media_type = media_type_map.get(ext, "application/octet-stream")

    secret_hex = request.headers.get("X-NG-Secret", "").strip()
    if secret_hex and len(secret_hex) == 64:
        key_bytes = _derive_media_key(secret_hex)
        encrypted = _encrypt_bytes(data, key_bytes)
        return _FastAPIResponse(
            content=encrypted,
            media_type="application/octet-stream",
            headers={
                "X-NG-Encrypted": "1",
                "X-NG-Original-Type": media_type,
                "X-NG-Filename": stored_filename,
                "Cache-Control": "no-store",
                "Content-Length": str(len(encrypted)),
            },
        )

    return _FastAPIResponse(
        content=data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{stored_filename}"',
            "Cache-Control": "no-store",
            "Content-Length": str(len(data)),
        },
    )


@demo.app.get("/logs/stream")
async def _logs_stream(request: _FastAPIRequest):
    """SSE endpoint that streams encrypted log lines to the browser.

    Each event is: data: <base64(nonce+ciphertext)>\\n\\n
    The browser decrypts with its HKDF-derived log key.
    Requires X-NG-Secret header (same localStorage secret as media).
    """
    from fastapi.responses import StreamingResponse as _StreamingResponse

    secret_hex = request.headers.get("X-NG-Secret", "").strip()
    if not secret_hex or len(secret_hex) != 64:
        return _FastAPIResponse(status_code=403, content=b"missing secret")

    log_key = _derive_log_key(secret_hex)

    async def _event_generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                line = _log_queue.get(timeout=0.5)
                enc = _encrypt_log_line(line, log_key)
                yield f"data: {enc}\n\n"
            except Exception:
                yield ": keepalive\n\n"

    return _StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    print(
        f" GRADIO LAUNCHING  Qwen on {PIC_DEVICE} "
        f"({GPU_NAME}, {GPU_VRAM_GB:.0f} GB) -- picgen ready immediately."
    )
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[SCRIPT_DIR, os.path.join(SCRIPT_DIR, "tmp", "gradio")],
    )
