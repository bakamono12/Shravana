"""Shravana Colab readiness checker.

Run standalone:
    python colab/check_ready.py

Or call from a notebook cell:
    import sys; sys.path.insert(0, "/content/Shravana")
    from colab.check_ready import run_checks
    run_checks()
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

REPO_DIR = Path("/content/Shravana")
MODELS_DIR = REPO_DIR / "storage" / "models"
FRONTEND_DIST = REPO_DIR / "frontend" / "dist" / "index.html"
DB_PATH = REPO_DIR / "shravana.db"

# (label, critical, check_fn)
_CHECKS: list[tuple[str, bool, "callable[[], tuple[bool, str]]"]] = []


def _check(label: str, critical: bool = True):
    def decorator(fn):
        _CHECKS.append((label, critical, fn))
        return fn
    return decorator


@_check("Python ≥ 3.10")
def _python_version():
    ok = sys.version_info >= (3, 10)
    return ok, f"{sys.version_info.major}.{sys.version_info.minor}"


@_check("CUDA available")
def _cuda():
    try:
        import torch
        ok = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if ok else "no GPU"
        return ok, name
    except ImportError:
        return False, "torch not installed"


@_check("FFmpeg in PATH")
def _ffmpeg():
    path = shutil.which("ffmpeg")
    return bool(path), path or "not found"


@_check("libmagic (python-magic)")
def _magic():
    try:
        import magic  # noqa: F401
        return True, "ok"
    except ImportError:
        return False, "pip install python-magic"


@_check("bitsandbytes")
def _bnb():
    try:
        import bitsandbytes  # noqa: F401
        return True, "ok"
    except ImportError:
        return False, 'pip install "bitsandbytes>=0.43.0"'


@_check("torch")
def _torch():
    try:
        import torch
        return True, torch.__version__
    except ImportError:
        return False, "not installed"


@_check("transformers")
def _transformers():
    try:
        import transformers
        return True, transformers.__version__
    except ImportError:
        return False, "not installed"


@_check("qwen-vl-utils")
def _qwen_vl_utils():
    try:
        from qwen_vl_utils import process_vision_info  # noqa: F401
        return True, "ok"
    except ImportError:
        return False, 'pip install "qwen-vl-utils>=0.0.14"'


@_check("faster-whisper")
def _faster_whisper():
    try:
        import faster_whisper  # noqa: F401
        return True, "ok"
    except ImportError:
        return False, "not installed"


@_check("backend package (app.config)")
def _backend():
    try:
        sys.path.insert(0, str(REPO_DIR / "backend"))
        from app.config import settings  # noqa: F401
        return True, f"QWEN_VL_MODEL_ID={settings.QWEN_VL_MODEL_ID}"
    except Exception as exc:
        return False, str(exc)


@_check("models downloaded", critical=False)
def _models():
    if not MODELS_DIR.exists():
        return False, f"{MODELS_DIR} missing — run Cell 6"
    names = ["qwen_lid", "qwen3_asr", "seamless_v2", "qwen2_5_vl"]
    missing = [n for n in names if not any((MODELS_DIR / n).rglob("*.safetensors"))]
    if missing:
        return False, f"missing: {', '.join(missing)}"
    return True, "all present"


@_check("database initialised", critical=False)
def _db():
    ok = DB_PATH.exists()
    return ok, str(DB_PATH) if ok else f"not found — run Cell 5"


@_check("frontend built", critical=False)
def _frontend():
    ok = FRONTEND_DIST.exists()
    return ok, str(FRONTEND_DIST) if ok else "not found — run Cell 3b"


def run_checks(raise_on_failure: bool = True) -> bool:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    print(f"\n{'─'*60}")
    print("  Shravana — Colab Readiness Check")
    print(f"{'─'*60}")

    any_critical_fail = False
    for label, critical, fn in _CHECKS:
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"error: {exc}"

        if ok:
            status = f"{GREEN}✓{RESET}"
        elif critical:
            status = f"{RED}✗{RESET}"
            any_critical_fail = True
        else:
            status = f"{YELLOW}⚠{RESET}"

        tag = "" if critical else " (optional)"
        print(f"  {status}  {label}{tag}  —  {detail}")

    print(f"{'─'*60}")
    if any_critical_fail:
        print(f"  {RED}Some critical checks failed. Fix them before proceeding.{RESET}")
    else:
        print(f"  {GREEN}All critical checks passed. Ready to start the server.{RESET}")
    print()

    if any_critical_fail and raise_on_failure:
        raise SystemExit(1)
    return not any_critical_fail


if __name__ == "__main__":
    run_checks()
