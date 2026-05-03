"""Lazy singleton model registry — loads each model only when first requested."""
from __future__ import annotations
import threading
from typing import Optional

from app.config import settings
from app.ml.base import ModelNotReady


_lock = threading.Lock()

_instances: dict = {}
_loaded: dict[str, bool] = {}

MODEL_CONFIGS = {
    "qwen_lid": {
        "repo_id": settings.QWEN_LID_MODEL_ID,
        "class": "QwenASRModel",
        "kwargs": {"model_id": settings.QWEN_LID_MODEL_ID, "lid_only": True},
    },
    "qwen3_asr": {
        "repo_id": settings.QWEN_ASR_MODEL_ID,
        "class": "QwenASRModel",
        "kwargs": {"model_id": settings.QWEN_ASR_MODEL_ID, "lid_only": False},
    },
    "parakeet": {
        "repo_id": settings.PARAKEET_MODEL_ID,
        "class": "ParakeetModel",
        "kwargs": {"model_id": settings.PARAKEET_MODEL_ID},
    },
    "whisper_turbo": {
        "repo_id": f"faster-whisper/{settings.WHISPER_MODEL_SIZE}",
        "class": "WhisperTurboModel",
        "kwargs": {"model_size": settings.WHISPER_MODEL_SIZE},
    },
    "forced_aligner": {
        "repo_id": settings.FORCED_ALIGNER_MODEL_ID,
        "class": "ForcedAlignerModel",
        "kwargs": {"model_id": settings.FORCED_ALIGNER_MODEL_ID},
    },
    # Translation models
    "seamless_v2": {
        "repo_id": settings.SEAMLESS_MODEL_ID,
        "class": "SeamlessM4TTranslator",
        "kwargs": {"model_id": settings.SEAMLESS_MODEL_ID},
    },
    "qwen2_5_vl": {
        "repo_id": settings.QWEN_VL_MODEL_ID,
        "class": "QwenVLTranslator",
        "kwargs": {"model_id": settings.QWEN_VL_MODEL_ID},
    },
}


# ------------------------------------------------------------------ #
# Remote routing helpers                                              #
# ------------------------------------------------------------------ #

def _route_remote(name: str) -> bool:
    """True when this model should be served by the remote GPU worker."""
    if not settings.REMOTE_GPU_URL:
        return False
    model_cfg = settings.REMOTE_GPU_MODELS.strip()
    if model_cfg.upper() == "ALL":
        return True
    return name in {m.strip() for m in model_cfg.split(",") if m.strip()}


def _remote_proxy_for(name: str):
    """Return (and cache) the appropriate remote proxy for the given model name."""
    if name in _instances:
        return _instances[name]

    from app.ml.remote_proxies import RemoteSTTProxy, RemoteTranslatorProxy, RemoteAlignerProxy

    cfg = MODEL_CONFIGS.get(name, {})
    cls_name = cfg.get("class", "")

    if cls_name in ("QwenASRModel", "ParakeetModel", "WhisperTurboModel"):
        instance = RemoteSTTProxy(model_name=name)
    elif cls_name in ("SeamlessM4TTranslator", "QwenVLTranslator"):
        instance = RemoteTranslatorProxy(model_name=name)
    elif cls_name == "ForcedAlignerModel":
        instance = RemoteAlignerProxy(model_name=name)
    else:
        instance = RemoteSTTProxy(model_name=name)  # generic fallback

    _instances[name] = instance
    return instance


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #

def _is_downloaded(name: str) -> bool:
    """Check whether model files are present on disk."""
    if name == "whisper_turbo":
        return True
    from pathlib import Path
    local_dir = Path(settings.BASE_DIR / settings.MODELS_DIR) / name
    if not local_dir.exists():
        return False
    return any(
        f for f in local_dir.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    )


def get(name: str):
    """Get a loaded model by name.

    Returns a remote proxy when REMOTE_GPU_URL is configured and the model is
    in the routing list. Otherwise loads the model locally (raises ModelNotReady
    if not downloaded).
    """
    with _lock:
        if _route_remote(name):
            return _remote_proxy_for(name)

        if name in _instances:
            return _instances[name]

        if not _is_downloaded(name):
            raise ModelNotReady(name)

        cfg = MODEL_CONFIGS[name]
        cls_name = cfg["class"]
        kwargs = cfg["kwargs"]

        if cls_name == "QwenASRModel":
            from app.ml.qwen_asr import QwenASRModel
            instance = QwenASRModel(**kwargs)
        elif cls_name == "ParakeetModel":
            from app.ml.parakeet import ParakeetModel
            instance = ParakeetModel(**kwargs)
        elif cls_name == "WhisperTurboModel":
            from app.ml.whisper_turbo import WhisperTurboModel
            instance = WhisperTurboModel(**kwargs)
        elif cls_name == "ForcedAlignerModel":
            from app.ml.forced_aligner import ForcedAlignerModel
            instance = ForcedAlignerModel(**kwargs)
        elif cls_name == "SeamlessM4TTranslator":
            from app.ml.seamless_translator import SeamlessM4TTranslator
            instance = SeamlessM4TTranslator(**kwargs)
        elif cls_name == "QwenVLTranslator":
            from app.ml.qwen_vl_translator import QwenVLTranslator
            instance = QwenVLTranslator(**kwargs)
        else:
            raise ValueError(f"Unknown model class: {cls_name}")

        instance.load()
        _instances[name] = instance
        return instance


def get_local(name: str):
    """Force local model loading, bypassing remote routing (used for fallback)."""
    with _lock:
        local_key = f"__local__{name}"
        if local_key in _instances:
            return _instances[local_key]

        if not _is_downloaded(name):
            raise ModelNotReady(name)

        cfg = MODEL_CONFIGS[name]
        cls_name = cfg["class"]
        kwargs = cfg["kwargs"]

        if cls_name == "QwenASRModel":
            from app.ml.qwen_asr import QwenASRModel
            instance = QwenASRModel(**kwargs)
        elif cls_name == "ParakeetModel":
            from app.ml.parakeet import ParakeetModel
            instance = ParakeetModel(**kwargs)
        elif cls_name == "WhisperTurboModel":
            from app.ml.whisper_turbo import WhisperTurboModel
            instance = WhisperTurboModel(**kwargs)
        elif cls_name == "ForcedAlignerModel":
            from app.ml.forced_aligner import ForcedAlignerModel
            instance = ForcedAlignerModel(**kwargs)
        elif cls_name == "SeamlessM4TTranslator":
            from app.ml.seamless_translator import SeamlessM4TTranslator
            instance = SeamlessM4TTranslator(**kwargs)
        elif cls_name == "QwenVLTranslator":
            from app.ml.qwen_vl_translator import QwenVLTranslator
            instance = QwenVLTranslator(**kwargs)
        else:
            raise ValueError(f"Unknown model class: {cls_name}")

        instance.load()
        _instances[local_key] = instance
        return instance


def get_demucs():
    """Return a DemucsSeparator (local) or RemoteDenoiser (when remote routing active)."""
    with _lock:
        if _route_remote("demucs"):
            if "demucs" not in _instances:
                from app.ml.remote_proxies import RemoteDenoiser
                _instances["demucs"] = RemoteDenoiser()
            return _instances["demucs"]

        if "demucs" not in _instances:
            from app.ml.demucs_separator import DemucsSeparator
            _instances["demucs"] = DemucsSeparator(model_name=settings.DEMUCS_MODEL_NAME)
        return _instances["demucs"]
