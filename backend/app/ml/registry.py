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
}


def _is_downloaded(name: str) -> bool:
    """Check whether model files are present on disk."""
    if name == "whisper_turbo":
        # faster-whisper fetches its own weights on first .load() call
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
    """Get a loaded model by name. Raises ModelNotReady if not downloaded yet."""
    with _lock:
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
        else:
            raise ValueError(f"Unknown model class: {cls_name}")

        instance.load()
        _instances[name] = instance
        return instance


def get_demucs():
    with _lock:
        if "demucs" not in _instances:
            from app.ml.demucs_separator import DemucsSeparator
            _instances["demucs"] = DemucsSeparator(model_name=settings.DEMUCS_MODEL_NAME)
        return _instances["demucs"]
