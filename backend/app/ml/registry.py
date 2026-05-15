"""Lazy singleton model registry — loads each model only when first requested."""
from __future__ import annotations
import threading
from pathlib import Path

from app.config import settings
from app.ml.base import ModelNotReady


SENTINEL_FILE = ".shravana_complete"


def model_complete(name: str) -> bool:
    sentinel = Path(settings.BASE_DIR / settings.MODELS_DIR) / name / SENTINEL_FILE
    return sentinel.exists()


_lock = threading.Lock()
_instances: dict = {}

MODEL_CONFIGS = {
    "qwen3_asr": {
        "repo_id": settings.QWEN_ASR_MODEL_ID,
        "class": "QwenASRModel",
        "kwargs": {"model_id": settings.QWEN_ASR_MODEL_ID, "lid_only": False},
    },
    "qwen2_5_vl": {
        "repo_id": settings.QWEN_VL_MODEL_ID,
        "class": "QwenVLTranslator",
        "kwargs": {"model_id": settings.QWEN_VL_MODEL_ID},
    },
}


def get(name: str):
    """Get a loaded model instance by name. Raises ModelNotReady if not downloaded."""
    with _lock:
        if name in _instances:
            return _instances[name]

        if not model_complete(name):
            raise ModelNotReady(name)

        cfg = MODEL_CONFIGS[name]
        cls_name = cfg["class"]
        kwargs = cfg["kwargs"]

        if cls_name == "QwenASRModel":
            from app.ml.qwen_asr import QwenASRModel
            instance = QwenASRModel(**kwargs)
        elif cls_name == "QwenVLTranslator":
            from app.ml.qwen_vl_translator import QwenVLTranslator
            instance = QwenVLTranslator(**kwargs)
        else:
            raise ValueError(f"Unknown model class: {cls_name}")

        instance.load()
        _instances[name] = instance
        return instance


def get_demucs():
    """Return a DemucsSeparator instance (lazy singleton)."""
    with _lock:
        if "demucs" not in _instances:
            from app.ml.demucs_separator import DemucsSeparator
            _instances["demucs"] = DemucsSeparator(model_name=settings.DEMUCS_MODEL_NAME)
        return _instances["demucs"]
