from pathlib import Path
from app.config import settings


def _base() -> Path:
    return settings.BASE_DIR / settings.STORAGE_DIR


def uploads_dir() -> Path:
    p = _base() / "uploads"
    p.mkdir(parents=True, exist_ok=True)
    return p


def job_dir(job_id: str) -> Path:
    p = _base() / "jobs" / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def chunks_dir(job_id: str) -> Path:
    p = job_dir(job_id) / "chunks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def subs_dir(job_id: str) -> Path:
    p = job_dir(job_id) / "subs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = _base() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def upload_path(job_id: str, original_filename: str) -> Path:
    ext = Path(original_filename).suffix
    p = uploads_dir() / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p / f"original{ext}"


def raw_audio_path(job_id: str) -> Path:
    return job_dir(job_id) / "raw.wav"


def clean_audio_path(job_id: str) -> Path:
    return job_dir(job_id) / "clean.wav"


def subtitle_path(job_id: str, fmt: str) -> Path:
    return subs_dir(job_id) / f"subtitles.{fmt}"
