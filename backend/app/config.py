from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Storage
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    STORAGE_DIR: str = "storage"
    UPLOAD_DIR: str = "storage/uploads"
    JOBS_DIR: str = "storage/jobs"
    MODELS_DIR: str = "storage/models"
    MAX_FILE_SIZE_GB: int = 5

    # Audio
    CHUNK_DURATION_SECONDS: int = 60
    CHUNK_OVERLAP_SECONDS: int = 5
    AUDIO_SAMPLE_RATE: int = 16000

    # Models
    PARAKEET_MODEL_ID: str = "nvidia/parakeet-tdt-1.1b"
    QWEN_LID_MODEL_ID: str = "Qwen/Qwen3-ASR-0.6B"
    QWEN_ASR_MODEL_ID: str = "Qwen/Qwen3-ASR-1.7B"
    WHISPER_MODEL_SIZE: str = "large-v3-turbo"
    FORCED_ALIGNER_MODEL_ID: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    DEMUCS_MODEL_NAME: str = "htdemucs"

    # Language routing
    LID_CONFIDENCE_THRESHOLD: float = 0.70
    CODE_SWITCH_THRESHOLD: float = 0.85

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./shravana.db"

    # Concurrency
    PIPELINE_CONCURRENCY: int = 1

    # Serving built UI from FastAPI
    SHRAVANA_SERVE_UI: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
