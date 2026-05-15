from pydantic import computed_field
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
    QWEN_ASR_MODEL_ID: str = "Qwen/Qwen3-ASR-1.7B"
    DEMUCS_MODEL_NAME: str = "htdemucs"

    # Translation models
    QWEN_VL_MODEL_ID: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Translation pipeline behaviour
    TRANSLATION_CONTEXT_WINDOW: int = 4       # rolling translated chunks kept in history
    TRANSLATION_MAX_CHUNK_SECONDS: int = 20   # semantic chunker max duration
    TRANSLATION_SILENCE_GAP: float = 1.5      # seconds gap that triggers a new semantic chunk
    VLM_KEYFRAMES_PER_CHUNK: int = 3          # keyframes sampled per translation chunk (video)
    CONTEXT_KEYFRAME_INTERVAL_S: int = 30     # seconds between keyframes for context build
    LLM_CALL_TIMEOUT_S: int = 30
    LLM_MAX_RETRIES: int = 2

    # VLM optimization
    VLM_USE_4BIT: bool = True       # 4-bit NF4 quant on CUDA (~2x VRAM saving); ignored on CPU
    VLM_MAX_PIXELS: int = 200704    # 256 × 28 × 28 visual patches — caps vision encoding cost

    # Language routing
    LID_CONFIDENCE_THRESHOLD: float = 0.70
    CODE_SWITCH_THRESHOLD: float = 0.85

    # Database — absolute path keeps things consistent regardless of CWD
    @computed_field  # type: ignore[prop-decorator]
    @property
    def DATABASE_URL(self) -> str:
        return f"sqlite+aiosqlite:///{self.BASE_DIR / 'shravana.db'}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def DATABASE_URL_SYNC(self) -> str:
        return f"sqlite:///{self.BASE_DIR / 'shravana.db'}"

    # Remote GPU executor
    REMOTE_GPU_URL: str = ""             # blank → pure local execution
    REMOTE_GPU_TOKEN: str = ""           # bearer token shared with the worker
    # "ALL" routes every model; comma-separated names for partial routing
    REMOTE_GPU_MODELS: str = "ALL"
    REMOTE_GPU_TIMEOUT_S: int = 180
    REMOTE_GPU_FALLBACK_LOCAL: bool = True   # on remote error, retry locally
    REMOTE_GPU_HEALTHCHECK_INTERVAL_S: int = 30

    # Concurrency
    PIPELINE_CONCURRENCY: int = 1

    # Serving built UI from FastAPI
    SHRAVANA_SERVE_UI: bool = False

    # Server port (consumed by backend/run.py and the Vite dev proxy)
    BACKEND_PORT: int = 8000
    BACKEND_HOST: str = "127.0.0.1"

    class Config:
        env_file = str(Path(__file__).parent.parent / ".env")
        env_file_encoding = "utf-8"


settings = Settings()
