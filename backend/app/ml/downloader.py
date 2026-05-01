"""Background model downloader — queues HuggingFace snapshot downloads on startup."""
from __future__ import annotations
import asyncio
import logging
import os
import threading
from pathlib import Path

from app.config import settings
from app.ml.registry import MODEL_CONFIGS

logger = logging.getLogger(__name__)

MODELS_TO_DOWNLOAD: dict[str, str] = {
    name: cfg["repo_id"]
    for name, cfg in MODEL_CONFIGS.items()
    if not cfg["repo_id"].startswith("faster-whisper/")
}


def _model_files_exist(name: str) -> bool:
    """Return True if the model directory is non-empty (files already downloaded)."""
    local_dir = Path(settings.BASE_DIR / settings.MODELS_DIR) / name
    if not local_dir.exists():
        return False
    # Any file other than hidden/lock files counts as a successful download
    return any(
        f for f in local_dir.rglob("*")
        if f.is_file() and not f.name.startswith(".")
    )


async def start_background_downloads() -> None:
    """Called at FastAPI lifespan start. Marks already-downloaded models done,
    resets genuinely stuck rows, and kicks off missing downloads."""
    from app.db import AsyncSessionLocal
    from app.models_db import ModelDownload
    from sqlalchemy import select

    main_loop = asyncio.get_event_loop()

    async with AsyncSessionLocal() as session:
        for name, repo_id in MODELS_TO_DOWNLOAD.items():
            result = await session.execute(select(ModelDownload).where(ModelDownload.name == name))
            row = result.scalar_one_or_none()
            if row is None:
                row = ModelDownload(name=name, repo_id=repo_id, status="queued")
                session.add(row)

            already_on_disk = _model_files_exist(name)

            if already_on_disk:
                # Files exist — mark done regardless of what the DB thinks
                if row.status != "done":
                    logger.info(f"Model '{name}' found on disk, marking done")
                    row.status = "done"
                    row.error_message = None
                # No download thread needed
            elif row.status in ("downloading", "failed"):
                # Genuinely incomplete — reset and retry
                row.status = "queued"
                row.error_message = None
                row.bytes_downloaded = 0

        await session.commit()

    # Only spawn download threads for models not already on disk
    for name, repo_id in MODELS_TO_DOWNLOAD.items():
        if not _model_files_exist(name):
            t = threading.Thread(
                target=_download_model,
                args=(name, repo_id, main_loop),
                daemon=True,
            )
            t.start()

    asyncio.create_task(_mark_whisper_ready())


async def _mark_whisper_ready() -> None:
    from app.db import AsyncSessionLocal
    from app.models_db import ModelDownload
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        result = await session.execute(select(ModelDownload).where(ModelDownload.name == "whisper_turbo"))
        row = result.scalar_one_or_none()
        if row is None:
            row = ModelDownload(
                name="whisper_turbo",
                repo_id=f"faster-whisper/{settings.WHISPER_MODEL_SIZE}",
                status="done",
            )
            session.add(row)
        elif row.status != "done":
            row.status = "done"
        await session.commit()


def _download_model(name: str, repo_id: str, main_loop: asyncio.AbstractEventLoop) -> None:
    """Run inside a daemon thread. Uses asyncio.run_coroutine_threadsafe to post
    status updates back to the main FastAPI event loop (safe from any thread)."""

    def post(coro):
        """Schedule a coroutine on the main loop and wait for it to finish."""
        future = asyncio.run_coroutine_threadsafe(coro, main_loop)
        try:
            future.result(timeout=10)
        except Exception as e:
            logger.warning(f"Status update failed for {name}: {e}")

    async def _update_status(status: str, error: str | None = None, bytes_dl: int = 0, bytes_total: int = 0):
        from app.db import AsyncSessionLocal
        from app.models_db import ModelDownload
        from app.jobs.progress_bus import publish_model_progress
        from sqlalchemy import select

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(ModelDownload).where(ModelDownload.name == name))
            row = result.scalar_one_or_none()
            if row:
                row.status = status
                row.bytes_downloaded = bytes_dl
                row.bytes_total = bytes_total
                row.error_message = error
                await session.commit()

        await publish_model_progress(name, status, bytes_dl, bytes_total)

    post(_update_status("downloading"))

    try:
        os.environ.setdefault("HF_HOME", str(settings.BASE_DIR / settings.MODELS_DIR))
        local_dir = Path(settings.BASE_DIR / settings.MODELS_DIR) / name

        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            # resume_download=True is default; ignore incomplete blobs automatically
        )
        post(_update_status("done", bytes_dl=1, bytes_total=1))
        logger.info(f"Model '{name}' ready at {local_dir}")
    except Exception as exc:
        logger.error(f"Model '{name}' download failed: {exc}")
        post(_update_status("failed", error=str(exc)))
