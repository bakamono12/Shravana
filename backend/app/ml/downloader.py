"""Background model downloader.

Sequentially downloads HuggingFace snapshots in a single worker thread, with:
  - per-model `.shravana_complete` sentinel (written only after a successful run)
  - subprocess-based download so a wedged C-extension can be killed
  - stall watchdog: if the on-disk dir size doesn't grow for STALL_TIMEOUT s,
    terminate and retry (HF resume picks up partial blobs)
  - bounded retries with exponential backoff; one failure does not block siblings
"""
from __future__ import annotations
import asyncio
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

from app.config import settings
from app.ml.registry import MODEL_CONFIGS, SENTINEL_FILE, model_complete

logger = logging.getLogger(__name__)

MODELS_TO_DOWNLOAD: dict[str, str] = {
    name: cfg["repo_id"]
    for name, cfg in MODEL_CONFIGS.items()
    if not cfg["repo_id"].startswith("faster-whisper/")
}

# Tuning
_STALL_TIMEOUT = 120        # seconds without disk-size growth before we kill
_POLL_INTERVAL = 10         # seconds between dir-size samples
_MAX_ATTEMPTS = 4
_BACKOFF_SECONDS = [0, 30, 120, 300]  # waited *before* attempt N (index N)

_download_queue: "queue.Queue[tuple[str, str]]" = queue.Queue()
_worker_started = False
_worker_lock = threading.Lock()


def _model_dir(name: str) -> Path:
    return Path(settings.BASE_DIR / settings.MODELS_DIR) / name


def _sentinel_path(name: str) -> Path:
    return _model_dir(name) / SENTINEL_FILE


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


def _spawn_download(name: str, repo_id: str) -> subprocess.Popen:
    """Run snapshot_download in a child process so we can kill it on stall."""
    local_dir = _model_dir(name)
    local_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(settings.BASE_DIR / settings.MODELS_DIR)

    code = (
        "from huggingface_hub import snapshot_download;"
        f"snapshot_download(repo_id={repo_id!r}, local_dir={str(local_dir)!r})"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _run_with_watchdog(name: str, repo_id: str, post) -> tuple[bool, str | None]:
    """Run a single download attempt with a stall watchdog. Returns (ok, error)."""
    local_dir = _model_dir(name)
    proc = _spawn_download(name, repo_id)

    last_size = _dir_size_bytes(local_dir)
    last_growth = time.monotonic()

    while True:
        try:
            rc = proc.wait(timeout=_POLL_INTERVAL)
            break
        except subprocess.TimeoutExpired:
            pass

        size = _dir_size_bytes(local_dir)
        if size > last_size:
            last_size = size
            last_growth = time.monotonic()
            post("downloading", bytes_dl=size)
        elif time.monotonic() - last_growth > _STALL_TIMEOUT:
            logger.warning(
                f"Model '{name}' stalled (no progress for {_STALL_TIMEOUT}s), terminating"
            )
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            return False, f"download stalled (no progress for {_STALL_TIMEOUT}s)"

    if rc == 0:
        try:
            _sentinel_path(name).touch()
        except OSError as e:
            return False, f"could not write sentinel: {e}"
        return True, None

    err_bytes = b""
    if proc.stderr is not None:
        try:
            err_bytes = proc.stderr.read() or b""
        except Exception:
            pass
    err_text = err_bytes.decode(errors="replace").strip()
    if len(err_text) > 500:
        err_text = "..." + err_text[-500:]
    return False, f"exit code {rc}: {err_text}" if err_text else f"exit code {rc}"


def _make_status_poster(main_loop: asyncio.AbstractEventLoop, name: str):
    """Build a thread-safe `post(status, ...)` that updates DB + WS bus."""

    async def _update(status: str, error: str | None, bytes_dl: int, bytes_total: int):
        from app.db import AsyncSessionLocal
        from app.models_db import ModelDownload
        from app.jobs.progress_bus import publish_model_progress
        from sqlalchemy import select

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(ModelDownload).where(ModelDownload.name == name)
            )
            row = result.scalar_one_or_none()
            if row:
                row.status = status
                row.bytes_downloaded = bytes_dl
                row.bytes_total = bytes_total
                row.error_message = error
                await session.commit()

        await publish_model_progress(name, status, bytes_dl, bytes_total)

    def post(status: str, error: str | None = None, bytes_dl: int = 0, bytes_total: int = 0):
        future = asyncio.run_coroutine_threadsafe(
            _update(status, error, bytes_dl, bytes_total), main_loop
        )
        try:
            future.result(timeout=10)
        except Exception as e:
            logger.warning(f"Status update failed for {name}: {e}")

    return post


def _download_worker(main_loop: asyncio.AbstractEventLoop) -> None:
    """Single worker thread; processes queued downloads sequentially."""
    while True:
        name, repo_id = _download_queue.get()
        try:
            if model_complete(name):
                continue

            post = _make_status_poster(main_loop, name)
            success = False
            last_err: str | None = None

            for attempt in range(_MAX_ATTEMPTS):
                wait = _BACKOFF_SECONDS[attempt]
                if wait:
                    logger.info(
                        f"Retrying '{name}' in {wait}s "
                        f"(attempt {attempt + 1}/{_MAX_ATTEMPTS})"
                    )
                    time.sleep(wait)

                post("downloading", bytes_dl=_dir_size_bytes(_model_dir(name)))
                try:
                    ok, err = _run_with_watchdog(name, repo_id, post)
                except Exception as exc:
                    ok, err = False, f"{type(exc).__name__}: {exc}"

                if ok:
                    post("done", bytes_dl=_dir_size_bytes(_model_dir(name)))
                    logger.info(f"Model '{name}' ready at {_model_dir(name)}")
                    success = True
                    break

                last_err = err
                logger.warning(
                    f"Model '{name}' attempt {attempt + 1}/{_MAX_ATTEMPTS} failed: {err}"
                )

            if not success:
                post("failed", error=last_err)
                logger.error(
                    f"Model '{name}' failed after {_MAX_ATTEMPTS} attempts: {last_err}"
                )
        except Exception as exc:
            logger.exception(f"Worker error while handling '{name}': {exc}")
        finally:
            _download_queue.task_done()


async def start_background_downloads() -> None:
    """Reconcile DB rows with on-disk state and enqueue missing models."""
    global _worker_started
    from app.db import AsyncSessionLocal
    from app.models_db import ModelDownload
    from sqlalchemy import select

    main_loop = asyncio.get_event_loop()

    async with AsyncSessionLocal() as session:
        for name, repo_id in MODELS_TO_DOWNLOAD.items():
            result = await session.execute(
                select(ModelDownload).where(ModelDownload.name == name)
            )
            row = result.scalar_one_or_none()
            if row is None:
                row = ModelDownload(name=name, repo_id=repo_id, status="queued")
                session.add(row)

            if model_complete(name):
                if row.status != "done":
                    logger.info(f"Model '{name}' sentinel found, marking done")
                    row.status = "done"
                    row.error_message = None
            else:
                if _model_dir(name).exists() and any(_model_dir(name).iterdir()):
                    logger.info(
                        f"Model '{name}' has incomplete prior download, will resume"
                    )
                row.status = "queued"
                row.error_message = None
                row.bytes_downloaded = 0

        await session.commit()

    with _worker_lock:
        if not _worker_started:
            t = threading.Thread(
                target=_download_worker, args=(main_loop,), daemon=True
            )
            t.start()
            _worker_started = True

    for name, repo_id in MODELS_TO_DOWNLOAD.items():
        if not model_complete(name):
            _download_queue.put((name, repo_id))

    asyncio.create_task(_mark_whisper_ready())


async def _mark_whisper_ready() -> None:
    from app.db import AsyncSessionLocal
    from app.models_db import ModelDownload
    from sqlalchemy import select

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ModelDownload).where(ModelDownload.name == "whisper_turbo")
        )
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
