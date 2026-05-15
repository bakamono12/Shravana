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
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

# Marker string embedded in the child's `python -c` so we can identify and kill
# orphan downloaders left over from a previous uvicorn run.
_CHILD_MARKER = "__SHRAVANA_DOWNLOADER_CHILD__"

from app.config import settings
from app.ml.registry import MODEL_CONFIGS, SENTINEL_FILE, model_complete

logger = logging.getLogger(__name__)

MODELS_TO_DOWNLOAD: dict[str, str] = {
    name: cfg["repo_id"]
    for name, cfg in MODEL_CONFIGS.items()
    if cfg["repo_id"]  # all models are HF downloads
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


def _kill_orphan_downloaders() -> int:
    """Kill any leftover downloader child processes from a previous run.

    Required because uvicorn --reload restarts the parent without killing our
    `subprocess.Popen` children — they survive as orphans and keep holding
    fcntl locks on `<model>/.cache/huggingface/download/*.lock`, blocking the
    fresh process indefinitely.
    """
    proc_root = Path("/proc")
    if not proc_root.exists():
        return 0  # not Linux
    me = os.getpid()
    killed = 0
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == me:
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
        except OSError:
            continue
        if _CHILD_MARKER not in cmdline:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed += 1
            logger.warning(f"Killed orphan downloader child PID {pid}")
        except OSError:
            pass
    if killed:
        # Give them a moment to release locks.
        time.sleep(0.5)
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                cmdline = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
            except OSError:
                continue
            if _CHILD_MARKER in cmdline:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
    return killed


def _clean_stale_locks(name: str, repo_id: str) -> int:
    """Remove `.lock` files left behind by a previously-killed download.

    HF uses filelock, and a stale lock file can keep a fresh attempt blocked
    even when the original holder is long dead. We're the sole writer, so it
    is safe to wipe these on retry.
    """
    removed = 0
    for root in (_model_dir(name), _hf_cache_dir_for(repo_id)):
        if not root.exists():
            continue
        for lock in root.rglob("*.lock"):
            try:
                lock.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        logger.info(f"Cleared {removed} stale lock file(s) for '{name}'")
    return removed


def _hf_cache_dir_for(repo_id: str) -> Path:
    """Path where huggingface_hub stores blobs for a given repo, given our HF_HOME."""
    safe = "models--" + repo_id.replace("/", "--")
    return Path(settings.BASE_DIR / settings.MODELS_DIR) / "hub" / safe


def _download_size_bytes(name: str, repo_id: str) -> int:
    """Total bytes on disk for this model = local_dir + the HF blob cache.

    `snapshot_download(local_dir=...)` writes blobs into the HF cache first and
    only links them into `local_dir` at the end of the run, so sampling only
    `local_dir` looks frozen for the entire download. Adding the per-repo hub
    cache makes the watchdog see real progress.
    """
    return _dir_size_bytes(_model_dir(name)) + _dir_size_bytes(_hf_cache_dir_for(repo_id))


def _spawn_download(name: str, repo_id: str) -> subprocess.Popen:
    """Run snapshot_download in a child process so we can kill it on stall."""
    local_dir = _model_dir(name)
    local_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = str(settings.BASE_DIR / settings.MODELS_DIR)
    # Force unbuffered stderr so HF log lines reach us in real time.
    env["PYTHONUNBUFFERED"] = "1"
    # Bail out of stuck TCP reads in 30s; HF then retries internally.
    env["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
    # Verbose hf_hub logging via Python logging (newline-terminated, parsable).
    env["HF_HUB_VERBOSITY"] = "debug"
    # Disable tqdm's \r-based bars — they block readline() in our stderr pump.
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    # Configure logging in the child so hf_hub's debug records actually print
    # to stderr; otherwise verbosity=debug is set but no handler is attached.
    code = (
        f"# {_CHILD_MARKER}\n"
        "import logging, sys\n"
        "logging.basicConfig(level=logging.INFO,"
        " format='%(levelname)s %(name)s: %(message)s', stream=sys.stderr)\n"
        "from huggingface_hub import snapshot_download\n"
        f"snapshot_download(repo_id={repo_id!r}, local_dir={str(local_dir)!r},"
        " max_workers=1)\n"
        "sys.exit(0)\n"
    )
    return subprocess.Popen(
        [sys.executable, "-c", code],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        # New session → killable as a process group, no inherited signal handlers.
        start_new_session=True,
    )


def _terminate_group(proc: subprocess.Popen) -> None:
    """Kill the child *and* anything it spawned (HF can spawn helper threads/forks)."""
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        pgid = None
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def _pump_stderr(name: str, proc: subprocess.Popen, sink: list[str]) -> threading.Thread:
    """Read child stderr line-by-line, log each line, keep last N lines for the error report."""
    def _run():
        if proc.stderr is None:
            return
        try:
            for raw in iter(proc.stderr.readline, b""):
                line = raw.decode(errors="replace").rstrip()
                if not line:
                    continue
                logger.info(f"[{name}] {line}")
                sink.append(line)
                if len(sink) > 50:
                    del sink[0:len(sink) - 50]
        except Exception:
            pass
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _run_with_watchdog(name: str, repo_id: str, post, attempt: int) -> tuple[bool, str | None]:
    """Run a single download attempt with a stall watchdog. Returns (ok, error)."""
    proc = _spawn_download(name, repo_id)
    stderr_lines: list[str] = []
    pump = _pump_stderr(name, proc, stderr_lines)

    last_size = _download_size_bytes(name, repo_id)
    last_growth = time.monotonic()

    while True:
        try:
            rc = proc.wait(timeout=_POLL_INTERVAL)
            break
        except subprocess.TimeoutExpired:
            pass

        size = _download_size_bytes(name, repo_id)
        if size > last_size:
            last_size = size
            last_growth = time.monotonic()
        elif time.monotonic() - last_growth > _STALL_TIMEOUT:
            logger.warning(
                f"Model '{name}' stalled (no progress for {_STALL_TIMEOUT}s), terminating"
            )
            post(
                "downloading",
                bytes_dl=size,
                attempt=attempt,
                note="stalled — restarting",
            )
            _terminate_group(proc)
            pump.join(timeout=2)
            return False, f"download stalled (no progress for {_STALL_TIMEOUT}s)"

        # Heartbeat every poll tick — even if size didn't change — so the UI
        # and logs see a steady signal during slow blob writes.
        post("downloading", bytes_dl=size, attempt=attempt)

    pump.join(timeout=5)

    if rc == 0:
        try:
            _sentinel_path(name).touch()
        except OSError as e:
            return False, f"could not write sentinel: {e}"
        return True, None

    err_text = "\n".join(stderr_lines[-10:]).strip()
    if len(err_text) > 500:
        err_text = "..." + err_text[-500:]
    return False, f"exit code {rc}: {err_text}" if err_text else f"exit code {rc}"


def _make_status_poster(main_loop: asyncio.AbstractEventLoop, name: str):
    """Build a thread-safe `post(status, ...)` that updates DB + WS bus."""

    async def _update(
        status: str,
        error: str | None,
        bytes_dl: int,
        bytes_total: int,
        attempt: int | None,
        note: str | None,
    ):
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

        await publish_model_progress(
            name, status, bytes_dl, bytes_total, attempt=attempt, note=note
        )

    def post(
        status: str,
        error: str | None = None,
        bytes_dl: int = 0,
        bytes_total: int = 0,
        attempt: int | None = None,
        note: str | None = None,
    ):
        future = asyncio.run_coroutine_threadsafe(
            _update(status, error, bytes_dl, bytes_total, attempt, note), main_loop
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
                attempt_n = attempt + 1
                wait = _BACKOFF_SECONDS[attempt]
                if wait:
                    logger.info(
                        f"Retrying '{name}' in {wait}s "
                        f"(attempt {attempt_n}/{_MAX_ATTEMPTS})"
                    )
                    post(
                        "downloading",
                        bytes_dl=_download_size_bytes(name, repo_id),
                        attempt=attempt_n,
                        note=f"retry {attempt_n}/{_MAX_ATTEMPTS} in {wait}s",
                    )
                    time.sleep(wait)

                _clean_stale_locks(name, repo_id)
                dir_size = _download_size_bytes(name, repo_id)
                start_note = "resuming" if dir_size > 0 else "starting"
                logger.info(
                    f"Model '{name}' attempt {attempt_n}/{_MAX_ATTEMPTS} {start_note}"
                )
                post(
                    "downloading",
                    bytes_dl=dir_size,
                    attempt=attempt_n,
                    note=start_note,
                )
                try:
                    ok, err = _run_with_watchdog(name, repo_id, post, attempt_n)
                except Exception as exc:
                    ok, err = False, f"{type(exc).__name__}: {exc}"

                if ok:
                    post(
                        "done",
                        bytes_dl=_download_size_bytes(name, repo_id),
                        attempt=attempt_n,
                    )
                    logger.info(f"Model '{name}' ready at {_model_dir(name)}")
                    success = True
                    break

                last_err = err
                logger.warning(
                    f"Model '{name}' attempt {attempt_n}/{_MAX_ATTEMPTS} failed: {err}"
                )

            if not success:
                post("failed", error=last_err, note=last_err)
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

    # Survivors of a previous uvicorn run can still hold fcntl locks on
    # storage/models/<x>/.cache/huggingface/download/*.lock. Reap them first.
    _kill_orphan_downloaders()

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

    pending = [
        (name, repo_id)
        for name, repo_id in MODELS_TO_DOWNLOAD.items()
        if not model_complete(name)
    ]
    # Publish queue-position events directly on this loop — calling the threaded
    # poster from inside the running loop deadlocks until its 10s timeout fires.
    from app.jobs.progress_bus import publish_model_progress
    for idx, (name, repo_id) in enumerate(pending, start=1):
        await publish_model_progress(
            name,
            "queued",
            _download_size_bytes(name, repo_id),
            0,
            note=f"queued, position {idx}/{len(pending)}",
        )
        logger.info(
            f"Queued model '{name}' (position {idx}/{len(pending)})"
        )
        _download_queue.put((name, repo_id))

