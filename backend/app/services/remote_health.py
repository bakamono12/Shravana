"""Background task that polls the remote GPU worker's /v1/health endpoint.

Caches the last result so the /api/system/executor route can respond instantly.
Started in main.py lifespan only when REMOTE_GPU_URL is configured.
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_cache: dict[str, Any] = {
    "healthy": False,
    "last_check": None,
    "api_version": 1,
    "gpu_info": {},
    "models_loaded": [],
    "error": None,
}


def get_cached() -> dict:
    return dict(_cache)


async def _poll_once() -> None:
    from app.config import settings
    from app.ml.remote_client import health as _health

    loop = asyncio.get_event_loop()
    result: dict = await loop.run_in_executor(None, _health)

    _cache["last_check"] = time.time()
    if result:
        _cache["healthy"] = True
        _cache["api_version"] = int(result.get("api_version", 1))
        _cache["gpu_info"] = result.get("gpu", {})
        _cache["models_loaded"] = result.get("models_loaded", [])
        _cache["error"] = None
    else:
        _cache["healthy"] = False
        _cache["gpu_info"] = {}
        _cache["models_loaded"] = []
        _cache["error"] = "health check returned empty response"


async def start_health_poller() -> None:
    """Run as a background asyncio task; poll until cancelled."""
    from app.config import settings

    logger.info(f"Remote health poller started — target: {settings.REMOTE_GPU_URL}")
    while True:
        try:
            await _poll_once()
        except Exception as exc:
            _cache["healthy"] = False
            _cache["last_check"] = time.time()
            _cache["error"] = str(exc)
            logger.debug(f"Remote health poll error: {exc}")
        await asyncio.sleep(settings.REMOTE_GPU_HEALTHCHECK_INTERVAL_S)
