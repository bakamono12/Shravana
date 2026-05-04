"""System/executor status and override routes."""
from typing import Literal, Optional
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ExecutorOverrideRequest(BaseModel):
    override: Optional[Literal["local", "remote"]] = None


@router.get("/system/executor")
async def executor_status() -> dict:
    """Returns the current executor config and (when remote) the worker's health."""
    from app.config import settings
    from app.services.executor_pref import get_override, resolve_executor
    from app.services.remote_health import get_cached

    override = get_override()
    effective = resolve_executor()
    remote_url_present = bool(settings.REMOTE_GPU_URL)

    base = {
        "env_default": "remote" if settings.REMOTE_GPU_URL else "local",
        "override": override,
        "effective": effective,
        "remote_url_present": remote_url_present,
        # Legacy fields kept for backward compat
        "mode": effective,
        "remote_url": settings.REMOTE_GPU_URL or None,
    }

    if not settings.REMOTE_GPU_URL:
        return {**base, "healthy": None, "last_check": None, "gpu_info": {}, "models_loaded": []}

    cached = get_cached()
    return {
        **base,
        "healthy": cached["healthy"],
        "api_version": cached.get("api_version", 1),
        "last_check": cached["last_check"],
        "gpu_info": cached["gpu_info"],
        "models_loaded": cached["models_loaded"],
        "error": cached.get("error"),
    }


@router.put("/system/executor")
async def set_executor_override(body: ExecutorOverrideRequest) -> dict:
    """Override the executor for new jobs. Pass null to follow env default."""
    from app.services.executor_pref import set_override, resolve_executor
    set_override(body.override)
    return {"override": body.override, "effective": resolve_executor()}
