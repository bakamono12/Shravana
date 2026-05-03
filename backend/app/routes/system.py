"""System/executor status route."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/system/executor")
async def executor_status() -> dict:
    """Returns the current executor mode and (when remote) the worker's health."""
    from app.config import settings
    from app.services.remote_health import get_cached

    if not settings.REMOTE_GPU_URL:
        return {
            "mode": "local",
            "remote_url": None,
            "healthy": None,
            "last_check": None,
            "gpu_info": {},
            "models_loaded": [],
        }

    cached = get_cached()
    return {
        "mode": "remote",
        "remote_url": settings.REMOTE_GPU_URL,
        "healthy": cached["healthy"],
        "last_check": cached["last_check"],
        "gpu_info": cached["gpu_info"],
        "models_loaded": cached["models_loaded"],
        "error": cached.get("error"),
    }
