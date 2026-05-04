"""In-process executor preference override.

The env var REMOTE_GPU_URL is the default. This module holds a runtime override
that takes precedence, letting the UI flip the executor without a restart.
Override is lost on backend restart — use REMOTE_GPU_URL in .env for persistence.
"""
from __future__ import annotations
from typing import Literal, Optional

_override: Optional[Literal["local", "remote"]] = None


def get_override() -> Optional[Literal["local", "remote"]]:
    return _override


def set_override(value: Optional[Literal["local", "remote"]]) -> None:
    global _override
    _override = value


def resolve_executor() -> Literal["local", "remote"]:
    """Return the effective executor for the next job."""
    from app.config import settings
    if _override == "local":
        return "local"
    if _override == "remote":
        return "remote" if settings.REMOTE_GPU_URL else "local"
    # Auto: follow env
    return "remote" if settings.REMOTE_GPU_URL else "local"
