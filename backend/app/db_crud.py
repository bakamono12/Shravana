"""Shared CRUD helpers that avoid duplicate rows."""
from __future__ import annotations
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models_db import Subtitle


async def upsert_subtitle(
    session: AsyncSession,
    job_id: str,
    fmt: str,
    path: str,
    language: str | None,
) -> None:
    """Delete any existing (job_id, format, language) row then insert a fresh one."""
    await session.execute(
        delete(Subtitle).where(
            Subtitle.job_id == job_id,
            Subtitle.format == fmt,
            Subtitle.language == language,
        )
    )
    session.add(Subtitle(job_id=job_id, format=fmt, path=path, language=language))
