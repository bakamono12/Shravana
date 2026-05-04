import mimetypes
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models_db import Subtitle, Job
import app.storage as storage

router = APIRouter()


@router.get("/subtitles/{job_id}/{fmt}")
async def download_subtitle(
    job_id: str,
    fmt: str,
    lang: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Download subtitle file.

    Query param ?lang= selects language variant (BCP-47 code).
    Omit to get the default: target language if translated, else source.
    """
    if fmt not in ("srt", "vtt", "json"):
        raise HTTPException(400, f"Unknown format '{fmt}'")

    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done":
        raise HTTPException(409, f"Job not done yet (status: {job.status})")

    # Determine effective language
    effective_lang = lang
    if not effective_lang:
        effective_lang = job.target_language if job.translate and job.target_language else None

    # Try exact language match first; fall back to any subtitle for this format
    if effective_lang:
        result = await db.execute(
            select(Subtitle).where(
                Subtitle.job_id == job_id,
                Subtitle.format == fmt,
                Subtitle.language == effective_lang,
            ).order_by(Subtitle.created_at.desc())
        )
        sub = result.scalars().first()
        if not sub:
            # Fall back to source-language subtitle (language IS NULL or doesn't match)
            result = await db.execute(
                select(Subtitle).where(
                    Subtitle.job_id == job_id,
                    Subtitle.format == fmt,
                ).order_by(Subtitle.created_at)
            )
            sub = result.scalars().first()
    else:
        result = await db.execute(
            select(Subtitle).where(
                Subtitle.job_id == job_id,
                Subtitle.format == fmt,
            ).order_by(Subtitle.created_at)
        )
        sub = result.scalars().first()

    if not sub:
        raise HTTPException(404, f"No {fmt} subtitle for this job")

    stem = Path(job.filename).stem or job_id
    suffix = f"_{sub.language}" if sub.language else ""
    media_types = {"srt": "text/plain", "vtt": "text/vtt", "json": "application/json"}
    return FileResponse(
        sub.path,
        media_type=media_types[fmt],
        filename=f"{stem}{suffix}.{fmt}",
    )


@router.get("/jobs/{job_id}/media")
async def stream_media(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    raw = job.video_path or job.audio_path or job.clean_audio_path
    if not raw or not Path(raw).exists():
        raise HTTPException(404, "Media not available")
    mime, _ = mimetypes.guess_type(raw)
    return FileResponse(raw, media_type=mime or "application/octet-stream")


@router.get("/jobs/{job_id}/clean.wav")
async def download_clean_wav(job_id: str, db: AsyncSession = Depends(get_db)):
    """Debug endpoint — download the denoised audio."""
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.clean_audio_path:
        raise HTTPException(404, "Clean audio not available yet")
    return FileResponse(job.clean_audio_path, media_type="audio/wav", filename=f"{job_id}_clean.wav")
