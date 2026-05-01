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
async def download_subtitle(job_id: str, fmt: str, db: AsyncSession = Depends(get_db)):
    if fmt not in ("srt", "vtt", "json"):
        raise HTTPException(400, f"Unknown format '{fmt}'")

    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status != "done":
        raise HTTPException(409, f"Job not done yet (status: {job.status})")

    result = await db.execute(
        select(Subtitle).where(Subtitle.job_id == job_id, Subtitle.format == fmt)
    )
    sub = result.scalar_one_or_none()
    if not sub:
        raise HTTPException(404, f"No {fmt} subtitle for this job")

    stem = Path(job.filename).stem or job_id
    media_types = {"srt": "text/plain", "vtt": "text/vtt", "json": "application/json"}
    return FileResponse(
        sub.path,
        media_type=media_types[fmt],
        filename=f"{stem}.{fmt}",
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
