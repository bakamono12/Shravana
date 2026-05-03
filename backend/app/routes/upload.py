import asyncio
import re
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, BackgroundTasks, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models_db import Job
from app.pipeline.audio_extractor import ACCEPTED_EXTENSIONS
from app.pipeline.translation_languages import is_supported
from app.schemas import UploadResponse
import app.storage as storage
from app.jobs import pipeline_runner

router = APIRouter()

MAX_SIZE_BYTES = 5 * 1024 ** 3  # 5 GB


def _safe_filename(raw: str) -> str:
    """Strip path components, collapse unsafe chars, preserve extension."""
    name = Path(raw).name  # drop any directory prefix
    stem, _, ext = name.rpartition(".")
    if not stem:
        stem, ext = ext, ""
    # keep letters, digits, spaces, hyphens, underscores, dots
    stem = re.sub(r"[^\w\s\-.]", "_", stem).strip()
    stem = re.sub(r"\s+", "_", stem)
    stem = stem[:200] or "upload"
    ext = re.sub(r"[^\w]", "", ext)[:10]
    return f"{stem}.{ext}" if ext else stem


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language_hint: str | None = Form(None),
    target_language: str | None = Form(None),
    translator_mode: str | None = Form(None),
    enable_refinement: bool = Form(True),
    glossary: str | None = Form(None),
    db: AsyncSession = Depends(get_db),
):
    raw_name = file.filename or "upload"
    safe_name = _safe_filename(raw_name)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in ACCEPTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Accepted: {', '.join(sorted(ACCEPTED_EXTENSIONS))}")

    if target_language and not is_supported(target_language):
        raise HTTPException(400, f"Unsupported target language '{target_language}'. See /api/translation/languages for supported codes.")

    if translator_mode and translator_mode not in ("vlm", "audio"):
        raise HTTPException(400, "translator_mode must be 'vlm' or 'audio'")

    job = Job(
        filename=safe_name,
        video_path="",
        language_hint=language_hint,
        translate=bool(target_language),
        target_language=target_language,
        translator_mode=translator_mode,
        enable_refinement=enable_refinement,
        glossary_json=glossary,
    )
    db.add(job)
    await db.flush()  # get the generated id

    dest = storage.upload_path(job.id, safe_name)
    job.video_path = str(dest)
    await db.commit()

    # Stream file to disk
    total = 0
    async with aiofiles.open(dest, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_SIZE_BYTES:
                await f.close()
                dest.unlink(missing_ok=True)
                await db.delete(job)
                await db.commit()
                raise HTTPException(413, "File too large (max 5 GB)")
            await f.write(chunk)

    background_tasks.add_task(_run_pipeline, job.id)
    return UploadResponse(job_id=job.id, filename=job.filename, status="pending")


async def _run_pipeline(job_id: str) -> None:
    await pipeline_runner.run(job_id)
