import json
import shutil

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.models_db import Job, Chunk, Subtitle, TranslationChunk
from app.schemas import JobOut, JobDetailOut, ChunkOut, ContextBundleOut
from app.jobs import pipeline_runner
from app.pipeline.translation_languages import is_supported

router = APIRouter()


class TranslateRequest(BaseModel):
    target_language: str
    translator_mode: str | None = None
    enable_refinement: bool = True
    glossary: str | None = None


@router.get("/jobs", response_model=list[JobOut])
async def list_jobs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).order_by(Job.created_at.desc()))
    return result.scalars().all()


@router.get("/jobs/{job_id}", response_model=JobDetailOut)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    result = await db.execute(select(Chunk).where(Chunk.job_id == job_id).order_by(Chunk.sequence))
    chunks = result.scalars().all()
    return JobDetailOut.model_validate({**JobOut.model_validate(job).model_dump(), "chunks": chunks})


@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status not in ("failed", "waiting_for_models"):
        raise HTTPException(400, f"Job status '{job.status}' cannot be retried")
    job.status = "pending"
    job.error_message = None
    job.waiting_for_model = None
    await db.commit()
    background_tasks.add_task(pipeline_runner.run, job_id)
    return {"job_id": job_id, "status": "pending"}


@router.post("/jobs/{job_id}/translate")
async def translate_job(
    job_id: str,
    req: TranslateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Retro-translate a completed STT job or re-run translation with new params."""
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status not in ("done", "failed"):
        raise HTTPException(409, f"Job must be done before translating (status: {job.status})")
    if not is_supported(req.target_language):
        raise HTTPException(400, f"Unsupported target language '{req.target_language}'")
    if req.translator_mode and req.translator_mode not in ("vlm", "audio"):
        raise HTTPException(400, "translator_mode must be 'vlm' or 'audio'")

    job.translate = True
    job.target_language = req.target_language
    job.translator_mode = req.translator_mode
    job.enable_refinement = req.enable_refinement
    job.glossary_json = req.glossary
    job.translation_status = "pending"
    # Reset to assembling so the runner re-enters at Phase 5.5
    job.status = "processing"
    job.error_message = None
    await db.commit()

    background_tasks.add_task(pipeline_runner.run_translation_only, job_id)
    return {"job_id": job_id, "status": "translating", "target_language": req.target_language}


@router.get("/jobs/{job_id}/context", response_model=ContextBundleOut)
async def get_job_context(job_id: str, db: AsyncSession = Depends(get_db)):
    """Return the ContextBundle built for a job (for inspection / debugging)."""
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.context_bundle_json:
        raise HTTPException(404, "No context bundle available for this job yet")
    data = json.loads(job.context_bundle_json)
    return ContextBundleOut(
        domain=data.get("domain", "casual"),
        format=data.get("format", "monologue"),
        source_language=job.detected_language or job.language_hint or "unknown",
        target_language=job.target_language or "unknown",
        named_entities=data.get("named_entities", []),
        idioms_detected=data.get("idioms_detected", []),
        scene_description=data.get("scene_description"),
        notes=data.get("notes"),
    )


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    active = {"extracting", "denoising", "chunking", "processing", "assembling"}
    if job.status in active:
        raise HTTPException(409, "Cannot delete a job that is actively running")
    await db.execute(delete(Chunk).where(Chunk.job_id == job_id))
    await db.execute(delete(Subtitle).where(Subtitle.job_id == job_id))
    await db.delete(job)
    await db.commit()
    for sub in ("jobs", "uploads"):
        p = settings.BASE_DIR / "storage" / sub / job_id
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
    return {"deleted": job_id}
