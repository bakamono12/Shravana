import shutil

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db import get_db
from app.models_db import Job, Chunk, Subtitle
from app.schemas import JobOut, JobDetailOut, ChunkOut
from app.jobs import pipeline_runner

router = APIRouter()


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
