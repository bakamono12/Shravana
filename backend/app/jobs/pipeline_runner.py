"""Orchestrate the full transcription pipeline for one job."""
from __future__ import annotations
import asyncio
import json
import logging
import time
from pathlib import Path

from sqlalchemy import select

from app.config import settings
from app.db import AsyncSessionLocal
from app.models_db import Job, Chunk, Subtitle
from app.ml.base import ModelNotReady
from app.jobs.progress_bus import publish_job_progress
import app.storage as storage

logger = logging.getLogger(__name__)

# Serialize pipeline runs (CPU + RAM constrained)
_pipeline_sem = asyncio.Semaphore(settings.PIPELINE_CONCURRENCY)


async def _update_job(session, job: Job, **kwargs) -> None:
    for k, v in kwargs.items():
        setattr(job, k, v)
    await session.commit()
    await _emit_progress(job)


async def _emit_progress(job: Job) -> None:
    total = job.total_chunks or 1
    done = job.completed_chunks or 0
    pct = int(done / total * 100)
    await publish_job_progress(job.id, {
        "job_id": job.id,
        "status": job.status,
        "phase": job.status,
        "total_chunks": job.total_chunks,
        "completed_chunks": job.completed_chunks,
        "percent": pct,
        "srt_url": f"/api/subtitles/{job.id}/srt" if job.status == "done" else None,
        "vtt_url": f"/api/subtitles/{job.id}/vtt" if job.status == "done" else None,
        "error": job.error_message,
    })


async def run(job_id: str) -> None:
    async with _pipeline_sem:
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if not job:
                return
            try:
                await _run_pipeline(session, job)
            except ModelNotReady as exc:
                job.status = "waiting_for_models"
                job.waiting_for_model = exc.model_name
                await session.commit()
                await _emit_progress(job)
            except Exception as exc:
                logger.exception(f"Pipeline failed for job {job_id}")
                job.status = "failed"
                job.error_message = str(exc)
                await session.commit()
                await _emit_progress(job)


async def run_translation_only(job_id: str) -> None:
    """Re-run Phase 5.5 on an already-transcribed job (retro-translate)."""
    async with _pipeline_sem:
        async with AsyncSessionLocal() as session:
            job = await session.get(Job, job_id)
            if not job:
                return
            try:
                await _run_translation_only(session, job)
            except ModelNotReady as exc:
                job.status = "waiting_for_models"
                job.waiting_for_model = exc.model_name
                await session.commit()
                await _emit_progress(job)
            except Exception as exc:
                logger.exception(f"Translation failed for job {job_id}")
                job.status = "failed"
                job.translation_status = "failed"
                job.error_message = str(exc)
                await session.commit()
                await _emit_progress(job)


async def _run_pipeline(session, job: Job) -> None:
    loop = asyncio.get_event_loop()

    # Stamp executor mode so the UI can show Local / Remote
    from app.config import settings as _cfg
    from app.services.executor_pref import resolve_executor
    executor = resolve_executor()
    job.executor = executor
    job.remote_url_snapshot = _cfg.REMOTE_GPU_URL or None
    await session.commit()

    # Phase 1: Extract audio
    await _update_job(session, job, status="extracting")
    from app.pipeline.audio_extractor import extract, normalize, detect_input_type
    raw_wav = str(storage.raw_audio_path(job.id))
    kind = detect_input_type(job.filename)
    if kind == "video":
        await loop.run_in_executor(None, extract, job.video_path, raw_wav)
    else:
        await loop.run_in_executor(None, normalize, job.video_path, raw_wav)
    job.audio_path = raw_wav
    await session.commit()

    # --- Delegate phases 2-7 to the remote GPU worker when configured --------
    if executor == "remote":
        from app.jobs.remote_pipeline import run_remote_pipeline, RemotePipelineError
        try:
            await run_remote_pipeline(session, job, raw_wav)
            return
        except RemotePipelineError as exc:
            if not settings.REMOTE_GPU_FALLBACK_LOCAL or exc.received_any_event:
                job.status = "failed"
                job.error_message = str(exc)
                job.executor = "remote"
                await session.commit()
                await _emit_progress(job)
                return
            logger.warning(f"Remote pipeline unavailable ({exc}), running locally")
            job.executor = "local"
            await session.commit()
    # --------------------------------------------------------------------------

    # Phase 2: Denoise
    await _update_job(session, job, status="denoising")
    from app.pipeline.denoiser import denoise, is_chunk_silent
    job_dir = str(storage.job_dir(job.id))

    def _denoise_progress(subphase: str, pct: int) -> None:
        asyncio.run_coroutine_threadsafe(
            publish_job_progress(job.id, {
                "job_id": job.id, "status": "denoising", "phase": "denoise",
                "subphase": subphase, "percent": pct,
                "stage_index": 2, "stage_total": 6,
            }),
            loop,
        )

    clean_wav, speech_intervals = await loop.run_in_executor(
        None, denoise, raw_wav, job_dir, job.id, _denoise_progress
    )
    job.clean_audio_path = clean_wav
    await session.commit()

    # Phase 3: Chunk
    await _update_job(session, job, status="chunking")
    await publish_job_progress(job.id, {
        "job_id": job.id, "status": "chunking", "phase": "chunk",
        "percent": 0, "stage_index": 3, "stage_total": 6,
    })
    from app.pipeline.chunker import chunk_audio
    chunks_dir = str(storage.chunks_dir(job.id))
    manifest = await loop.run_in_executor(None, chunk_audio, clean_wav, chunks_dir)

    for entry in manifest:
        silent = is_chunk_silent(entry.start_time, entry.end_time, speech_intervals)
        chunk = Chunk(
            job_id=job.id,
            sequence=entry.sequence,
            filename=entry.filename,
            path=entry.path,
            start_time=entry.start_time,
            end_time=entry.end_time,
            duration=entry.duration,
            status="skipped" if silent else "pending",
        )
        session.add(chunk)
    job.total_chunks = len(manifest)
    job.completed_chunks = sum(1 for e in manifest if is_chunk_silent(e.start_time, e.end_time, speech_intervals))
    await session.commit()
    await publish_job_progress(job.id, {
        "job_id": job.id, "status": "chunking", "phase": "chunk",
        "percent": 100, "stage_index": 3, "stage_total": 6,
        "total_chunks": job.total_chunks,
    })

    # Phase 4-5: Detect language + transcribe each chunk
    await _update_job(session, job, status="processing")
    from app.pipeline.language_detector import detect
    from app.pipeline.router import route
    from app.pipeline.transcriber import transcribe_chunk
    from app.pipeline.reassembler import serialize_segments

    result = await session.execute(
        select(Chunk).where(Chunk.job_id == job.id, Chunk.status == "pending").order_by(Chunk.sequence)
    )
    pending_chunks = result.scalars().all()

    # Detect language from first chunk
    first_lang = job.language_hint or "en"
    first_conf = 0.0
    if pending_chunks:
        first_lang, first_conf = await loop.run_in_executor(None, detect, pending_chunks[0].path, job.language_hint)
        job.detected_language = first_lang
        await session.commit()

    start_time = time.time()
    for i, chunk in enumerate(pending_chunks):
        chunk.status = "processing"
        chunk.detected_language = first_lang
        model_name = route(first_lang, first_conf)
        chunk.assigned_model = model_name
        await session.commit()

        try:
            segs = await loop.run_in_executor(None, transcribe_chunk, chunk.path, first_lang)
            chunk.transcript_json = serialize_segments(segs)
            chunk.status = "done"
        except ModelNotReady:
            raise
        except Exception as exc:
            chunk.status = "failed"
            chunk.error_message = str(exc)
            logger.error(f"Chunk {chunk.id} failed: {exc}")

        job.completed_chunks += 1
        elapsed = time.time() - start_time
        remaining = len(pending_chunks) - (i + 1)
        eta = int(elapsed / (i + 1) * remaining) if i > 0 else None
        await session.commit()
        await publish_job_progress(job.id, {
            "job_id": job.id,
            "status": "processing",
            "phase": "transcribing",
            "total_chunks": job.total_chunks,
            "completed_chunks": job.completed_chunks,
            "percent": int(job.completed_chunks / job.total_chunks * 100) if job.total_chunks else 0,
            "eta_seconds": eta,
            "stage_index": 4, "stage_total": 6,
        })

    # Phase 5.5: Context-aware translation (no-op when translate=False)
    if job.translate and job.target_language:
        source_lang = first_lang
        await _run_translation_phase(session, job, loop, source_lang)

    # Phase 6: Reassemble (source language subtitles)
    await _update_job(session, job, status="assembling")
    await publish_job_progress(job.id, {
        "job_id": job.id, "status": "assembling", "phase": "assemble",
        "percent": 0, "stage_index": 6, "stage_total": 6,
    })
    from app.pipeline.reassembler import reassemble

    result = await session.execute(
        select(Chunk).where(Chunk.job_id == job.id).order_by(Chunk.sequence)
    )
    all_chunks = result.scalars().all()
    chunk_data = [
        {"sequence": c.sequence, "start_time": c.start_time, "transcript_json": c.transcript_json}
        for c in all_chunks if c.transcript_json
    ]
    merged_segments = await loop.run_in_executor(None, reassemble, chunk_data)

    # Phase 7: Generate source-language subtitles
    from app.pipeline.subtitle_generator import generate_srt, generate_vtt
    srt_path = str(storage.subtitle_path(job.id, "srt"))
    vtt_path = str(storage.subtitle_path(job.id, "vtt"))
    await loop.run_in_executor(None, generate_srt, merged_segments, srt_path)
    await loop.run_in_executor(None, generate_vtt, merged_segments, vtt_path)

    src_lang_code = job.detected_language or job.language_hint
    from app.db_crud import upsert_subtitle
    for fmt, path in [("srt", srt_path), ("vtt", vtt_path)]:
        await upsert_subtitle(session, job.id, fmt, path, src_lang_code)

    job.status = "done"
    await session.commit()
    await _emit_progress(job)
    logger.info(f"Job {job.id} completed successfully")


async def _run_translation_phase(session, job: Job, loop, source_lang: str) -> None:
    """Phase 5.5: context build → semantic chunk → translate → emit translated subtitles."""
    import json as _json
    import dataclasses
    from app.pipeline.reassembler import reassemble as _reassemble_all
    from app.pipeline.subtitle_generator import generate_srt, generate_vtt
    from app.pipeline.translation_orchestrator import run_translation_compute
    from app.models_db import TranslationChunk as TranslationChunkRow

    await _update_job(session, job, status="translating")

    stt_result = await session.execute(
        select(Chunk).where(Chunk.job_id == job.id).order_by(Chunk.sequence)
    )
    stt_chunks = stt_result.scalars().all()
    stt_chunk_data = [
        {"sequence": c.sequence, "start_time": c.start_time, "transcript_json": c.transcript_json}
        for c in stt_chunks if c.transcript_json
    ]

    await publish_job_progress(job.id, {
        "job_id": job.id, "status": "translating", "phase": "context",
        "total_chunks": job.total_chunks, "completed_chunks": 0, "percent": 0,
    })

    source_segments = await loop.run_in_executor(None, _reassemble_all, stt_chunk_data)
    job_dir = str(storage.job_dir(job.id))

    def _on_progress(done: int, total: int) -> None:
        asyncio.run_coroutine_threadsafe(
            publish_job_progress(job.id, {
                "job_id": job.id, "status": "translating", "phase": "translating",
                "total_chunks": total, "completed_chunks": done,
                "percent": int(done / total * 100),
                "stage_index": 5, "stage_total": 6,
            }),
            loop,
        )

    translated_segs, unit_results, context = await loop.run_in_executor(
        None,
        lambda: run_translation_compute(
            source_segments, source_lang, job.target_language,
            job.enable_refinement, job.glossary_json or "",
            job.video_path, job_dir,
            on_progress=_on_progress,
        ),
    )

    # Persist context bundle
    job.context_bundle_json = _json.dumps({
        "domain": context.domain, "format": context.format,
        "named_entities": context.named_entities, "idioms_detected": context.idioms_detected,
        "scene_description": context.scene_description, "notes": context.notes,
    })

    # Persist translation units
    for ur in unit_results:
        session.add(TranslationChunkRow(
            job_id=job.id, sequence=ur["sequence"],
            start_time=ur["start_time"], end_time=ur["end_time"],
            source_text=ur["source_text"],
            first_pass=ur["first_pass"],
            translated_text=ur["final_text"],
            status="done",
        ))

    job.translation_status = "done"
    await session.commit()

    await publish_job_progress(job.id, {
        "job_id": job.id, "status": "translating", "phase": "assembling",
        "total_chunks": len(unit_results), "completed_chunks": len(unit_results), "percent": 100,
    })

    tl_srt = str(storage.subtitle_path(job.id, "srt")).replace(".srt", f"_{job.target_language}.srt")
    tl_vtt = str(storage.subtitle_path(job.id, "vtt")).replace(".vtt", f"_{job.target_language}.vtt")
    await loop.run_in_executor(None, generate_srt, translated_segs, tl_srt)
    await loop.run_in_executor(None, generate_vtt, translated_segs, tl_vtt)
    from app.db_crud import upsert_subtitle
    for fmt, path in [("srt", tl_srt), ("vtt", tl_vtt)]:
        await upsert_subtitle(session, job.id, fmt, path, job.target_language)
    await session.commit()


async def _run_translation_only(session, job: Job) -> None:
    """Retro-translate: run Phase 5.5 + emit done. Assumes STT is already complete."""
    loop = asyncio.get_event_loop()
    source_lang = job.detected_language or job.language_hint or "en"
    await _run_translation_phase(session, job, loop, source_lang)
    job.status = "done"
    await session.commit()
    await _emit_progress(job)
    logger.info(f"Job {job.id} translation-only completed")
