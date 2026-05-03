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
    executor = "remote" if _cfg.REMOTE_GPU_URL else "local"
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

    # Phase 2: Denoise
    await _update_job(session, job, status="denoising")
    from app.pipeline.denoiser import denoise, is_chunk_silent
    job_dir = str(storage.job_dir(job.id))
    clean_wav, speech_intervals = await loop.run_in_executor(None, denoise, raw_wav, job_dir, job.id)
    job.clean_audio_path = clean_wav
    await session.commit()

    # Phase 3: Chunk
    await _update_job(session, job, status="chunking")
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
    await _emit_progress(job)

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
            segs = await loop.run_in_executor(None, transcribe_chunk, chunk.path, model_name, first_lang)
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
        })

    # Phase 5.5: Context-aware translation (no-op when translate=False)
    if job.translate and job.target_language:
        source_lang = first_lang
        await _run_translation_phase(session, job, loop, source_lang)

    # Phase 6: Reassemble (source language subtitles)
    await _update_job(session, job, status="assembling")
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
    for fmt, path in [("srt", srt_path), ("vtt", vtt_path)]:
        session.add(Subtitle(job_id=job.id, format=fmt, path=path, language=src_lang_code))

    job.status = "done"
    await session.commit()
    await _emit_progress(job)
    logger.info(f"Job {job.id} completed successfully")


async def _run_translation_phase(session, job: Job, loop, source_lang: str) -> None:
    """Phase 5.5: context build → semantic chunk → translate → emit translated subtitles."""
    import json as _json
    from app.pipeline.reassembler import reassemble as _reassemble_all
    from app.pipeline.semantic_chunker import chunk_for_translation
    from app.pipeline.context_builder import build_context
    from app.pipeline.translator import translate_unit, select_translator_mode
    from app.pipeline.translation_reassembler import distribute_translation_to_segments
    from app.pipeline.glossary import Glossary
    from app.pipeline.subtitle_generator import generate_srt, generate_vtt
    from app.models_db import TranslationChunk as TranslationChunkRow
    from app.ml.base import TranslatedSegment as _TlSeg

    await _update_job(session, job, status="translating")

    mode = job.translator_mode or select_translator_mode(job.filename, None)
    job.translator_mode = mode
    await session.commit()

    stt_result = await session.execute(
        select(Chunk).where(Chunk.job_id == job.id).order_by(Chunk.sequence)
    )
    stt_chunks = stt_result.scalars().all()
    stt_chunk_data = [
        {"sequence": c.sequence, "start_time": c.start_time, "transcript_json": c.transcript_json}
        for c in stt_chunks if c.transcript_json
    ]

    all_segs_dicts: list = []
    for c in stt_chunks:
        if c.transcript_json:
            parsed = _json.loads(c.transcript_json)
            offset = c.start_time
            for s in parsed:
                all_segs_dicts.append({
                    "text": s.get("text", ""),
                    "start": s.get("start", 0) + offset,
                    "end": s.get("end", 0) + offset,
                })

    is_video = mode == "vlm"
    job_dir = str(storage.job_dir(job.id))

    await publish_job_progress(job.id, {
        "job_id": job.id, "status": "translating", "phase": "context",
        "total_chunks": job.total_chunks, "completed_chunks": 0, "percent": 0,
    })

    context = await loop.run_in_executor(
        None, build_context,
        source_lang, job.target_language, all_segs_dicts,
        job.video_path if is_video else None, job_dir, is_video,
    )
    job.context_bundle_json = _json.dumps({
        "domain": context.domain, "format": context.format,
        "named_entities": context.named_entities, "idioms_detected": context.idioms_detected,
        "scene_description": context.scene_description, "notes": context.notes,
    })
    await session.commit()

    glossary = Glossary.from_domain(context.domain)
    if job.glossary_json:
        try:
            glossary = glossary.merge(Glossary.from_user_text(job.glossary_json))
        except Exception:
            pass

    merged_for_chunking = await loop.run_in_executor(None, _reassemble_all, stt_chunk_data)
    seg_dicts = [{"text": s.text, "start": s.start, "end": s.end} for s in merged_for_chunking]
    translation_units = await loop.run_in_executor(
        None, chunk_for_translation, seg_dicts,
        settings.TRANSLATION_MAX_CHUNK_SECONDS, settings.TRANSLATION_SILENCE_GAP,
    )

    for unit in translation_units:
        session.add(TranslationChunkRow(
            job_id=job.id, sequence=unit.sequence,
            start_time=unit.start_time, end_time=unit.end_time,
            source_text=unit.source_text, status="pending",
        ))
    await session.commit()

    history: list = []
    translated_texts: list[str] = []
    n_units = len(translation_units)

    for i, unit in enumerate(translation_units):
        prev_text = translation_units[i - 1].source_text if i > 0 else ""
        next_text = translation_units[i + 1].source_text if i < n_units - 1 else ""

        await publish_job_progress(job.id, {
            "job_id": job.id, "status": "translating", "phase": "translating",
            "total_chunks": n_units, "completed_chunks": i, "percent": int(i / n_units * 100),
        })

        first_pass = unit.source_text
        final_text = unit.source_text
        try:
            first_pass, final_text = await loop.run_in_executor(
                None, translate_unit,
                unit, job.video_path if is_video else None,
                source_lang, job.target_language,
                mode, history, prev_text, next_text,
                context, glossary, job.enable_refinement,
            )
        except ModelNotReady:
            raise
        except Exception as exc:
            logger.error(f"Translation unit {unit.sequence} failed: {exc}")

        translated_texts.append(final_text)

        tc_result = await session.execute(
            select(TranslationChunkRow).where(
                TranslationChunkRow.job_id == job.id,
                TranslationChunkRow.sequence == unit.sequence,
            )
        )
        tc_row = tc_result.scalar_one_or_none()
        if tc_row:
            tc_row.first_pass = first_pass
            tc_row.translated_text = final_text
            tc_row.status = "done"
        await session.commit()

        dummy_ts = _TlSeg(
            text=final_text, start=unit.start_time, end=unit.end_time,
            source_text=unit.source_text, source_language=source_lang,
            target_language=job.target_language,
        )
        history = (history + [dummy_ts])[-settings.TRANSLATION_CONTEXT_WINDOW:]

    job.translation_status = "done"
    await session.commit()

    translated_segs = await loop.run_in_executor(
        None, distribute_translation_to_segments,
        merged_for_chunking, translation_units, translated_texts, job.target_language,
    )

    await publish_job_progress(job.id, {
        "job_id": job.id, "status": "translating", "phase": "assembling",
        "total_chunks": n_units, "completed_chunks": n_units, "percent": 100,
    })

    tl_srt = str(storage.subtitle_path(job.id, "srt")).replace(".srt", f"_{job.target_language}.srt")
    tl_vtt = str(storage.subtitle_path(job.id, "vtt")).replace(".vtt", f"_{job.target_language}.vtt")
    await loop.run_in_executor(None, generate_srt, translated_segs, tl_srt)
    await loop.run_in_executor(None, generate_vtt, translated_segs, tl_vtt)
    for fmt, path in [("srt", tl_srt), ("vtt", tl_vtt)]:
        session.add(Subtitle(job_id=job.id, format=fmt, path=path, language=job.target_language))
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
