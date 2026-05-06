"""Two-stage remote pipeline client.

Stage 1: run_remote_transcribe() — uploads audio to /v1/pipeline/transcribe,
         streams progress, persists Chunk + source Subtitle rows, stores workdir_token.
Stage 2: run_remote_translate()  — sends workdir_token to /v1/pipeline/translate,
         streams progress, persists translated Subtitle rows.

Legacy: run_remote_pipeline() calls both stages sequentially for backward compat.
"""
from __future__ import annotations
import asyncio
import json
import logging
from pathlib import Path

from app.config import settings
from app.jobs.progress_bus import publish_job_progress
from app.models_db import Job, Chunk
import app.storage as storage

logger = logging.getLogger(__name__)


class RemotePipelineError(Exception):
    def __init__(self, message: str, received_any_event: bool = False):
        super().__init__(message)
        self.received_any_event = received_any_event


# ------------------------------------------------------------------ #
# Generic streaming helper                                            #
# ------------------------------------------------------------------ #

def _make_stream_thread(endpoint: str, files: dict | None, data: dict | None,
                        json_body: dict | None, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """Return a callable that streams from the worker endpoint into queue."""
    from app.ml.remote_client import call_streaming_multipart, call_streaming_json, RemoteCallError

    def _thread() -> None:
        received_any = False
        try:
            if files is not None:
                stream_fn = lambda: call_streaming_multipart(endpoint, files, data or {})
            else:
                stream_fn = lambda: call_streaming_json(endpoint, json_body or {})

            for event in stream_fn():
                received_any = True
                asyncio.run_coroutine_threadsafe(queue.put(("event", event)), loop).result(timeout=60)
        except RemoteCallError as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", str(exc), received_any)), loop
            ).result(timeout=5)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", f"Streaming failed: {exc}", received_any)), loop
            ).result(timeout=5)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done",)), loop).result(timeout=5)

    return _thread


async def _consume_stream(queue: asyncio.Queue, job: "Job", session,
                           on_event=None) -> dict:
    """
    Drain queue until done/error, forwarding progress to UI.
    Returns the final 'result' event dict.
    Raises RemotePipelineError on error.
    """
    received_any = False
    while True:
        msg = await queue.get()
        kind = msg[0]

        if kind == "done":
            break

        if kind == "error":
            _, errmsg, had_events = msg
            raise RemotePipelineError(errmsg, received_any_event=had_events)

        event = msg[1]
        received_any = True
        ev_type = event.get("event", "progress")

        if ev_type == "error":
            raise RemotePipelineError(event.get("message", "Worker error"), received_any_event=True)

        if ev_type == "result":
            return event

        if on_event:
            await on_event(event, job, session)
        else:
            await _forward_progress(event, job, session)

    return {}


async def _forward_progress(event: dict, job: "Job", session) -> None:
    phase = event.get("phase", "")
    status_map = {
        "denoising": "denoising",
        "chunking": "chunking",
        "processing": "processing",
        "translating": "translating",
        "assembling": "assembling",
    }
    ui_status = status_map.get(phase, phase)
    if ui_status and job.status != ui_status:
        job.status = ui_status
        await session.commit()

    await publish_job_progress(job.id, {
        "job_id": job.id,
        "status": ui_status or job.status,
        **{k: v for k, v in event.items() if k != "event"},
    })


# ------------------------------------------------------------------ #
# Stage 1: Transcribe                                                 #
# ------------------------------------------------------------------ #

async def run_remote_transcribe(session, job: "Job", raw_wav_path: str) -> None:
    """
    Upload raw_wav to /v1/pipeline/transcribe, stream progress, persist results.
    Stores workdir_token in job.workdir_token for stage 2.
    """
    from app.ml.remote_client import call_streaming_multipart, RemoteCallError

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    params = {
        "job_id": job.id,
        "language_hint": job.language_hint,
        "filename": job.filename,
    }

    def _thread() -> None:
        received_any = False
        try:
            with open(raw_wav_path, "rb") as f:
                files = {"audio": ("raw.wav", f, "audio/wav")}
                data = {"payload": json.dumps(params)}
                for event in call_streaming_multipart("/v1/pipeline/transcribe", files, data):
                    received_any = True
                    asyncio.run_coroutine_threadsafe(queue.put(("event", event)), loop).result(timeout=60)
        except RemoteCallError as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", str(exc), received_any)), loop
            ).result(timeout=5)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", f"Streaming failed: {exc}", received_any)), loop
            ).result(timeout=5)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done",)), loop).result(timeout=5)

    loop.run_in_executor(None, _thread)

    result = await _consume_stream(queue, job, session)
    await _persist_transcribe_result(session, job, result, loop)


async def _persist_transcribe_result(session, job: "Job", result: dict, loop) -> None:
    from app.pipeline.subtitle_generator import generate_srt, generate_vtt
    from app.db_crud import upsert_subtitle

    # Defensive: if the stream ended without a result event, surface a clear error
    # (received_any_event=True prevents the caller from falling back to legacy).
    if not result.get("workdir_token") and not result.get("segments_src"):
        raise RemotePipelineError(
            "Stage 1 finished without a result event (stream cut off?)",
            received_any_event=True,
        )

    detected_lang = result.get("detected_language") or job.language_hint or "en"
    job.detected_language = detected_lang
    job.workdir_token = result.get("workdir_token")
    job.status = "assembling"
    await session.commit()

    await publish_job_progress(job.id, {
        "job_id": job.id, "status": "assembling", "phase": "assemble", "percent": 50,
    })

    # Persist chunk rows
    chunk_list = result.get("chunks", [])
    job.total_chunks = len(chunk_list)
    job.completed_chunks = sum(1 for c in chunk_list if c.get("status") in ("done", "skipped"))

    for c in chunk_list:
        session.add(Chunk(
            job_id=job.id,
            sequence=c["sequence"],
            filename=f"chunk_{c['sequence']:04d}.wav",
            path="",
            start_time=c["start"],
            end_time=c["end"],
            duration=c["end"] - c["start"],
            status=c.get("status", "done"),
            detected_language=c.get("language"),
            assigned_model=c.get("model"),
            transcript_json=c.get("transcript_json"),
        ))
    await session.commit()

    # Generate source-language subtitles
    segments_src = result.get("segments_src", [])
    if segments_src:
        from app.ml.base import TranscriptSegment, WordTimestamp

        def _to_seg(d: dict) -> TranscriptSegment:
            words = [WordTimestamp(**w) for w in d.get("words", [])]
            return TranscriptSegment(
                text=d["text"], start=d["start"], end=d["end"],
                words=words, language=d.get("language", detected_lang),
            )

        segs = [_to_seg(d) for d in segments_src]
        srt_path = str(storage.subtitle_path(job.id, "srt"))
        vtt_path = str(storage.subtitle_path(job.id, "vtt"))
        await loop.run_in_executor(None, generate_srt, segs, srt_path)
        await loop.run_in_executor(None, generate_vtt, segs, vtt_path)
        await upsert_subtitle(session, job.id, "srt", srt_path, detected_lang)
        await upsert_subtitle(session, job.id, "vtt", vtt_path, detected_lang)

    # Mark as done at the transcribe stage (translation may follow)
    job.status = "done"
    job.executor = "remote"
    await session.commit()

    await publish_job_progress(job.id, {
        "job_id": job.id, "status": "done", "phase": "done",
        "total_chunks": job.total_chunks, "completed_chunks": job.completed_chunks,
        "percent": 100,
        "srt_url": f"/api/subtitles/{job.id}/srt",
        "vtt_url": f"/api/subtitles/{job.id}/vtt",
    })


# ------------------------------------------------------------------ #
# Stage 2: Translate                                                  #
# ------------------------------------------------------------------ #

async def run_remote_translate(session, job: "Job") -> None:
    """
    Send workdir_token to /v1/pipeline/translate, stream progress, persist results.
    Falls back to stateless path if token is gone (expired).
    """
    from app.ml.remote_client import call_streaming_json, RemoteCallError

    if not job.workdir_token:
        raise RemotePipelineError("No workdir_token — cannot run remote translate", received_any_event=False)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    payload = {
        "workdir_token": job.workdir_token,
        "target_language": job.target_language,
        "translator_mode": job.translator_mode,
        "enable_refinement": job.enable_refinement,
        "glossary_text": job.glossary_json or "",
    }

    def _thread() -> None:
        received_any = False
        try:
            for event in call_streaming_json("/v1/pipeline/translate", payload):
                received_any = True
                asyncio.run_coroutine_threadsafe(queue.put(("event", event)), loop).result(timeout=60)
        except RemoteCallError as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", str(exc), received_any)), loop
            ).result(timeout=5)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", f"Streaming failed: {exc}", received_any)), loop
            ).result(timeout=5)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done",)), loop).result(timeout=5)

    loop.run_in_executor(None, _thread)

    result = await _consume_stream(queue, job, session)
    await _persist_translate_result(session, job, result, loop)


async def _persist_translate_result(session, job: "Job", result: dict, loop) -> None:
    from app.pipeline.subtitle_generator import generate_srt, generate_vtt
    from app.db_crud import upsert_subtitle

    segments_tl = result.get("segments_tl", [])
    if segments_tl and job.target_language:
        from app.ml.base import TranslatedSegment, WordTimestamp

        def _to_tl(d: dict) -> TranslatedSegment:
            words = [WordTimestamp(**w) for w in d.get("words", [])]
            return TranslatedSegment(
                text=d["text"], start=d["start"], end=d["end"],
                words=words,
                source_text=d.get("source_text", ""),
                source_language=d.get("source_language", job.detected_language or ""),
                target_language=d.get("target_language", job.target_language or ""),
            )

        tl_segs = [_to_tl(d) for d in segments_tl]
        tl_srt = str(storage.subtitle_path(job.id, "srt")).replace(".srt", f"_{job.target_language}.srt")
        tl_vtt = str(storage.subtitle_path(job.id, "vtt")).replace(".vtt", f"_{job.target_language}.vtt")
        await loop.run_in_executor(None, generate_srt, tl_segs, tl_srt)
        await loop.run_in_executor(None, generate_vtt, tl_segs, tl_vtt)
        await upsert_subtitle(session, job.id, "srt", tl_srt, job.target_language)
        await upsert_subtitle(session, job.id, "vtt", tl_vtt, job.target_language)
        job.translation_status = "done"

    job.status = "done"
    await session.commit()

    await publish_job_progress(job.id, {
        "job_id": job.id, "status": "done", "phase": "done", "percent": 100,
        "srt_url": f"/api/subtitles/{job.id}/srt",
        "vtt_url": f"/api/subtitles/{job.id}/vtt",
    })


# ------------------------------------------------------------------ #
# Legacy all-in-one (used when worker doesn't have two-stage support) #
# ------------------------------------------------------------------ #

_MIN_WORKER_API_VERSION = 2


async def run_remote_pipeline(session, job: "Job", raw_wav_path: str) -> None:
    """
    Try /v1/pipeline/transcribe + /v1/pipeline/translate (worker api_version >= 2).
    Falls back to /v1/pipeline/run if the worker is older.
    Raises RemotePipelineError so pipeline_runner can fall back to local.
    """
    from app.ml.remote_client import RemoteCallError
    from app.services.remote_health import get_cached as _get_health

    # Fast-fail with a clear message when the worker is reachable but too old.
    health = _get_health()
    if health.get("healthy") and int(health.get("api_version", 1)) < _MIN_WORKER_API_VERSION:
        raise RemotePipelineError(
            f"Worker api_version={health.get('api_version', 1)} < {_MIN_WORKER_API_VERSION} "
            f"— commit & push the latest code, re-run Colab Cell 1 (git pull) + Cell 3 (restart worker).",
            received_any_event=False,
        )

    # Stage 1: only this stage can trigger the legacy fallback.
    # If stage 1 produced any events the worker has the new endpoint — don't fall back.
    try:
        await run_remote_transcribe(session, job, raw_wav_path)
    except RemotePipelineError as exc:
        if exc.received_any_event:
            raise
        logger.info("Worker /v1/pipeline/transcribe unavailable — trying legacy /v1/pipeline/run")
        await _run_legacy_pipeline(session, job, raw_wav_path)
        return

    # Stage 2: never falls back to legacy — stage 1 already did the heavy work.
    if job.translate and job.target_language:
        await run_remote_translate(session, job)


async def _run_legacy_pipeline(session, job: "Job", raw_wav_path: str) -> None:
    """All-in-one legacy fallback using /v1/pipeline/run."""
    from app.ml.remote_client import call_streaming_multipart

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    params = {
        "job_id": job.id,
        "language_hint": job.language_hint,
        "target_language": job.target_language if job.translate else None,
        "translator_mode": job.translator_mode,
        "enable_refinement": job.enable_refinement,
        "glossary_text": job.glossary_json or "",
        "filename": job.filename,
    }

    def _thread() -> None:
        received_any = False
        try:
            with open(raw_wav_path, "rb") as f:
                files = {"audio": ("raw.wav", f, "audio/wav")}
                data = {"payload": json.dumps(params)}
                for event in call_streaming_multipart("/v1/pipeline/run", files, data):
                    received_any = True
                    asyncio.run_coroutine_threadsafe(queue.put(("event", event)), loop).result(timeout=60)
        except Exception as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put(("error", str(exc), received_any)), loop
            ).result(timeout=5)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(("done",)), loop).result(timeout=5)

    loop.run_in_executor(None, _thread)

    while True:
        msg = await queue.get()
        kind = msg[0]
        if kind == "done":
            break
        if kind == "error":
            _, errmsg, had_events = msg
            raise RemotePipelineError(errmsg, received_any_event=had_events)

        event = msg[1]
        ev_type = event.get("event", "progress")

        if ev_type == "error":
            raise RemotePipelineError(event.get("message", "Worker error"), received_any_event=True)

        if ev_type == "result":
            await _persist_legacy_result(session, job, event, loop)
            break

        await _forward_progress(event, job, session)


async def _persist_legacy_result(session, job: "Job", result: dict, loop) -> None:
    from app.pipeline.reassembler import reassemble
    from app.pipeline.subtitle_generator import generate_srt, generate_vtt
    from app.db_crud import upsert_subtitle

    detected_lang = result.get("detected_language") or job.language_hint or "en"
    job.detected_language = detected_lang
    job.status = "assembling"
    await session.commit()

    await publish_job_progress(job.id, {"job_id": job.id, "status": "assembling",
                                         "phase": "assembling", "percent": 90})

    chunk_list = result.get("chunks", [])
    job.total_chunks = len(chunk_list)
    job.completed_chunks = sum(1 for c in chunk_list if c.get("status") in ("done", "skipped"))

    for c in chunk_list:
        session.add(Chunk(
            job_id=job.id, sequence=c["sequence"],
            filename=f"chunk_{c['sequence']:04d}.wav", path="",
            start_time=c["start"], end_time=c["end"],
            duration=c["end"] - c["start"],
            status=c.get("status", "done"),
            detected_language=c.get("language"),
            assigned_model=c.get("model"),
            transcript_json=c.get("transcript_json"),
        ))
    await session.commit()

    segments_src = result.get("segments_src", [])
    if segments_src:
        from app.ml.base import TranscriptSegment, WordTimestamp

        def _to_seg(d):
            words = [WordTimestamp(**w) for w in d.get("words", [])]
            return TranscriptSegment(text=d["text"], start=d["start"], end=d["end"],
                                     words=words, language=d.get("language", detected_lang))

        segs = [_to_seg(d) for d in segments_src]
        srt_path = str(storage.subtitle_path(job.id, "srt"))
        vtt_path = str(storage.subtitle_path(job.id, "vtt"))
        await loop.run_in_executor(None, generate_srt, segs, srt_path)
        await loop.run_in_executor(None, generate_vtt, segs, vtt_path)
        await upsert_subtitle(session, job.id, "srt", srt_path, detected_lang)
        await upsert_subtitle(session, job.id, "vtt", vtt_path, detected_lang)

    segments_tl = result.get("segments_tl", [])
    if segments_tl and job.target_language:
        from app.ml.base import TranslatedSegment, WordTimestamp

        def _to_tl(d):
            words = [WordTimestamp(**w) for w in d.get("words", [])]
            return TranslatedSegment(text=d["text"], start=d["start"], end=d["end"],
                                     words=words, source_text=d.get("source_text", ""),
                                     source_language=d.get("source_language", detected_lang),
                                     target_language=d.get("target_language", job.target_language))

        tl_segs = [_to_tl(d) for d in segments_tl]
        tl_srt = str(storage.subtitle_path(job.id, "srt")).replace(".srt", f"_{job.target_language}.srt")
        tl_vtt = str(storage.subtitle_path(job.id, "vtt")).replace(".vtt", f"_{job.target_language}.vtt")
        await loop.run_in_executor(None, generate_srt, tl_segs, tl_srt)
        await loop.run_in_executor(None, generate_vtt, tl_segs, tl_vtt)
        await upsert_subtitle(session, job.id, "srt", tl_srt, job.target_language)
        await upsert_subtitle(session, job.id, "vtt", tl_vtt, job.target_language)
        job.translation_status = "done" if tl_segs else None

    job.status = "done"
    job.executor = "remote"
    await session.commit()

    await publish_job_progress(job.id, {
        "job_id": job.id, "status": "done", "phase": "done",
        "total_chunks": job.total_chunks, "completed_chunks": job.completed_chunks,
        "percent": 100,
        "srt_url": f"/api/subtitles/{job.id}/srt",
        "vtt_url": f"/api/subtitles/{job.id}/vtt",
    })
