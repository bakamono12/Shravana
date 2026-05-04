"""Full pipeline handler for the remote GPU worker.

Exposes three entry points:
  run_pipeline()    — legacy all-in-one (phases 2-7), used by /v1/pipeline/run
  run_transcribe()  — stage 1: denoise → chunk → transcribe, keeps workdir alive
  run_translate()   — stage 2: translate using stored workdir
"""
from __future__ import annotations
import dataclasses
import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

import torch

logger = logging.getLogger("worker.pipeline")

# ----------------------------------------------------------------------- #
# Workdir store (token → {path, state, created_at})                       #
# TTL 30 min — cleaned by background thread.                              #
# ----------------------------------------------------------------------- #

_store: dict[str, dict] = {}
_store_lock = threading.Lock()
_WORKDIR_TTL_S = 1800  # 30 minutes


def _cleanup_expired() -> None:
    now = time.monotonic()
    with _store_lock:
        expired = [t for t, v in _store.items() if now - v["created_at"] > _WORKDIR_TTL_S]
    for token in expired:
        _evict(token)


def _evict(token: str) -> None:
    with _store_lock:
        entry = _store.pop(token, None)
    if entry:
        try:
            shutil.rmtree(entry["path"], ignore_errors=True)
        except Exception:
            pass


def _start_cleanup_thread() -> None:
    def _loop():
        while True:
            time.sleep(300)
            _cleanup_expired()
    t = threading.Thread(target=_loop, daemon=True)
    t.start()


_start_cleanup_thread()


# ----------------------------------------------------------------------- #
# Serialisation helpers                                                    #
# ----------------------------------------------------------------------- #

def _seg_to_dict(s) -> dict:
    return dataclasses.asdict(s)


# ----------------------------------------------------------------------- #
# Stage 1: Denoise → Chunk → Transcribe                                   #
# ----------------------------------------------------------------------- #

def run_transcribe(raw_wav_path: str, params: dict, emit: Callable[[dict], None]) -> str:
    """
    Run phases 2-5 (denoise → chunk → transcribe) and stream NDJSON events.
    Final event has event='result' and includes workdir_token.

    The work_dir is NOT cleaned up — caller must call run_translate() later
    which will evict it, or the TTL-based cleanup will handle it.
    """
    language_hint = params.get("language_hint")
    filename = params.get("filename", "")
    job_id = params.get("job_id", "unknown")
    video_path = params.get("video_path")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    work_dir = tempfile.mkdtemp(prefix=f"shravana_{job_id}_")

    try:
        # Phase 2: Denoise
        emit({"event": "progress", "phase": "denoising", "subphase": "demucs",
              "percent": 0, "stage_index": 1, "stage_total": 3})

        from app.pipeline.denoiser import _run_demucs, _run_noisereduce, _run_vad, is_chunk_silent

        def _demucs_progress(pct: int) -> None:
            emit({"event": "progress", "phase": "denoising", "subphase": "demucs",
                  "percent": pct, "stage_index": 1, "stage_total": 3})

        vocals_wav = _run_demucs(raw_wav_path, work_dir, on_progress=_demucs_progress, device=device)

        emit({"event": "progress", "phase": "denoising", "subphase": "noisereduce",
              "percent": 70, "stage_index": 1, "stage_total": 3})
        clean_wav = _run_noisereduce(vocals_wav, os.path.join(work_dir, "clean.wav"))

        emit({"event": "progress", "phase": "denoising", "subphase": "vad",
              "percent": 85, "stage_index": 1, "stage_total": 3})
        speech_intervals = _run_vad(clean_wav)

        # Phase 3: Chunk
        from app.pipeline.chunker import chunk_audio
        chunks_dir = os.path.join(work_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        manifest = chunk_audio(clean_wav, chunks_dir)

        pending = [
            e for e in manifest
            if not is_chunk_silent(e.start_time, e.end_time, speech_intervals)
        ]
        total_chunks = len(manifest)
        emit({"event": "progress", "phase": "chunking", "total_chunks": total_chunks,
              "pending_chunks": len(pending), "percent": 100,
              "stage_index": 2, "stage_total": 3})

        # Phase 4-5: Language detect + transcribe
        from app.pipeline.language_detector import detect
        from app.pipeline.router import route
        from app.pipeline.transcriber import transcribe_chunk
        from app.pipeline.reassembler import serialize_segments

        first_lang = language_hint or "en"
        first_conf = 0.0
        if pending:
            first_lang, first_conf = detect(pending[0].path, language_hint)

        chunk_results = []
        for i, entry in enumerate(manifest):
            silent = is_chunk_silent(entry.start_time, entry.end_time, speech_intervals)
            if silent:
                chunk_results.append({
                    "sequence": entry.sequence,
                    "start": entry.start_time,
                    "end": entry.end_time,
                    "duration": entry.duration,
                    "status": "skipped",
                    "language": first_lang,
                    "model": None,
                    "transcript_json": None,
                })
                continue

            model_name = route(first_lang, first_conf)
            try:
                segs = transcribe_chunk(entry.path, model_name, first_lang)
                transcript_json = serialize_segments(segs)
                status = "done"
            except Exception as exc:
                logger.error(f"Chunk {entry.sequence} transcription failed: {exc}")
                transcript_json = None
                status = "failed"

            chunk_results.append({
                "sequence": entry.sequence,
                "start": entry.start_time,
                "end": entry.end_time,
                "duration": entry.duration,
                "status": status,
                "language": first_lang,
                "model": model_name,
                "transcript_json": transcript_json,
            })

            done_count = sum(1 for c in chunk_results if c["status"] in ("done", "skipped", "failed"))
            emit({
                "event": "progress", "phase": "processing",
                "chunk_done": done_count, "total_chunks": total_chunks,
                "detected_language": first_lang, "model": model_name,
                "percent": int(done_count / total_chunks * 100),
                "stage_index": 3, "stage_total": 3,
            })

        # Reassemble transcript
        from app.pipeline.reassembler import reassemble
        stt_chunk_data = [
            {"sequence": c["sequence"], "start_time": c["start"],
             "transcript_json": c["transcript_json"]}
            for c in chunk_results if c["transcript_json"]
        ]
        source_segments = reassemble(stt_chunk_data)

        # Register workdir so stage 2 can use it
        token = str(uuid.uuid4())
        with _store_lock:
            _store[token] = {
                "path": work_dir,
                "source_segments": source_segments,
                "chunk_results": chunk_results,
                "first_lang": first_lang,
                "filename": filename,
                "video_path": video_path,
                "created_at": time.monotonic(),
            }

        result = {
            "event": "result",
            "workdir_token": token,
            "detected_language": first_lang,
            "chunks": chunk_results,
            "segments_src": [_seg_to_dict(s) for s in source_segments],
        }
        emit(result)
        return token

    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


# ----------------------------------------------------------------------- #
# Stage 2: Translate                                                       #
# ----------------------------------------------------------------------- #

def run_translate(token: str, params: dict, emit: Callable[[dict], None]) -> None:
    """
    Run phase 5.5 (translate) using the workdir stored by run_transcribe().
    Evicts the workdir on completion.

    If token is not found (expired), raises KeyError.
    """
    with _store_lock:
        entry = _store.get(token)
    if not entry:
        raise KeyError(f"Workdir token not found or expired: {token}")

    target_language = params.get("target_language", "en")
    translator_mode = params.get("translator_mode")
    enable_refinement = params.get("enable_refinement", True)
    glossary_text = params.get("glossary_text", "")

    source_segments = entry["source_segments"]
    first_lang = entry["first_lang"]
    filename = entry["filename"]
    video_path = entry["video_path"]
    work_dir = entry["path"]

    try:
        from app.pipeline.translator import select_translator_mode
        from app.pipeline.translation_orchestrator import run_translation_compute

        mode = translator_mode or select_translator_mode(filename, None)

        def _on_tl_progress(done: int, total: int) -> None:
            emit({
                "event": "progress", "phase": "translating",
                "unit_done": done, "total_units": total,
                "percent": int(done / total * 100),
                "stage_index": 1, "stage_total": 1,
            })

        emit({"event": "progress", "phase": "translating", "unit_done": 0,
              "total_units": 0, "percent": 0, "stage_index": 1, "stage_total": 1})

        translated_segs, _, _ = run_translation_compute(
            source_segments,
            source_lang=first_lang,
            target_lang=target_language,
            translator_mode=mode,
            enable_refinement=enable_refinement,
            user_glossary_text=glossary_text,
            video_path=video_path,
            job_dir=work_dir,
            on_progress=_on_tl_progress,
        )

        emit({"event": "progress", "phase": "assembling", "percent": 100})

        result = {
            "event": "result",
            "segments_tl": [_seg_to_dict(s) for s in translated_segs],
        }
        emit(result)

    finally:
        _evict(token)


# ----------------------------------------------------------------------- #
# Legacy all-in-one (for /v1/pipeline/run backward compat)                #
# ----------------------------------------------------------------------- #

def run_pipeline(raw_wav_path: str, params: dict, emit: Callable[[dict], None]) -> None:
    """Run phases 2-7 in one call. Kept for backward compatibility."""
    language_hint = params.get("language_hint")
    target_language = params.get("target_language")
    translator_mode = params.get("translator_mode")
    enable_refinement = params.get("enable_refinement", True)
    glossary_text = params.get("glossary_text", "")
    filename = params.get("filename", "")
    job_id = params.get("job_id", "unknown")
    video_path = params.get("video_path")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tempfile.TemporaryDirectory() as work_dir:
        emit({"event": "progress", "phase": "denoising", "subphase": "demucs", "percent": 0})

        from app.pipeline.denoiser import _run_demucs, _run_noisereduce, _run_vad, is_chunk_silent

        def _demucs_progress(pct: int) -> None:
            emit({"event": "progress", "phase": "denoising", "subphase": "demucs", "percent": pct})

        vocals_wav = _run_demucs(raw_wav_path, work_dir, on_progress=_demucs_progress, device=device)

        emit({"event": "progress", "phase": "denoising", "subphase": "noisereduce"})
        clean_wav = _run_noisereduce(vocals_wav, os.path.join(work_dir, "clean.wav"))

        emit({"event": "progress", "phase": "denoising", "subphase": "vad"})
        speech_intervals = _run_vad(clean_wav)

        from app.pipeline.chunker import chunk_audio
        chunks_dir = os.path.join(work_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        manifest = chunk_audio(clean_wav, chunks_dir)

        pending = [e for e in manifest if not is_chunk_silent(e.start_time, e.end_time, speech_intervals)]
        total_chunks = len(manifest)
        emit({"event": "progress", "phase": "chunking", "total_chunks": total_chunks,
              "pending_chunks": len(pending)})

        from app.pipeline.language_detector import detect
        from app.pipeline.router import route
        from app.pipeline.transcriber import transcribe_chunk
        from app.pipeline.reassembler import serialize_segments

        first_lang = language_hint or "en"
        first_conf = 0.0
        if pending:
            first_lang, first_conf = detect(pending[0].path, language_hint)

        chunk_results = []
        for i, entry in enumerate(manifest):
            silent = is_chunk_silent(entry.start_time, entry.end_time, speech_intervals)
            if silent:
                chunk_results.append({"sequence": entry.sequence, "start": entry.start_time,
                                       "end": entry.end_time, "duration": entry.duration,
                                       "status": "skipped", "language": first_lang,
                                       "model": None, "transcript_json": None})
                continue

            model_name = route(first_lang, first_conf)
            try:
                segs = transcribe_chunk(entry.path, model_name, first_lang)
                transcript_json = serialize_segments(segs)
                status = "done"
            except Exception as exc:
                logger.error(f"Chunk {entry.sequence} transcription failed: {exc}")
                transcript_json = None
                status = "failed"

            chunk_results.append({"sequence": entry.sequence, "start": entry.start_time,
                                   "end": entry.end_time, "duration": entry.duration,
                                   "status": status, "language": first_lang,
                                   "model": model_name, "transcript_json": transcript_json})

            done_count = sum(1 for c in chunk_results if c["status"] in ("done", "skipped", "failed"))
            emit({"event": "progress", "phase": "processing", "chunk_done": done_count,
                  "total_chunks": total_chunks, "detected_language": first_lang,
                  "model": model_name, "percent": int(done_count / total_chunks * 100)})

        from app.pipeline.reassembler import reassemble
        stt_chunk_data = [{"sequence": c["sequence"], "start_time": c["start"],
                            "transcript_json": c["transcript_json"]}
                           for c in chunk_results if c["transcript_json"]]
        source_segments = reassemble(stt_chunk_data)

        translated_segs = []
        if target_language and source_segments:
            from app.pipeline.translator import select_translator_mode
            from app.pipeline.translation_orchestrator import run_translation_compute

            mode = translator_mode or select_translator_mode(filename, None)

            def _on_tl_progress(done: int, total: int) -> None:
                emit({"event": "progress", "phase": "translating", "unit_done": done,
                      "total_units": total, "mode": mode, "percent": int(done / total * 100)})

            emit({"event": "progress", "phase": "translating", "unit_done": 0,
                  "total_units": 0, "percent": 0})

            try:
                translated_segs, _, _ = run_translation_compute(
                    source_segments, source_lang=first_lang, target_lang=target_language,
                    translator_mode=mode, enable_refinement=enable_refinement,
                    user_glossary_text=glossary_text, video_path=video_path,
                    job_dir=work_dir, on_progress=_on_tl_progress,
                )
            except Exception as exc:
                logger.error(f"Translation failed: {exc}")

        emit({"event": "progress", "phase": "assembling"})
        emit({
            "event": "result",
            "detected_language": first_lang,
            "chunks": chunk_results,
            "segments_src": [_seg_to_dict(s) for s in source_segments],
            "segments_tl": [_seg_to_dict(s) for s in translated_segs],
        })
