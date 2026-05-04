"""Shravana Remote GPU Worker.

A FastAPI server that exposes the same ML model logic from backend/app/ml
over HTTP. Run this on any machine with a GPU (Colab, RunPod, dedicated server)
and point the local backend at it via REMOTE_GPU_URL.

Start:
    cd /path/to/Shravana
    WORKER_TOKEN=mysecret uvicorn worker.server:app --host 0.0.0.0 --port 8001

Environment variables:
    WORKER_TOKEN        Bearer token the local backend must send (required in prod)
    WORKER_MODELS_DIR   Where to store/load model weights (default: storage/models)
    WORKER_DEVICE       torch device override (default: auto-detect cuda/cpu)
"""
from __future__ import annotations
import base64
import dataclasses
import json
import logging
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, List, Optional

import asyncio
import threading

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile, Depends
from fastapi.responses import Response, StreamingResponse

# ── Make backend.app importable when running from repo root ──────────────────
import sys
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── Minimal settings shim (worker doesn't use pydantic-settings) ─────────────
WORKER_TOKEN: str = os.getenv("WORKER_TOKEN", "")
MODELS_DIR: Path = Path(os.getenv("WORKER_MODELS_DIR", str(_REPO_ROOT / "storage" / "models")))
DEVICE: str = os.getenv("WORKER_DEVICE", "")
WORKER_API_VERSION: int = int(os.getenv("WORKER_API_VERSION", "2"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
logger = logging.getLogger("worker")

app = FastAPI(title="Shravana GPU Worker", version="0.1.0")

# ── Auth ─────────────────────────────────────────────────────────────────────

def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not WORKER_TOKEN:
        return  # no token configured → open (dev mode)
    if authorization != f"Bearer {WORKER_TOKEN}":
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


# ── Model registry (local, no remote loop) ───────────────────────────────────

_instances: dict = {}
_lock = __import__("threading").Lock()


def _get_model(name: str):
    with _lock:
        if name in _instances:
            return _instances[name]

    # Patch settings so backend/app/ml modules find their model files
    _patch_settings()
    from app.ml.registry import MODEL_CONFIGS, _is_downloaded
    from app.ml.base import ModelNotReady

    if name not in MODEL_CONFIGS:
        raise HTTPException(status_code=404, detail=f"Unknown model: {name}")
    if not _is_downloaded(name):
        raise HTTPException(status_code=503, detail={"model_loading": name})

    cfg = MODEL_CONFIGS[name]
    cls_name = cfg["class"]
    kwargs = cfg["kwargs"]

    if cls_name == "QwenASRModel":
        from app.ml.qwen_asr import QwenASRModel
        instance = QwenASRModel(**kwargs)
    elif cls_name == "WhisperTurboModel":
        from app.ml.whisper_turbo import WhisperTurboModel
        instance = WhisperTurboModel(**kwargs)
    elif cls_name == "SeamlessM4TTranslator":
        from app.ml.seamless_translator import SeamlessM4TTranslator
        instance = SeamlessM4TTranslator(**kwargs)
    elif cls_name == "QwenVLTranslator":
        from app.ml.qwen_vl_translator import QwenVLTranslator
        instance = QwenVLTranslator(**kwargs)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported model class: {cls_name}")

    instance.load()
    with _lock:
        _instances[name] = instance
    return instance


def _patch_settings():
    """Point the backend settings at this worker's model storage directory."""
    try:
        from app.config import settings
        settings.MODELS_DIR = str(MODELS_DIR)
        if DEVICE:
            os.environ["WORKER_DEVICE"] = DEVICE
    except Exception:
        pass


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _dict_to_seg(d: dict):
    from app.ml.base import TranscriptSegment, WordTimestamp
    words = [WordTimestamp(**w) for w in d.get("words", [])]
    return TranscriptSegment(
        text=d["text"], start=d["start"], end=d["end"],
        words=words, language=d.get("language", "en"),
    )


def _dict_to_tl(d: dict):
    from app.ml.base import TranslatedSegment, WordTimestamp
    words = [WordTimestamp(**w) for w in d.get("words", [])]
    return TranslatedSegment(
        text=d["text"], start=d["start"], end=d["end"],
        words=words,
        source_text=d.get("source_text", ""),
        source_language=d.get("source_language", ""),
        target_language=d.get("target_language", ""),
    )


def _dict_to_ctx(d: Optional[dict]):
    if not d:
        return None
    from app.ml.base import ContextBundle
    return ContextBundle(
        domain=d.get("domain", "casual"),
        format=d.get("format", "monologue"),
        source_language=d.get("source_language", "en"),
        target_language=d.get("target_language", "en"),
        named_entities=d.get("named_entities", []),
        idioms_detected=d.get("idioms_detected", []),
        scene_description=d.get("scene_description"),
        notes=d.get("notes"),
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/v1/health")
async def health(_: None = Depends(_check_auth)) -> dict:
    gpu_info: dict = {}
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total = props.total_memory / 1024 ** 3
            used = (props.total_memory - torch.cuda.mem_get_info(idx)[0]) / 1024 ** 3
            gpu_info = {
                "name": props.name,
                "total_gb": round(total, 2),
                "used_gb": round(used, 2),
                "free_gb": round(total - used, 2),
            }
    except Exception:
        pass

    return {
        "status": "ok",
        "api_version": WORKER_API_VERSION,
        "gpu": gpu_info,
        "models_loaded": list(_instances.keys()),
    }


# ── Models list ───────────────────────────────────────────────────────────────

@app.get("/v1/models")
async def list_models(_: None = Depends(_check_auth)) -> dict:
    _patch_settings()
    from app.ml.registry import MODEL_CONFIGS, _is_downloaded
    models = []
    for name, cfg in MODEL_CONFIGS.items():
        models.append({
            "name": name,
            "repo_id": cfg["repo_id"],
            "downloaded": _is_downloaded(name),
            "loaded": name in _instances,
        })
    return {"models": models}


@app.post("/v1/models/{name}/load")
async def load_model(name: str, _: None = Depends(_check_auth)) -> dict:
    _get_model(name)
    return {"status": "loaded", "model": name}


# ── ASR endpoints ─────────────────────────────────────────────────────────────

@app.post("/v1/asr/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(default=None),
    _: None = Depends(_check_auth),
) -> dict:
    m = _get_model(model)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        if language and hasattr(m, "transcribe"):
            import inspect
            sig = inspect.signature(m.transcribe)
            if "language" in sig.parameters:
                segs = m.transcribe(tmp_path, language=language)
            else:
                segs = m.transcribe(tmp_path)
        else:
            segs = m.transcribe(tmp_path)
        return {"segments": [dataclasses.asdict(s) for s in segs]}
    finally:
        os.unlink(tmp_path)


@app.post("/v1/asr/detect_language")
async def detect_language(
    audio: UploadFile = File(...),
    model: str = Form(...),
    _: None = Depends(_check_auth),
) -> dict:
    m = _get_model(model)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        lang, conf = m.detect_language(tmp_path)
        return {"language": lang, "confidence": conf}
    finally:
        os.unlink(tmp_path)


# ── Translation endpoints ─────────────────────────────────────────────────────

@app.post("/v1/translate")
async def translate(
    request: "Request",
    _: None = Depends(_check_auth),
) -> dict:
    form = await request.form()
    payload_raw = form.get("payload")
    if not payload_raw:
        raise HTTPException(status_code=422, detail="Missing 'payload' form field")
    body = json.loads(str(payload_raw))

    model_name = body["model"]
    segments = [_dict_to_seg(s) for s in body.get("segments", [])]
    source_lang = body["source_lang"]
    target_lang = body["target_lang"]
    history = [_dict_to_tl(h) for h in body.get("history", [])]
    context = _dict_to_ctx(body.get("context"))
    glossary_text = body.get("glossary_text", "")

    m = _get_model(model_name)

    _MAX_SEGS_PER_BATCH = 4  # keep each model call under ~30s to avoid Cloudflare 524

    audio_tmp: Optional[str] = None
    frame_tmps: list[str] = []
    try:
        audio_field = form.get("audio")
        if audio_field and hasattr(audio_field, "read"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(await audio_field.read())
                audio_tmp = tmp.name

        # Collect frame_0, frame_1, … frame_n uploads (for VLM mode)
        i = 0
        while True:
            frame_field = form.get(f"frame_{i}")
            if not frame_field or not hasattr(frame_field, "read"):
                break
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp.write(await frame_field.read())
                frame_tmps.append(tmp.name)
            i += 1

        frames = [Path(p) for p in frame_tmps] if frame_tmps else None

        # Batch segments to keep each translate() call ≤_MAX_SEGS_PER_BATCH so
        # SeamlessM4T never runs for >100 s in a single call (Cloudflare 524 threshold).
        all_results = []
        rolling_history = list(history)
        for batch_start in range(0, max(1, len(segments)), _MAX_SEGS_PER_BATCH):
            batch = segments[batch_start:batch_start + _MAX_SEGS_PER_BATCH]
            batch_segs = m.translate(
                segments=batch,
                source_lang=source_lang,
                target_lang=target_lang,
                history=rolling_history,
                audio_path=audio_tmp,
                frames=frames,
                context=context,
                glossary_text=glossary_text,
            )
            all_results.extend(batch_segs)
            rolling_history = rolling_history + batch_segs

        return {"segments": [dataclasses.asdict(s) for s in all_results]}
    finally:
        if audio_tmp:
            try:
                os.unlink(audio_tmp)
            except Exception:
                pass
        for fp in frame_tmps:
            try:
                os.unlink(fp)
            except Exception:
                pass


@app.post("/v1/translate/refine")
async def refine(
    body: dict,
    _: None = Depends(_check_auth),
) -> dict:
    model_name = body["model"]
    context = _dict_to_ctx(body.get("context"))
    glossary_text = body.get("glossary_text", "")

    m = _get_model(model_name)
    result = m.refine(
        first_pass=body.get("first_pass", ""),
        source_text=body.get("source_text", ""),
        prev_chunk=body.get("prev_text", ""),
        next_chunk=body.get("next_text", ""),
        source_lang=body.get("source_lang", "en"),
        target_lang=body.get("target_lang", "en"),
        start=float(body.get("start", 0.0)),
        end=float(body.get("end", 0.0)),
        context=context,
        glossary_text=glossary_text,
    )
    return {"text": result}


@app.post("/v1/generate")
async def generate(
    body: dict,
    _: None = Depends(_check_auth),
) -> dict:
    """Raw text generation (used by context builder when remote)."""
    model_name = body["model"]
    m = _get_model(model_name)
    result = m._call_model(
        source_text=body.get("source_text", ""),
        system_prompt=body.get("system_prompt", ""),
        history_text=body.get("history_text", ""),
        frames=None,
    )
    return {"text": result}


# ── Denoiser endpoint ─────────────────────────────────────────────────────────

@app.post("/v1/denoise")
async def denoise_audio(
    audio: UploadFile = File(...),
    job_id: str = Form(default=""),
    _: None = Depends(_check_auth),
) -> Response:
    """
    Run Demucs + noisereduce + Silero VAD on the uploaded audio.
    Returns the clean WAV as the response body.
    Speech intervals are JSON-encoded (base64) in the X-Speech-Intervals header.
    """
    with tempfile.TemporaryDirectory() as work_dir:
        raw_path = os.path.join(work_dir, "raw.wav")
        with open(raw_path, "wb") as f:
            f.write(await audio.read())

        from app.pipeline.denoiser import (
            _run_demucs, _run_noisereduce, _run_vad,
        )
        # Force local execution (never recurse back to remote)
        vocals_wav = _run_demucs(raw_path, work_dir)
        clean_wav = _run_noisereduce(vocals_wav, os.path.join(work_dir, "clean.wav"))
        speech_intervals = _run_vad(clean_wav)

        with open(clean_wav, "rb") as f:
            clean_bytes = f.read()

    intervals_b64 = base64.b64encode(json.dumps(speech_intervals).encode()).decode()
    return Response(
        content=clean_bytes,
        media_type="audio/wav",
        headers={"X-Speech-Intervals": intervals_b64},
    )


# ── Full pipeline endpoint (streaming NDJSON) ─────────────────────────────────

@app.post("/v1/pipeline/run")
async def pipeline_run(
    request: Request,
    _: None = Depends(_check_auth),
) -> StreamingResponse:
    """
    Run phases 2-7 (denoise → chunk → transcribe → translate → assemble) on the
    worker and stream NDJSON progress events back.

    Multipart fields:
      audio    (required) — raw WAV file from local Phase 1 extraction
      video    (optional) — original video file for VLM keyframe extraction
      payload  (required) — JSON string with job parameters:
                  {job_id, language_hint, target_language, translator_mode,
                   enable_refinement, glossary_text, filename}

    Each NDJSON line is a JSON object. The final line has event="result".
    """
    form = await request.form()

    audio_field = form.get("audio")
    if not audio_field or not hasattr(audio_field, "read"):
        raise HTTPException(status_code=422, detail="Missing 'audio' form field")
    payload_raw = form.get("payload")
    if not payload_raw:
        raise HTTPException(status_code=422, detail="Missing 'payload' form field")
    params = json.loads(str(payload_raw))

    # Save uploads to a persistent tempdir (pipeline runs in a thread; can't use async ctx)
    work_dir = tempfile.mkdtemp(prefix="shravana_pipeline_")
    raw_wav = os.path.join(work_dir, "raw.wav")
    with open(raw_wav, "wb") as f:
        f.write(await audio_field.read())

    video_path: Optional[str] = None
    video_field = form.get("video")
    if video_field and hasattr(video_field, "read"):
        ext = Path(getattr(video_field, "filename", "video.mp4")).suffix or ".mp4"
        video_path = os.path.join(work_dir, f"video{ext}")
        with open(video_path, "wb") as f:
            f.write(await video_field.read())
        params["video_path"] = video_path

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _emit(event: dict) -> None:
        """Called from the pipeline thread; marshals event onto the async queue."""
        fut = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
        fut.result(timeout=60)

    def _run() -> None:
        try:
            _patch_settings()
            from worker.pipeline_handler import run_pipeline
            run_pipeline(raw_wav, params, _emit)
        except Exception as exc:
            logger.error(f"Pipeline run failed: {exc}", exc_info=True)
            asyncio.run_coroutine_threadsafe(
                queue.put({"event": "error", "message": str(exc)}), loop
            ).result(timeout=5)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=5)
            import shutil
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass

    threading.Thread(target=_run, daemon=True).start()

    async def _ndjson_generator():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield (json.dumps(event) + "\n").encode()

    return StreamingResponse(_ndjson_generator(), media_type="application/x-ndjson")


# ── Two-stage pipeline: transcribe endpoint ───────────────────────────────────

@app.post("/v1/pipeline/transcribe")
async def pipeline_transcribe(
    request: Request,
    _: None = Depends(_check_auth),
) -> StreamingResponse:
    """
    Stage 1: denoise → chunk → transcribe.
    Returns NDJSON stream; final event has event='result' and workdir_token.
    The workdir is kept alive for ~30 min so /v1/pipeline/translate can use it.
    """
    form = await request.form()
    audio_field = form.get("audio")
    if not audio_field or not hasattr(audio_field, "read"):
        raise HTTPException(status_code=422, detail="Missing 'audio' form field")
    payload_raw = form.get("payload")
    if not payload_raw:
        raise HTTPException(status_code=422, detail="Missing 'payload' form field")
    params = json.loads(str(payload_raw))

    # Stage 1 keeps its own workdir (managed by pipeline_handler._store)
    staging_dir = tempfile.mkdtemp(prefix="shravana_stage1_")
    raw_wav = os.path.join(staging_dir, "raw.wav")
    with open(raw_wav, "wb") as f:
        f.write(await audio_field.read())

    video_path: Optional[str] = None
    video_field = form.get("video")
    if video_field and hasattr(video_field, "read"):
        ext = Path(getattr(video_field, "filename", "video.mp4")).suffix or ".mp4"
        video_path = os.path.join(staging_dir, f"video{ext}")
        with open(video_path, "wb") as f:
            f.write(await video_field.read())
        params["video_path"] = video_path

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _emit(event: dict) -> None:
        fut = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
        fut.result(timeout=60)

    def _run() -> None:
        try:
            _patch_settings()
            from worker.pipeline_handler import run_transcribe
            run_transcribe(raw_wav, params, _emit)
        except Exception as exc:
            logger.error(f"Transcribe stage failed: {exc}", exc_info=True)
            asyncio.run_coroutine_threadsafe(
                queue.put({"event": "error", "message": str(exc)}), loop
            ).result(timeout=5)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=5)
            import shutil
            shutil.rmtree(staging_dir, ignore_errors=True)

    threading.Thread(target=_run, daemon=True).start()

    async def _gen():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield (json.dumps(event) + "\n").encode()

    return StreamingResponse(_gen(), media_type="application/x-ndjson")


# ── Two-stage pipeline: translate endpoint ────────────────────────────────────

@app.post("/v1/pipeline/translate")
async def pipeline_translate(
    body: dict,
    _: None = Depends(_check_auth),
) -> StreamingResponse:
    """
    Stage 2: translate segments stored by /v1/pipeline/transcribe.
    Body JSON: {workdir_token, target_language, translator_mode, enable_refinement, glossary_text}.
    Returns NDJSON stream; final event has event='result' with segments_tl.
    """
    token = body.get("workdir_token")
    if not token:
        raise HTTPException(status_code=422, detail="Missing workdir_token")

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _emit(event: dict) -> None:
        fut = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
        fut.result(timeout=60)

    def _run() -> None:
        try:
            _patch_settings()
            from worker.pipeline_handler import run_translate
            run_translate(token, body, _emit)
        except KeyError as exc:
            asyncio.run_coroutine_threadsafe(
                queue.put({"event": "error", "message": f"Workdir token expired: {exc}"}), loop
            ).result(timeout=5)
        except Exception as exc:
            logger.error(f"Translate stage failed: {exc}", exc_info=True)
            asyncio.run_coroutine_threadsafe(
                queue.put({"event": "error", "message": str(exc)}), loop
            ).result(timeout=5)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=5)

    threading.Thread(target=_run, daemon=True).start()

    async def _gen():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield (json.dumps(event) + "\n").encode()

    return StreamingResponse(_gen(), media_type="application/x-ndjson")


# ── Delete workdir (cleanup before TTL) ──────────────────────────────────────

@app.delete("/v1/pipeline/workdir/{token}")
async def delete_workdir(
    token: str,
    _: None = Depends(_check_auth),
) -> dict:
    from worker.pipeline_handler import _evict
    _evict(token)
    return {"status": "evicted"}
