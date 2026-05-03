# Shravana — Agent Guide

Shravana converts video/audio files into subtitles via a multi-phase ML pipeline.
Stack: **FastAPI + SQLAlchemy async (SQLite) + Alembic** backend · **Vite + React + Tailwind** frontend.

---

## Architecture Overview

```
Upload → Job row created → pipeline_runner.run() (background task)
  Phase 1  audio_extractor   – ffmpeg extract/normalize → raw.wav
  Phase 2  denoiser          – Demucs + noisereduce → clean.wav + speech intervals
  Phase 3  chunker           – 60 s chunks (5 s overlap) → Chunk rows
  Phase 4  language_detector – Qwen LID → Whisper LID fallback → user hint
  Phase 5  router + transcriber – per-chunk STT (model chosen by language)
  Phase 5.5 (optional) translator – semantic chunking → context → translate → refine
  Phase 6-7 reassembler + subtitle_generator → .srt/.vtt files → Subtitle rows
```

Progress events flow through the **in-process asyncio pub/sub** (`jobs/progress_bus.py`) → WebSocket (`routes/ws.py`) → frontend.

---

## Key Files

| File | Role |
|---|---|
| `backend/app/jobs/pipeline_runner.py` | Full pipeline orchestration; the single source of truth for phase order |
| `backend/app/ml/registry.py` | Lazy singleton model loader; raises `ModelNotReady` if weights absent |
| `backend/app/pipeline/router.py` | Language → STT model mapping (`en`→parakeet, `hi`→qwen3_asr, `mr`→whisper_turbo) |
| `backend/app/config.py` | All tunable constants (`CHUNK_DURATION_SECONDS`, thresholds, model IDs) via pydantic-settings |
| `backend/app/storage.py` | Canonical path helpers — always use these, never construct paths manually |
| `backend/app/models_db.py` | SQLAlchemy ORM: `Job`, `Chunk`, `TranslationChunk`, `Subtitle`, `ModelDownload` |
| `backend/app/pipeline/translator.py` | Translation orchestrator: mode selection (`vlm`/`audio`), first-pass + optional refinement pass |
| `backend/app/pipeline/translation_languages.py` | Supported translation target languages registry — **add new languages here only** |
| `backend/app/pipeline/glossary.py` | Domain-seeded + user-supplied glossary injected into translation prompts |
| `backend/app/schemas.py` | Pydantic API schemas (`JobOut`, `ChunkOut`, `ContextBundleOut`, `ProgressEvent`, etc.) |
| `backend/app/routes/upload.py` | `POST /api/upload` — accepts `language_hint`, `target_language`, `translator_mode`, `enable_refinement`, `glossary` form fields |
| `backend/app/routes/subtitles.py` | Subtitle download (`GET /api/subtitles/{job_id}/{fmt}?lang=`), media stream, clean-wav debug |
| `backend/app/routes/translation.py` | `GET /api/translation/languages` — returns all supported target languages for the frontend dropdown |

---

## Dev Workflows

```bash
# Backend (from repo root)
cd backend && source .venv/bin/activate
uvicorn app.main:app --reload --port 8000   # API at :8000, Swagger at /docs

# Frontend (separate terminal)
cd frontend && npm run dev                  # Vite dev server at :5173 (/api proxied to :8000)

# Tests
cd backend && pytest tests/ -v

# DB migrations — after editing models_db.py
alembic revision --autogenerate -m "describe change"
alembic upgrade head

# Reset DB
rm shravana.db && python -c "import asyncio; from app.db import init_db; asyncio.run(init_db())"
```

---

## Critical Patterns

**Blocking ML calls must use `run_in_executor`.**  
All STT/translation/audio processing is synchronous. In `pipeline_runner.py` every heavy call follows:
```python
result = await loop.run_in_executor(None, sync_function, arg1, arg2)
```

**`ModelNotReady` must be re-raised, never swallowed.**  
When a required model hasn't finished downloading, the pipeline re-raises `ModelNotReady` so the job transitions to `waiting_for_models` and resumes automatically once the download completes.

**Adding a new STT model — three-file change:**
1. Implement `BaseSTTModel` in `backend/app/ml/` (`.load()` + `.transcribe() → List[TranscriptSegment]`)
2. Register in `ml/registry.py` `MODEL_CONFIGS` dict
3. Map a language code in `pipeline/router.py` `LANGUAGE_MODEL_MAP`

**Translation subtitles use a language-suffixed filename:**
Source: `subtitles.srt` · Translated: `subtitles_{lang}.srt` (e.g. `subtitles_en.srt`)  
Both stored as separate `Subtitle` rows with the `language` column populated.

**Translator mode is auto-detected from file extension:**  
`select_translator_mode()` in `pipeline/translator.py` returns `"vlm"` for video extensions (`.mp4`, `.mkv`, …) and `"audio"` otherwise. Override at upload via the `translator_mode` form field (`"vlm"` or `"audio"`).  
- **`vlm` mode**: Qwen2.5-VL processes keyframes + transcript per semantic chunk.  
- **`audio` mode**: SeamlessM4T-v2 translates transcript text only.

**Refinement pass runs after first-pass translation (when `enable_refinement=True`):**  
A second text-only Qwen2.5-VL call refines the first-pass result using prev/next context and the glossary. If `qwen2_5_vl` is unavailable, the pipeline logs a warning and keeps the first-pass result (`degraded_no_refiner`).

**Code-switching routes to `qwen3_asr`:**  
In `pipeline/router.py`, if `language == "en"` and `confidence < CODE_SWITCH_THRESHOLD` (default `0.85`), the chunk is routed to `qwen3_asr` instead of `parakeet` to handle Hindi-English mixing.

**Adding a new translation language — single-file change:**  
Append one entry to `SUPPORTED_LANGUAGES` in `backend/app/pipeline/translation_languages.py` with the BCP-47 code, display name, and `seamless_code` (leave `seamless=False` if not supported by SeamlessM4T).

**`translation_status` is independent of `job.status`:**  
`Job.translation_status` tracks the translation sub-pipeline: `pending → context → translating → done` (or `failed`/`degraded_no_refiner`). It is `None` for non-translated jobs.

**`PIPELINE_CONCURRENCY=1` is intentional** — CPU-only machines can't run two model inferences in parallel. Do not increase without profiling.

**Config is environment-driven** — set any `Settings` field via env var or `backend/.env`. No hardcoded paths outside `storage.py`.

---

## Storage Layout

```
storage/
  uploads/{job_id}/original.<ext>      # uploaded file
  jobs/{job_id}/raw.wav                 # extracted/normalised audio
  jobs/{job_id}/clean.wav               # denoised vocals
  jobs/{job_id}/chunks/                 # 60-second WAV chunks
  jobs/{job_id}/subs/subtitles.{srt,vtt}
  jobs/{job_id}/subs/subtitles_{lang}.{srt,vtt}  # translated
  models/{model_name}/                  # HuggingFace cache
```

---

## Job Status Lifecycle

`pending → extracting → denoising → chunking → processing → [translating →] assembling → done`  
Error paths: `failed` · `waiting_for_models` (auto-retried when model ready)

`translation_status` (independent field): `pending → context → translating → done` · `failed` · `degraded_no_refiner`

To manually retry: `POST /api/jobs/{job_id}/retry`  
To retro-translate a completed job: `POST /api/jobs/{job_id}/translate` with `{ "target_language": "en" }`  
To delete a job (not actively running): `DELETE /api/jobs/{job_id}` — removes DB rows and storage files  
To inspect context bundle: `GET /api/jobs/{job_id}/context`  
To download subtitles with language selection: `GET /api/subtitles/{job_id}/srt?lang=en`

