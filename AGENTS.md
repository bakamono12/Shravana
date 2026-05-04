# Shravana — Agent Guide

Shravana converts video/audio files into subtitles via a multi-phase ML pipeline.
Stack: **FastAPI + SQLAlchemy async (SQLite) + Alembic** backend · **Vite + React + Tailwind** frontend.

---

## Architecture Overview

```
Upload → Job row created → pipeline_runner.run() (background task)
  Phase 1  audio_extractor   – ffmpeg extract/normalize → raw.wav
  Phase 2  denoiser          – Demucs + noisereduce → clean.wav + speech intervals (with per-sub-phase %)
  Phase 3  chunker           – 60 s chunks (5 s overlap) → Chunk rows (emits per-chunk %)
  Phase 4  language_detector – Qwen LID → Whisper LID fallback → user hint
  Phase 5  router + transcriber – per-chunk STT (model chosen by language)
  Phase 5.5 (optional) translator – semantic chunking → context → translate → refine
  Phase 6-7 reassembler + subtitle_generator → .srt/.vtt files → Subtitle rows
```

**Executor mode** is resolved via `services/executor_pref.py:resolve_executor()`:  
`REMOTE_GPU_URL` set → "remote" by default; UI override (`PUT /api/system/executor`) can force local or remote for new jobs without restart.

**Remote execution** uses a two-stage handoff when the worker is updated:
1. `/v1/pipeline/transcribe` — uploads audio once, streams denoise→chunk→transcribe progress, returns `workdir_token` + source SRT/VTT (downloadable immediately).
2. `/v1/pipeline/translate` — passes `workdir_token` (no re-upload), streams translate progress, returns translated SRT/VTT.
Falls back to legacy `/v1/pipeline/run` if the worker is older (no two-stage endpoints).

Progress events flow through the **in-process asyncio pub/sub** (`jobs/progress_bus.py`) → WebSocket (`routes/ws.py`) → frontend.  
Every event now includes `stage_index` (1–6) and `stage_total` (6) so the frontend can render a per-stage stepper and an overall progress bar (`((stage_index-1) + pct/100) / stage_total * 100`).

Every GPU-heavy model call is routed through `ml/registry.get(name)`. When executor is remote, the registry returns a **remote proxy** instead of the local singleton — the rest of the pipeline is unchanged.

**Active models** (2026-05-04):
| Name | Purpose | Languages |
|---|---|---|
| `qwen_lid` | Language identification before STT routing | 52+ |
| `qwen3_asr` | Primary STT — Indic + multilingual | 52 incl. Hindi/Marathi/Tamil |
| `whisper_turbo` | Universal STT default | 99 |
| `seamless_v2` | Audio-grounded first-pass translation | 100+ |
| `qwen2_5_vl` | VLM refinement + video-context translation | Any |

`parakeet` (English-only, redundant) and `forced_aligner` (no callers) have been removed.

```
┌────────────────────────────┐         ┌──────────────────────────────────┐
│ backend (local)            │         │ Colab / remote worker            │
│                            │         │                                  │
│ pipeline_runner.py         │         │ FastAPI server (worker/server.py)│
│   ↓ resolve_executor()     │         │   loads same backend.app.ml      │
│   if remote:               │  HTTPS  │   modules, exposes endpoints     │
│   remote_pipeline.py       │         │                                  │
│   run_remote_transcribe ───┼────────►│ POST /v1/pipeline/transcribe     │
│   ← workdir_token + SRT   ◄┼─────────┤   streams NDJSON progress        │
│   run_remote_translate ────┼────────►│ POST /v1/pipeline/translate      │
│   ← translated SRT        ◄┼─────────┤   streams NDJSON progress        │
│                            │         │                                  │
│   if local:                │         │ GET  /v1/health → gpu/models     │
│   existing local pipeline  │         └──────────────────────────────────┘
└────────────────────────────┘
```

---

## Key Files

| File | Role |
|---|---|
| `backend/app/jobs/pipeline_runner.py` | Full pipeline orchestration; the single source of truth for phase order |
| `backend/app/ml/registry.py` | Lazy singleton model loader; raises `ModelNotReady` if weights absent. **Routing seam**: returns remote proxy when `REMOTE_GPU_URL` is set |
| `backend/app/ml/remote_client.py` | Single `httpx` client — bearer auth, retry/backoff, `health()`, `call_json()`. Translates HTTP errors to `RemoteCallError` / `ModelNotReady` |
| `backend/app/ml/remote_proxies.py` | `RemoteSTTProxy`, `RemoteTranslatorProxy`, `RemoteDenoiser` — implement the same ABCs as local models |
| `backend/app/jobs/remote_pipeline.py` | Two-stage remote handoff: `run_remote_transcribe`, `run_remote_translate`, legacy `run_remote_pipeline` |
| `backend/app/services/executor_pref.py` | In-process override store (`resolve_executor()`, `set_override()`). Lost on restart — env is the durable default |
| `backend/app/db_crud.py` | `upsert_subtitle(session, job_id, fmt, path, language)` — atomic delete+insert to prevent duplicate Subtitle rows |
| `backend/app/services/remote_health.py` | Background poller for `/v1/health`; caches result for `/api/system/executor` |
| `backend/app/routes/system.py` | `GET /api/system/executor` → `{env_default, override, effective, remote_url_present, ...}`; `PUT` sets override |
| `backend/app/pipeline/router.py` | Language → STT model mapping (`hi`→qwen3_asr; default→whisper_turbo) |
| `backend/app/config.py` | All tunable constants (`CHUNK_DURATION_SECONDS`, thresholds, model IDs, `REMOTE_GPU_*`) via pydantic-settings |
| `backend/app/storage.py` | Canonical path helpers — always use these, never construct paths manually |
is th| `backend/app/models_db.py` | SQLAlchemy ORM: `Job`, `Chunk`, `TranslationChunk`, `Subtitle`, `ModelDownload`. `Job` has `executor` + `remote_url_snapshot`; `Chunk` has `remote_job_id` |
| `backend/app/pipeline/translator.py` | Translation orchestrator: mode selection (`vlm`/`audio`), first-pass + optional refinement pass |
| `backend/app/pipeline/translation_languages.py` | Supported translation target languages registry — **add new languages here only** |
| `backend/app/pipeline/glossary.py` | Domain-seeded + user-supplied glossary injected into translation prompts |
| `backend/app/schemas.py` | Pydantic API schemas (`JobOut`, `ChunkOut`, `ContextBundleOut`, `ProgressEvent`, etc.). `JobOut` includes `executor: str \| None` |
| `backend/app/routes/upload.py` | `POST /api/upload` — accepts `language_hint`, `target_language`, `translator_mode`, `enable_refinement`, `glossary` form fields |
| `backend/app/routes/subtitles.py` | Subtitle download (`GET /api/subtitles/{job_id}/{fmt}?lang=`), media stream, clean-wav debug |
| `backend/app/routes/translation.py` | `GET /api/translation/languages` — returns all supported target languages for the frontend dropdown |
| `worker/server.py` | Remote worker FastAPI app — imports `backend.app.ml` directly; exposes `/v1/health`, `/v1/asr/*`, `/v1/translate`, `/v1/pipeline/*`, `/v1/models` |
| `worker/pipeline_handler.py` | Worker-side pipeline stages: `run_transcribe`, `run_translate`, `run_pipeline` (legacy). Manages workdir token store (30-min TTL). |
| `worker/colab_setup.ipynb` | Four-cell Colab notebook: clone → install → download models → start worker + Cloudflare tunnel |
| `worker/requirements.txt` | Pinned deps for the remote worker |
| `frontend/src/components/ExecutorBadge.tsx` | Popover toggle: Auto (env) / Force Local / Force Remote. Calls `PUT /api/system/executor`. "Force Remote" disabled when `remote_url_present=false`. |

---

## Dev Workflows

```bash
# Backend (from repo root)
cd backend && source .venv/bin/activate
uvicorn app.main:app --reload --port 8000   # API at :8000, Swagger at /docs

# Frontend (separate terminal)
cd frontend && npm run dev                  # Vite dev server at :5173 (/api proxied to :8000)

# Run remote worker locally (for testing)
cd backend && source .venv/bin/activate
uvicorn worker.server:app --port 8001       # worker at :8001
# Then set in backend/.env:
#   REMOTE_GPU_URL=http://127.0.0.1:8001
#   REMOTE_GPU_TOKEN=test
#   REMOTE_GPU_MODELS=ALL

# Tests
cd backend && pytest tests/ -v

# DB migrations — after editing models_db.py
alembic revision --autogenerate -m "describe change"
alembic upgrade head

# Reset DB
rm shravana.db && python -c "import asyncio; from app.db import init_db; asyncio.run(init_db())"
```

---

## Remote GPU Config (env vars)

| Variable | Default | Description |
|---|---|---|
| `REMOTE_GPU_URL` | `None` | Remote worker base URL. Blank = pure local execution |
| `REMOTE_GPU_TOKEN` | `None` | Bearer token for the worker. 401 → fast-fail, no fallback |
| `REMOTE_GPU_MODELS` | `"ALL"` | `"ALL"` routes every model; CSV (e.g. `"qwen3_asr,seamless_v2"`) = partial routing |
| `REMOTE_GPU_TIMEOUT_S` | `180` | Per-request timeout in seconds |
| `REMOTE_GPU_FALLBACK_LOCAL` | `true` | On remote error/timeout, retry locally and log a warning. Set `false` to fail-fast |
| `REMOTE_GPU_HEALTHCHECK_INTERVAL_S` | `30` | How often `remote_health.py` polls `/v1/health` |

---

## Critical Patterns

**Blocking ML calls must use `run_in_executor`.**  
All STT/translation/audio processing is synchronous. In `pipeline_runner.py` every heavy call follows:
```python
result = await loop.run_in_executor(None, sync_function, arg1, arg2)
```

**`ModelNotReady` must be re-raised, never swallowed.**  
When a required model hasn't finished downloading, the pipeline re-raises `ModelNotReady` so the job transitions to `waiting_for_models` and resumes automatically once the download completes. Remote 503 with `{"model_loading": "..."}` is also mapped to `ModelNotReady` by `remote_client.py`.

**Remote routing seam is `ml/registry.get(name)`.**  
If `REMOTE_GPU_URL` is set and the model name matches `REMOTE_GPU_MODELS`, `get(name)` returns a proxy from `remote_proxies.py`. `pipeline_runner.py`, `transcriber.py`, `translator.py`, `denoiser.py` do not need to know which side runs the inference.  
Use `registry.get_local(name)` (forced-local accessor) only inside fallback error handlers.

**`RemoteCallError` triggers the fallback path.**  
In `pipeline_runner.py` remote model calls are wrapped:
```python
try:
    result = await loop.run_in_executor(None, proxy.transcribe, audio_path)
except RemoteCallError:
    if settings.REMOTE_GPU_FALLBACK_LOCAL:
        model = registry.get_local(name)
        result = await loop.run_in_executor(None, model.transcribe, audio_path)
    else:
        raise
```

**Token mismatch (401) → fast-fail, no fallback.**  
A 401 from the remote worker means misconfiguration; the pipeline fails immediately with a clear log message.

**`job.executor` is stamped once, not per-chunk.**  
When the first model call is made, `pipeline_runner.py` writes `job.executor = "remote" | "local"` and `job.remote_url_snapshot`. This is cosmetic for the UI; it does not affect control flow.

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
In `pipeline/router.py`, if the detected language confidence is below `CODE_SWITCH_THRESHOLD` (default `0.85`), the chunk is routed to `qwen3_asr` (52-lang model) regardless of the top language, to handle code-switching like Hindi-English mixing.

**Adding a new translation language — single-file change:**  
Append one entry to `SUPPORTED_LANGUAGES` in `backend/app/pipeline/translation_languages.py` with the BCP-47 code, display name, and `seamless_code` (leave `seamless=False` if not supported by SeamlessM4T).

**`translation_status` is independent of `job.status`:**  
`Job.translation_status` tracks the translation sub-pipeline: `pending → context → translating → done` (or `failed`/`degraded_no_refiner`). It is `None` for non-translated jobs.

**`PIPELINE_CONCURRENCY=1` is intentional** — CPU-only machines can't run two model inferences in parallel. Do not increase without profiling.

**Config is environment-driven** — set any `Settings` field via env var or `backend/.env`. No hardcoded paths outside `storage.py`.

**Worker imports `backend.app.ml` directly** — the worker is not a separate model reimplementation. Clone the repo on the GPU machine, add the backend package to `PYTHONPATH`, and run `worker/server.py`. This keeps model logic as a single source of truth.

---

## Remote Worker Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/health` | GPU info, loaded models, free VRAM |
| `POST` | `/v1/asr/transcribe?model=<name>` | Multipart: `audio` WAV → `List[TranscriptSegment]` |
| `POST` | `/v1/asr/detect_language?model=<name>` | Multipart: `audio` WAV → `{language, confidence}` |
| `POST` | `/v1/translate` | Multipart: `audio?`, `frame_0..N?`, `payload` JSON → `List[TranslatedSegment]`. Batches ≤4 segs per model call to avoid Cloudflare 524 |
| `POST` | `/v1/translate/refine` | JSON: `{first_pass, source_text, prev_text, next_text, …}` → `{text}` |
| `POST` | `/v1/generate` | JSON: `{model, source_text, system_prompt, history_text}` → `{text}` (used by context builder) |
| `POST` | `/v1/denoise` | Multipart: `audio` WAV → cleaned WAV + `X-Speech-Intervals` header |
| `POST` | `/v1/pipeline/transcribe` | Multipart streaming: audio → NDJSON events → `event=result` with chunks, SRT, `workdir_token` |
| `POST` | `/v1/pipeline/translate` | JSON streaming: `{workdir_token, target_language, …}` → NDJSON events → `event=result` with translated SRT |
| `DELETE` | `/v1/pipeline/workdir/{token}` | Early workdir cleanup (before TTL expiry) |
| `GET` | `/v1/models` | List models with `loaded` flag |
| `POST` | `/v1/models/{name}/load` | Admin: pre-load a model |

---

## Failure Modes

| Case | Behaviour |
|---|---|
| `REMOTE_GPU_URL` blank | Pure local execution — zero behaviour change |
| Remote down at job start | `healthy=false`. If `REMOTE_GPU_FALLBACK_LOCAL=true` (default), runs locally; otherwise job fails with `error_message="remote unavailable"` |
| Remote returns 503 + `{"model_loading": "..."}` | Mapped to `ModelNotReady` → reuses the existing `waiting_for_models` flow |
| Remote times out mid-call | Same fallback path; chunk retried once locally before failing the chunk |
| Token mismatch (401) | Fast-fail with a clear log message; no fallback |

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
  models/{model_name}/                  # HuggingFace cache (local backend)
```

The remote worker maintains its own `storage/models/` on the GPU machine. Model weights are not shared between the local backend and the worker.

---

## Job Status Lifecycle

`pending → extracting → denoising → chunking → processing → [translating →] assembling → done`  
Error paths: `failed` · `waiting_for_models` (auto-retried when model ready)

`translation_status` (independent field): `pending → context → translating → done` · `failed` · `degraded_no_refiner`

`executor` field on `Job`: `"local"` (default) | `"remote"` — set once when the first model call is made.

To manually retry: `POST /api/jobs/{job_id}/retry`  
To retro-translate a completed job: `POST /api/jobs/{job_id}/translate` with `{ "target_language": "en" }`  
To delete a job (not actively running): `DELETE /api/jobs/{job_id}` — removes DB rows and storage files  
To inspect context bundle: `GET /api/jobs/{job_id}/context`  
To download subtitles with language selection: `GET /api/subtitles/{job_id}/srt?lang=en`  
To check remote executor health: `GET /api/system/executor`

---

## Out of Scope (v2)

- SSE streaming progress for single long remote calls
- Server-side cancellation of in-flight remote requests
- Multi-region or load-balanced workers (today: one URL, one worker)
- Sharing model weight files between a co-located backend and worker
