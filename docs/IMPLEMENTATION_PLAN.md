# Shravana — Subtitle Pipeline + Reactive UI

## Context

`subtitle_pipeline_plan.md` describes a self-hosted, multi-model ASR pipeline (English → Parakeet, Hindi → Qwen3-ASR, Marathi → Whisper-large-v3-turbo) but is **API-only**, assumes **NVIDIA GPU**, and assumes **video input only**. The repo currently contains only that plan and an empty venv.

The user wants:

1. A working pipeline that ingests **audio or video** uploaded from a UI, denoises/cleans audio when needed, transcribes via the plan's models, returns `.srt`/`.vtt`.
2. A **reactive React UI** with dark/light theme support.
3. **Heavy model downloads must run as background tasks** — app starts instantly, models populate over time with visible progress.
4. Light infra (SQLite + FastAPI BackgroundTasks, no Docker/Celery/Postgres yet).
5. Models stay as in the plan (Parakeet 1.1B, Qwen3-ASR-1.7B, Whisper-large-v3-turbo, Qwen3-ASR-0.6B LID, Qwen3-ForcedAligner-0.6B) — running on CPU, accepting the slowness.

This plan **supersedes** `subtitle_pipeline_plan.md` where they conflict (input mode, infra, UI, denoising). The model choices and pipeline phases from the original plan are kept.

---

## Architecture

```
Shravana/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI entrypoint (mounts API + serves built UI)
│   │   ├── config.py              # pydantic-settings
│   │   ├── db.py                  # SQLAlchemy + SQLite engine, session
│   │   ├── models_db.py           # Job, Chunk, Subtitle, ModelDownload ORM
│   │   ├── schemas.py             # Pydantic request/response
│   │   ├── routes/
│   │   │   ├── upload.py          # POST /api/upload (multipart, accepts audio+video)
│   │   │   ├── jobs.py            # GET /api/jobs, /api/jobs/{id}
│   │   │   ├── subtitles.py       # GET /api/subtitles/{id}/{srt|vtt|json}
│   │   │   ├── models_admin.py    # GET /api/models  (download status)
│   │   │   └── ws.py              # WS /api/ws/jobs/{id}, /api/ws/models
│   │   ├── pipeline/
│   │   │   ├── audio_extractor.py # ffmpeg: video → wav 16k mono
│   │   │   ├── denoiser.py        # NEW — Demucs vocals + noisereduce + Silero VAD
│   │   │   ├── chunker.py         # 60s chunks, 5s overlap, on normalized wav
│   │   │   ├── language_detector.py
│   │   │   ├── router.py
│   │   │   ├── transcriber.py
│   │   │   ├── reassembler.py
│   │   │   └── subtitle_generator.py
│   │   ├── ml/
│   │   │   ├── base.py            # BaseSTTModel ABC + dataclasses
│   │   │   ├── parakeet.py
│   │   │   ├── qwen_asr.py        # used for both LID-0.6B and ASR-1.7B
│   │   │   ├── whisper_turbo.py
│   │   │   ├── forced_aligner.py
│   │   │   ├── demucs_separator.py
│   │   │   ├── registry.py        # lazy singleton — only loads what's downloaded
│   │   │   └── downloader.py      # background HF snapshot_download w/ progress
│   │   ├── jobs/
│   │   │   ├── pipeline_runner.py # orchestrates all phases for one job
│   │   │   └── progress_bus.py    # asyncio pub/sub for WS broadcast
│   │   └── storage.py             # path conventions under storage/
│   ├── alembic/                   # migrations (sqlite-friendly)
│   ├── pyproject.toml
│   └── tests/
│       ├── test_audio_extractor.py
│       ├── test_chunker.py
│       ├── test_denoiser.py
│       ├── test_reassembler.py
│       └── test_subtitle_generator.py
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts             # proxy /api → http://localhost:8000
│   ├── tailwind.config.ts         # darkMode: 'class'
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── lib/api.ts             # fetch wrappers
│   │   ├── lib/ws.ts              # WS client w/ reconnect
│   │   ├── hooks/useJob.ts
│   │   ├── hooks/useTheme.ts      # dark/light/system, localStorage
│   │   ├── hooks/useModelStatus.ts
│   │   ├── components/
│   │   │   ├── ThemeToggle.tsx
│   │   │   ├── Uploader.tsx       # drag-drop, audio+video, chunked POST
│   │   │   ├── JobCard.tsx        # progress, phase, ETA
│   │   │   ├── JobList.tsx
│   │   │   ├── SubtitlePreview.tsx
│   │   │   ├── ModelStatusPanel.tsx  # shows background download progress
│   │   │   ├── LanguageHint.tsx      # optional override (en/hi/mr)
│   │   │   └── ui/                # shadcn/ui generated components
│   │   └── pages/
│   │       ├── Home.tsx           # uploader + job list
│   │       └── JobDetail.tsx      # detailed progress + preview + downloads
└── storage/                       # gitignored
    ├── uploads/{job_id}/...
    ├── jobs/{job_id}/{raw,clean,chunks,subs}/
    └── models/                    # HF cache (HF_HOME points here)
```

---

## Key design decisions

### 1. Denoising / "clear audio" layer (new vs. original plan)
For both audio and video uploads, after extraction normalize to 16k mono WAV, then:

1. **Demucs (`htdemucs`)** to separate the **vocals** stem — strips music, ambient noise, crowd. MIT licensed. Heavy but high quality; on CPU runs ~3-5x slower than realtime — acceptable since user accepted slowness.
2. **`noisereduce`** spectral gating on the vocals stem to clean residual hiss/static.
3. **Silero VAD** to mark non-speech regions; chunks falling entirely in silence are skipped (avoids Whisper hallucination — gotcha #5 in original plan).

`denoiser.py` produces `clean.wav` which feeds the chunker. Pipeline DB tracks `clean_audio_path` separately from `audio_path`.

### 2. Background model downloads
- On API startup, `ml/downloader.py` queues HF `snapshot_download` for each required model into `storage/models/` (set `HF_HOME` env). Runs in a `BackgroundTasks` worker thread per model.
- Each download has a `ModelDownload` row: `(name, repo_id, status, bytes_downloaded, bytes_total, error)`. Progress emitted via WS `/api/ws/models`.
- `registry.get(name)` raises `ModelNotReady` if download incomplete; pipeline runner catches this and parks the job in `waiting_for_models` status with the missing model name surfaced in the UI.
- UI `ModelStatusPanel` shows per-model bars; uploads work immediately, jobs queue until needed model is ready.

### 3. Job execution (no Celery yet)
- `pipeline_runner.run(job_id)` is a single async coroutine that walks all phases sequentially for one job, awaiting CPU-bound steps via `run_in_threadpool`.
- A module-level `asyncio.Semaphore(1)` serializes pipeline runs (CPU + RAM constrained — running two big models concurrently would OOM).
- Triggered by `BackgroundTasks` from the upload route.
- `progress_bus.publish(job_id, event)` → WS clients on `/api/ws/jobs/{id}`.
- Migration path: the function is structured as discrete phase calls, each idempotent on `(job_id, chunk_id)` — converting to Celery later means wrapping each phase as a `@shared_task`.

### 4. Audio vs video input
Upload route inspects MIME / extension:
- Video (`mp4, mkv, mov, avi, webm`) → `audio_extractor.extract()` → wav.
- Audio (`mp3, wav, m4a, flac, ogg, aac`) → `audio_extractor.normalize()` (just resample to 16k mono).
- Then both paths go through `denoiser` → `chunker` → rest of pipeline.

### 5. UI theming
- Tailwind `darkMode: 'class'`. `useTheme` hook stores preference in `localStorage`, defaults to `prefers-color-scheme`. `ThemeToggle` cycles light → dark → system.
- shadcn/ui's neutral palette + a single accent color (defined in `tailwind.config.ts`) used in both modes.
- All custom components use `bg-background text-foreground` semantic classes — dark mode "just works" via CSS vars defined in `index.css`.

### 6. Real-time progress
- WS `/api/ws/jobs/{id}` pushes `{phase, percent, completed_chunks, total_chunks, eta_seconds}` on each phase boundary and per chunk.
- WS `/api/ws/models` pushes per-model download progress.
- `lib/ws.ts` reconnects with exponential backoff; `useJob` falls back to polling `GET /api/jobs/{id}` if WS is down.

---

## Files to be created (critical)

**Backend new files:** every file listed under `backend/app/` above. None already exist.
**Frontend new files:** entire `frontend/` tree.
**Modify:** `.gitignore` (add `storage/`, `node_modules/`, `frontend/dist/`, `__pycache__/`, `*.db`).
**Replace/keep:** `subtitle_pipeline_plan.md` stays as a reference doc; this plan supersedes it.

## Reused / external libraries (no internal code to reuse — fresh repo)
- `fastapi`, `uvicorn[standard]`, `pydantic-settings`, `sqlalchemy`, `alembic`, `python-multipart`, `aiosqlite`
- `ffmpeg-python` (+ system `ffmpeg` — must `apt install ffmpeg`)
- `torch` (CPU build), `transformers`, `huggingface-hub`, `nemo_toolkit[asr]`, `faster-whisper`
- `demucs`, `noisereduce`, `silero-vad` (or `torch.hub` load), `soundfile`, `librosa`
- Frontend: `react`, `react-dom`, `react-router-dom`, `vite`, `typescript`, `tailwindcss`, `@radix-ui/*` (via shadcn), `lucide-react`, `clsx`, `tailwind-merge`

---

## Implementation order

1. **Repo skeleton** — `backend/pyproject.toml`, `frontend/` scaffold via `npm create vite`, install Tailwind + shadcn init, update `.gitignore`.
2. **Backend bones** — `config.py`, `db.py`, ORM models, alembic init + initial migration, `storage.py`, FastAPI app with health check.
3. **Frontend bones** — Vite + Tailwind + shadcn, `useTheme` + `ThemeToggle`, empty `Home`/`JobDetail` pages, dev proxy to `:8000`. **Verify dark/light toggle in browser.**
4. **Model downloader** — `ml/downloader.py` + `ModelDownload` table + `/api/models` + WS. UI `ModelStatusPanel`. **Verify**: kill backend mid-download, restart, downloads resume.
5. **Audio extractor + normalizer** — handle both video and audio inputs. Unit tests with a sample mp4 and mp3.
6. **Denoiser** — Demucs vocals → noisereduce → VAD mask. Test on a noisy clip; spot-check `clean.wav` audibly.
7. **Chunker** — 60s/5s overlap on normalized clean wav. Test that durations and DB rows match.
8. **Model wrappers** — `base.py` first, then Parakeet, Qwen-ASR, Whisper-turbo, ForcedAligner. Each gated on its `ModelDownload` being `done`. Test each with a known-language clip.
9. **Registry** — lazy singleton; raises `ModelNotReady` cleanly.
10. **LID + router** — implement, unit-test routing table and code-switch heuristic.
11. **Transcriber + reassembler** — overlap dedup logic from original plan §7. Test on a 3-chunk synthetic case.
12. **Subtitle generator** — SRT + VTT. Test format compliance with `pysubs2` round-trip.
13. **Upload route + pipeline_runner** — wire it all together, single-job e2e on a 30s sample.
14. **WS progress + job list UI** — `Uploader`, `JobList`, `JobCard`, `JobDetail`, `SubtitlePreview`. **Verify in browser:** upload a video, watch progress live, download srt.
15. **Polish** — language hint dropdown, error toasts, empty states, loading skeletons, mobile layout pass.
16. **Persist this plan into the repo** — copy this plan file to `docs/IMPLEMENTATION_PLAN.md` (create `docs/`) so it lives alongside the code and can be referenced after the Claude session ends. `subtitle_pipeline_plan.md` stays as the original reference.
17. **Write `RUNNING.md` at repo root** — the run-helper described below.

---

## `RUNNING.md` (final deliverable — content to write at step 17)

The file should contain, in order:

1. **Prerequisites** — Ubuntu/Debian system packages: `ffmpeg`, `python3.12-venv`, `nodejs >= 20`, `npm`. Disk: ~25 GB free for models + Torch wheels. RAM: 16 GB recommended (models run on CPU).
2. **First-time setup** — exact commands:
   ```bash
   sudo apt install -y ffmpeg python3-venv
   # backend
   cd backend
   python3 -m venv .venv && source .venv/bin/activate
   pip install -e .
   alembic upgrade head
   # frontend
   cd ../frontend
   npm install
   ```
3. **Run (development)** — two-terminal workflow:
   ```bash
   # terminal 1 — API + background workers
   cd backend && source .venv/bin/activate
   uvicorn app.main:app --reload --port 8000

   # terminal 2 — UI dev server
   cd frontend && npm run dev          # http://localhost:5173
   ```
4. **Run (single-process / "production-ish")**:
   ```bash
   cd frontend && npm run build         # outputs to frontend/dist
   cd ../backend && source .venv/bin/activate
   SHRAVANA_SERVE_UI=1 uvicorn app.main:app --port 8000
   # open http://localhost:8000
   ```
   `app.main` mounts `frontend/dist` as static when `SHRAVANA_SERVE_UI=1`.
5. **Models** — first launch downloads ~20 GB to `storage/models/` in the background. The UI's Model Status panel shows progress; uploads queue until needed models are ready. To pre-download from the CLI: `python -m app.ml.downloader --all`.
6. **Where things live** — `storage/uploads/`, `storage/jobs/<id>/{raw,clean,chunks,subs}/`, `storage/models/`, `backend/shravana.db` (SQLite).
7. **Common tasks** —
   - Reset DB: `rm backend/shravana.db && alembic upgrade head`
   - Wipe a stuck job: `rm -rf storage/jobs/<id>` + delete row
   - Re-trigger a failed job: `POST /api/jobs/<id>/retry`
   - Run tests: `cd backend && pytest`; `cd frontend && npm run build`
8. **Troubleshooting** —
   - `ModelNotReady` in job error: wait for the model panel to show 100% for that model.
   - `ffmpeg not found`: install via apt.
   - Out-of-memory during transcription: reduce `CHUNK_DURATION_SECONDS` in `.env` (default 60 → try 30).
   - Whisper hallucinating on silence: confirm `denoiser.py` VAD is enabled; check `clean.wav` for actual content.
9. **Pointers** — link to `docs/IMPLEMENTATION_PLAN.md` (full plan) and `subtitle_pipeline_plan.md` (original spec).

---

## Verification (end-to-end)

Run from repo root:

```bash
# one-time
sudo apt install -y ffmpeg
cd backend && python -m venv .venv && source .venv/bin/activate && pip install -e .
alembic upgrade head
cd ../frontend && npm install

# run
# terminal 1
cd backend && uvicorn app.main:app --reload --port 8000
# terminal 2
cd frontend && npm run dev   # opens http://localhost:5173
```

Manual test plan:

1. Open UI — confirm dark/light toggle works, persists across reload, respects `prefers-color-scheme` on first visit.
2. Open the model status panel — confirm 5 models begin downloading on first launch, progress streams over WS.
3. Upload a short MP4 (English speech with background music). While models still downloading, job should show `waiting_for_models`. After Parakeet + Demucs ready, job should auto-start.
4. Watch phase progression: `extracting → denoising → chunking → detecting_language → transcribing (N/M) → assembling → done`.
5. Play the produced `clean.wav` (exposed at `/api/jobs/{id}/clean.wav` for debugging) — music/noise should be substantially reduced.
6. Download `.srt` and `.vtt`, verify timestamps line up with the source video in VLC.
7. Repeat with: a Hindi audio-only `.m4a`; a noisy Marathi `.wav`; a Hindi-English code-switched clip — verify routing picks `qwen3_asr` for the last.
8. Backend tests: `cd backend && pytest`. Frontend type-check: `cd frontend && npm run build`.

Open risk to flag during implementation:
- Demucs + Whisper-large-v3-turbo together on this Iris Xe / CPU laptop will be **very slow** (a 5-min video may take 15-30 min). User accepted this; surface ETA prominently in UI so it's not mistaken for a hang.
- Total disk for all 5 models is ~20 GB — call this out in the model panel before downloads begin.
