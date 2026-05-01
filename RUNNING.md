# Shravana — How to Run

## Prerequisites

| Requirement | Version | Install |
|---|---|---|
| Python | 3.10 – 3.12 | system |
| Node.js | ≥ 18 | system |
| npm | ≥ 8 | bundled with Node |
| FFmpeg | any recent | `sudo apt install -y ffmpeg` |
| Free disk | ~25 GB | for ML models + Torch |
| RAM | ≥ 16 GB | models run on CPU |

```bash
sudo apt install -y ffmpeg python3-venv
```

---

## First-time setup

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# Initialise SQLite database (creates shravana.db)
python -c "import asyncio; from app.db import init_db; asyncio.run(init_db())"
```

> **Note:** `nemo_toolkit[asr]` (for Parakeet) is optional and has heavy
> system deps. Install separately only if you need English ASR:
> `pip install 'nemo_toolkit[asr]>=1.23.0'`

### Frontend

```bash
cd frontend
npm install
```

---

## Run — development (two terminals)

```bash
# Terminal 1: API + background model downloader
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2: Vite dev server with hot-reload
cd frontend
npm run dev
# Open http://localhost:5173
```

The Vite dev server proxies `/api` → `http://localhost:8000`.

---

## Run — single-process (production-ish)

```bash
# Build frontend
cd frontend
npm run build            # outputs to frontend/dist/

# Serve everything from FastAPI
cd ../backend
source .venv/bin/activate
SHRAVANA_SERVE_UI=1 uvicorn app.main:app --port 8000
# Open http://localhost:8000
```

---

## Models

On **first launch** the backend automatically begins downloading all 5 ML
models (~20 GB total) to `storage/models/` in background threads.

| Model | Size | Used for |
|---|---|---|
| Qwen3-ASR-0.6B | ~2 GB | Language identification |
| Qwen3-ASR-1.7B | ~5 GB | Hindi / code-switched ASR |
| Parakeet TDT 1.1B | ~4 GB | English ASR |
| Whisper large-v3-turbo | ~6 GB | Marathi / fallback ASR |
| Qwen3-ForcedAligner-0.6B | ~2 GB | Word-level timestamps for Whisper |

Progress is shown in the **Model Status** panel in the UI.

Uploaded jobs will queue in `waiting_for_models` status until the required
model finishes downloading, then resume automatically.

To pre-download from the CLI before starting the server:

```bash
cd backend && source .venv/bin/activate
python -c "
import asyncio, os
os.environ.setdefault('HF_HOME', 'storage/models')
from huggingface_hub import snapshot_download
for repo in [
    'Qwen/Qwen3-ASR-0.6B',
    'Qwen/Qwen3-ASR-1.7B',
    'nvidia/parakeet-tdt-1.1b',
    'Qwen/Qwen3-ForcedAligner-0.6B',
]:
    print(f'Downloading {repo}...')
    snapshot_download(repo)
print('All done.')
"
```

---

## Where things live

| Path | Contents |
|---|---|
| `backend/shravana.db` | SQLite database (jobs, chunks, subtitles, model download status) |
| `storage/uploads/{job_id}/` | Original uploaded files |
| `storage/jobs/{job_id}/raw.wav` | Extracted & normalised audio |
| `storage/jobs/{job_id}/clean.wav` | Denoised vocals (after Demucs + noisereduce) |
| `storage/jobs/{job_id}/chunks/` | 60-second audio chunks |
| `storage/jobs/{job_id}/subs/` | Generated `.srt` and `.vtt` files |
| `storage/models/` | HuggingFace model cache |

---

## Configuration

Copy `.env.example` → `.env` in the `backend/` directory and adjust:

```env
# storage paths (defaults are relative to repo root)
UPLOAD_DIR=storage/uploads
JOBS_DIR=storage/jobs
MODELS_DIR=storage/models

# audio chunking
CHUNK_DURATION_SECONDS=60   # reduce to 30 if RAM is tight
CHUNK_OVERLAP_SECONDS=5

# concurrency (keep 1 on CPU-only machines)
PIPELINE_CONCURRENCY=1

# serve built UI from FastAPI
SHRAVANA_SERVE_UI=false
```

---

## Common tasks

**Reset database:**
```bash
rm backend/shravana.db
python -c "import asyncio; from app.db import init_db; asyncio.run(init_db())"
```

**Wipe a stuck job:**
```bash
rm -rf storage/jobs/<job_id>
# then delete the row in shravana.db (sqlite3 CLI or DB browser)
```

**Retry a failed job via API:**
```bash
curl -X POST http://localhost:8000/api/jobs/<job_id>/retry
```

**Download the denoised audio for debugging:**
```
GET http://localhost:8000/api/jobs/<job_id>/clean.wav
```

**Run backend tests:**
```bash
cd backend && source .venv/bin/activate
pytest tests/ -v
```

**Type-check + build frontend:**
```bash
cd frontend && npm run build
```

---

## Performance expectations (CPU-only, Intel Iris Xe)

| Input | Demucs | Whisper-large / Parakeet / Qwen | Total |
|---|---|---|---|
| 1-min audio (English) | ~3 min | ~5 min | ~8 min |
| 5-min video (Hindi) | ~12 min | ~20 min | ~32 min |
| 10-min video (Marathi) | ~25 min | ~45 min | ~70 min |

The UI shows an ETA based on elapsed time per chunk. A long ETA is normal — it
is not a hang.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModelNotReady` in job error | Wait for that model to show 100% in Model Status panel |
| `ffmpeg not found` | `sudo apt install -y ffmpeg` |
| Out-of-memory crash during transcription | Set `CHUNK_DURATION_SECONDS=30` in `.env` |
| Whisper hallucinating on silent segments | Silero VAD should skip these; check `clean.wav` for audio content |
| Demucs not found | `pip install demucs` inside the backend venv |
| `ModuleNotFoundError: nemo` | Install separately: `pip install 'nemo_toolkit[asr]'` |
| Port 8000 in use | `SHRAVANA_PORT=8001 uvicorn app.main:app --port 8001` |

---

## References

- Full implementation plan: [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md)
- Original pipeline spec: [`subtitle_pipeline_plan.md`](subtitle_pipeline_plan.md)
- FastAPI docs: http://localhost:8000/docs (Swagger UI, auto-generated)
