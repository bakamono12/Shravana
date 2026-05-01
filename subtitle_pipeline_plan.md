# Video Transcription & Subtitle Generation Pipeline
## Comprehensive Technical Plan for Claude Code

---

## 1. Project Overview

Build a production-grade pipeline that:
1. Accepts a video file upload from a user (low-bandwidth friendly)
2. Extracts audio from the video
3. Splits audio into sequential chunks
4. Detects language per chunk
5. Routes each chunk to the best STT model for that language
6. Reassembles transcripts in order
7. Outputs `.srt` / `.vtt` subtitle files

**Target Languages (Phase 1):** English, Hindi, Marathi  
**Architecture:** Python backend, async job queue, self-hosted open-source models only  
**License Constraint:** All models must be Apache 2.0 or MIT (commercial use allowed)

---

## 2. Technology Stack

### Core
| Component | Technology |
|---|---|
| Language | Python 3.11+ |
| Web Framework | FastAPI |
| Job Queue | Celery + Redis |
| Database | PostgreSQL (job tracking) |
| File Storage | Local filesystem (S3-compatible interface for future) |
| Audio Processing | FFmpeg (via subprocess) |

### ML Models
| Language | Model | License | VRAM |
|---|---|---|---|
| Language Detection | Qwen3-ASR-0.6B | Apache 2.0 | ~2GB |
| English | Parakeet TDT v2 (NVIDIA) | Apache 2.0 | ~4GB |
| Hindi | Qwen3-ASR-1.7B | Apache 2.0 | ~5GB |
| Marathi | Whisper Large V3 Turbo | MIT | ~6GB |
| Timestamp Alignment | Qwen3-ForcedAligner-0.6B | Apache 2.0 | ~2GB |

### Infrastructure
| Component | Technology |
|---|---|
| Resumable Upload | tus protocol (`tusd` server or `tus-py-server`) |
| Model Serving | Local inference via `nemo` / `transformers` / `faster-whisper` |
| Progress Tracking | Redis pub/sub → WebSocket to client |

---

## 3. Project Structure

```
subtitle-pipeline/
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entrypoint
│   ├── routes/
│   │   ├── upload.py            # Resumable upload endpoint
│   │   ├── jobs.py              # Job status, progress, results
│   │   └── subtitles.py         # Download .srt / .vtt
│   └── websocket.py             # Real-time progress push
├── pipeline/
│   ├── __init__.py
│   ├── audio_extractor.py       # Phase 1: FFmpeg strip audio
│   ├── chunker.py               # Phase 2: Chunk + manifest
│   ├── language_detector.py     # Phase 3: Qwen3-ASR-0.6B LID
│   ├── router.py                # Phase 4: Route chunk to model
│   ├── transcriber.py           # Phase 5: Run STT per chunk
│   ├── reassembler.py           # Phase 6: Merge + offset timestamps
│   └── subtitle_generator.py   # Phase 7: Write .srt / .vtt
├── models/
│   ├── __init__.py
│   ├── base.py                  # Abstract base class for all STT models
│   ├── parakeet.py              # NVIDIA Parakeet TDT v2 wrapper
│   ├── qwen_asr.py              # Qwen3-ASR-1.7B wrapper
│   ├── whisper_turbo.py         # Whisper Large V3 Turbo wrapper
│   ├── forced_aligner.py        # Qwen3-ForcedAligner-0.6B wrapper
│   └── model_registry.py        # Singleton model loader
├── workers/
│   ├── __init__.py
│   ├── celery_app.py            # Celery config
│   └── tasks.py                 # Celery task definitions
├── db/
│   ├── __init__.py
│   ├── models.py                # SQLAlchemy ORM models
│   └── crud.py                  # DB operations
├── storage/
│   ├── __init__.py
│   └── file_manager.py          # File path management
├── config.py                    # Settings via pydantic-settings
├── requirements.txt
├── docker-compose.yml
└── .env.example
```

---

## 4. Database Schema

```sql
-- Jobs table: tracks one job per uploaded video
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    video_path TEXT NOT NULL,
    audio_path TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    -- status: pending | extracting | chunking | processing | assembling | done | failed
    total_chunks INT DEFAULT 0,
    completed_chunks INT DEFAULT 0,
    detected_language TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Chunks table: tracks each audio chunk independently
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    sequence INT NOT NULL,
    filename TEXT NOT NULL,
    path TEXT NOT NULL,
    start_time FLOAT NOT NULL,       -- seconds from start of full audio
    end_time FLOAT NOT NULL,
    duration FLOAT NOT NULL,
    detected_language TEXT,
    assigned_model TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    -- status: pending | processing | done | failed
    transcript JSONB,                -- raw STT output with word timestamps
    retry_count INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Subtitles table: final output per job
CREATE TABLE subtitles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES jobs(id) ON DELETE CASCADE,
    format TEXT NOT NULL,            -- srt | vtt
    path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 5. Pipeline — Phase by Phase

### Phase 1: Resumable Upload

**File:** `api/routes/upload.py`

- Implement `tus` protocol endpoint
- On upload complete, create a `Job` record in DB with status `pending`
- Trigger the main pipeline Celery task: `process_video.delay(job_id)`
- Store uploaded file at: `storage/uploads/{job_id}/original.{ext}`

**Key points:**
- Do NOT process anything during upload
- Support chunked uploads so low-bandwidth users can resume
- Validate file type (accept: mp4, mkv, avi, mov, webm)
- Set max file size limit in config (e.g. 5GB)

---

### Phase 2: Audio Extraction

**File:** `pipeline/audio_extractor.py`

```python
# Goal: strip audio from video via stream copy — near instant, no re-encoding
# Command to run:
# ffmpeg -i {video_path} -vn -acodec copy -y {audio_path}
# Output: storage/jobs/{job_id}/raw_audio.aac

# IMPORTANT: if source video has non-AAC audio (e.g. MP3, FLAC in MKV),
# fall back to re-encoding to WAV 16kHz mono — required by all STT models:
# ffmpeg -i {video_path} -vn -ar 16000 -ac 1 -acodec pcm_s16le -y {audio_path}
```

**Update job status to:** `extracting` → `chunking`

---

### Phase 3: Audio Chunking + Manifest

**File:** `pipeline/chunker.py`

```python
# Goal: split raw audio into 60-second chunks with 5-second overlap
# The overlap prevents sentences being cut at chunk boundaries

# FFmpeg command:
# ffmpeg -i {audio_path} \
#   -f segment \
#   -segment_time 60 \
#   -segment_time_overlap 5 \       <-- 5s overlap for boundary stitching
#   -reset_timestamps 1 \
#   -acodec copy \
#   -y {output_dir}/chunk_%04d.aac

# After chunking, create a manifest and insert chunk records into DB:
# manifest = [
#   {
#     "sequence": 0,
#     "filename": "chunk_0000.aac",
#     "path": "...",
#     "start_time": 0,
#     "end_time": 65,               # 60 + 5 overlap
#     "duration": 65.0
#   },
#   ...
# ]
```

**Store chunks at:** `storage/jobs/{job_id}/chunks/chunk_XXXX.aac`  
**Update job:** `total_chunks = N`, status → `processing`

---

### Phase 4: Language Detection

**File:** `pipeline/language_detector.py`

```python
# Use Qwen3-ASR-0.6B in LID-only mode on first 10 seconds of each chunk
# This is fast (~92ms per chunk) and avoids full transcription just for LID

# Returns: ISO 639-1 language code e.g. "en", "hi", "mr"

# Marathi note: Qwen3-ASR may not detect Marathi reliably as it's not in its
# primary language list. Handle this by:
# 1. If LID returns Hindi ("hi") but script analysis suggests Marathi vocabulary,
#    treat as Marathi
# 2. OR: accept a user-provided language hint at upload time as override
# 3. Store detected language per chunk in DB

# Fallback logic:
# - If LID confidence < 0.7, default to the job-level detected language
# - If still uncertain, default to "hi" (Hindi) for Devanagari script
```

---

### Phase 5: Model Routing

**File:** `pipeline/router.py`

```python
LANGUAGE_MODEL_MAP = {
    "en": "parakeet",          # NVIDIA Parakeet TDT v2
    "hi": "qwen3_asr",         # Qwen3-ASR-1.7B
    "mr": "whisper_turbo",     # Whisper Large V3 Turbo
    "default": "whisper_turbo" # fallback for unknown languages
}

# Code-switching detection:
# If a chunk contains mixed language (e.g. Hindi+English which is very common
# in Bollywood), route to qwen3_asr — it handles code-switching better
# Detection: if LID confidence < 0.85 and one of the languages is "en",
# assume code-switching and route to qwen3_asr
```

---

### Phase 6: Transcription Workers

**File:** `models/base.py` — Abstract base class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

@dataclass
class WordTimestamp:
    word: str
    start: float   # seconds from chunk start
    end: float     # seconds from chunk start
    confidence: float

@dataclass
class TranscriptSegment:
    text: str
    start: float
    end: float
    words: List[WordTimestamp]
    language: str

class BaseSTTModel(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> List[TranscriptSegment]:
        pass
    
    @abstractmethod
    def load(self):
        pass
```

**File:** `models/parakeet.py` — NVIDIA Parakeet TDT v2

```python
# Install: pip install nemo_toolkit[asr]
# Model: nvidia/parakeet-tdt-1.1b
# Expects: 16kHz mono WAV
# Returns: word-level timestamps natively
# Auto-adds punctuation and capitalization
# English only
```

**File:** `models/qwen_asr.py` — Qwen3-ASR-1.7B

```python
# Install: pip install transformers torch
# Model: Qwen/Qwen3-ASR-1.7B (HuggingFace)
# Expects: 16kHz mono audio
# Returns: word-level timestamps
# Supports Hindi natively, handles Hindi-English code-switching
```

**File:** `models/whisper_turbo.py` — Whisper Large V3 Turbo

```python
# Install: pip install faster-whisper
# Use faster-whisper for 4x speed improvement over original whisper
# Model: large-v3-turbo
# Expects: 16kHz mono WAV
# Returns: segment-level timestamps (use Qwen3-ForcedAligner for word-level)
# Use for Marathi — widest language coverage
```

**File:** `models/model_registry.py`

```python
# Singleton pattern — load models once at worker startup, not per request
# Models are large (2-8GB each), never reload per chunk
# Use a dict to hold loaded model instances:
# registry = {
#   "parakeet": ParakeetModel(),
#   "qwen3_asr": Qwen3ASRModel(),
#   "whisper_turbo": WhisperTurboModel(),
#   "forced_aligner": ForcedAlignerModel()
# }
```

---

### Phase 7: Transcript Reassembly

**File:** `pipeline/reassembler.py`

```python
# Goal: merge all chunk transcripts into one timeline-correct transcript

# Step 1: Sort chunks by sequence number
# Step 2: For each chunk, offset all timestamps:
#   actual_start = chunk.start_time + word.start
#   actual_end = chunk.start_time + word.end
#   BUT: for overlapping chunks (last 5 seconds of chunk N overlaps
#        with first 5 seconds of chunk N+1), deduplicate words at seam

# Overlap deduplication strategy:
#   - Take chunk N's transcript up to (chunk_duration - overlap/2) seconds
#   - Take chunk N+1's transcript starting from (overlap/2) seconds
#   - This drops the overlapping middle from both sides cleanly

# Step 3: Run Qwen3-ForcedAligner on Whisper outputs (Marathi)
#   because Whisper returns segment-level timestamps, not word-level
#   Parakeet and Qwen3-ASR return word-level natively

# Step 4: Store final merged transcript as JSONB in job record
```

---

### Phase 8: Subtitle File Generation

**File:** `pipeline/subtitle_generator.py`

**SRT format:**
```
1
00:00:01,000 --> 00:00:04,200
This is the first subtitle line.

2
00:00:04,500 --> 00:00:07,800
This is the second subtitle line.
```

**VTT format:**
```
WEBVTT

00:00:01.000 --> 00:00:04.200
This is the first subtitle line.
```

**Subtitle segmentation rules:**
- Max 42 characters per line
- Max 2 lines per subtitle block
- Min display duration: 1 second
- Max display duration: 7 seconds
- Group words into subtitle blocks based on natural pause points (end of sentence, comma, etc.)

---

## 6. Celery Task Flow

**File:** `workers/tasks.py`

```python
# Task 1: process_video(job_id)
#   - extract audio
#   - chunk audio
#   - for each chunk, dispatch: transcribe_chunk.delay(chunk_id)
#   - dispatch: monitor_job.delay(job_id) — waits for all chunks

# Task 2: transcribe_chunk(chunk_id)
#   - detect language
#   - route to model
#   - transcribe
#   - store result in DB
#   - increment job.completed_chunks
#   - publish progress to Redis pub/sub

# Task 3: assemble_job(job_id)  [triggered when all chunks done]
#   - reassemble transcripts
#   - generate .srt and .vtt
#   - update job status to "done"

# Retry config per task:
#   max_retries = 3
#   retry_backoff = True
#   retry_backoff_max = 60  # seconds
```

---

## 7. API Endpoints

```
POST   /upload/                     # Init tus upload session
PATCH  /upload/{upload_id}          # tus chunk upload
HEAD   /upload/{upload_id}          # tus offset check

GET    /jobs/{job_id}               # Job status + progress
GET    /jobs/{job_id}/chunks        # Per-chunk status breakdown
WS     /jobs/{job_id}/progress      # WebSocket real-time progress

GET    /subtitles/{job_id}/srt      # Download .srt file
GET    /subtitles/{job_id}/vtt      # Download .vtt file
GET    /subtitles/{job_id}/json     # Raw transcript JSON (for custom use)
```

---

## 8. Progress Tracking (WebSocket)

```json
// Message pushed to client on each chunk completion:
{
  "job_id": "uuid",
  "status": "processing",
  "total_chunks": 12,
  "completed_chunks": 7,
  "percent": 58,
  "current_phase": "transcribing",
  "estimated_remaining_seconds": 45
}

// Final message:
{
  "job_id": "uuid",
  "status": "done",
  "total_chunks": 12,
  "completed_chunks": 12,
  "percent": 100,
  "srt_url": "/subtitles/{job_id}/srt",
  "vtt_url": "/subtitles/{job_id}/vtt"
}
```

---

## 9. Audio Preprocessing (Critical)

All STT models require **16kHz mono audio**. Add a preprocessing step in the chunker:

```python
# Before chunking, normalize audio to 16kHz mono:
# ffmpeg -i raw_audio.aac \
#   -ar 16000 \        # resample to 16kHz
#   -ac 1 \            # convert to mono
#   -acodec pcm_s16le \ # PCM WAV for model compatibility
#   -y normalized.wav

# Then chunk the normalized WAV, not the original AAC
# This ensures every model gets properly formatted input
```

---

## 10. Configuration (`config.py`)

```python
# Use pydantic-settings for all config:

class Settings(BaseSettings):
    # Storage
    UPLOAD_DIR: str = "storage/uploads"
    JOBS_DIR: str = "storage/jobs"
    MAX_FILE_SIZE_GB: int = 5

    # Audio processing
    CHUNK_DURATION_SECONDS: int = 60
    CHUNK_OVERLAP_SECONDS: int = 5
    AUDIO_SAMPLE_RATE: int = 16000

    # Models
    PARAKEET_MODEL_ID: str = "nvidia/parakeet-tdt-1.1b"
    QWEN_ASR_MODEL_ID: str = "Qwen/Qwen3-ASR-1.7B"
    QWEN_LID_MODEL_ID: str = "Qwen/Qwen3-ASR-0.6B"
    WHISPER_MODEL_SIZE: str = "large-v3-turbo"
    FORCED_ALIGNER_MODEL_ID: str = "Qwen/Qwen3-ForcedAligner-0.6B"

    # Language routing
    LID_CONFIDENCE_THRESHOLD: float = 0.70
    CODE_SWITCH_THRESHOLD: float = 0.85

    # Queue
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_CONCURRENCY: int = 2   # limit concurrent GPU jobs

    # DB
    DATABASE_URL: str = "postgresql://user:pass@localhost:5432/subtitles"

    class Config:
        env_file = ".env"
```

---

## 11. Docker Compose

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./storage:/app/storage
    depends_on:
      - redis
      - postgres

  worker:
    build: .
    command: celery -A workers.celery_app worker --loglevel=info --concurrency=2
    env_file: .env
    volumes:
      - ./storage:/app/storage
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: subtitles
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

---

## 12. Implementation Order (for Claude Code)

Build in this exact sequence — each phase is independently testable:

1. **Setup** — project structure, config, Docker Compose, DB migrations (use Alembic)
2. **Storage layer** — file manager, path conventions
3. **DB layer** — SQLAlchemy models, CRUD operations
4. **Audio extraction** — `audio_extractor.py`, test with a sample MP4
5. **Chunker** — `chunker.py` + manifest, verify chunk files and DB records
6. **Model wrappers** — implement `base.py` first, then each model wrapper, test each independently
7. **Model registry** — singleton loader, verify GPU memory usage
8. **Language detector** — `language_detector.py`, test with Hindi/English/Marathi samples
9. **Router** — `router.py`, unit test routing logic
10. **Transcription worker** — `transcriber.py`, end-to-end single chunk test
11. **Celery tasks** — wire up tasks, test full chunk pipeline
12. **Reassembler** — `reassembler.py`, test overlap deduplication
13. **Subtitle generator** — `subtitle_generator.py`, verify SRT/VTT format
14. **FastAPI routes** — upload, jobs, subtitles endpoints
15. **WebSocket progress** — real-time updates
16. **End-to-end test** — full video → subtitles flow

---

## 13. Key Gotchas to Handle in Code

- **AAC chunking boundary issue** — AAC is not always frame-accurate when splitting. Consider converting to WAV before chunking if segment boundaries are imprecise.
- **Marathi LID failure** — Qwen3-ASR-0.6B may misidentify Marathi as Hindi. Add a user-supplied language hint in the upload request as an override.
- **GPU OOM** — Models share GPU. Use `CELERY_CONCURRENCY=1` if running all models on a single GPU. Implement model unloading between tasks if VRAM is tight.
- **Empty chunks** — Last chunk may be very short (a few seconds). Skip transcription if chunk duration < 2 seconds.
- **Whisper hallucination** — Whisper sometimes generates phantom text on silence. Add a silence detection step (using FFmpeg `silencedetect` filter) and skip transcription for silent chunks.
- **Timestamp offset drift** — When reassembling, use `chunk.start_time` from DB (not computed from sequence × duration) because FFmpeg segment timestamps can drift slightly on non-uniform audio.

---

## 14. Dependencies (`requirements.txt`)

```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
celery[redis]>=5.3.0
redis>=5.0.0
sqlalchemy>=2.0.0
alembic>=1.13.0
asyncpg>=0.29.0
pydantic-settings>=2.0.0
python-multipart>=0.0.9

# Audio
ffmpeg-python>=0.2.0

# ML
torch>=2.2.0
transformers>=4.40.0
nemo_toolkit[asr]>=1.23.0
faster-whisper>=1.0.0

# Utils
python-dotenv>=1.0.0
httpx>=0.27.0
websockets>=12.0
```

---

## 15. Out of Scope for Phase 1

- Translation (subtitles in a different language than audio)
- Speaker diarization (who said what)
- Frontend UI (Phase 1 is API-only)
- Cloud deployment / S3 storage
- Fine-tuning models on custom data
- Marathi-specific model fine-tuning (use Whisper base for now)
- Billing / auth / rate limiting
