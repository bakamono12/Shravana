# Shravana — Context-Aware Translation Module
## Implementation Plan for Claude Code

---

## Project Background

**Shravana** (श्रवण) is an STT/subtitle generation pipeline supporting Hindi, Marathi, and English.
It takes video or audio input, extracts audio, transcribes using Whisper/Parakeet/Qwen3-ASR,
and produces `.srt` / `.vtt` subtitle files.

Stack: **Python, FastAPI, Celery, Redis, SQLite/Postgres, FFmpeg, faster-whisper**

We are adding a **context-aware translation module** as a post-STT step.
Translation must be context-sensitive — not plain word-for-word output.
It must understand domain, speaker register, idioms, code-switching, and named entities
so the output reads naturally rather than like a machine translation.

The module must handle **two input types** differently:
1. **Video** — has visual frames for scene/domain context extraction
2. **Audio only** (podcasts, lectures, voice notes) — must derive all context from audio features + transcript alone

---

## Goals

- Translate transcribed subtitles from Hindi/Marathi/English → any target language
- Translation must be context-aware (domain, register, idioms, entities, speaker style)
- Support both video and audio-only inputs
- Slot cleanly into the existing Shravana pipeline as an optional post-STT phase
- Output: translated `.srt` / `.vtt` with same timestamps, speaker labels preserved

---

## New Dependencies to Add

```
# requirements.txt additions

# Translation
ctranslate2>=4.0.0          # fast IndicTrans2 / OPUS-MT inference
sentencepiece>=0.2.0         # tokenizer for IndicTrans2

# Audio context extraction
pyannote.audio>=3.1.0        # speaker diarization
librosa>=0.10.0              # audio feature extraction (pace, energy, silence)

# Video context extraction
Pillow>=10.0.0               # keyframe decoding

# NLP / entity extraction
spacy>=3.7.0                 # NER for entity list building
# run after install: python -m spacy download en_core_web_sm

# LLM for context extraction + translation refinement
# Use whichever is already in the project, in priority order:
# 1. Local Ollama endpoint (preferred, no cost)
# 2. OpenAI-compatible endpoint (vLLM, LiteLLM proxy)
# 3. Direct HuggingFace transformers (fallback, heavy)
```

---

## Folder Structure to Create

Add these inside the existing Shravana project root:

```
shravana/
└── translation/
    ├── __init__.py
    ├── config.py              # TranslationConfig dataclass
    ├── context/
    │   ├── __init__.py
    │   ├── base.py            # Abstract ContextExtractor
    │   ├── video_extractor.py # Keyframe-based context (for video input)
    │   ├── audio_extractor.py # Audio-feature + diarization context (for audio-only)
    │   └── models.py          # ContextBundle dataclass
    ├── translator/
    │   ├── __init__.py
    │   ├── base.py            # Abstract Translator
    │   ├── indicTrans.py      # IndicTrans2 fast first-pass
    │   ├── llm_refiner.py     # LLM context-aware refinement pass
    │   └── pipeline.py        # Orchestrates first-pass + refine
    ├── chunker.py             # Semantic chunking (topic-aware, not just time-based)
    ├── glossary.py            # Domain-specific term overrides
    └── tasks.py               # Celery tasks for translation jobs
```

---

## Core Data Models

### `translation/context/models.py`

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class SpeakerProfile:
    speaker_id: str                  # e.g. "SPEAKER_00"
    dominant_language: str           # "hi" | "mr" | "en"
    register: str                    # "formal" | "conversational" | "colloquial" | "technical"
    code_switch_tendency: str        # "heavy" | "moderate" | "none"
    translation_style: str           # instruction for LLM e.g. "casual, uses slang"

@dataclass
class ContextBundle:
    # Global (once per file)
    domain: str                      # "tech" | "finance" | "spirituality" | "sports" | "news" | "casual" | "academic" | "legal" | "medical" | "entertainment"
    format: str                      # "interview" | "monologue" | "panel" | "lecture" | "debate" | "storytelling" | "podcast"
    source_language: str             # primary language of source content
    target_language: str             # translation target
    named_entities: list[str]        # ["Zerodha", "Nifty", "Mumbai"] — do not translate these
    idioms_detected: list[str]       # source idioms found — translate culturally not literally
    speakers: dict[str, SpeakerProfile]  # keyed by speaker_id
    
    # Video-specific (None for audio-only)
    scene_description: Optional[str] = None   # from VLM keyframe analysis
    
    # Audio-specific
    speaking_rate: Optional[str] = None       # "fast" | "moderate" | "slow"
    audio_quality: Optional[str] = None       # "studio" | "field" | "phone"
    background_context: Optional[str] = None  # "crowd" | "music" | "silent"

@dataclass
class TranslatedChunk:
    start_time: float
    end_time: float
    speaker_id: Optional[str]
    source_text: str
    translated_text: str
    first_pass: str              # raw IndicTrans2 output before refinement
    confidence: float
```

---

## Phase 1 — Context Extraction

### `translation/context/audio_extractor.py`

Implement `AudioContextExtractor` with these methods:

**`extract_audio_features(audio_path: str) -> dict`**
- Use `librosa` to compute:
  - `speaking_rate` — average syllables/sec (fast > 4, slow < 2)
  - `silence_ratio` — fraction of audio that is silence (pyannote VAD or librosa)
  - `energy_variance` — high variance = emotional/dynamic, low = monotone/lecture
  - `music_detected` — bool, detect via onset strength + chroma patterns
- Return a dict of these raw signals

**`run_diarization(audio_path: str) -> list[dict]`**
- Use `pyannote.audio` pipeline `"pyannote/speaker-diarization-3.1"`
- Return list of `{speaker_id, start, end}` segments
- If diarization fails or only 1 speaker found, return single speaker "SPEAKER_00"

**`build_speaker_profiles(diarization_segments, transcript_chunks) -> dict[str, SpeakerProfile]`**
- Match transcript chunks to speaker segments by timestamp overlap
- For each speaker, collect their transcript lines
- Use LLM to classify their register, code-switch tendency, and dominant language
- Return dict of SpeakerProfile per speaker

**`extract_context_from_transcript(full_transcript: str, target_language: str) -> ContextBundle`**
- Send full transcript to LLM with this system prompt structure:
  ```
  You are a media analyst. Analyze this transcript and return ONLY a JSON object with:
  {
    "domain": one of [tech, finance, spirituality, sports, news, casual, academic, legal, medical, entertainment],
    "format": one of [interview, monologue, panel, lecture, debate, storytelling, podcast],
    "named_entities": [list of proper nouns, brand names, place names],
    "idioms_detected": [list of idiomatic phrases found],
    "notes": "any other context useful for a translator"
  }
  Transcript:
  {transcript}
  ```
- Parse JSON response, build and return ContextBundle

### `translation/context/video_extractor.py`

Implement `VideoContextExtractor` — extends audio extractor with keyframe analysis.

**`sample_keyframes(video_path: str, interval_seconds: int = 30) -> list[str]`**
- Use FFmpeg to extract one frame every `interval_seconds`
- Command: `ffmpeg -i {video_path} -vf fps=1/{interval_seconds} /tmp/frames/frame_%04d.jpg`
- Return list of frame file paths

**`describe_keyframes(frame_paths: list[str]) -> str`**
- For each frame, send to a VLM (Qwen2.5-VL-3B-Instruct via HuggingFace, 4-bit quant if available, else skip)
- Prompt: `"Describe this scene in one sentence: setting, activity, and apparent domain/subject matter"`
- Concatenate descriptions into a single scene summary string
- If VLM not available: return `None` and log a warning — video falls back to audio-only context

**`build_context(video_path: str, audio_path: str, transcript: str, target_language: str) -> ContextBundle`**
- Run `sample_keyframes` + `describe_keyframes` → scene_description
- Run audio features + diarization (inherited)
- Run `extract_context_from_transcript` with scene description injected into the prompt
- Return ContextBundle with scene_description populated

---

## Phase 2 — Semantic Chunking

### `translation/chunker.py`

Implement `SemanticChunker` — splits transcript into topic-coherent chunks for translation.

**Why not just use subtitle segments?**
Individual subtitle lines are too short for context. Translation quality improves
when the LLM sees a full topic cluster (10–30 seconds of speech) at once.

**`chunk_by_topic(segments: list[dict], max_chunk_seconds: int = 20) -> list[list[dict]]`**
- Group consecutive subtitle segments into chunks
- Start a new chunk when ANY of these conditions are true:
  1. Accumulated duration exceeds `max_chunk_seconds`
  2. Speaker changes (if diarization data is available)
  3. Long silence (> 1.5s gap between segment end and next start)
  4. Topic shift detected (optional: use sentence embeddings cosine distance)
- Return list of segment groups — each group = one translation unit
- Each chunk carries: combined source text, start/end timestamps, speaker_id

**Sliding window for context:**
When translating chunk N, also pass chunk N-1 (previous) and chunk N+1 (next, if available)
as context strings. Do NOT translate these — they are context-only.

---

## Phase 3 — Translation Pipeline

### `translation/translator/indicTrans.py`

Implement `IndicTranslator` — fast first-pass translation.

- Use `IndicTrans2` model via `ctranslate2` for speed
- Model: `ai4bharat/indictrans2-indic-en-dist-200M` (small, fast, good quality)
- Supported source langs: Hindi (`hin_Deva`), Marathi (`mar_Deva`), English (`eng_Latn`)
- This is the **first pass** — speed-optimized, not context-aware
- For each chunk: translate and store as `first_pass` in TranslatedChunk
- If IndicTrans2 is not available (model not downloaded): fall back to returning source text

**`translate_batch(chunks: list[str], src_lang: str, tgt_lang: str) -> list[str]`**
- Batch all chunks together for efficiency
- Handle max batch size (16 chunks at a time)
- Return list of translated strings

### `translation/translator/llm_refiner.py`

Implement `LLMRefiner` — context-aware refinement of first-pass output.

This is the core intelligence layer. Takes IndicTrans2 output and improves it using
the ContextBundle.

**`build_system_prompt(context: ContextBundle) -> str`**

Build this system prompt dynamically:
```
You are a professional subtitle translator and cultural localization expert.

CONTENT CONTEXT:
- Domain: {context.domain}
- Format: {context.format}  
- Source Language: {context.source_language}
- Target Language: {context.target_language}

TRANSLATION RULES:
1. Preserve the speaker's energy and register — do NOT make casual speech formal or vice versa
2. These are NAMED ENTITIES — never translate them: {', '.join(context.named_entities)}
3. Handle code-switching naturally — if Hindi sentence has English words, keep them English in output
4. Translate idioms culturally, not literally
5. Match the {context.domain} domain — use appropriate terminology
6. Keep subtitles concise — max 2 lines, ~42 chars per line ideally
{speaker_note if speaker_id in context.speakers else ""}

For scene context: {context.scene_description or "audio-only content"}
```

**`refine_chunk(chunk_text: str, first_pass: str, context: ContextBundle, prev_chunk: str, next_chunk: str, speaker_id: str) -> str`**

Build user message:
```
PREVIOUS CONTEXT (do not translate): {prev_chunk}

CURRENT SEGMENT to translate:
Source: {chunk_text}
Machine translation draft: {first_pass}

NEXT CONTEXT (do not translate): {next_chunk}

Provide ONLY the improved translation. No explanation. No quotes. Just the translated text.
```

Send to LLM endpoint (Ollama/vLLM/LiteLLM — configurable in config.py).
Return refined translation string.

**`refine_batch(chunks: list[TranslatedChunk], context: ContextBundle) -> list[TranslatedChunk]`**
- Process chunks with sliding window (prev, current, next)
- Handle LLM failures gracefully — if refine fails, keep `first_pass` as final
- Return updated list of TranslatedChunk with `translated_text` set

### `translation/translator/pipeline.py`

Implement `TranslationPipeline` — orchestrates the full flow.

```python
class TranslationPipeline:
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.indic = IndicTranslator(config)
        self.refiner = LLMRefiner(config)
        self.chunker = SemanticChunker()

    def run(
        self,
        segments: list[dict],          # subtitle segments from STT
        context: ContextBundle,
        media_path: str,               # video or audio path
    ) -> list[TranslatedChunk]:
        
        # Step 1: semantic chunking
        chunk_groups = self.chunker.chunk_by_topic(segments)
        
        # Step 2: first pass (IndicTrans2, batched)
        source_texts = [" ".join(s["text"] for s in g) for g in chunk_groups]
        first_passes = self.indic.translate_batch(
            source_texts, context.source_language, context.target_language
        )
        
        # Step 3: build TranslatedChunk list
        chunks = [
            TranslatedChunk(
                start_time=group[0]["start"],
                end_time=group[-1]["end"],
                speaker_id=group[0].get("speaker_id"),
                source_text=source_texts[i],
                translated_text="",   # filled by refiner
                first_pass=first_passes[i],
                confidence=0.0,
            )
            for i, group in enumerate(chunk_groups)
        ]
        
        # Step 4: LLM refinement (with context)
        if self.config.enable_llm_refinement:
            chunks = self.refiner.refine_batch(chunks, context)
        else:
            for chunk in chunks:
                chunk.translated_text = chunk.first_pass
        
        return chunks
```

---

## Phase 4 — Subtitle Reassembly

### `translation/reassembler.py`

Implement `TranslationReassembler` — maps translated chunks back to original subtitle timing.

**`map_to_segments(original_segments: list[dict], translated_chunks: list[TranslatedChunk]) -> list[dict]`**
- Each chunk covers multiple original subtitle segments
- Distribute translated text back to original timestamps
- If chunk has 3 original segments → split translated text across 3 with same time boundaries
- Use simple line-count split or sentence boundary detection
- Each output segment: `{index, start, end, speaker_id, text (translated)}`

**`to_srt(segments: list[dict]) -> str`**
- Standard SRT format with speaker label prepended if multi-speaker:
  `[SPEAKER_00] Translated text here.`

**`to_vtt(segments: list[dict]) -> str`**
- Standard WebVTT format

---

## Phase 5 — Config

### `translation/config.py`

```python
from dataclasses import dataclass

@dataclass
class TranslationConfig:
    # LLM endpoint for context extraction + refinement
    llm_endpoint: str = "http://localhost:11434/api/chat"   # Ollama default
    llm_model: str = "qwen3:8b"                             # or whatever is available
    llm_type: str = "ollama"    # "ollama" | "openai_compat" | "anthropic"
    llm_api_key: str = ""       # for openai_compat or anthropic
    
    # IndicTrans2
    indic_model_id: str = "ai4bharat/indictrans2-indic-en-dist-200M"
    indic_device: str = "cpu"   # "cpu" | "cuda"
    
    # Pyannote (diarization)
    pyannote_token: str = ""    # HuggingFace token for pyannote model access
    enable_diarization: bool = True
    
    # VLM for keyframe analysis (video only)
    vlm_model: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    enable_vlm: bool = True
    
    # Pipeline behaviour
    enable_llm_refinement: bool = True   # set False to use IndicTrans2 only (faster)
    max_chunk_seconds: int = 20
    sliding_window: bool = True
    
    # Glossary overrides (domain-specific terms that must not be mistranslated)
    custom_glossary: dict[str, str] = None   # {"SIP": "SIP", "demat": "demat"}
```

---

## Phase 6 — Celery Task

### `translation/tasks.py`

Add a Celery task that runs after the existing STT task completes:

```python
@celery_app.task(bind=True, max_retries=2, name="translation.translate_job")
def translate_job(self, job_id: str, target_language: str, translation_config: dict):
    """
    Runs after transcription is complete.
    Reads STT output from DB, runs context-aware translation, writes results.
    """
    config = TranslationConfig(**translation_config)
    pipeline = TranslationPipeline(config)
    
    # 1. Load STT segments from DB for this job_id
    # 2. Load media file path from DB
    # 3. Detect if video or audio-only → pick extractor
    # 4. Run context extraction → ContextBundle
    # 5. Run TranslationPipeline.run()
    # 6. Run TranslationReassembler
    # 7. Write translated .srt and .vtt to output dir
    # 8. Update job status in DB
```

---

## Phase 7 — API Endpoint

Add to existing FastAPI router:

**`POST /jobs/{job_id}/translate`**

Request body:
```json
{
  "target_language": "en",
  "enable_llm_refinement": true,
  "enable_diarization": true,
  "llm_model": "qwen3:8b",
  "custom_glossary": {}
}
```

Response:
```json
{
  "translation_job_id": "uuid",
  "status": "queued",
  "estimated_seconds": 120
}
```

**`GET /jobs/{job_id}/translation/status`**
- Returns `{status, progress_pct, error}`

**`GET /jobs/{job_id}/translation/download`**
- Returns translated `.srt` or `.vtt` file as download
- Query param: `?format=srt` or `?format=vtt`

---

## Implementation Order

Build in this exact order — each step is independently testable:

1. **`translation/config.py`** — dataclass, no dependencies
2. **`translation/context/models.py`** — dataclasses, no dependencies
3. **`translation/context/audio_extractor.py`** — librosa + pyannote + LLM call
4. **`translation/context/video_extractor.py`** — extends audio, adds FFmpeg + VLM
5. **`translation/chunker.py`** — pure Python, no ML
6. **`translation/glossary.py`** — simple dict + domain preset glossaries
7. **`translation/translator/indicTrans.py`** — ctranslate2 wrapper
8. **`translation/translator/llm_refiner.py`** — LLM prompt builder + caller
9. **`translation/translator/pipeline.py`** — orchestrator
10. **`translation/reassembler.py`** — SRT/VTT output
11. **`translation/tasks.py`** — Celery task wiring
12. **API endpoints** — FastAPI router additions
13. **Tests** — one test per step using a short Hindi sample audio/video

---

## Graceful Degradation Rules

The module must never crash the parent Shravana job. Implement these fallbacks:

| Failure | Fallback |
|---|---|
| Pyannote diarization fails | Single speaker, no profiles |
| VLM not available | Skip scene description, audio-only context |
| LLM endpoint unreachable | Skip refinement, use IndicTrans2 first-pass only |
| IndicTrans2 model not downloaded | Return source text unchanged, log error |
| librosa extraction fails | Skip audio features, use defaults |
| LLM returns invalid JSON | Regex-extract what you can, fill defaults |

Each component must log its fallback clearly with `logger.warning(...)`.

---

## Test Cases to Implement

1. **Hindi podcast mono** — single speaker, no diarization expected
2. **Hindi-English code-switching** — verify named entities not translated
3. **Multi-speaker interview** — verify per-speaker register consistency
4. **Video with visual context** — verify scene description influences translation
5. **Short audio (< 30s)** — edge case for chunker and reassembler
6. **Audio with music intro** — verify music detection, intro not treated as speech

---

## Notes for Claude Code

- Do NOT install pyannote without asking user to accept the HuggingFace token requirement first — gated model
- The VLM (qwen2-vl) is optional and heavy — default `enable_vlm: False`, document how to enable
- Use `loguru` or Python stdlib `logging` consistently with the rest of Shravana
- All LLM calls must have a timeout (30s default) and retry (max 2)
- Translation output files go into the same job output directory as `.srt`/`.vtt` from STT, named `{job_id}_translated_{lang}.srt`
- Respect existing DB schema — add new columns to jobs table if needed via Alembic migration, do not break existing columns
