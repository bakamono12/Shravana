import uuid
from datetime import datetime
from sqlalchemy import String, Float, Integer, DateTime, ForeignKey, Text, Boolean, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.utcnow()


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    video_path: Mapped[str] = mapped_column(String, nullable=False)
    audio_path: Mapped[str | None] = mapped_column(String, nullable=True)
    clean_audio_path: Mapped[str | None] = mapped_column(String, nullable=True)
    # pending|extracting|denoising|chunking|processing|assembling|done|failed|waiting_for_models
    status: Mapped[str] = mapped_column(String, default="pending")
    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    completed_chunks: Mapped[int] = mapped_column(Integer, default=0)
    detected_language: Mapped[str | None] = mapped_column(String, nullable=True)
    language_hint: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    waiting_for_model: Mapped[str | None] = mapped_column(String, nullable=True)
    # translation fields
    translate: Mapped[bool] = mapped_column(Boolean, default=False)
    target_language: Mapped[str | None] = mapped_column(String, nullable=True)
    # audio|vlm|none — auto-detected if None
    translator_mode: Mapped[str | None] = mapped_column(String, nullable=True)
    enable_refinement: Mapped[bool] = mapped_column(Boolean, default=True)
    context_bundle_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    glossary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # pending|context|translating|refining|done|failed|degraded_no_refiner — independent of status
    translation_status: Mapped[str | None] = mapped_column(String, nullable=True)
    # local|remote — which executor ran this job
    executor: Mapped[str] = mapped_column(String, default="local")
    # snapshot of REMOTE_GPU_URL at job time (for debugging/audit)
    remote_url_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    # token referencing worker-side temp workdir (two-stage remote handoff)
    workdir_token: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    chunks: Mapped[list["Chunk"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    translation_chunks: Mapped[list["TranslationChunk"]] = relationship(back_populates="job", cascade="all, delete-orphan")
    subtitles: Mapped[list["Subtitle"]] = relationship(back_populates="job", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    duration: Mapped[float] = mapped_column(Float, nullable=False)
    detected_language: Mapped[str | None] = mapped_column(String, nullable=True)
    assigned_model: Mapped[str | None] = mapped_column(String, nullable=True)
    # pending|processing|done|failed|skipped
    status: Mapped[str] = mapped_column(String, default="pending")
    transcript_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    translation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    job: Mapped["Job"] = relationship(back_populates="chunks")


class TranslationChunk(Base):
    """Semantic translation unit — one or more STT segments grouped by topic/silence/speaker."""
    __tablename__ = "translation_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    first_pass: Mapped[str | None] = mapped_column(Text, nullable=True)
    translated_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    speaker_id: Mapped[str | None] = mapped_column(String, nullable=True)
    # pending|translating|done|failed
    status: Mapped[str] = mapped_column(String, default="pending")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    job: Mapped["Job"] = relationship(back_populates="translation_chunks")


class Subtitle(Base):
    __tablename__ = "subtitles"
    __table_args__ = (
        UniqueConstraint("job_id", "format", "language", name="uq_subtitle_job_fmt_lang"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"))
    format: Mapped[str] = mapped_column(String, nullable=False)  # srt|vtt|json
    path: Mapped[str] = mapped_column(String, nullable=False)
    # BCP-47 language code
    language: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    job: Mapped["Job"] = relationship(back_populates="subtitles")


class ModelDownload(Base):
    __tablename__ = "model_downloads"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    repo_id: Mapped[str] = mapped_column(String, nullable=False)
    # queued|downloading|done|failed
    status: Mapped[str] = mapped_column(String, default="queued")
    bytes_downloaded: Mapped[int] = mapped_column(Integer, default=0)
    bytes_total: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)
