from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ChunkOut(BaseModel):
    id: str
    sequence: int
    start_time: float
    end_time: float
    duration: float
    status: str
    detected_language: Optional[str]
    assigned_model: Optional[str]

    class Config:
        from_attributes = True


class JobOut(BaseModel):
    id: str
    filename: str
    status: str
    total_chunks: int
    completed_chunks: int
    detected_language: Optional[str]
    language_hint: Optional[str]
    error_message: Optional[str]
    waiting_for_model: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class JobDetailOut(JobOut):
    chunks: list[ChunkOut] = []


class SubtitleOut(BaseModel):
    id: str
    job_id: str
    format: str
    created_at: datetime

    class Config:
        from_attributes = True


class ModelDownloadOut(BaseModel):
    id: str
    name: str
    repo_id: str
    status: str
    bytes_downloaded: int
    bytes_total: int
    error_message: Optional[str]
    updated_at: datetime

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    status: str


class ProgressEvent(BaseModel):
    job_id: str
    status: str
    phase: str
    total_chunks: int
    completed_chunks: int
    percent: int
    eta_seconds: Optional[int] = None
    srt_url: Optional[str] = None
    vtt_url: Optional[str] = None
    error: Optional[str] = None


class ModelProgressEvent(BaseModel):
    name: str
    status: str
    bytes_downloaded: int
    bytes_total: int
    percent: int
