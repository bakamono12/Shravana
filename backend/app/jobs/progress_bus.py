"""In-process asyncio pub/sub for WebSocket broadcast."""
from __future__ import annotations
import asyncio
import json
from typing import Dict, Set


# job_id → set of asyncio.Queue for active WS connections
_job_subscribers: Dict[str, Set[asyncio.Queue]] = {}
_model_subscribers: Set[asyncio.Queue] = set()


def subscribe_job(job_id: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _job_subscribers.setdefault(job_id, set()).add(q)
    return q


def unsubscribe_job(job_id: str, q: asyncio.Queue) -> None:
    subs = _job_subscribers.get(job_id, set())
    subs.discard(q)


def subscribe_models() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _model_subscribers.add(q)
    return q


def unsubscribe_models(q: asyncio.Queue) -> None:
    _model_subscribers.discard(q)


async def publish_job_progress(job_id: str, event: dict) -> None:
    payload = json.dumps(event)
    dead = set()
    for q in _job_subscribers.get(job_id, set()):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.add(q)
    for q in dead:
        _job_subscribers.get(job_id, set()).discard(q)


async def publish_model_progress(
    name: str,
    status: str,
    bytes_dl: int,
    bytes_total: int,
    *,
    attempt: int | None = None,
    note: str | None = None,
) -> None:
    percent = int(bytes_dl / bytes_total * 100) if bytes_total else 0
    event: dict = {
        "name": name,
        "status": status,
        "bytes_downloaded": bytes_dl,
        "bytes_total": bytes_total,
        "percent": percent,
    }
    if attempt is not None:
        event["attempt"] = attempt
    if note is not None:
        event["note"] = note
    payload = json.dumps(event)
    dead = set()
    for q in _model_subscribers:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.add(q)
    for q in dead:
        _model_subscribers.discard(q)
