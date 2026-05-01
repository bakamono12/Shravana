"""WebSocket endpoints for real-time progress streaming."""
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.jobs.progress_bus import subscribe_job, unsubscribe_job, subscribe_models, unsubscribe_models

router = APIRouter()


@router.websocket("/ws/jobs/{job_id}")
async def job_progress_ws(websocket: WebSocket, job_id: str):
    await websocket.accept()
    q = subscribe_job(job_id)
    try:
        while True:
            # Send queued events; also heartbeat every 15s
            try:
                msg = await asyncio.wait_for(q.get(), timeout=15.0)
                await websocket.send_text(msg)
            except asyncio.TimeoutError:
                await websocket.send_text('{"type":"ping"}')
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        unsubscribe_job(job_id, q)


@router.websocket("/ws/models")
async def models_progress_ws(websocket: WebSocket):
    await websocket.accept()
    q = subscribe_models()
    try:
        while True:
            try:
                msg = await asyncio.wait_for(q.get(), timeout=15.0)
                await websocket.send_text(msg)
            except asyncio.TimeoutError:
                await websocket.send_text('{"type":"ping"}')
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        unsubscribe_models(q)
