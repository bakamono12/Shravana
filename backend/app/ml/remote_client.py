"""Sync HTTP client for the remote GPU worker.

All model proxy classes call these helpers from executor threads (sync context).
The remote_health module polls /v1/health from async context via run_in_executor.
"""
from __future__ import annotations
import json
import logging
import time
from typing import Any, Optional

import httpx

from app.ml.base import ModelNotReady

logger = logging.getLogger(__name__)


class RemoteCallError(Exception):
    pass


_client: Optional[httpx.Client] = None


def _get_client() -> httpx.Client:
    global _client
    if _client is None:
        from app.config import settings
        if not settings.REMOTE_GPU_URL:
            raise RemoteCallError("REMOTE_GPU_URL is not configured")
        headers = {}
        if settings.REMOTE_GPU_TOKEN:
            headers["Authorization"] = f"Bearer {settings.REMOTE_GPU_TOKEN}"
        _client = httpx.Client(
            base_url=settings.REMOTE_GPU_URL.rstrip("/"),
            headers=headers,
            timeout=settings.REMOTE_GPU_TIMEOUT_S,
        )
    return _client


def reset_client() -> None:
    """Discard the cached client (call after changing REMOTE_GPU_URL at runtime)."""
    global _client
    _client = None


def _retry_call(fn, *args, retries: int = 2, **kwargs) -> Any:
    last_exc: Exception = RuntimeError("no attempt made")
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except RemoteCallError:
            raise
        except ModelNotReady:
            raise
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RemoteCallError(f"Remote GPU unreachable after {retries + 1} attempt(s): {last_exc}") from last_exc


def _parse_response(resp: httpx.Response) -> Any:
    if resp.status_code == 401:
        raise RemoteCallError("Remote GPU: authentication failed — check REMOTE_GPU_TOKEN")
    if resp.status_code == 503:
        try:
            body = resp.json()
        except Exception:
            body = {}
        model = body.get("model_loading")
        if model:
            raise ModelNotReady(model)
        raise RemoteCallError(f"Remote GPU service unavailable: {body}")
    resp.raise_for_status()
    return resp.json()


def call_json(path: str, json_data: Any = None) -> Any:
    """POST JSON payload, return parsed JSON response. Sync."""
    from app.config import settings

    def _do():
        client = _get_client()
        resp = client.post(path, json=json_data, timeout=settings.REMOTE_GPU_TIMEOUT_S)
        return _parse_response(resp)

    return _retry_call(_do, retries=settings.LLM_MAX_RETRIES)


def call_multipart(path: str, files: dict, data: Optional[dict] = None) -> Any:
    """POST multipart/form-data with file uploads, return parsed JSON. Sync."""
    from app.config import settings

    def _do():
        client = _get_client()
        resp = client.post(path, files=files, data=data or {}, timeout=settings.REMOTE_GPU_TIMEOUT_S)
        return _parse_response(resp)

    return _retry_call(_do, retries=settings.LLM_MAX_RETRIES)


def call_multipart_binary(path: str, files: dict, data: Optional[dict] = None) -> bytes:
    """POST multipart, return raw bytes response body (used for file downloads). Sync."""
    from app.config import settings

    def _do():
        client = _get_client()
        resp = client.post(path, files=files, data=data or {}, timeout=settings.REMOTE_GPU_TIMEOUT_S)
        if resp.status_code == 401:
            raise RemoteCallError("Remote GPU: authentication failed — check REMOTE_GPU_TOKEN")
        if resp.status_code == 503:
            raise RemoteCallError(f"Remote GPU service unavailable: {resp.text[:200]}")
        resp.raise_for_status()
        return resp.content

    return _retry_call(_do, retries=settings.LLM_MAX_RETRIES)


def health() -> dict:
    """GET /v1/health. Returns {} on any error (caller handles missing keys)."""
    try:
        from app.config import settings
        client = _get_client()
        resp = client.get("/v1/health", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        logger.debug(f"Remote health check failed: {exc}")
    return {}
