"""Request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    uri: str | int | None = Field(
        None, description="Webcam index, file path or RTSP/HTTP URL. Omit to restart the configured source."
    )
    loop: bool | None = None
    realtime: bool | None = None


class SourceStatus(BaseModel):
    session_id: str | None
    status: str
    source: str | None
    error: str | None = None


class Health(BaseModel):
    status: str = "ok"
    version: str
    pipeline: str
