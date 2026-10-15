"""FastAPI application: dashboard, MJPEG stream, REST, WebSocket."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rva import __version__
from rva.analytics.types import EventType
from rva.api.schemas import Health, SourceStatus, StartRequest
from rva.config import AppSettings, PipelineConfig, SourceConfig
from rva.service import AnalyticsService, DetectorFactory
from rva.storage import EventStore

log = logging.getLogger(__name__)

STATIC_DIR = files("rva") / "web" / "static"
BOUNDARY = "frame"


def _service(request: Request | WebSocket) -> AnalyticsService:
    return request.app.state.service


def _status(service: AnalyticsService) -> SourceStatus:
    p = service.pipeline
    if p is None:
        return SourceStatus(session_id=None, status="idle", source=None)
    return SourceStatus(session_id=p.session_id, status=str(p.status), source=str(p.cfg.source.uri), error=p.error)


async def mjpeg_frames(service: AnalyticsService, poll_s: float = 0.01) -> AsyncIterator[bytes]:
    """Yield multipart JPEG parts whenever the pipeline publishes a new frame."""
    last_seq = -1
    while True:
        pipeline = service.pipeline
        if pipeline is not None:
            seq, jpeg = pipeline.latest_jpeg()
            if jpeg is not None and seq != last_seq:
                last_seq = seq
                yield (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg
                    + b"\r\n"
                )
                continue
        await asyncio.sleep(poll_s)


def create_app(
    settings: AppSettings | None = None,
    pipeline_cfg: PipelineConfig | None = None,
    detector_factory: DetectorFactory | None = None,
) -> FastAPI:
    settings = settings or AppSettings()
    cfg = pipeline_cfg or PipelineConfig.from_yaml(settings.config_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        kwargs = {"detector_factory": detector_factory} if detector_factory else {}
        service = AnalyticsService(cfg, EventStore(settings.db_path), **kwargs)
        app.state.service = service
        if settings.autostart:
            service.start()
        yield
        service.shutdown()

    app = FastAPI(
        title="Realtime Vision Analytics",
        version=__version__,
        description="People/vehicle counting, zone occupancy and dwell time on live video.",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/api/health", response_model=Health, tags=["system"])
    def health(request: Request) -> Health:
        return Health(version=__version__, pipeline=_status(_service(request)).status)

    @app.get("/api/config", tags=["system"])
    def get_config(request: Request) -> dict[str, Any]:
        """Active scenario: source, detector settings, lines and zones (normalised coords)."""
        return _service(request).cfg.model_dump(mode="json")

    @app.get("/api/stats", tags=["analytics"])
    def stats(request: Request) -> dict[str, Any]:
        """Current counts, occupancy, dwell times and FPS of the running session."""
        p = _service(request).pipeline
        if p is None:
            return {"status": "idle"}
        return p.stats()

    @app.get("/api/history", tags=["analytics"])
    def history(request: Request, limit: Annotated[int, Query(ge=1, le=10_000)] = 300) -> list[dict[str, Any]]:
        """Time series of totals sampled every ``processing.stats_interval_s`` seconds."""
        p = _service(request).pipeline
        return p.history(limit) if p else []

    @app.get("/api/events", tags=["events"])
    def events(
        request: Request,
        limit: Annotated[int, Query(ge=1, le=5000)] = 100,
        type: EventType | None = None,  # noqa: A002 - public query parameter name
        session_id: str | None = None,
        after_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Persisted line-crossing and zone events, newest first."""
        return _service(request).store.query(limit, type, session_id, after_id)

    @app.get("/api/events/export.csv", tags=["events"])
    def export_csv(request: Request, type: EventType | None = None, session_id: str | None = None) -> Response:  # noqa: A002
        body = _service(request).store.export_csv(type, session_id)
        return Response(
            body, media_type="text/csv", headers={"Content-Disposition": 'attachment; filename="events.csv"'}
        )

    @app.post("/api/source/start", response_model=SourceStatus, tags=["source"])
    def start(request: Request, body: StartRequest | None = None) -> SourceStatus:
        """Start (or restart) processing. Optionally switch to a new source."""
        service = _service(request)
        source: SourceConfig | None = None
        if body is not None and (body.uri is not None or body.loop is not None or body.realtime is not None):
            update = {k: v for k, v in body.model_dump().items() if v is not None}
            try:
                source = SourceConfig.model_validate({**service.cfg.source.model_dump(), **update})
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        service.start(source)
        return _status(service)

    @app.post("/api/source/stop", response_model=SourceStatus, tags=["source"])
    def stop(request: Request) -> SourceStatus:
        service = _service(request)
        service.stop()
        return _status(service)

    @app.get("/stream.mjpg", tags=["video"])
    def stream(request: Request) -> StreamingResponse:
        """Annotated video as an MJPEG stream (use directly in an ``<img>`` tag)."""
        return StreamingResponse(
            mjpeg_frames(_service(request)),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/snapshot.jpg", tags=["video"])
    def snapshot(request: Request) -> Response:
        """Latest annotated frame as a single JPEG."""
        p = _service(request).pipeline
        jpeg = p.latest_jpeg()[1] if p else None
        if jpeg is None:
            raise HTTPException(404, "no frame available yet")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.websocket("/ws/stats")
    async def ws_stats(websocket: WebSocket, interval: float = 0.5) -> None:
        """Pushes the same payload as ``GET /api/stats`` every ``interval`` seconds."""
        await websocket.accept()
        service = _service(websocket)
        interval = min(max(interval, 0.1), 10.0)
        try:
            while True:
                p = service.pipeline
                await websocket.send_json(p.stats() if p else {"status": "idle"})
                await asyncio.sleep(interval)
        except WebSocketDisconnect:
            log.debug("websocket client disconnected")

    return app
