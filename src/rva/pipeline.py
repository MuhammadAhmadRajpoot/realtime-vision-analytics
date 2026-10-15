"""Frame processing and the background pipeline thread."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

import cv2
import numpy as np
import supervision as sv

from rva.analytics.engine import AnalyticsEngine
from rva.analytics.types import Event
from rva.annotate import Annotator
from rva.config import PipelineConfig
from rva.detection import Detector, Tracker, to_tracked_objects
from rva.sources import Frame, VideoSource, resize_to_width
from rva.storage import EventStore

log = logging.getLogger(__name__)


class FpsMeter:
    """Exponential moving average of the processing rate."""

    def __init__(self, smoothing: float = 0.9) -> None:
        self.smoothing = smoothing
        self.fps = 0.0
        self._last: float | None = None

    def tick(self, now: float | None = None) -> float:
        now = time.perf_counter() if now is None else now
        if self._last is not None and now > self._last:
            inst = 1.0 / (now - self._last)
            self.fps = inst if self.fps == 0 else self.smoothing * self.fps + (1 - self.smoothing) * inst
        self._last = now
        return self.fps


@dataclass(slots=True)
class ProcessResult:
    annotated: np.ndarray
    tracked: sv.Detections
    events: list[Event]


class FrameProcessor:
    """Resize -> detect -> track -> analytics -> annotate, for one frame at a time."""

    def __init__(self, cfg: PipelineConfig, detector: Detector, frame_rate: float) -> None:
        self.cfg = cfg
        self.detector = detector
        self.tracker = Tracker(cfg.tracker, frame_rate)
        self.engine = AnalyticsEngine(cfg)
        self.annotator = Annotator(cfg.processing.heatmap_alpha, cfg.processing.heatmap)
        self.fps = FpsMeter()

    def process(self, frame: Frame) -> ProcessResult:
        events: list[Event] = []
        if frame.restarted:
            self.tracker.reset()
            events.extend(self.engine.restart_tracks())
        image = resize_to_width(frame.image, self.cfg.processing.max_width)
        tracked = self.tracker.update(self.detector.detect(image), frame.t)
        events.extend(self.engine.update(to_tracked_objects(tracked, image.shape), frame.t))
        fps = self.fps.tick()
        annotated = self.annotator.draw(image, tracked, self.engine, fps)
        now = datetime.now(UTC).isoformat(timespec="milliseconds")
        for e in events:
            e.ts = now
        return ProcessResult(annotated, tracked, events)


class Status(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class _Shared:
    """State written by the worker thread and read by the API."""

    jpeg: bytes | None = None
    frame_seq: int = 0
    snapshot: dict[str, Any] = field(default_factory=dict)
    recent_events: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=50))
    history: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=600))


class Pipeline:
    """Runs one source through a :class:`FrameProcessor` on a background thread."""

    def __init__(
        self,
        cfg: PipelineConfig,
        detector_provider: Callable[[], Detector],
        store: EventStore,
    ) -> None:
        self.cfg = cfg
        self.session_id = uuid.uuid4().hex[:12]
        self._detector_provider = detector_provider
        self._store = store
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._shared = _Shared(history=deque(maxlen=cfg.processing.history_size))
        self._thread: threading.Thread | None = None
        self.status = Status.IDLE
        self.error: str | None = None
        self.started_at: str | None = None
        self.frames_processed = 0
        self.fps = 0.0

    # -- lifecycle -----------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.status = Status.STARTING
        self.started_at = datetime.now(UTC).isoformat(timespec="seconds")
        self._thread = threading.Thread(target=self._run, name=f"pipeline-{self.session_id}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout)
        if self.status in (Status.RUNNING, Status.STARTING):
            self.status = Status.STOPPED

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def wait(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    # -- worker --------------------------------------------------------------------
    def _run(self) -> None:
        log.info("pipeline starting", extra={"session": self.session_id, "source": str(self.cfg.source.uri)})
        source = VideoSource(self.cfg.source, self.cfg.processing.frame_skip)
        processor: FrameProcessor | None = None
        try:
            detector = self._detector_provider()
            source.open()
            processor = FrameProcessor(self.cfg, detector, source.effective_fps)
            self.status = Status.RUNNING
            next_sample = 0.0
            while not self._stop.is_set():
                frame = source.read(self._stop)
                if frame is None:
                    break
                result = processor.process(frame)
                self._publish(processor, result)
                now = time.monotonic()
                if now >= next_sample:
                    self._sample_history(processor)
                    next_sample = now + self.cfg.processing.stats_interval_s
            if not self._stop.is_set():
                self.status = Status.FINISHED
        except Exception as exc:
            log.exception("pipeline failed", extra={"session": self.session_id})
            self.status = Status.ERROR
            self.error = str(exc)
        finally:
            if processor is not None:
                self._persist(processor.engine.restart_tracks())  # close open zone visits
                self._update_snapshot(processor)
            source.close()
            log.info("pipeline ended", extra={"session": self.session_id, "status": str(self.status)})

    def _persist(self, events: list[Event]) -> None:
        if not events:
            return
        now = datetime.now(UTC).isoformat(timespec="milliseconds")
        for e in events:
            e.ts = e.ts or now
        self._store.add(self.session_id, events)
        with self._lock:
            self._shared.recent_events.extendleft(e.to_dict() for e in events)

    def _publish(self, processor: FrameProcessor, result: ProcessResult) -> None:
        self._persist(result.events)
        ok, buf = cv2.imencode(".jpg", result.annotated, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.processing.jpeg_quality])
        self.frames_processed += 1
        self.fps = processor.fps.fps
        with self._lock:
            if ok:
                self._shared.jpeg = buf.tobytes()
                self._shared.frame_seq += 1
        self._update_snapshot(processor)

    def _update_snapshot(self, processor: FrameProcessor) -> None:
        snap = processor.engine.snapshot()
        snap["fps"] = round(self.fps, 2)
        with self._lock:
            self._shared.snapshot = snap

    def _sample_history(self, processor: FrameProcessor) -> None:
        totals = processor.engine.snapshot()["totals"]
        sample = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "video_t": round(processor.engine.t, 2),
            "fps": round(self.fps, 2),
            **totals,
        }
        with self._lock:
            self._shared.history.append(sample)

    # -- readers (thread-safe) -----------------------------------------------------
    def latest_jpeg(self) -> tuple[int, bytes | None]:
        with self._lock:
            return self._shared.frame_seq, self._shared.jpeg

    def stats(self) -> dict[str, Any]:
        with self._lock:
            snap = dict(self._shared.snapshot)
            recent = list(self._shared.recent_events)[:20]
        return {
            "session_id": self.session_id,
            "status": str(self.status),
            "error": self.error,
            "source": str(self.cfg.source.uri),
            "started_at": self.started_at,
            "frames_processed": self.frames_processed,
            **snap,
            "recent_events": recent,
        }

    def history(self, limit: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = list(self._shared.history)
        return items[-limit:] if limit else items
