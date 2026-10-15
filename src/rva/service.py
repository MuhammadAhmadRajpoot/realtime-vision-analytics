"""Owns the event store, the cached detector and the currently running pipeline."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from rva.config import DetectorConfig, PipelineConfig, SourceConfig
from rva.detection import Detector, YoloDetector
from rva.pipeline import Pipeline
from rva.storage import EventStore

log = logging.getLogger(__name__)

DetectorFactory = Callable[[DetectorConfig], Detector]


class AnalyticsService:
    def __init__(
        self,
        cfg: PipelineConfig,
        store: EventStore,
        detector_factory: DetectorFactory = YoloDetector,
    ) -> None:
        self.cfg = cfg
        self.store = store
        self._factory = detector_factory
        self._detector: Detector | None = None
        self._detector_lock = threading.Lock()
        self._lock = threading.Lock()
        self.pipeline: Pipeline | None = None

    def _get_detector(self) -> Detector:
        # Loaded lazily on the pipeline thread and reused across restarts.
        with self._detector_lock:
            if self._detector is None:
                self._detector = self._factory(self.cfg.detector)
            return self._detector

    def start(self, source: SourceConfig | None = None) -> Pipeline:
        with self._lock:
            if self.pipeline is not None:
                self.pipeline.stop()
            if source is not None:
                self.cfg = self.cfg.model_copy(update={"source": source})
            self.pipeline = Pipeline(self.cfg, self._get_detector, self.store)
            self.pipeline.start()
            log.info("source started", extra={"uri": str(self.cfg.source.uri), "session": self.pipeline.session_id})
            return self.pipeline

    def stop(self) -> Pipeline | None:
        with self._lock:
            if self.pipeline is not None:
                self.pipeline.stop()
                log.info("source stopped", extra={"session": self.pipeline.session_id})
            return self.pipeline

    def shutdown(self) -> None:
        self.stop()
        self.store.close()
