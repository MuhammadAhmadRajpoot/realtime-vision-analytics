"""Object detection and multi-object tracking.

The :class:`Detector` protocol is the seam used by tests: anything that turns a
BGR frame into :class:`supervision.Detections` (with a ``class_name`` data field)
can drive the pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np
import supervision as sv

from rva.analytics.types import TrackedObject
from rva.config import DetectorConfig, TrackerConfig

log = logging.getLogger(__name__)


class DetectorError(RuntimeError):
    """Raised when the model cannot be loaded or run."""


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> sv.Detections: ...


class YoloDetector:
    """Ultralytics YOLO detector. Accepts PyTorch (``.pt``) or ONNX (``.onnx``) weights.

    Official weights (e.g. ``yolo11n.pt``) are downloaded on first use into the
    directory of the configured path.
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        from ultralytics import YOLO  # deferred: heavy import (torch)

        self.cfg = cfg
        path = Path(cfg.model)
        if not path.exists():
            path = download_official_weights(path)
        try:
            self.model = YOLO(str(path), task="detect")
        except Exception as exc:  # ultralytics raises a variety of types
            raise DetectorError(f"failed to load model {path}: {exc}") from exc
        self.backend = "onnx" if path.suffix == ".onnx" else "pytorch"
        # The first inference is much slower (lazy init); pay it before frames arrive
        # so real-time sources do not drop their first seconds.
        self.detect(np.zeros((cfg.imgsz, cfg.imgsz, 3), dtype=np.uint8))
        log.info("detector loaded", extra={"model": str(path), "backend": self.backend, "imgsz": cfg.imgsz})

    def detect(self, frame: np.ndarray) -> sv.Detections:
        result = self.model.predict(
            frame,
            imgsz=self.cfg.imgsz,
            conf=self.cfg.confidence,
            iou=self.cfg.iou,
            classes=self.cfg.class_ids,
            device=self.cfg.device,
            verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(result)


def download_official_weights(path: Path) -> Path:
    """Download an official Ultralytics asset (e.g. ``yolo11n.pt``) into ``path``."""
    if path.suffix != ".pt":
        raise DetectorError(f"model not found: {path} (export ONNX first with `rva export-onnx`)")
    from ultralytics.utils.downloads import attempt_download_asset

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading weights", extra={"model": path.name})
    try:
        downloaded = Path(attempt_download_asset(str(path)))
    except Exception as exc:
        raise DetectorError(f"could not download {path.name}: {exc}") from exc
    if not downloaded.exists():
        raise DetectorError(f"weights not found and not an official asset: {path}")
    return downloaded


class Tracker:
    """ByteTrack (Roboflow ``trackers`` package) driven by frame timestamps.

    Passing timestamps (instead of assuming a fixed frame step) keeps the Kalman
    motion model and the lost-track budget correct when frames are dropped for
    real-time pacing or ``frame_skip``. Detections that are not (yet) confirmed
    tracks are dropped, so analytics only ever see stable track ids.
    """

    def __init__(self, cfg: TrackerConfig, frame_rate: float) -> None:
        self.cfg = cfg
        self.frame_rate = frame_rate
        self.reset()

    def reset(self) -> None:
        from trackers import ByteTrackTracker

        self._tracker = ByteTrackTracker(
            lost_track_buffer=self.cfg.lost_track_buffer,
            frame_rate=self.frame_rate,
            track_activation_threshold=self.cfg.track_activation_threshold,
            high_conf_det_threshold=self.cfg.high_conf_det_threshold,
            minimum_consecutive_frames=self.cfg.minimum_consecutive_frames,
            minimum_iou_threshold=self.cfg.minimum_iou_threshold,
        )

    def update(self, detections: sv.Detections, t: float | None = None) -> sv.Detections:
        tracked = self._tracker.update(detections, timestamp=t)
        if tracked.tracker_id is None:
            return sv.Detections.empty()
        return tracked[tracked.tracker_id >= 0]


def to_tracked_objects(tracked: sv.Detections, frame_shape: tuple[int, ...]) -> list[TrackedObject]:
    """Convert tracked detections to normalised analytics objects (bottom-centre anchors)."""
    if len(tracked) == 0 or tracked.tracker_id is None:
        return []
    h, w = frame_shape[:2]
    anchors = tracked.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    names = tracked.data.get("class_name")
    confs = tracked.confidence if tracked.confidence is not None else np.ones(len(tracked))
    objects = []
    for i, (x, y) in enumerate(anchors):
        name = str(names[i]) if names is not None else str(int(tracked.class_id[i]))
        objects.append(
            TrackedObject(
                track_id=int(tracked.tracker_id[i]),
                class_name=name,
                anchor=(float(x) / w, float(y) / h),
                confidence=float(confs[i]),
            )
        )
    return objects
