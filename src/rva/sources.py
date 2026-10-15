"""Frame sources: video files, webcams and RTSP/HTTP streams behind one API."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from rva.config import SourceConfig

log = logging.getLogger(__name__)


class SourceError(RuntimeError):
    """Raised when a source cannot be opened."""


@dataclass(slots=True)
class Frame:
    image: np.ndarray
    t: float
    """Seconds on the source clock: media time for files, wall time since open for live."""
    restarted: bool = False
    """True for the first frame after a file loop or stream reconnect."""


class VideoSource:
    """Wraps :class:`cv2.VideoCapture` with looping, real-time pacing and reconnects.

    ``frame_skip`` drops N frames between returned frames using ``grab()`` (no
    decode cost). With ``realtime`` enabled, files are paced at their native FPS
    and frames are dropped when the consumer falls behind, so a file behaves like
    a live camera.
    """

    def __init__(self, cfg: SourceConfig, frame_skip: int = 0) -> None:
        self.cfg = cfg
        self.frame_skip = frame_skip
        self._cap: cv2.VideoCapture | None = None
        self.fps = 25.0
        self._opened_at = 0.0
        self._loop_offset = 0.0
        self._pending_restart = False

    @property
    def is_file(self) -> bool:
        return self.cfg.kind == "file"

    @property
    def effective_fps(self) -> float:
        return self.fps / (self.frame_skip + 1)

    def open(self) -> None:
        uri: str | int = int(self.cfg.uri) if self.cfg.kind == "webcam" else self.cfg.uri
        cap = cv2.VideoCapture(uri)
        if not cap.isOpened():
            raise SourceError(f"cannot open source {self.cfg.uri!r}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if 1.0 <= fps <= 240.0 else 25.0
        self._cap = cap
        self._opened_at = time.monotonic()
        log.info("source opened", extra={"uri": str(self.cfg.uri), "kind": self.cfg.kind, "fps": self.fps})

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> VideoSource:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _media_t(self) -> float:
        assert self._cap is not None
        return self._cap.get(cv2.CAP_PROP_POS_FRAMES) / self.fps

    def _pace(self) -> None:
        """Real-time file playback: sleep if early, skip frames if late."""
        assert self._cap is not None
        target = (time.monotonic() - self._opened_at) * self.fps
        pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        if pos > target:
            time.sleep((pos - target) / self.fps)
            return
        behind = int(target - pos)
        for _ in range(min(behind, int(self.fps))):  # bounded catch-up
            if not self._cap.grab():
                break

    def read(self, stop: threading.Event | None = None) -> Frame | None:
        """Return the next frame, or ``None`` when a non-looping file ended or ``stop`` is set.

        Live sources reconnect (every ``reconnect_delay_s``) until they recover.
        """
        if self._cap is None:
            raise SourceError("source is not open")
        if self.is_file and self.cfg.realtime:
            self._pace()
        for _ in range(self.frame_skip):
            self._cap.grab()
        ok, image = self._cap.read()
        if not ok:
            if not self._recover(stop):
                return None
            ok, image = self._cap.read()
            if not ok:
                raise SourceError(f"source {self.cfg.uri!r} produced no frames")
        restarted, self._pending_restart = self._pending_restart, False
        if self.is_file:
            t = self._loop_offset + self._media_t()
        else:
            t = time.monotonic() - self._opened_at + self._loop_offset
        return Frame(image=image, t=t, restarted=restarted)

    def _recover(self, stop: threading.Event | None) -> bool:
        """Loop a file or reconnect a stream. Returns False if reading should end."""
        assert self._cap is not None
        if self.is_file:
            if not self.cfg.loop:
                return False
            self._loop_offset += self._media_t() + 1.0 / self.fps
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._opened_at = time.monotonic()
            self._pending_restart = True
            log.info("source looped", extra={"uri": str(self.cfg.uri)})
            return True
        # Live source: keep the clock monotonic across reconnects.
        self._loop_offset += time.monotonic() - self._opened_at
        while stop is None or not stop.is_set():
            log.warning("stream read failed, reconnecting", extra={"uri": str(self.cfg.uri)})
            self.close()
            time.sleep(self.cfg.reconnect_delay_s)
            try:
                self.open()
            except SourceError as exc:
                log.warning("reconnect failed", extra={"error": str(exc)})
                continue
            self._pending_restart = True
            return True
        return False


def resize_to_width(image: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA)
