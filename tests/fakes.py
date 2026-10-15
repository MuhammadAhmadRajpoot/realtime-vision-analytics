"""Deterministic test doubles: a synthetic video and a blob 'detector'."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

W, H, FPS, N_FRAMES = 320, 240, 20, 60


class BlobDetector:
    """Deterministic stand-in for YOLO: every bright blob is a 'person'."""

    def __init__(self, *_: object) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> sv.Detections:
        self.calls += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 30]
        if not boxes:
            return sv.Detections.empty()
        xyxy = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], dtype=np.float32)
        return sv.Detections(
            xyxy=xyxy,
            confidence=np.full(len(boxes), 0.9, dtype=np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
            data={"class_name": np.array(["person"] * len(boxes))},
        )


def blob_frame(i: int) -> np.ndarray:
    """A 30x30 white square moving straight down from y=10 to y~220 over N_FRAMES."""
    img = np.full((H, W, 3), 40, dtype=np.uint8)
    y = 10 + int(i * (H - 50) / (N_FRAMES - 1))
    cv2.rectangle(img, (145, y), (175, y + 30), (255, 255, 255), -1)
    return img
