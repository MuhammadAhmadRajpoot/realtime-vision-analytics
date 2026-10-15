"""Tests that run the real YOLO11n model on CPU.

The weights (~5.4 MB) are downloaded to ``models/`` on first run; afterwards the
test is fully offline. Skipped automatically if the weights are unavailable.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from rva.config import DetectorConfig, PipelineConfig
from rva.detection import DetectorError, YoloDetector
from rva.pipeline import FrameProcessor
from rva.sources import Frame

pytestmark = pytest.mark.model

WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "yolo11n.pt"


@pytest.fixture(scope="module")
def detector() -> YoloDetector:
    try:
        return YoloDetector(DetectorConfig(model=str(WEIGHTS), classes=["person", "bus"], confidence=0.4))
    except DetectorError as exc:
        pytest.skip(f"YOLO weights unavailable: {exc}")


@pytest.fixture(scope="module")
def bus_image():
    from ultralytics.utils import ASSETS

    img = cv2.imread(str(ASSETS / "bus.jpg"))  # sample image bundled with ultralytics
    assert img is not None
    return img


def test_yolo_detects_people_and_bus(detector: YoloDetector, bus_image) -> None:
    dets = detector.detect(bus_image)
    names = list(dets.data["class_name"])
    assert names.count("person") >= 3
    assert "bus" in names
    assert set(names) <= {"person", "bus"}, "class filter is applied"


def test_real_model_through_processor(detector: YoloDetector, bus_image) -> None:
    cfg = PipelineConfig.model_validate(
        {"zones": [{"name": "all", "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}], "processing": {"max_width": 640}}
    )
    proc = FrameProcessor(cfg, detector, frame_rate=10)
    for i in range(3):  # ByteTrack confirms tracks after consecutive hits
        result = proc.process(Frame(image=bus_image, t=i / 10))
    assert result.tracked.tracker_id is not None
    assert len(result.tracked) >= 3
    assert proc.engine.zones[0].occupancy >= 3
    assert result.annotated.shape[1] == 640
