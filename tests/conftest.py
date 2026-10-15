from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from fakes import FPS, N_FRAMES, H, W, blob_frame
from rva.config import PipelineConfig


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("video") / "blob.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"MJPG"), FPS, (W, H))
    assert writer.isOpened()
    for i in range(N_FRAMES):
        writer.write(blob_frame(i))
    writer.release()
    return path


@pytest.fixture
def scenario(synthetic_video: Path) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "source": {"uri": str(synthetic_video), "loop": False, "realtime": False},
            "processing": {"max_width": 640, "stats_interval_s": 0.01},
            "lines": [{"name": "middle", "start": [0.0, 0.5], "end": [1.0, 0.5]}],
            "zones": [{"name": "lower", "polygon": [[0.2, 0.6], [0.8, 0.6], [0.8, 0.99], [0.2, 0.99]]}],
            "zone_exit_timeout_s": 0.5,
        }
    )
