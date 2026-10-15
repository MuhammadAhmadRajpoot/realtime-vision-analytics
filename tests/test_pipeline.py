from __future__ import annotations

from pathlib import Path

import numpy as np

from fakes import N_FRAMES, BlobDetector, blob_frame
from rva.analytics import EventType
from rva.config import PipelineConfig, SourceConfig
from rva.pipeline import FpsMeter, FrameProcessor, Pipeline, Status
from rva.sources import Frame, VideoSource, resize_to_width
from rva.storage import EventStore


def test_video_source_reads_file_and_skips(synthetic_video: Path) -> None:
    cfg = SourceConfig(uri=str(synthetic_video), loop=False, realtime=False)
    with VideoSource(cfg) as src:
        frames = list(iter(src.read, None))
    assert len(frames) == N_FRAMES
    assert frames[1].t > frames[0].t
    with VideoSource(cfg, frame_skip=2) as src:
        assert len(list(iter(src.read, None))) == N_FRAMES // 3


def test_video_source_loops(synthetic_video: Path) -> None:
    cfg = SourceConfig(uri=str(synthetic_video), loop=True, realtime=False)
    with VideoSource(cfg) as src:
        frames = [src.read() for _ in range(N_FRAMES + 5)]
    assert all(f is not None for f in frames)
    restarted = [i for i, f in enumerate(frames) if f and f.restarted]
    assert restarted == [N_FRAMES]
    times = [f.t for f in frames if f]
    assert times == sorted(times), "clock must stay monotonic across loops"


def test_resize_to_width() -> None:
    img = np.zeros((100, 400, 3), dtype=np.uint8)
    assert resize_to_width(img, 200).shape == (50, 200, 3)
    assert resize_to_width(img, 800) is img


def test_fps_meter() -> None:
    m = FpsMeter(smoothing=0.0)
    m.tick(0.0)
    assert m.tick(0.1) == 10.0


def test_frame_processor_counts_blob(scenario: PipelineConfig) -> None:
    proc = FrameProcessor(scenario, BlobDetector(), frame_rate=20)
    events = []
    for i in range(N_FRAMES):
        result = proc.process(Frame(image=blob_frame(i), t=i / 20))
        events += result.events
        assert result.annotated.shape == (240, 320, 3)
    events += proc.engine.restart_tracks()
    kinds = [(e.type, e.direction) for e in events]
    assert kinds == [(EventType.LINE_CROSS, "in"), (EventType.ZONE_ENTER, None), (EventType.ZONE_EXIT, None)]
    assert all(e.ts for e in events[:2])
    assert len({e.track_id for e in events}) == 1, "the blob keeps one track id"
    assert events[-1].dwell_s > 0.5
    assert proc.engine.heatmap.acc.max() > 0


def test_pipeline_thread_end_to_end(scenario: PipelineConfig, tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")
    pipe = Pipeline(scenario, BlobDetector, store)
    pipe.start()
    pipe.wait(timeout=30)
    assert pipe.status == Status.FINISHED, pipe.error
    assert pipe.frames_processed == N_FRAMES
    seq, jpeg = pipe.latest_jpeg()
    assert seq == N_FRAMES
    assert jpeg is not None
    assert jpeg[:2] == b"\xff\xd8"
    stats = pipe.stats()
    assert stats["totals"]["in"] == 1
    assert stats["zones"][0]["total_visits"] == 1
    assert pipe.history(), "history is sampled"
    rows = store.query(session_id=pipe.session_id)
    assert sorted(r["type"] for r in rows) == ["line_cross", "zone_enter", "zone_exit"]


def test_pipeline_reports_errors(tmp_path: Path) -> None:
    cfg = PipelineConfig(source=SourceConfig(uri=str(tmp_path / "missing.mp4"), loop=False))
    pipe = Pipeline(cfg, BlobDetector, EventStore(":memory:"))
    pipe.start()
    pipe.wait(timeout=10)
    assert pipe.status == Status.ERROR
    assert "cannot open source" in (pipe.error or "")
