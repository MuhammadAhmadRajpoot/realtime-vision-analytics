from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest
from pydantic import ValidationError

from rva.analytics import Event, EventType
from rva.config import DetectorConfig, PipelineConfig, SourceConfig
from rva.storage import EventStore

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_are_valid(path: Path) -> None:
    cfg = PipelineConfig.from_yaml(path)
    assert cfg.detector.classes


def test_from_yaml_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text(
        "source: {uri: 0}\n"
        "detector: {classes: [person, car], confidence: 0.5}\n"
        "lines: [{name: a, start: [0, 0.5], end: [1, 0.5]}]\n",
        encoding="utf-8",
    )
    cfg = PipelineConfig.from_yaml(p)
    assert cfg.source.kind == "webcam"
    assert cfg.detector.class_ids == [0, 2]


@pytest.mark.parametrize(
    ("uri", "kind"),
    [(0, "webcam"), ("1", "webcam"), ("rtsp://cam/stream", "stream"), ("http://x/y.mjpg", "stream"), ("a.mp4", "file")],
)
def test_source_kind(uri: str | int, kind: str) -> None:
    assert SourceConfig(uri=uri).kind == kind


@pytest.mark.parametrize(
    "data",
    [
        {"detector": {"classes": ["unicorn"]}},
        {"detector": {"imgsz": 333}},
        {"zones": [{"name": "z", "polygon": [[0, 0], [1, 1]]}]},
        {"lines": [{"name": "l", "start": [0.5, 0.5], "end": [0.5, 0.5]}]},
        {"lines": [{"name": "l", "start": [0, 1.5], "end": [1, 1]}]},
        {"zones": [{"name": "z", "polygon": [[0, 0], [1, 0], [1, 1]]}] * 2},
        {"unexpected": True},
    ],
)
def test_invalid_configs_are_rejected(data: dict) -> None:
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(data)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PipelineConfig.from_yaml(tmp_path / "nope.yaml")


def test_detector_defaults() -> None:
    assert DetectorConfig().class_ids == [0]


def _event(kind: EventType, tid: int, **kw: object) -> Event:
    return Event(type=kind, name="x", track_id=tid, class_name="person", video_t=1.0, ts="2026-01-01T00:00:00", **kw)


def test_event_store_query_filter_and_csv(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "e.db")
    store.add("s1", [_event(EventType.LINE_CROSS, 1, direction="in"), _event(EventType.ZONE_ENTER, 2)])
    store.add("s2", [_event(EventType.ZONE_EXIT, 2, dwell_s=3.5)])
    assert store.count() == 3
    assert store.count(session_id="s1") == 2
    newest = store.query(limit=1)
    assert newest[0]["type"] == "zone_exit"
    assert newest[0]["dwell_s"] == 3.5
    assert [r["track_id"] for r in store.query(event_type="line_cross")] == [1]
    assert len(store.query(after_id=2)) == 1

    rows = list(csv.DictReader(io.StringIO(store.export_csv(session_id="s1"))))
    assert [r["type"] for r in rows] == ["line_cross", "zone_enter"]
    assert rows[0]["direction"] == "in"
    store.close()


def test_event_store_persists_across_connections(tmp_path: Path) -> None:
    db = tmp_path / "p.db"
    s = EventStore(db)
    s.add("s", [_event(EventType.ZONE_ENTER, 1)])
    s.close()
    assert EventStore(db).count() == 1
