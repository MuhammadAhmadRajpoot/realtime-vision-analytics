from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import N_FRAMES, BlobDetector
from rva.api.app import create_app, mjpeg_frames
from rva.config import AppSettings, PipelineConfig


@pytest.fixture
def client(scenario: PipelineConfig, tmp_path: Path) -> Iterator[TestClient]:
    settings = AppSettings(db_path=tmp_path / "api.db", autostart=False)
    app = create_app(settings, scenario, detector_factory=BlobDetector)
    with TestClient(app) as c:
        yield c


def _wait_finished(client: TestClient, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        stats = client.get("/api/stats").json()
        if stats.get("status") in ("finished", "error"):
            return stats
        time.sleep(0.05)
    raise AssertionError("pipeline did not finish")


def test_idle_state(client: TestClient) -> None:
    assert client.get("/api/health").json()["pipeline"] == "idle"
    assert client.get("/api/stats").json() == {"status": "idle"}
    assert client.get("/api/history").json() == []
    assert client.get("/api/snapshot.jpg").status_code == 404
    assert client.get("/").headers["content-type"].startswith("text/html")
    assert client.get("/static/app.js").status_code == 200


def test_config_endpoint(client: TestClient) -> None:
    cfg = client.get("/api/config").json()
    assert cfg["lines"][0]["name"] == "middle"
    assert cfg["zones"][0]["name"] == "lower"


def test_run_source_and_read_results(client: TestClient) -> None:
    started = client.post("/api/source/start").json()
    assert started["status"] in ("starting", "running", "finished")
    stats = _wait_finished(client)
    assert stats["status"] == "finished", stats
    assert stats["frames_processed"] == N_FRAMES
    assert stats["totals"]["in"] == 1

    events = client.get("/api/events", params={"session_id": started["session_id"]}).json()
    assert sorted(e["type"] for e in events) == ["line_cross", "zone_enter", "zone_exit"]
    crossings = client.get("/api/events", params={"type": "line_cross"}).json()
    assert [e["direction"] for e in crossings] == ["in"]
    assert client.get("/api/events", params={"type": "bogus"}).status_code == 422

    csv_resp = client.get("/api/events/export.csv")
    assert csv_resp.headers["content-type"].startswith("text/csv")
    assert csv_resp.text.splitlines()[0].startswith("id,session_id,ts")
    assert len(csv_resp.text.splitlines()) == 4

    assert client.get("/api/history").json()
    snap = client.get("/api/snapshot.jpg")
    assert snap.status_code == 200
    assert snap.content[:2] == b"\xff\xd8"


def test_start_with_new_source_and_stop(client: TestClient, synthetic_video: Path) -> None:
    r = client.post("/api/source/start", json={"uri": str(synthetic_video), "loop": True, "realtime": True})
    assert r.status_code == 200
    first = r.json()["session_id"]
    assert client.get("/api/config").json()["source"]["loop"] is True
    stopped = client.post("/api/source/stop").json()
    assert stopped["session_id"] == first
    assert stopped["status"] in ("stopped", "finished")


def test_start_rejects_bad_source(client: TestClient) -> None:
    assert client.post("/api/source/start", json={"uri": {"x": 1}}).status_code == 422


def test_websocket_pushes_stats(client: TestClient) -> None:
    client.post("/api/source/start")
    _wait_finished(client)
    with client.websocket_connect("/ws/stats?interval=0.1") as ws:
        msg = ws.receive_json()
        assert msg["status"] == "finished"
        assert msg["totals"]["in"] == 1


def test_mjpeg_generator_yields_multipart_jpeg(client: TestClient) -> None:
    client.post("/api/source/start")
    _wait_finished(client)
    service = client.app.state.service

    async def first_part() -> bytes:
        gen = mjpeg_frames(service)
        part = await asyncio.wait_for(anext(gen), timeout=5)
        await gen.aclose()
        return part

    part = asyncio.run(first_part())
    assert part.startswith(b"--frame\r\nContent-Type: image/jpeg")
    assert b"\xff\xd8" in part
