from __future__ import annotations

import numpy as np
import pytest

from rva.analytics import AnalyticsEngine, EventType, Heatmap, LineCounter, TrackedObject, ZoneMonitor, point_in_polygon
from rva.config import LineConfig, PipelineConfig, ZoneConfig

HLINE = LineConfig(name="door", start=(0.2, 0.5), end=(0.8, 0.5))
SQUARE = ZoneConfig(name="shelf", polygon=[(0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5)])


def obj(tid: int, x: float, y: float, cls: str = "person") -> TrackedObject:
    return TrackedObject(track_id=tid, class_name=cls, anchor=(x, y))


def walk(counter: LineCounter | ZoneMonitor, tid: int, ys: list[float], x: float = 0.5, cls: str = "person"):
    events = []
    for i, y in enumerate(ys):
        events += counter.update([obj(tid, x, y, cls)], t=float(i))
    return events


class TestLineCounter:
    def test_downward_crossing_counts_in(self) -> None:
        lc = LineCounter(HLINE)
        events = walk(lc, 1, [0.3, 0.4, 0.6, 0.7])
        assert [e.direction for e in events] == ["in"]
        assert lc.totals() == {"in": 1, "out": 0}

    def test_upward_crossing_counts_out_per_class(self) -> None:
        lc = LineCounter(HLINE)
        walk(lc, 1, [0.7, 0.3], cls="car")
        walk(lc, 2, [0.3, 0.7], cls="person")
        assert lc.counts["car"] == {"in": 0, "out": 1}
        assert lc.counts["person"] == {"in": 1, "out": 0}

    def test_jitter_inside_margin_is_ignored(self) -> None:
        lc = LineCounter(HLINE, margin=0.02)
        events = walk(lc, 1, [0.4, 0.49, 0.51, 0.49, 0.51, 0.495])
        assert events == []

    def test_crossing_outside_segment_extent_is_not_counted(self) -> None:
        lc = LineCounter(HLINE)
        assert walk(lc, 1, [0.3, 0.7], x=0.95) == []

    def test_round_trip_counts_both_directions(self) -> None:
        lc = LineCounter(HLINE)
        walk(lc, 1, [0.3, 0.7, 0.3])
        assert lc.totals() == {"in": 1, "out": 1}

    def test_reset_tracks_keeps_counts_but_forgets_sides(self) -> None:
        lc = LineCounter(HLINE)
        walk(lc, 1, [0.3, 0.7])
        lc.reset_tracks()
        # Same id reappearing on the other side after a reset must not count.
        assert lc.update([obj(1, 0.5, 0.3)], t=10.0) == []
        assert lc.totals()["in"] == 1


class TestZoneMonitor:
    def test_point_in_polygon(self) -> None:
        poly = [(0, 0), (1, 0), (1, 1), (0, 1)]
        assert point_in_polygon((0.5, 0.5), poly)
        assert not point_in_polygon((1.5, 0.5), poly)
        concave = [(0, 0), (1, 0), (1, 1), (0.5, 0.4), (0, 1)]
        assert not point_in_polygon((0.5, 0.8), concave)

    def test_enter_dwell_exit(self) -> None:
        zm = ZoneMonitor(SQUARE, exit_timeout_s=1.0)
        events = []
        for t in range(6):  # inside for t=0..5
            events += zm.update([obj(7, 0.3, 0.3)], t=float(t))
        assert zm.occupancy == 1
        assert zm.dwell_times(5.0) == {7: 5.0}
        events += zm.update([obj(7, 0.9, 0.9)], t=6.0)  # outside, still within timeout
        assert zm.occupancy == 1
        events += zm.update([], t=7.0)  # 2 s since last seen inside
        types = [e.type for e in events]
        assert types == [EventType.ZONE_ENTER, EventType.ZONE_EXIT]
        assert events[-1].dwell_s == pytest.approx(5.0)
        assert zm.occupancy == 0
        snap = zm.snapshot(7.0)
        assert snap["total_visits"] == 1
        assert snap["avg_dwell_s"] == pytest.approx(5.0)

    def test_short_occlusion_does_not_split_visit(self) -> None:
        zm = ZoneMonitor(SQUARE, exit_timeout_s=1.5)
        events = zm.update([obj(1, 0.3, 0.3)], t=0.0)
        events += zm.update([], t=1.0)
        events += zm.update([obj(1, 0.3, 0.3)], t=2.0)
        assert [e.type for e in events] == [EventType.ZONE_ENTER]
        assert zm.total_visits == 1

    def test_flush_closes_all_visits(self) -> None:
        zm = ZoneMonitor(SQUARE)
        zm.update([obj(1, 0.2, 0.2), obj(2, 0.3, 0.3, "dog")], t=0.0)
        assert zm.snapshot(0.0)["by_class"] == {"person": 1, "dog": 1}
        exits = zm.flush()
        assert {e.track_id for e in exits} == {1, 2}
        assert zm.occupancy == 0


class TestHeatmap:
    def test_accumulates_and_renders(self) -> None:
        hm = Heatmap(grid=(20, 20), sigma=1.0)
        hm.update([(0.5, 0.5), (0.5, 0.5), (0.0, 1.0)])
        peak = np.unravel_index(hm.acc.argmax(), hm.acc.shape)
        assert peak == (9, 9)
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        out = hm.render(frame)
        assert out.shape == frame.shape
        assert out.sum() > 0

    def test_decay_and_empty_render(self) -> None:
        hm = Heatmap(grid=(10, 10), decay=0.5)
        frame = np.full((10, 10, 3), 7, dtype=np.uint8)
        assert hm.render(frame) is frame
        hm.update([(0.5, 0.5)])
        first = hm.acc.max()
        hm.update([])
        assert hm.acc.max() == pytest.approx(first * 0.5)


def test_engine_snapshot_totals() -> None:
    cfg = PipelineConfig(lines=[HLINE], zones=[SQUARE])
    eng = AnalyticsEngine(cfg)
    eng.update([obj(1, 0.3, 0.3), obj(2, 0.45, 0.45)], t=0.0)
    events = eng.update([obj(1, 0.3, 0.3), obj(2, 0.45, 0.6)], t=1.0)
    assert [e.type for e in events] == [EventType.LINE_CROSS]
    snap = eng.snapshot()
    assert snap["totals"] == {"in": 1, "out": 0, "occupancy": 2, "visible": 2}
    # Track 2 just left the square but is still within the exit timeout; a restart closes both.
    closed = eng.restart_tracks()
    assert {e.track_id for e in closed} == {1, 2}
