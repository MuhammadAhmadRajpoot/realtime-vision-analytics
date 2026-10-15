"""Polygon zone monitor: occupancy, enter/exit events and dwell time."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rva.analytics.types import Event, EventType, TrackedObject
from rva.config import ZoneConfig

Point = tuple[float, float]


def point_in_polygon(p: Point, polygon: list[Point]) -> bool:
    """Even-odd ray casting test. Points exactly on an edge may go either way."""
    x, y = p
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_at_y = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_at_y:
                inside = not inside
    return inside


@dataclass(slots=True)
class _Visit:
    class_name: str
    entered_at: float
    last_inside: float


class ZoneMonitor:
    """Tracks which objects are inside a polygon and for how long.

    An object leaves the zone once it has not been observed *inside* for
    ``exit_timeout_s`` seconds, whether it walked out or the tracker lost it.
    This single rule debounces boundary jitter and short occlusions.
    """

    def __init__(self, cfg: ZoneConfig, exit_timeout_s: float = 1.5) -> None:
        self.cfg = cfg
        self.polygon: list[Point] = list(cfg.polygon)
        self.exit_timeout_s = exit_timeout_s
        self._visits: dict[int, _Visit] = {}
        self.total_visits = 0
        self._completed_dwell_sum = 0.0
        self._completed = 0
        self.max_dwell_s = 0.0

    @property
    def name(self) -> str:
        return self.cfg.name

    def contains(self, p: Point) -> bool:
        return point_in_polygon(p, self.polygon)

    def update(self, objects: list[TrackedObject], t: float) -> list[Event]:
        events: list[Event] = []
        for obj in objects:
            if not self.contains(obj.anchor):
                continue
            visit = self._visits.get(obj.track_id)
            if visit is None:
                self._visits[obj.track_id] = _Visit(obj.class_name, t, t)
                self.total_visits += 1
                events.append(self._event(EventType.ZONE_ENTER, obj.track_id, obj.class_name, t))
            else:
                visit.last_inside = t

        expired = [tid for tid, v in self._visits.items() if t - v.last_inside > self.exit_timeout_s]
        events.extend(self._close(tid) for tid in expired)
        return events

    def flush(self) -> list[Event]:
        """Close every open visit (used when the source restarts or stops)."""
        return [self._close(tid) for tid in list(self._visits)]

    def _close(self, track_id: int) -> Event:
        visit = self._visits.pop(track_id)
        dwell = visit.last_inside - visit.entered_at
        self._completed += 1
        self._completed_dwell_sum += dwell
        self.max_dwell_s = max(self.max_dwell_s, dwell)
        return self._event(EventType.ZONE_EXIT, track_id, visit.class_name, visit.last_inside, dwell)

    def _event(self, kind: EventType, track_id: int, class_name: str, t: float, dwell: float | None = None) -> Event:
        return Event(type=kind, name=self.name, track_id=track_id, class_name=class_name, video_t=t, dwell_s=dwell)

    @property
    def occupancy(self) -> int:
        return len(self._visits)

    def dwell_times(self, t: float) -> dict[int, float]:
        """Current dwell (seconds) for every object inside the zone."""
        return {tid: t - v.entered_at for tid, v in self._visits.items()}

    def snapshot(self, t: float) -> dict[str, object]:
        dwell = self.dwell_times(t)
        avg_completed = self._completed_dwell_sum / self._completed if self._completed else 0.0
        return {
            "name": self.name,
            "occupancy": self.occupancy,
            "by_class": dict(Counter(v.class_name for v in self._visits.values())),
            "total_visits": self.total_visits,
            "avg_dwell_s": round(avg_completed, 2),
            "max_dwell_s": round(max([self.max_dwell_s, *dwell.values()]), 2),
            "current_dwell_s": {str(k): round(v, 2) for k, v in dwell.items()},
        }
