"""Directional line-crossing counter with hysteresis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from rva.analytics.types import Event, EventType, TrackedObject
from rva.config import LineConfig

Point = tuple[float, float]


def _cross(a: Point, b: Point, p: Point) -> float:
    """Z component of (b - a) x (p - a). Positive = right-hand side in image coords."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """True if segment p1-p2 properly intersects (or touches) segment q1-q2."""
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    return d1 * d2 <= 0 and d3 * d4 <= 0


@dataclass(slots=True)
class _TrackState:
    side: int
    point: Point
    last_seen: float


class LineCounter:
    """Counts objects crossing a line segment, per class and direction.

    A track's side of the line only changes once its anchor is more than
    ``margin`` (normalised, perpendicular distance) away from the line, which
    suppresses double counts from box jitter around the line. A crossing is only
    counted when the movement actually passes through the segment, not its
    infinite extension.
    """

    def __init__(self, cfg: LineConfig, margin: float = 0.01, forget_after_s: float = 5.0) -> None:
        self.cfg = cfg
        self.start: Point = cfg.start
        self.end: Point = cfg.end
        self.margin = margin
        self.forget_after_s = forget_after_s
        self._length = ((self.end[0] - self.start[0]) ** 2 + (self.end[1] - self.start[1]) ** 2) ** 0.5
        self._tracks: dict[int, _TrackState] = {}
        self.counts: dict[str, dict[str, int]] = defaultdict(lambda: {"in": 0, "out": 0})

    @property
    def name(self) -> str:
        return self.cfg.name

    def _side(self, p: Point) -> int:
        dist = _cross(self.start, self.end, p) / self._length
        if dist > self.margin:
            return 1
        if dist < -self.margin:
            return -1
        return 0

    def update(self, objects: list[TrackedObject], t: float) -> list[Event]:
        events: list[Event] = []
        for obj in objects:
            side = self._side(obj.anchor)
            state = self._tracks.get(obj.track_id)
            if state is None:
                if side != 0:
                    self._tracks[obj.track_id] = _TrackState(side, obj.anchor, t)
                continue
            state.last_seen = t
            if side in (0, state.side):
                if side != 0:
                    state.point = obj.anchor
                continue
            if _segments_intersect(state.point, obj.anchor, self.start, self.end):
                direction = "in" if side > 0 else "out"
                self.counts[obj.class_name][direction] += 1
                events.append(
                    Event(
                        type=EventType.LINE_CROSS,
                        name=self.name,
                        track_id=obj.track_id,
                        class_name=obj.class_name,
                        video_t=t,
                        direction=direction,
                    )
                )
            state.side, state.point = side, obj.anchor

        stale = [tid for tid, s in self._tracks.items() if t - s.last_seen > self.forget_after_s]
        for tid in stale:
            del self._tracks[tid]
        return events

    def reset_tracks(self) -> None:
        """Forget per-track state (e.g. when the tracker restarts) but keep counts."""
        self._tracks.clear()

    def totals(self) -> dict[str, int]:
        return {
            "in": sum(c["in"] for c in self.counts.values()),
            "out": sum(c["out"] for c in self.counts.values()),
        }

    def snapshot(self) -> dict[str, object]:
        return {"name": self.name, **self.totals(), "by_class": {k: dict(v) for k, v in self.counts.items()}}
