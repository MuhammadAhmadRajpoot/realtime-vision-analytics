"""Plain data types shared by the analytics components.

Analytics operate on normalised coordinates so they are independent of the
detector, the tracker library and the processing resolution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


@dataclass(frozen=True, slots=True)
class TrackedObject:
    track_id: int
    class_name: str
    anchor: tuple[float, float]
    """Normalised bottom-centre of the box (where the object touches the ground)."""
    confidence: float = 1.0


class EventType(StrEnum):
    LINE_CROSS = "line_cross"
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"


@dataclass(slots=True)
class Event:
    type: EventType
    name: str
    """Name of the line or zone that produced the event."""
    track_id: int
    class_name: str
    video_t: float
    """Seconds on the pipeline clock (media time for files, wall time for live)."""
    direction: str | None = None
    dwell_s: float | None = None
    ts: str = field(default="")
    """ISO-8601 wall-clock timestamp, filled in by the pipeline."""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["type"] = str(self.type)
        return d
