"""Combines line counters, zone monitors and the heatmap behind one interface."""

from __future__ import annotations

from typing import Any

from rva.analytics.heatmap import Heatmap
from rva.analytics.lines import LineCounter
from rva.analytics.types import Event, TrackedObject
from rva.analytics.zones import ZoneMonitor
from rva.config import PipelineConfig


class AnalyticsEngine:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.lines = [LineCounter(lc) for lc in cfg.lines]
        self.zones = [ZoneMonitor(zc, exit_timeout_s=cfg.zone_exit_timeout_s) for zc in cfg.zones]
        self.heatmap = Heatmap(decay=cfg.processing.heatmap_decay)
        self.t = 0.0
        self.visible: dict[str, int] = {}

    def update(self, objects: list[TrackedObject], t: float) -> list[Event]:
        self.t = t
        visible: dict[str, int] = {}
        for o in objects:
            visible[o.class_name] = visible.get(o.class_name, 0) + 1
        self.visible = visible
        if self.cfg.processing.heatmap:
            self.heatmap.update([o.anchor for o in objects])
        events: list[Event] = []
        for line in self.lines:
            events.extend(line.update(objects, t))
        for zone in self.zones:
            events.extend(zone.update(objects, t))
        return events

    def restart_tracks(self) -> list[Event]:
        """Called when the tracker is reset (source loop/reconnect): close open state."""
        for line in self.lines:
            line.reset_tracks()
        events: list[Event] = []
        for zone in self.zones:
            events.extend(zone.flush())
        return events

    def zone_dwell(self) -> dict[int, float]:
        """Longest current dwell per track across all zones (for on-frame labels)."""
        out: dict[int, float] = {}
        for zone in self.zones:
            for tid, d in zone.dwell_times(self.t).items():
                out[tid] = max(d, out.get(tid, 0.0))
        return out

    def snapshot(self) -> dict[str, Any]:
        line_totals = [line.totals() for line in self.lines]
        return {
            "video_t": round(self.t, 2),
            "totals": {
                "in": sum(lt["in"] for lt in line_totals),
                "out": sum(lt["out"] for lt in line_totals),
                "occupancy": sum(zone.occupancy for zone in self.zones),
                "visible": sum(self.visible.values()),
            },
            "visible_by_class": dict(self.visible),
            "lines": [line.snapshot() for line in self.lines],
            "zones": [zone.snapshot(self.t) for zone in self.zones],
        }
