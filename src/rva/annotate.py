"""Draws detections, tracks, lines, zones and the heatmap onto frames."""

from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from rva.analytics.engine import AnalyticsEngine

# BGR colours.
_BOX = (229, 135, 57)  # blue
_LINE = (38, 89, 217)  # orange
_ZONE = (112, 158, 25)  # aqua-green
_TEXT_BG = (25, 26, 26)
_WHITE = (240, 239, 236)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _px(p: tuple[float, float], w: int, h: int) -> tuple[int, int]:
    return round(p[0] * (w - 1)), round(p[1] * (h - 1))


def _label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float = 0.45) -> None:
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = org
    x = min(max(x, 0), img.shape[1] - tw - 6)  # keep labels inside the frame
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + base - 1), color, -1)
    cv2.putText(img, text, (x + 3, y - 2), _FONT, scale, _WHITE, 1, cv2.LINE_AA)


class Annotator:
    def __init__(self, heatmap_alpha: float = 0.45, draw_heatmap: bool = True) -> None:
        self.heatmap_alpha = heatmap_alpha
        self.draw_heatmap = draw_heatmap

    def draw(self, frame: np.ndarray, tracked: sv.Detections, engine: AnalyticsEngine, fps: float) -> np.ndarray:
        h, w = frame.shape[:2]
        out = engine.heatmap.render(frame, self.heatmap_alpha) if self.draw_heatmap else frame.copy()
        self._zones(out, engine, w, h)
        self._lines(out, engine, w, h)
        self._boxes(out, tracked, engine.zone_dwell())
        self._hud(out, engine, fps)
        return out

    def _zones(self, img: np.ndarray, engine: AnalyticsEngine, w: int, h: int) -> None:
        overlay = img.copy()
        for zone in engine.zones:
            pts = np.array([_px(p, w, h) for p in zone.polygon], dtype=np.int32)
            cv2.fillPoly(overlay, [pts], _ZONE)
        cv2.addWeighted(overlay, 0.18, img, 0.82, 0, dst=img)
        for zone in engine.zones:
            pts = np.array([_px(p, w, h) for p in zone.polygon], dtype=np.int32)
            cv2.polylines(img, [pts], True, _ZONE, 2, cv2.LINE_AA)
            x, y = pts.min(axis=0)
            _label(img, f"{zone.name}: {zone.occupancy} inside", (int(x) + 4, int(y) + 18), _ZONE)

    def _lines(self, img: np.ndarray, engine: AnalyticsEngine, w: int, h: int) -> None:
        for line in engine.lines:
            a, b = _px(line.start, w, h), _px(line.end, w, h)
            cv2.line(img, a, b, _LINE, 2, cv2.LINE_AA)
            cv2.circle(img, a, 4, _LINE, -1)
            cv2.circle(img, b, 4, _LINE, -1)
            totals = line.totals()
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2 - 8)
            _label(img, f"{line.name}  in {totals['in']}  out {totals['out']}", mid, _LINE, 0.5)

    def _boxes(self, img: np.ndarray, tracked: sv.Detections, dwell: dict[int, float]) -> None:
        if len(tracked) == 0 or tracked.tracker_id is None:
            return
        names = tracked.data.get("class_name")
        for i, (x1, y1, x2, y2) in enumerate(tracked.xyxy.astype(int)):
            tid = int(tracked.tracker_id[i])
            cv2.rectangle(img, (x1, y1), (x2, y2), _BOX, 2)
            text = f"#{tid} {names[i] if names is not None else ''}".strip()
            if tid in dwell:
                text += f" {dwell[tid]:.1f}s"
            _label(img, text, (x1, y1 - 2), _BOX, 0.4)
            cv2.circle(img, ((x1 + x2) // 2, y2), 3, _WHITE, -1)

    def _hud(self, img: np.ndarray, engine: AnalyticsEngine, fps: float) -> None:
        snap = engine.snapshot()["totals"]
        text = f"FPS {fps:4.1f} | visible {snap['visible']} | in {snap['in']} out {snap['out']}"
        _label(img, text, (8, 22), _TEXT_BG, 0.5)
