"""Cumulative presence heatmap rendered as a colour overlay."""

from __future__ import annotations

import cv2
import numpy as np


class Heatmap:
    """Accumulates object anchor positions on a low-resolution grid.

    Each update adds a Gaussian blob per object; an optional per-frame decay lets
    old activity fade. The grid is resized to the frame only when rendering.
    """

    def __init__(self, grid: tuple[int, int] = (96, 160), sigma: float = 2.5, decay: float = 1.0) -> None:
        self.rows, self.cols = grid
        self.decay = decay
        self.acc = np.zeros(grid, dtype=np.float32)
        radius = max(1, int(3 * sigma))
        k = cv2.getGaussianKernel(2 * radius + 1, sigma)
        self._kernel = (k @ k.T).astype(np.float32)
        self._radius = radius

    def update(self, anchors: list[tuple[float, float]]) -> None:
        if self.decay < 1.0:
            self.acc *= self.decay
        r = self._radius
        for x, y in anchors:
            cx = int(np.clip(x, 0, 1) * (self.cols - 1))
            cy = int(np.clip(y, 0, 1) * (self.rows - 1))
            y0, y1 = max(cy - r, 0), min(cy + r + 1, self.rows)
            x0, x1 = max(cx - r, 0), min(cx + r + 1, self.cols)
            self.acc[y0:y1, x0:x1] += self._kernel[y0 - cy + r : y1 - cy + r, x0 - cx + r : x1 - cx + r]

    def reset(self) -> None:
        self.acc.fill(0.0)

    def render(self, frame: np.ndarray, alpha: float = 0.45, threshold: float = 0.15) -> np.ndarray:
        """Blend the heatmap onto ``frame`` (BGR). Cold cells stay untouched."""
        peak = float(self.acc.max())
        if peak <= 0:
            return frame
        h, w = frame.shape[:2]
        # sqrt compresses the dynamic range so walking paths stay visible next to hot spots.
        norm = cv2.resize(np.sqrt(self.acc / peak), (w, h), interpolation=cv2.INTER_LINEAR)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        weight = np.clip((norm - threshold) * (alpha / (1 - threshold)), 0, alpha).astype(np.float32)
        return cv2.blendLinear(frame, colored, 1.0 - weight, weight)
