"""Export Ultralytics YOLO weights to ONNX for faster CPU inference via onnxruntime."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from rva.config import DetectorConfig
from rva.detection import DetectorError, download_official_weights

log = logging.getLogger(__name__)


def export_onnx(weights: Path, imgsz: int = 640, output: Path | None = None) -> Path:
    """Export ``weights`` (.pt) to a static-shape ONNX graph and return its path.

    The default output name embeds the input size (``yolo11n_640.onnx``) because a
    static graph only accepts the size it was exported with.
    """
    from ultralytics import YOLO

    DetectorConfig(imgsz=imgsz)  # validates the size (multiple of 32)
    if not weights.exists():
        weights = download_official_weights(weights)
    output = output or weights.with_name(f"{weights.stem}_{imgsz}.onnx")
    if output.exists():
        log.info("onnx model already exists", extra={"path": str(output)})
        return output
    try:
        exported = Path(YOLO(str(weights)).export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True))
    except Exception as exc:
        raise DetectorError(f"ONNX export failed: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), output)
    log.info("onnx exported", extra={"path": str(output), "imgsz": imgsz})
    return output
