"""Configuration models.

Two layers:

* :class:`PipelineConfig` - the analytics scenario (source, detector, lines, zones),
  loaded from a YAML file.
* :class:`AppSettings` - process-level settings (paths, host, port), read from
  environment variables prefixed with ``RVA_`` or a ``.env`` file.

All geometry (lines and zone polygons) is expressed in *normalised* coordinates
in ``[0, 1]`` relative to the frame width/height, so a scenario keeps working when
the processing resolution changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

NormCoord = Annotated[float, Field(ge=0.0, le=1.0)]
NormPoint = tuple[NormCoord, NormCoord]

# COCO class names used by the pretrained YOLO models.
COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
)  # fmt: skip


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceConfig(_Strict):
    """Where frames come from.

    ``uri`` may be a webcam index (``0``), a file path, or an RTSP/HTTP URL.
    """

    uri: str | int = "data/people-detection.mp4"
    loop: bool = Field(True, description="Restart video files when they end.")
    realtime: bool = Field(
        True,
        description="Pace video files at their native FPS, dropping frames when "
        "processing is slower (emulates a live camera).",
    )
    reconnect_delay_s: float = Field(2.0, gt=0)

    @property
    def kind(self) -> Literal["webcam", "stream", "file"]:
        uri = self.uri
        if isinstance(uri, int) or (isinstance(uri, str) and uri.isdigit()):
            return "webcam"
        if "://" in str(uri):
            return "stream"
        return "file"


class DetectorConfig(_Strict):
    model: str = Field("models/yolo11n.pt", description="Path to .pt or .onnx weights.")
    imgsz: int = Field(640, ge=128, le=1920, description="Inference size (multiple of 32).")
    confidence: float = Field(0.35, ge=0.0, le=1.0)
    iou: float = Field(0.5, ge=0.0, le=1.0)
    classes: list[str] = Field(default_factory=lambda: ["person"])
    device: str = "cpu"

    @field_validator("imgsz")
    @classmethod
    def _multiple_of_32(cls, v: int) -> int:
        if v % 32:
            raise ValueError("imgsz must be a multiple of 32")
        return v

    @field_validator("classes")
    @classmethod
    def _known_classes(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(COCO_CLASSES))
        if unknown:
            raise ValueError(f"unknown COCO classes: {unknown}")
        if not v:
            raise ValueError("at least one class is required")
        return v

    @property
    def class_ids(self) -> list[int]:
        return [COCO_CLASSES.index(c) for c in self.classes]


class TrackerConfig(_Strict):
    """ByteTrack parameters (see the Roboflow ``trackers`` documentation)."""

    track_activation_threshold: float = Field(0.4, ge=0.0, le=1.0, description="Min confidence to start a track.")
    high_conf_det_threshold: float = Field(0.5, ge=0.0, le=1.0, description="First-stage association threshold.")
    lost_track_buffer: int = Field(30, ge=1, description="Frames (at source FPS) a lost track is kept.")
    minimum_consecutive_frames: int = Field(2, ge=1, description="Hits before a track id is assigned.")
    minimum_iou_threshold: float = Field(0.1, ge=0.0, le=1.0)


class ProcessingConfig(_Strict):
    frame_skip: int = Field(0, ge=0, description="Skip N frames between processed frames.")
    max_width: int = Field(960, ge=160, description="Resize frames wider than this.")
    jpeg_quality: int = Field(80, ge=30, le=100)
    heatmap: bool = True
    heatmap_alpha: float = Field(0.45, ge=0.0, le=1.0)
    heatmap_decay: float = Field(
        0.999, gt=0.0, le=1.0, description="Per-frame multiplicative decay (1 = never forget)."
    )
    stats_interval_s: float = Field(1.0, gt=0, description="History sampling interval.")
    history_size: int = Field(600, ge=10, description="Number of history samples kept.")


class LineConfig(_Strict):
    """A counting line from ``start`` to ``end``.

    Crossing from the line's left side to its right side (looking from ``start``
    towards ``end``, in image coordinates) counts as ``in``; the reverse is ``out``.
    """

    name: str
    start: NormPoint
    end: NormPoint

    @model_validator(mode="after")
    def _non_degenerate(self) -> LineConfig:
        if self.start == self.end:
            raise ValueError(f"line {self.name!r}: start and end must differ")
        return self


class ZoneConfig(_Strict):
    name: str
    polygon: list[NormPoint] = Field(min_length=3)


class PipelineConfig(_Strict):
    source: SourceConfig = Field(default_factory=SourceConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    lines: list[LineConfig] = Field(default_factory=list)
    zones: list[ZoneConfig] = Field(default_factory=list)
    zone_exit_timeout_s: float = Field(
        1.5, gt=0, description="A track unseen this long is considered to have left its zone."
    )

    @model_validator(mode="after")
    def _unique_names(self) -> PipelineConfig:
        for kind, items in (("line", self.lines), ("zone", self.zones)):
            names = [i.name for i in items]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate {kind} names: {names}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"config file not found: {path}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)


class AppSettings(BaseSettings):
    """Process settings, overridable with ``RVA_*`` environment variables."""

    model_config = SettingsConfigDict(env_prefix="RVA_", env_file=".env", extra="ignore")

    config_path: Path = Path("configs/default.yaml")
    db_path: Path = Path("data/events.db")
    host: str = "127.0.0.1"
    port: int = 8103
    autostart: bool = Field(True, description="Start processing the configured source on boot.")
    log_level: str = "INFO"
    log_json: bool = False
