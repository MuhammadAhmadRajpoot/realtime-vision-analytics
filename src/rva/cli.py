"""Command line entry point: ``rva serve | process | export-onnx``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2

from rva.config import AppSettings, PipelineConfig
from rva.logging_setup import configure_logging

log = logging.getLogger("rva")


def _serve(args: argparse.Namespace, settings: AppSettings) -> int:
    import uvicorn

    from rva.api.app import create_app

    if args.config:
        settings.config_path = Path(args.config)
    app = create_app(settings)
    uvicorn.run(app, host=args.host or settings.host, port=args.port or settings.port, log_config=None)
    return 0


def _process(args: argparse.Namespace, settings: AppSettings) -> int:
    """Run the pipeline headless over a file and print the final stats as JSON."""
    from rva.detection import YoloDetector
    from rva.pipeline import FrameProcessor
    from rva.sources import VideoSource

    cfg = PipelineConfig.from_yaml(args.config or settings.config_path)
    source_cfg = cfg.source.model_copy(update={"loop": False, "realtime": False})
    if args.source:
        source_cfg = source_cfg.model_copy(update={"uri": args.source})
    writer: cv2.VideoWriter | None = None
    n = 0
    events = 0
    started = time.perf_counter()
    with VideoSource(source_cfg, cfg.processing.frame_skip) as source:
        processor = FrameProcessor(cfg, YoloDetector(cfg.detector), source.effective_fps)
        while args.max_frames is None or n < args.max_frames:
            frame = source.read()
            if frame is None:
                break
            result = processor.process(frame)
            events += len(result.events)
            n += 1
            if args.output:
                if writer is None:
                    h, w = result.annotated.shape[:2]
                    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.output, fourcc, source.effective_fps, (w, h))
                writer.write(result.annotated)
            if args.frame_out and n == (args.frame_at or 1):
                Path(args.frame_out).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(args.frame_out, result.annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        events += len(processor.engine.restart_tracks())
    if writer is not None:
        writer.release()
    elapsed = time.perf_counter() - started
    summary = {
        "frames": n,
        "elapsed_s": round(elapsed, 2),
        "avg_fps": round(n / elapsed, 2) if elapsed else 0.0,
        "events": events,
        **processor.engine.snapshot(),
    }
    print(json.dumps(summary, indent=2))
    return 0


def _export_onnx(args: argparse.Namespace, settings: AppSettings) -> int:
    from rva.onnx_export import export_onnx

    out = export_onnx(Path(args.weights), args.imgsz, Path(args.output) if args.output else None)
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rva", description="Realtime vision analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="Run the API + dashboard")
    s.add_argument("--config", help="Scenario YAML (default: RVA_CONFIG_PATH or configs/default.yaml)")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.set_defaults(func=_serve)

    p = sub.add_parser("process", help="Process a video headless and print stats")
    p.add_argument("--config")
    p.add_argument("--source", help="Override the source URI from the config")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--output", help="Write annotated video (.mp4)")
    p.add_argument("--frame-out", help="Save one annotated frame as an image")
    p.add_argument("--frame-at", type=int, help="1-based processed-frame index for --frame-out")
    p.set_defaults(func=_process)

    e = sub.add_parser("export-onnx", help="Export YOLO weights to ONNX at a fixed input size")
    e.add_argument("--weights", default="models/yolo11n.pt")
    e.add_argument("--imgsz", type=int, default=640)
    e.add_argument("--output", help="Destination .onnx path (default: models/<name>_<imgsz>.onnx)")
    e.set_defaults(func=_export_onnx)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = AppSettings()
    configure_logging(settings.log_level, settings.log_json)
    try:
        return int(args.func(args, settings))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        log.error("command failed", extra={"error": str(exc)})
        return 1


if __name__ == "__main__":
    sys.exit(main())
