"""CPU benchmark: PyTorch vs ONNX Runtime at several input sizes.

For each (backend, imgsz) it measures
  * detector-only throughput (model.predict incl. pre/post-processing), and
  * end-to-end pipeline throughput (resize + detect + ByteTrack + analytics + drawing),
on the same pre-decoded frames of the demo video, after a warm-up. Cases are
interleaved over several rounds so slow drift (thermal throttling) is shared. Detector FPS is
1 / median latency, which is robust to the occasional OS scheduling stall; mean and
p95 are reported alongside it.

Usage:
    python scripts/benchmark.py --frames 60 --rounds 3 --sizes 320 416 640
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from rva.config import PipelineConfig, SourceConfig
from rva.detection import YoloDetector
from rva.onnx_export import export_onnx
from rva.pipeline import FrameProcessor
from rva.sources import Frame, VideoSource, resize_to_width


def load_frames(path: str, n: int, max_width: int) -> list[np.ndarray]:
    frames = []
    with VideoSource(SourceConfig(uri=path, loop=False, realtime=False)) as src:
        while len(frames) < n and (f := src.read()) is not None:
            frames.append(resize_to_width(f.image, max_width))
    if not frames:
        raise SystemExit(f"no frames read from {path}; run scripts/download_demo.py first")
    return frames


@dataclass
class Case:
    backend: str
    imgsz: int
    cfg: PipelineConfig
    detector: YoloDetector
    latencies: list[float] = field(default_factory=list)
    e2e_fps: list[float] = field(default_factory=list)

    def run_round(self, frames: list[np.ndarray]) -> None:
        for f in frames:
            t0 = time.perf_counter()
            self.detector.detect(f)
            self.latencies.append(time.perf_counter() - t0)
        proc = FrameProcessor(self.cfg, self.detector, frame_rate=12)
        t0 = time.perf_counter()
        for i, f in enumerate(frames):
            proc.process(Frame(image=f, t=i / 12))
        self.e2e_fps.append(len(frames) / (time.perf_counter() - t0))

    def summary(self) -> dict[str, float | int | str]:
        lat = self.latencies
        return {
            "backend": self.backend,
            "imgsz": self.imgsz,
            "detector_fps": round(1 / statistics.median(lat), 2),
            "detector_ms_median": round(1000 * statistics.median(lat), 1),
            "detector_ms_mean": round(1000 * statistics.mean(lat), 1),
            "detector_ms_p95": round(1000 * float(np.percentile(lat, 95)), 1),
            "pipeline_fps": round(statistics.median(self.e2e_fps), 2),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--frames", type=int, default=60, help="Frames per case per round")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--sizes", type=int, nargs="+", default=[320, 640])
    ap.add_argument("--out", type=Path, default=Path("docs/benchmarks.json"))
    args = ap.parse_args()

    base = PipelineConfig.from_yaml(args.config)
    frames = load_frames(str(base.source.uri), args.frames, base.processing.max_width)
    weights = Path(base.detector.model)
    h, w = frames[0].shape[:2]
    print(f"{len(frames)} frames at {w}x{h}, {args.rounds} interleaved rounds, model {weights.name}")

    cases: list[Case] = []
    for size in args.sizes:
        onnx_path = export_onnx(weights, size)
        for backend, model in (("pytorch", weights), ("onnx", onnx_path)):
            cfg = base.model_copy(deep=True)
            cfg.detector = cfg.detector.model_copy(update={"model": str(model), "imgsz": size})
            detector = YoloDetector(cfg.detector)
            for f in frames[: args.warmup]:
                detector.detect(f)
            cases.append(Case(backend, size, cfg, detector))

    # Interleave cases (rotating the order each round) so thermal throttling and
    # background load affect every configuration equally.
    for r in range(args.rounds):
        order = cases[r % len(cases) :] + cases[: r % len(cases)]
        for case in order:
            case.run_round(frames)
        print(f"round {r + 1}/{args.rounds} done")
    results = [c.summary() for c in cases]

    import onnxruntime
    import torch
    import ultralytics

    report = {
        "hardware": {
            "cpu": platform.processor(),
            "logical_cores": os.cpu_count(),
            "os": f"{platform.system()} {platform.release()}",
            "gpu": "none (CPU only)",
        },
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnxruntime": onnxruntime.__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
        },
        "input": {
            "video": str(base.source.uri),
            "frames_per_round": len(frames),
            "rounds": args.rounds,
            "frame_size": f"{w}x{h}",
        },
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n| Backend | imgsz | Detector FPS | Median ms | Mean ms | p95 ms | End-to-end FPS |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(
            f"| {r['backend']} | {r['imgsz']} | {r['detector_fps']} | {r['detector_ms_median']} "
            f"| {r['detector_ms_mean']} | {r['detector_ms_p95']} | {r['pipeline_fps']} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
