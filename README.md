# Realtime Vision Analytics

[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/MuhammadAhmadRajpoot/realtime-vision-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/MuhammadAhmadRajpoot/realtime-vision-analytics/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)

Turn any camera, video file or RTSP stream into live footfall and traffic metrics on an ordinary CPU.
Realtime Vision Analytics detects people and vehicles with **YOLO11n**, follows them with **ByteTrack**,
and turns the tracks into business signals: directional **line-crossing counts** (in/out per class),
**polygon zones** with live occupancy and per-person **dwell time**, and a cumulative **heatmap**.
Everything is exposed through a **FastAPI** service (MJPEG video, REST, WebSocket), every event is
persisted to **SQLite** with CSV export, and a self-contained dashboard shows it all live. Scenarios are
plain YAML, so the same code serves retail footfall, security perimeters and traffic counting.

![Dashboard processing the demo video live](docs/screenshot.png)

<p align="center"><img src="docs/sample_output.jpg" width="640" alt="Annotated output frame: tracks, counting lines, zone and heatmap"><br>
<em>Annotated output frame: track ids, counting lines with in/out totals, zone occupancy, dwell time and heatmap.</em></p>

## Features

- **Detection**: Ultralytics YOLO11n (nano), class filter by COCO name (`person`, `car`, `bicycle`, ...), confidence/IoU thresholds.
- **Tracking**: ByteTrack (Roboflow `trackers`), driven by frame timestamps so dropped/skipped frames do not break motion prediction.
- **Line counting**: directional in/out counts per class, segment-accurate intersection test, hysteresis band against box jitter.
- **Zones**: arbitrary polygons, live occupancy, enter/exit events, current/average/max dwell time, debounced exits (short occlusions do not split a visit).
- **Heatmap**: cumulative presence map with optional decay, blended onto the stream.
- **Sources**: video files (looping, real-time pacing), webcams (`0`), RTSP/HTTP streams with automatic reconnect.
- **CPU speed knobs**: inference size, frame skip, max processing width, ONNX Runtime backend (`rva export-onnx`).
- **API**: MJPEG stream, REST stats/history/events, WebSocket live stats, start/stop/switch source at runtime, CSV export.
- **Persistence**: every line crossing and zone enter/exit (with dwell) stored in SQLite, tagged by session.
- **Dashboard**: live video, KPI tiles (in / out / occupancy / visible / FPS), zone and line tables, occupancy chart with hover tooltips, event log. No CDN or build step.
- **Engineering**: typed Pydantic config with validation, structured logging (key=value or JSON), 48 pytest tests (offline after the one-time weights download), Ruff, CI, Docker.

## Architecture

```mermaid
flowchart LR
    subgraph Sources
        F[Video file]
        W[Webcam]
        R[RTSP / HTTP]
    end
    Sources --> VS["VideoSource<br/>pacing, frame skip,<br/>loop / reconnect"]
    VS --> RS[Resize to max_width]
    RS --> DET["YOLO11n detector<br/>PyTorch or ONNX Runtime"]
    DET --> TRK["ByteTrack<br/>timestamp driven"]
    TRK --> AN["AnalyticsEngine"]
    AN --> LC[Line counters]
    AN --> ZM[Zone monitors]
    AN --> HM[Heatmap]
    AN --> ANN[Annotator]
    LC & ZM --> EV[(SQLite event log)]
    ANN --> JPG[Latest JPEG]
    AN --> SNAP[Stats snapshot + history]

    subgraph FastAPI
        MJ["/stream.mjpg"]
        REST["/api/stats, /api/history,<br/>/api/events, export.csv"]
        WS["/ws/stats"]
        CTRL["/api/source/start | stop"]
    end
    JPG --> MJ
    SNAP --> REST
    SNAP --> WS
    EV --> REST
    CTRL --> VS
    FastAPI --> UI[Dashboard]
```

The pipeline runs on a background thread (`rva.pipeline.Pipeline`); the API only reads thread-safe
snapshots, so slow clients never stall inference. All geometry is normalised to `[0, 1]`, so a scenario
keeps working if you change the processing resolution. Analytics (`rva.analytics`) are pure Python on
plain dataclasses and are tested without any model.

## Tech stack

| Area | Tools |
|---|---|
| Detection | Ultralytics YOLO11n, PyTorch (CPU), ONNX / ONNX Runtime |
| Tracking | ByteTrack via Roboflow `trackers`, `supervision` detections |
| Vision | OpenCV, NumPy |
| Service | FastAPI, Uvicorn, WebSockets, MJPEG |
| Config / data | Pydantic v2, pydantic-settings, YAML, SQLite |
| Frontend | Vanilla HTML/CSS/JS, hand-rolled SVG chart |
| Quality | pytest, Ruff, GitHub Actions, Docker |

## Quickstart

### Local (Windows / Linux / macOS)

```bash
git clone https://github.com/MuhammadAhmadRajpoot/realtime-vision-analytics.git
cd realtime-vision-analytics
python -m venv .venv
source .venv/Scripts/activate          # Windows Git Bash;  Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt    # CPU-only PyTorch wheels
pip install -e . --no-deps

python scripts/download_demo.py        # demo clip -> data/ (CC BY 4.0, Intel)
rva serve                              # http://127.0.0.1:8103
```

YOLO11n weights (5.4 MB) download automatically into `models/` on first start.
Open <http://127.0.0.1:8103> and the default scenario starts processing the demo video immediately.

Faster CPU inference with ONNX Runtime:

```bash
rva export-onnx --imgsz 416            # -> models/yolo11n_416.onnx
# then set detector.model: models/yolo11n_416.onnx (and imgsz: 416) in your YAML
```

Headless processing (no server), e.g. to produce an annotated video and a JSON summary:

```bash
rva process --source data/people-detection.mp4 --output out.mp4 --frame-out docs/sample_output.jpg --frame-at 300
```

Other sources at runtime, without restarting the server:

```bash
curl -X POST localhost:8103/api/source/start -H "Content-Type: application/json" -d '{"uri": 0}'
curl -X POST localhost:8103/api/source/start -H "Content-Type: application/json" -d '{"uri": "rtsp://user:pass@camera/stream1"}'
```

### Docker

```bash
docker compose up --build              # http://localhost:8103
```

The image is CPU-only (`python:3.12-slim`), runs as a non-root user, downloads the demo clip and weights
on first start, and keeps them (plus the SQLite log) in the mounted `./data` and `./models` volumes.

### Task shortcuts

`make install | demo | serve | test | lint | format | export-onnx | benchmark | sample | docker`
(see the [Makefile](Makefile); every target is a one-line command you can also run directly).

## Configuration

A scenario is one YAML file (`configs/default.yaml`, `configs/webcam.yaml`). Coordinates are normalised
(`x` right, `y` down). A line counts **in** when an object's bottom-centre crosses to the right-hand side
of the direction `start -> end`, and **out** in the opposite direction.

```yaml
source:   { uri: data/people-detection.mp4, loop: true, realtime: true }
detector: { model: models/yolo11n.pt, imgsz: 416, confidence: 0.35, classes: [person] }
processing: { frame_skip: 0, max_width: 960, heatmap: true }
lines:
  - { name: entrance, start: [0.92, 0.86], end: [0.08, 0.86] }
zones:
  - { name: center floor, polygon: [[0.22, 0.52], [0.78, 0.52], [0.84, 0.82], [0.16, 0.82]] }
```

| Key | Default | Description |
|---|---|---|
| `source.uri` | `data/people-detection.mp4` | Webcam index, file path, or `rtsp://` / `http://` URL |
| `source.loop` | `true` | Restart files when they end (trackers and open zone visits are reset) |
| `source.realtime` | `true` | Pace files at native FPS and drop frames when processing is slower |
| `source.reconnect_delay_s` | `2.0` | Delay between reconnect attempts for live streams |
| `detector.model` | `models/yolo11n.pt` | `.pt` (PyTorch) or `.onnx` weights |
| `detector.imgsz` | `640` (`416` in default.yaml) | Inference size, multiple of 32 |
| `detector.confidence` / `iou` | `0.35` / `0.5` | Detection thresholds |
| `detector.classes` | `[person]` | COCO class names to keep |
| `tracker.track_activation_threshold` | `0.4` | Min confidence to start a track |
| `tracker.lost_track_buffer` | `30` | Frames a lost track survives (scaled by source FPS) |
| `tracker.minimum_consecutive_frames` | `2` | Hits before a track id is issued |
| `processing.frame_skip` | `0` | Process every (N+1)-th frame |
| `processing.max_width` | `960` | Downscale wider frames before inference and drawing |
| `processing.heatmap` / `heatmap_alpha` / `heatmap_decay` | `true` / `0.45` / `0.999` | Heatmap overlay |
| `processing.jpeg_quality` | `80` | MJPEG quality |
| `processing.stats_interval_s` / `history_size` | `1.0` / `600` | History sampling |
| `zone_exit_timeout_s` | `1.5` | Unseen-inside time before a zone exit is emitted |
| `lines[]` | `[]` | `{name, start: [x, y], end: [x, y]}` |
| `zones[]` | `[]` | `{name, polygon: [[x, y], ...]}` (3+ points) |

Process settings come from environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `RVA_CONFIG_PATH` | `configs/default.yaml` | Scenario file |
| `RVA_DB_PATH` | `data/events.db` | SQLite event log |
| `RVA_HOST` / `RVA_PORT` | `127.0.0.1` / `8103` | Bind address |
| `RVA_AUTOSTART` | `true` | Start the configured source on boot |
| `RVA_LOG_LEVEL` / `RVA_LOG_JSON` | `INFO` / `false` | Logging |

Invalid configs fail fast with a clear message (unknown class names, degenerate lines, polygons with
fewer than 3 points, coordinates outside `[0, 1]`, duplicate names, unknown keys).

## API reference

Interactive docs at `/docs` (OpenAPI).

| Method | Path | Description |
|---|---|---|
| GET | `/` | Dashboard |
| GET | `/stream.mjpg` | Annotated MJPEG stream (`<img src="/stream.mjpg">`) |
| GET | `/api/snapshot.jpg` | Latest annotated frame |
| GET | `/api/stats` | Current session: status, FPS, totals, per-line and per-zone stats, current dwell per track, recent events |
| GET | `/api/history?limit=300` | Time series of totals (in, out, occupancy, visible, FPS) |
| GET | `/api/events?limit=&type=&session_id=&after_id=` | Persisted events, newest first (`type` = `line_cross`, `zone_enter`, `zone_exit`) |
| GET | `/api/events/export.csv?type=&session_id=` | CSV export of the event log |
| POST | `/api/source/start` | Start/restart; optional body `{"uri": ..., "loop": ..., "realtime": ...}` |
| POST | `/api/source/stop` | Stop processing |
| GET | `/api/config` | Active scenario |
| GET | `/api/health` | Liveness + pipeline status |
| WS | `/ws/stats?interval=0.5` | Pushes the `/api/stats` payload every `interval` seconds |

Example `/api/stats` response, captured from a live run on the demo clip (abridged):

```json
{
  "session_id": "657d8eb7d340", "status": "running", "source": "data/people-detection.mp4",
  "frames_processed": 118, "video_t": 21.17,
  "totals": { "in": 2, "out": 2, "occupancy": 0, "visible": 0 },
  "lines": [
    { "name": "entrance", "in": 2, "out": 0, "by_class": { "person": { "in": 2, "out": 0 } } },
    { "name": "west exit", "in": 0, "out": 2, "by_class": { "person": { "in": 0, "out": 2 } } },
    { "name": "east exit", "in": 0, "out": 0, "by_class": {} }
  ],
  "zones": [
    { "name": "center floor", "occupancy": 0, "by_class": {}, "total_visits": 2,
      "avg_dwell_s": 1.38, "max_dwell_s": 1.67, "current_dwell_s": {} }
  ]
}
```

## Tests

```bash
pytest -q            # 48 tests, offline after the first run
pytest -m model      # only the real-model tests
ruff check . && ruff format --check .
```

- **Unit tests** cover line-crossing geometry (direction, hysteresis, segment extent, per-class counts),
  zone enter/exit/dwell/occlusion debounce, heatmap accumulation/decay, config validation and the SQLite store.
- **Pipeline and API tests** run end-to-end on a synthetic video with a deterministic blob detector:
  real OpenCV decoding, real ByteTrack, the background thread, REST, WebSocket, MJPEG framing and CSV export.
- **Real-model tests** run YOLO11n on the sample image bundled with Ultralytics and check people/bus
  detections, class filtering and tracking through the full frame processor. They download the 5.4 MB
  weights once and are skipped if the weights cannot be obtained.

## Benchmarks

All numbers below were measured by me with `python scripts/benchmark.py --frames 60 --rounds 3 --sizes 320 416 640`
on a **CPU-only Windows 11 laptop (Intel Core i7-7500U, 2 cores / 4 threads, no GPU)**, Python 3.13, PyTorch 2.14.1+cpu,
ONNX Runtime 1.31.0, Ultralytics 8.4.177, YOLO11n. Input: the first 60 frames of the demo clip (768x432), each
configuration run 3 times, interleaved so thermal throttling affects all of them equally. Raw output:
[`docs/benchmarks.json`](docs/benchmarks.json).

| Backend | imgsz | Detector FPS (1 / median) | Median latency | p95 latency | End-to-end FPS |
|---|---:|---:|---:|---:|---:|
| PyTorch | 320 | 12.7 | 78.5 ms | 113.8 ms | 11.4 |
| ONNX Runtime | 320 | **19.3** | **51.8 ms** | 61.6 ms | **16.8** |
| PyTorch | 416 | 9.2 | 109.1 ms | 188.2 ms | 8.5 |
| ONNX Runtime | 416 | 12.0 | 83.6 ms | 96.9 ms | 11.3 |
| PyTorch | 640 | 4.6 | 216.7 ms | 374.5 ms | 5.2 |
| ONNX Runtime | 640 | 5.3 | 189.2 ms | 328.2 ms | 4.7 |

*Detector* = `model.predict` including YOLO pre/post-processing. *End-to-end* = resize + detect + ByteTrack +
line/zone analytics + heatmap + drawing (excludes video decoding and JPEG encoding).

What I take from it:

- ONNX Runtime is about 1.5x faster than PyTorch at 320 and 1.3x at 416 on this CPU; at 640 the gap mostly
  disappears and run-to-run noise dominates.
- On this laptop, 416 px with ONNX keeps up with the 12 FPS demo clip; 320 px leaves headroom for 15 to 25 FPS
  sources. The default config uses PyTorch at 416 so it works without an export step.
- Results on a 2-core laptop are noisy: the machine was not idle during the run (other processes were
  using the CPU), and an earlier non-interleaved run of the same script gave visibly different numbers for
  the later configurations. Re-run the script on your own
  hardware before relying on these figures.

**Live service.** With the default config (PyTorch, 416 px) the running server reported 8.7 to 9.4
processing FPS on the dashboard and WebSocket while streaming MJPEG to a browser (the screenshot above was
taken during that run).

**Counting sanity check.** Running `rva process` over the full 50 s demo clip (596 frames, default config)
produced 7 `entrance` in-crossings, 4 `west exit` and 3 `east exit` out-crossings, and 7 zone visits. I
counted the clip by hand (contact sheet at 1 frame per second): 7 people enter from the bottom, 4 leave
to the left and 3 to the right, so the counts match on this clip. That is one short, easy clip, not an
accuracy benchmark.

## Project structure

```
realtime-vision-analytics/
├── src/rva/
│   ├── analytics/        # line counter, zone monitor, heatmap, engine (pure, model-free)
│   ├── api/              # FastAPI app + schemas
│   ├── web/static/       # dashboard (HTML/CSS/JS, no build step)
│   ├── annotate.py       # drawing of boxes, lines, zones, HUD
│   ├── cli.py            # rva serve | process | export-onnx
│   ├── config.py         # Pydantic scenario + settings models
│   ├── detection.py      # YOLO detector, ByteTrack wrapper
│   ├── logging_setup.py  # key=value / JSON structured logs
│   ├── onnx_export.py    # static-shape ONNX export
│   ├── pipeline.py       # frame processor + background pipeline thread
│   ├── service.py        # owns pipeline lifecycle, detector cache, event store
│   ├── sources.py        # file / webcam / RTSP capture with pacing and reconnect
│   └── storage.py        # SQLite event log + CSV export
├── configs/              # default.yaml (demo hallway), webcam.yaml
├── scripts/              # download_demo.py, benchmark.py
├── tests/                # pytest suite (synthetic video, fakes, real-model tests)
├── docs/                 # screenshot, sample output, benchmarks.json
├── Dockerfile, docker-compose.yml, Makefile
└── .github/workflows/ci.yml
```

## Roadmap

- Perspective calibration (homography) for real-world distances and speed estimation.
- Multi-camera support with one pipeline per source and a camera selector in the dashboard.
- Polygon/line editor in the dashboard (draw on the live frame, save to YAML).
- Alert rules (occupancy above N, dwell above T) with webhook notifications.
- OpenVINO / INT8 quantised export for further CPU speed-up.
- Prometheus metrics endpoint.

## License

The source code in this repository is released under the [MIT License](LICENSE),
Copyright (c) 2026 Muhammad Ahmad.

**Important third-party license note:** this project uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
which is licensed under **AGPL-3.0**, and downloads Ultralytics' pretrained weights (also AGPL-3.0) at runtime.
The MIT license covers only the original code here. If you distribute this software or run it as a network
service together with Ultralytics, the AGPL-3.0 obligations apply to that combined work, unless you hold an
[Ultralytics Enterprise License](https://www.ultralytics.com/license). The demo videos come from
[intel-iot-devkit/sample-videos](https://github.com/intel-iot-devkit/sample-videos) (CC BY 4.0, Intel Corporation)
and are downloaded at runtime, not redistributed.

## Author

**Muhammad Ahmad**, AI/ML Engineer

- Portfolio: <https://muhammad-ahmad-rajpoot.vercel.app/>
- LinkedIn: <https://www.linkedin.com/in/muhammad-ahmad-a31b04322/>
- GitHub: [@MuhammadAhmadRajpoot](https://github.com/MuhammadAhmadRajpoot)
