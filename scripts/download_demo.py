"""Download the demo videos used by the default configs.

Source: https://github.com/intel-iot-devkit/sample-videos
License: Creative Commons Attribution 4.0 International (CC BY 4.0),
(c) Intel Corporation. The files are fetched at runtime and are not
redistributed with this repository.

Usage:
    python scripts/download_demo.py            # people-detection.mp4 (default demo)
    python scripts/download_demo.py --all      # also the person/bicycle/car clip
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/"
VIDEOS = {
    "people-detection.mp4": "People walking through a hallway (default demo).",
    "person-bicycle-car-detection.mp4": "Street scene with people, bicycles and cars.",
}
DEFAULT = "people-detection.mp4"


def download(name: str, dest_dir: Path, force: bool = False) -> Path:
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[skip] {dest} already exists")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {BASE_URL + name}")
    with urllib.request.urlopen(BASE_URL + name, timeout=60) as resp, tmp.open("wb") as fh:
        while chunk := resp.read(1 << 16):
            fh.write(chunk)
    tmp.replace(dest)
    print(f"[done] {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path("data"))
    parser.add_argument("--all", action="store_true", help="Download every demo clip")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()
    names = list(VIDEOS) if args.all else [DEFAULT]
    try:
        for name in names:
            download(name, args.dest, args.force)
    except OSError as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        return 1
    print("Videos: intel-iot-devkit/sample-videos, CC BY 4.0 (c) Intel Corporation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
