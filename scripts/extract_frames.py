#!/usr/bin/env python3
"""Extract UCF-Crime videos to 1-FPS JPG frames with ffmpeg.

Layout produced (data contract):
    <data_root>/frames/<video_stem>/000001.jpg ...

Usage:
    python scripts/extract_frames.py --data-root /root/autodl-tmp/data/ucf_crime
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv")


def extract_video(video_path: str, out_dir: str, fps: int = 1) -> bool:
    os.makedirs(out_dir, exist_ok=True)
    if os.listdir(out_dir):  # already extracted
        return True
    out_pattern = os.path.join(out_dir, "%06d.jpg")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}",
           "-q:v", "2", out_pattern]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        logger.error("ffmpeg failed for %s: %s", video_path, proc.stderr[-500:])
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="1-FPS frame extraction")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data/ucf_crime")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="debug: only N videos")
    args = parser.parse_args()

    videos_root = os.path.join(args.data_root, "videos")
    frames_root = os.path.join(args.data_root, "frames")
    os.makedirs(frames_root, exist_ok=True)

    videos = []
    for root, _dirs, files in os.walk(videos_root):
        for name in sorted(files):
            if name.lower().endswith(VIDEO_EXTS):
                videos.append(os.path.join(root, name))
    videos.sort()
    if args.limit:
        videos = videos[:args.limit]
    logger.info("%d videos to extract at %d FPS", len(videos), args.fps)

    ok, fail = 0, 0
    for i, vpath in enumerate(videos):
        stem = os.path.splitext(os.path.basename(vpath))[0]
        out_dir = os.path.join(frames_root, stem)
        if extract_video(vpath, out_dir, args.fps):
            ok += 1
        else:
            fail += 1
        if (i + 1) % 10 == 0:
            logger.info("progress: %d/%d (ok=%d fail=%d)", i + 1, len(videos), ok, fail)
    logger.info("done: %d ok, %d failed", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
