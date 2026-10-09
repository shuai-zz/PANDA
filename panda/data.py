"""UCF-Crime data pipeline: annotation parsing, 1-FPS frame loading, clip
iteration and frame-level ground truth.

Data contract (prepared by the dataset-prep agent, do not change):
    <data_root>/videos/<category>/*.avi            (test set, 290 videos)
    <data_root>/annotations/Temporal_Anomaly_Annotation_for_Testing.txt
    <data_root>/frames/<video_stem>/000001.jpg ... (ffmpeg 1 FPS)

Annotation line: `<video_name>.avi <category> <s1> <e1> <s2> <e2>`
(frame numbers in the ORIGINAL video; normal videos are `-1 -1 -1 -1`).

Frame-level GT: the i-th extracted (1 FPS) frame corresponds to original frame
i * fps (1-based, i.e. second i-1..i). fps is read from the video file's
metadata; falls back to 30.0 (UCF-Crime) when unreadable.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ANNOTATION_FILE = "Temporal_Anomaly_Annotation_for_Testing.txt"
DEFAULT_FPS = 30.0


@dataclass
class VideoMeta:
    video_id: str           # <stem> (e.g. "Robbery001")
    stem: str
    category: str           # annotation category ("Normal" for normal videos)
    intervals: list = field(default_factory=list)  # [(start, end), ...] original frame numbers
    is_normal: bool = True
    video_path: str = ""    # original video file (may be missing)
    frame_paths: list = field(default_factory=list)  # extracted 1-FPS frames sorted
    fps: float = DEFAULT_FPS


def parse_annotations(annotations_path: str) -> dict:
    """Parse Temporal_Anomaly_Annotation_for_Testing.txt -> {stem: (category, intervals)}."""
    out = {}
    with open(annotations_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            m = re.match(r"(.+)\.(?:avi|mp4)$", parts[0], re.I)
            stem = m.group(1) if m else parts[0]
            category = parts[1]
            nums = [int(x) for x in parts[2:]]
            intervals = []
            for i in range(0, len(nums) - 1, 2):
                s, e = nums[i], nums[i + 1]
                if s >= 0 and e >= 0:
                    intervals.append((s, e))
            out[stem] = (category, intervals)
    return out


def read_video_fps(video_path: str, default: float = DEFAULT_FPS) -> float:
    """Read fps from video metadata; fall back to `default`."""
    if not video_path or not os.path.exists(video_path):
        return default
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps and fps > 1.0:
            return float(fps)
    except Exception as exc:
        logger.warning("could not read fps from %s (%s); using %.1f",
                       video_path, exc, default)
    return default


def find_video_file(videos_root: str, stem: str) -> str:
    for root, _dirs, files in os.walk(videos_root):
        for name in files:
            if os.path.splitext(name)[0] == stem:
                return os.path.join(root, name)
    return ""


def load_dataset(data_root: str) -> list:
    """Load all test videos that have extracted frames, sorted by video id."""
    annotations_path = os.path.join(data_root, "annotations", ANNOTATION_FILE)
    annotations = parse_annotations(annotations_path)
    frames_root = os.path.join(data_root, "frames")
    videos_root = os.path.join(data_root, "videos")

    metas = []
    for stem in sorted(os.listdir(frames_root)):
        fdir = os.path.join(frames_root, stem)
        if not os.path.isdir(fdir):
            continue
        frame_paths = sorted(
            os.path.join(fdir, name) for name in os.listdir(fdir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not frame_paths:
            continue
        category, intervals = annotations.get(stem, ("Unknown", []))
        video_path = find_video_file(videos_root, stem)
        metas.append(VideoMeta(
            video_id=stem, stem=stem, category=category, intervals=intervals,
            is_normal=(not intervals), video_path=video_path,
            frame_paths=frame_paths, fps=DEFAULT_FPS,
        ))
    return metas


def with_fps(metas: list) -> list:
    """Fill in per-video fps from video metadata (batched, tolerates failures)."""
    for meta in metas:
        meta.fps = read_video_fps(meta.video_path, DEFAULT_FPS)
    return metas


def select_subset(metas: list, n_normal: int = 10, n_abnormal: int = 10) -> list:
    """Smoke-test subset: first N normal + first N abnormal videos (sorted)."""
    normals = [m for m in metas if m.is_normal][:n_normal]
    abnormals = [m for m in metas if not m.is_normal][:n_abnormal]
    selected = normals + abnormals
    logger.info("subset selected: %d normal + %d abnormal", len(normals), len(abnormals))
    return selected


def iter_clips(frame_paths: list, clip_frames: int):
    """Group consecutive sampled frames into clips of s frames."""
    for i in range(0, len(frame_paths), clip_frames):
        yield i // clip_frames, frame_paths[i:i + clip_frames]


def frame_labels(meta: VideoMeta) -> list:
    """Frame-level GT over the extracted 1-FPS frames.

    Extracted frame i (1-based) covers original frames ~((i-1)*fps, i*fps];
    we label by original frame i*fps (spec data contract).
    """
    labels = []
    for i in range(1, len(meta.frame_paths) + 1):
        orig = i * meta.fps
        label = int(any(s <= orig <= e for s, e in meta.intervals))
        labels.append(label)
    return labels
