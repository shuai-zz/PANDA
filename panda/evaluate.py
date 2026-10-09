"""Evaluation: frame-level AUC / AP and mean-filter temporal smoothing.

Metric protocol: following the standard UCF-Crime / XD-Violence frame-level
protocol, the PRIMARY number is computed by pooling the frames of ALL videos
into one list before the ROC/AP curve (micro-average). Per-video macro means
are also reported for reference.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def mean_filter(scores: list, window: int = 10) -> list:
    """Temporal mean smoothing (paper implementation detail: window size = 10)."""
    if window <= 1 or len(scores) == 0:
        return list(scores)
    kernel = np.ones(window) / window
    padded = np.pad(np.asarray(scores, dtype=np.float64), window // 2, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    n = len(scores)
    return smoothed[:n].tolist()


def frame_auc(labels: list, scores: list) -> float:
    from sklearn.metrics import roc_auc_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores)) * 100.0


def frame_ap(labels: list, scores: list) -> float:
    from sklearn.metrics import average_precision_score

    if len(set(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores)) * 100.0


def evaluate_video(labels: list, scores: list, smooth_window: int = 10) -> dict:
    """Per-video metrics (raw + smoothed scores). NaN when single-class."""
    smoothed = mean_filter(scores, smooth_window)
    return {
        "auc_raw": frame_auc(labels, scores),
        "auc_smoothed": frame_auc(labels, smoothed),
        "ap_raw": frame_ap(labels, scores),
        "ap_smoothed": frame_ap(labels, smoothed),
        "scores_smoothed": smoothed,
        "n_frames": len(labels),
        "n_anomaly_frames": int(sum(labels)),
    }


def aggregate(all_labels: list, all_scores: list, all_smoothed: list,
              video_metrics: list, metric: str = "auc") -> dict:
    """Pooled (micro) primary metric + per-video macro means.

    `all_*` are frame-level lists concatenated across videos.
    """
    key = "ap" if metric.lower() == "ap" else "auc"
    fn = frame_ap if key == "ap" else frame_auc

    def _mean(values):
        valid = [v for v in values if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    return {
        "n_videos": len(video_metrics),
        "n_frames": len(all_labels),
        f"pooled_{key}_raw": fn(all_labels, all_scores),
        f"pooled_{key}_smoothed": fn(all_labels, all_smoothed),
        f"macro_{key}_raw_mean": _mean([m[f"{key}_raw"] for m in video_metrics]),
        f"macro_{key}_smoothed_mean": _mean([m[f"{key}_smoothed"] for m in video_metrics]),
    }
