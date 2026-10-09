"""Chain-of-Memory (CoM): short CoM and long CoM (paper §3.4).

Short CoM: textual trace of the last l=5 reasoning steps plus the corresponding
visual frames (one representative frame per step, to control VLM tokens).

Long CoM: all memory units Mt = {reasoning, reflection, refined reasoning} of the
current video; RetrieveTop1 uses the insufficient reason as query (式7).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np

from .config import Config
from .knowledge import Embedder

logger = logging.getLogger(__name__)


def format_reasoning_step(clip_index: int, result: dict) -> str:
    """One textual short-CoM entry (reasoning-stage memory)."""
    return (f"Clip {clip_index}: score={result.get('score')}, "
            f"status={result.get('status')}, reason={result.get('reason', '')}")


def format_reflection_step(round_idx: int, reflection: dict) -> str:
    """One textual short-CoM entry (reflection-stage memory)."""
    tools = reflection.get("tools_to_use") or []
    tool_names = ", ".join(t.get("tool_name", "") for t in tools if isinstance(t, dict))
    return (f"Reflection round {round_idx}: reason={reflection.get('reason', '')}; "
            f"tools_to_use=[{tool_names}]; "
            f"new_anomaly_rule={reflection.get('new_anomaly_rule', '')}")


class ShortCoM:
    """Sliding window of the last l steps (text + aligned visual frames)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.text_entries: deque = deque(maxlen=cfg.short_com_len)
        self.visual_entries: deque = deque(maxlen=cfg.short_com_len)

    def add_reasoning(self, clip_index: int, result: dict, frame_paths: list) -> None:
        self.text_entries.append(format_reasoning_step(clip_index, result))
        if frame_paths:
            mid = frame_paths[len(frame_paths) // 2]
            self.visual_entries.append([mid])

    def add_reflection(self, round_idx: int, reflection: dict) -> None:
        # reflection-stage short CoM = the set of past reflection outputs
        self.text_entries.append(format_reflection_step(round_idx, reflection))

    def text_prompt(self) -> str:
        if not self.text_entries:
            return "No reliable historical detection information available."
        return "\n".join(self.text_entries)

    def visual_frames(self, max_total: Optional[int] = None) -> list:
        """Flattened representative frame paths from the last l steps."""
        per_step = max(1, self.cfg.short_com_visual_per_step)
        paths: list = []
        for frames in self.visual_entries:
            paths.extend(frames[:per_step])
        limit = max_total or self.cfg.short_com_len * per_step
        return paths[-limit:]

    def clear(self) -> None:
        self.text_entries.clear()
        self.visual_entries.clear()


class LongCoM:
    """Temporally evolving memory LongCoM = {M1, ..., MT} with FAISS retrieval.

    Each unit Mt = {reasoning, reflection, refined_reasoning}. Retrieval is over
    the insufficient reason of each unit (encoded with all-MiniLM-L6-v2).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.units: list = []
        self._embedder = None
        self._emb: Optional[np.ndarray] = None

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = Embedder.get(self.cfg)
        return self._embedder

    def add(self, unit: dict) -> None:
        """Append Mt and invalidate the embedding cache."""
        self.units.append(unit)
        self._emb = None

    def _embeddings(self) -> Optional[np.ndarray]:
        if self._emb is not None:
            return self._emb
        if not self.units:
            return None
        reasons = []
        for u in self.units:
            r = ""
            refl = u.get("reflection") or {}
            if isinstance(refl, dict):
                r = str(refl.get("reason", ""))
            if not r:
                res = u.get("reasoning") or {}
                r = str(res.get("reason", ""))
            reasons.append(r)
        self._emb = self.embedder.encode(reasons)
        return self._emb

    def retrieve_top1(self, insufficient_reason: str) -> Optional[dict]:
        """RetrieveTop1(Insufficient Reason, LongCoM) (式7)."""
        emb = self._embeddings()
        if emb is None or not insufficient_reason.strip():
            return None
        q = self.embedder.encode([insufficient_reason])
        sims = emb @ q[0]
        idx = int(np.argmax(sims))
        logger.info("LongCoM retrieve top-1: unit %d, similarity %.3f",
                    idx, float(sims[idx]))
        return self.units[idx]

    def format_experience(self, unit: Optional[dict]) -> str:
        if unit is None:
            return "No relevant history experience."
        reasoning = unit.get("reasoning") or {}
        reflection = unit.get("reflection") or {}
        refined = unit.get("refined_reasoning") or {}
        parts = [
            f"Past reasoning: score={reasoning.get('score')}, "
            f"status={reasoning.get('status')}, reason={reasoning.get('reason', '')}",
            f"Past reflection: reason={reflection.get('reason', '')}; "
            f"new_anomaly_rule={reflection.get('new_anomaly_rule', '')}; "
            f"new_heuristic_prompt={reflection.get('new_heuristic_prompt', '')}",
        ]
        if refined:
            parts.append(
                f"Past re-reasoning: score={refined.get('score')}, "
                f"status={refined.get('status')}, reason={refined.get('reason', '')}")
        return "\n".join(parts)

    def clear(self) -> None:
        self.units.clear()
        self._emb = None
