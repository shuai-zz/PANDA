"""Model clients: VLM (Qwen2.5-VL-7B via local vLLM OpenAI-compatible API)
and MLLM (Kimi, planning/reflection).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from typing import Any, Optional

import numpy as np
from openai import OpenAI

from .config import Config

logger = logging.getLogger(__name__)

SCORE_JSON_RE = re.compile(r"\{.*\"score\".*\}", re.S)
STATUS_CHOICES = ("normal", "abnormal", "insufficient")


def parse_score_json(text: str) -> Optional[dict]:
    """Parse a VLM {score, status, reason} JSON object.

    Strategy: strip markdown fences -> json.loads -> regex extraction -> None.
    """
    if not text:
        return None
    cleaned = text.strip()
    # strip ```json ... ``` fences if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    candidates = [cleaned]
    m = SCORE_JSON_RE.search(cleaned)
    if m:
        candidates.append(m.group(0))
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        parsed = _normalize_score_obj(obj)
        if parsed is not None:
            return parsed
    return None


def _normalize_score_obj(obj: Any) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    score = obj.get("score", obj.get("Score"))
    status = obj.get("status", obj.get("Status"))
    reason = obj.get("reason", obj.get("Reason", ""))
    try:
        score = float(score)
    except (TypeError, ValueError):
        return None
    score = min(1.0, max(0.0, score))
    if not isinstance(status, str):
        return None
    status = status.strip().lower()
    if status not in STATUS_CHOICES:
        # tolerate variants like "Normal." / "insufficient evidence"
        for choice in STATUS_CHOICES:
            if status.startswith(choice):
                status = choice
                break
        else:
            return None
    return {"score": score, "status": status, "reason": str(reason)}


def extract_json_object(text: str) -> Optional[Any]:
    """Generic JSON extraction for MLLM outputs (may be a list or object)."""
    if not text:
        return None
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    # greedy match from first {/[ to last }/]
    start = min([i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0], default=-1)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))
    if 0 <= start < end:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            return None
    return None


def encode_image_b64(image: np.ndarray, long_side: int = 448, quality: int = 85) -> str:
    """BGR ndarray -> base64 JPEG data URL, long side resized to `long_side`."""
    import cv2

    h, w = image.shape[:2]
    scale = long_side / max(h, w)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("cv2.imencode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class VLMClient:
    """OpenAI-compatible client for the local Qwen2.5-VL vLLM server."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.vlm_base_url, api_key=cfg.vlm_api_key,
                             timeout=300, max_retries=2)
        self.model = cfg.vlm_model

    def chat(self, prompt: str, images: Optional[list] = None,
             system: Optional[str] = None) -> str:
        """Single-turn chat. `images` is a list of BGR ndarrays or file paths."""
        max_imgs = getattr(self.cfg, "vlm_max_images", 0)
        if max_imgs and len(images or []) > max_imgs:
            logger.warning("VLM prompt capped from %d to %d images",
                           len(images), max_imgs)
            images = list(images)[:max_imgs]
        content: list = [{"type": "text", "text": prompt}]
        for img in images or []:
            content.append({"type": "image_url", "image_url": {"url": self._to_url(img)}})
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=self.cfg.vlm_temperature,
        )
        return resp.choices[0].message.content or ""

    def _to_url(self, img) -> str:
        if isinstance(img, str):
            with open(img, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("ascii")
            return f"data:image/jpeg;base64,{b64}"
        return encode_image_b64(img, self.cfg.vlm_long_side, self.cfg.vlm_jpeg_quality)

    def detect(self, prompt: str, images: list,
               system: Optional[str] = None) -> dict:
        """Reasoning/perception call returning {score, status, reason}.

        Retry once on parse failure (spec §7 risk table); final fallback is
        score=0.5 / status=insufficient so the clip can still be scored.
        """
        for attempt in range(self.cfg.vlm_max_retries + 1):
            try:
                text = self.chat(prompt, images, system=system)
            except Exception as exc:  # network/server error
                logger.warning("VLM request failed (attempt %d): %s", attempt + 1, exc)
                continue
            parsed = parse_score_json(text)
            if parsed is not None:
                if attempt > 0:
                    logger.info("VLM JSON parsed on retry (attempt %d)", attempt + 1)
                return parsed
            logger.warning("VLM output not parseable (attempt %d): %.300s",
                           attempt + 1, text)
        logger.error("VLM failed after retries; falling back to insufficient/0.5")
        return {"score": 0.5, "status": "insufficient",
                "reason": "VLM output could not be parsed; fallback score assigned."}


class MLLMClient:
    """Kimi client for knowledge-base construction, planning and reflection.

    Kimi is a forced-reasoning model: max_tokens must stay >= 4096. We pass
    ``think_effort`` via extra_body and degrade gracefully if unsupported.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = cfg.mllm_model
        self._think_supported = True  # flipped off after first rejection
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Created lazily so that MLLM-free stages (e.g. M0) work without a key."""
        if self._client is None:
            if not self.cfg.mllm_api_key:
                raise RuntimeError(
                    "MLLM API key missing: set KIMI_API_KEY (and optionally "
                    "KIMI_BASE_URL) in the environment.")
            self._client = OpenAI(base_url=self.cfg.mllm_base_url,
                                  api_key=self.cfg.mllm_api_key,
                                  timeout=600, max_retries=2)
        return self._client

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        kwargs = dict(model=self.model, messages=messages,
                      max_tokens=self.cfg.mllm_max_tokens)
        if self.cfg.mllm_temperature is not None:
            kwargs["temperature"] = self.cfg.mllm_temperature
        if self._think_supported and self.cfg.mllm_think_effort:
            kwargs["extra_body"] = {"think_effort": self.cfg.mllm_think_effort}
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if self._think_supported and ("think" in msg or "extra" in msg or
                                          "unsupported" in msg or "unknown" in msg):
                logger.warning("think_effort unsupported by API (%s); disabling", exc)
                self._think_supported = False
                kwargs.pop("extra_body", None)
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        return resp.choices[0].message.content or ""

    def generate_json(self, prompt: str, system: Optional[str] = None) -> Any:
        """Generate and parse JSON; retry with backoff on rate limits, then return None."""
        max_attempts = max(self.cfg.mllm_max_retries + 1, 6)
        for attempt in range(max_attempts):
            try:
                text = self.generate(prompt, system=system)
            except Exception as exc:
                msg = str(exc).lower()
                logger.warning("MLLM request failed (attempt %d): %s", attempt + 1, exc)
                if "429" in msg or "overload" in msg or "rate limit" in msg:
                    wait = min(30 * (2 ** attempt), 300)
                    logger.info("MLLM rate limited; backing off %ds", wait)
                    time.sleep(wait)
                continue
            obj = extract_json_object(text)
            if obj is not None:
                return obj
            logger.warning("MLLM output not parseable (attempt %d): %.300s",
                           attempt + 1, text)
        logger.error("MLLM failed to produce parseable JSON after retries")
        return None
