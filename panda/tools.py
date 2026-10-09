"""The 8 information-enhancement tools of PANDA (paper Appendix D).

Unified signature: (frames: list[np.ndarray], query: str) -> dict.
Every tool returns a dict:
    {"ok": bool,
     "text_info": str,          # semantic/Text Enhancement Info (式9)
     "frames": list[np.ndarray] or None,  # Visual Enhancement Info (processed clip)
     "error": str or None}
Tools run on CPU (the GPU is occupied by the vLLM server); heavy models are
loaded lazily and cached. All failures are caught and reported, never raised.

image_retrieval additionally needs the already-extracted frames of the current
video; the agent registers them once per video via ``set_tool_context``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .config import Config

logger = logging.getLogger(__name__)

ToolFn = Callable[[list, str], dict]


@dataclass
class ToolContext:
    """Per-video context shared with tools that need video-level information."""
    video_id: str = ""
    frame_paths: list = field(default_factory=list)  # all extracted frames of the video


_CONTEXT = ToolContext()
_LAZY_CACHE: dict = {}


def set_tool_context(video_id: str, frame_paths: list) -> None:
    _CONTEXT.video_id = video_id
    _CONTEXT.frame_paths = list(frame_paths)
    _LAZY_CACHE.pop("clip_video_id", None)  # invalidate CLIP per-video embedding cache


def _ok(text_info: str = "", frames: Optional[list] = None) -> dict:
    return {"ok": True, "text_info": text_info, "frames": frames, "error": None}


def _err(exc: Exception) -> dict:
    logger.warning("tool failed: %s", exc)
    return {"ok": False, "text_info": "", "frames": None,
            "error": f"{type(exc).__name__}: {exc}"}


def _tool(fn):
    """Decorator: run tool with try/except, always return a dict."""
    def wrapper(frames: list, query: str) -> dict:
        try:
            return fn(frames, query)
        except Exception as exc:
            return _err(exc)
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    return wrapper


# ---------------------------------------------------------------------------
# 1. image_denoise — cv2.fastNlMeansDenoisingColored
# ---------------------------------------------------------------------------
@_tool
def image_denoise(frames: list, query: str) -> dict:
    import cv2

    out = [cv2.fastNlMeansDenoisingColored(f, None, 10, 10, 7, 21) for f in frames]
    return _ok(f"Applied fastNlMeansDenoisingColored to {len(out)} frame(s).", out)


# ---------------------------------------------------------------------------
# 2. image_deblur — unsharp masking (Gaussian blur subtraction)
# ---------------------------------------------------------------------------
@_tool
def image_deblur(frames: list, query: str) -> dict:
    import cv2

    amount, sigma = 1.5, 3.0
    out = []
    for f in frames:
        blur = cv2.GaussianBlur(f, (0, 0), sigma)
        out.append(cv2.addWeighted(f, 1.0 + amount, blur, -amount, 0))
    return _ok(f"Applied unsharp masking (amount={amount}, sigma={sigma}) to "
               f"{len(out)} frame(s).", out)


# ---------------------------------------------------------------------------
# 3. image_brightness — CLAHE on LAB L channel
# ---------------------------------------------------------------------------
@_tool
def image_brightness(frames: list, query: str) -> dict:
    import cv2

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = []
    for f in frames:
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        lab = cv2.merge([clahe.apply(l), a, b])
        out.append(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    return _ok(f"Applied CLAHE brightness enhancement to {len(out)} frame(s).", out)


# ---------------------------------------------------------------------------
# 4. image_zoom — center-crop + INTER_CUBIC upscale
# ---------------------------------------------------------------------------
@_tool
def image_zoom(frames: list, query: str) -> dict:
    import cv2

    ratio = 0.5  # crop central 50% region, upscale back to full resolution
    out = []
    for f in frames:
        h, w = f.shape[:2]
        ch, cw = int(h * ratio), int(w * ratio)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        crop = f[y0:y0 + ch, x0:x0 + cw]
        out.append(cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC))
    return _ok(f"Zoomed into the central {int(ratio*100)}% region of "
               f"{len(out)} frame(s).", out)


# ---------------------------------------------------------------------------
# 5. image_super_resolution — Real-ESRGAN (fallback: 4x bicubic + warning)
# ---------------------------------------------------------------------------
def _load_realesrgan(cfg: Config):
    key = "realesrgan"
    if key in _LAZY_CACHE:
        return _LAZY_CACHE[key]
    try:
        from modelscope import snapshot_download
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet

        weights = snapshot_download(cfg.realesrgan_weights_id)
        wpath = None
        for root, _dirs, files in os.walk(weights):
            for name in files:
                if name.endswith(".pth"):
                    wpath = os.path.join(root, name)
                    break
            if wpath:
                break
        if wpath is None:
            raise RuntimeError("no .pth weights found for RealESRGAN")
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(scale=4, model_path=wpath, model=model,
                                 tile=0, pre_pad=0, device="cpu")
        _LAZY_CACHE[key] = ("realesrgan", upsampler)
    except Exception as exc:
        logger.warning("Real-ESRGAN unavailable (%s); falling back to 4x bicubic", exc)
        _LAZY_CACHE[key] = ("bicubic", None)
    return _LAZY_CACHE[key]


@_tool
def image_super_resolution(frames: list, query: str) -> dict:
    import cv2

    mode, upsampler = _load_realesrgan(get_config())
    out = []
    for f in frames:
        if mode == "realesrgan":
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img, _ = upsampler.enhance(rgb, outscale=4)
            out.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            h, w = f.shape[:2]
            out.append(cv2.resize(f, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC))
    note = ("Real-ESRGAN x4" if mode == "realesrgan"
            else "4x bicubic (Real-ESRGAN unavailable)")
    return _ok(f"Applied {note} to {len(out)} frame(s).", out)


# ---------------------------------------------------------------------------
# 6. object_detection — YOLO-World (open-vocabulary; query = classes)
# ---------------------------------------------------------------------------
def _load_yolo_world(cfg: Config):
    key = "yolo_world"
    if key in _LAZY_CACHE:
        return _LAZY_CACHE[key]
    from ultralytics import YOLO

    wpath = None
    try:
        from modelscope import snapshot_download
        d = snapshot_download(cfg.yolo_world_modelscope_id)
        for root, _dirs, files in os.walk(d):
            for name in files:
                if name.endswith(".pt"):
                    wpath = os.path.join(root, name)
                    break
            if wpath:
                break
    except Exception as exc:
        logger.warning("modelscope YOLO-World download failed (%s); "
                       "trying ultralytics auto-download", exc)
    model = YOLO(wpath or "yolov8s-worldv2.pt")  # falls back to github release
    _LAZY_CACHE[key] = model
    return model


@_tool
def object_detection(frames: list, query: str) -> dict:
    cfg = get_config()
    model = _load_yolo_world(cfg)
    classes = [c.strip() for c in query.split(",") if c.strip()]
    if not classes:
        classes = ["person"]
    model.set_classes(classes)
    summaries = []
    for i, f in enumerate(frames):
        results = model.predict(f, verbose=False, device="cpu")
        boxes = results[0].boxes
        found = []
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            found.append(f"{classes[int(cls_id)]}({conf:.2f})")
        summaries.append(f"frame {i}: detected [{', '.join(found) or 'nothing'}]")
    return _ok("YOLO-World open-vocabulary detection with classes "
               f"[{', '.join(classes)}]:\n" + "\n".join(summaries), None)


# ---------------------------------------------------------------------------
# 7. image_retrieval — CLIP retrieval over the current video's extracted frames
# ---------------------------------------------------------------------------
def _load_clip(cfg: Config):
    key = "clip_model"
    if key in _LAZY_CACHE:
        return _LAZY_CACHE[key]
    import torch
    from transformers import CLIPModel, CLIPProcessor

    from .knowledge import _download_from_modelscope

    path = _download_from_modelscope(cfg.clip_modelscope_id, cfg.clip_fallback_ids)
    model = CLIPModel.from_pretrained(path).eval()
    processor = CLIPProcessor.from_pretrained(path)
    _LAZY_CACHE[key] = (model, processor)
    return _LAZY_CACHE[key]


def _clip_video_embeddings(cfg: Config):
    """Lazily embed all extracted frames of the current video (CPU, batched)."""
    import torch
    from PIL import Image

    if _LAZY_CACHE.get("clip_video_id") == _CONTEXT.video_id and \
            "clip_video_emb" in _LAZY_CACHE:
        return _LAZY_CACHE["clip_video_emb"]
    if not _CONTEXT.frame_paths:
        raise RuntimeError("no extracted frames registered for the current video")
    model, processor = _load_clip(cfg)
    embs = []
    batch, batch_paths = [], []

    def _flush():
        if not batch:
            return
        inputs = processor(images=batch, return_tensors="pt")
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        out = out / out.norm(dim=-1, keepdim=True)
        embs.append(out.numpy().astype(np.float32))
        batch_paths.clear()
        batch.clear()

    for p in _CONTEXT.frame_paths:
        batch.append(Image.open(p).convert("RGB"))
        if len(batch) >= 16:
            _flush()
    _flush()
    emb = np.concatenate(embs, axis=0)
    _LAZY_CACHE["clip_video_id"] = _CONTEXT.video_id
    _LAZY_CACHE["clip_video_emb"] = emb
    return emb


@_tool
def image_retrieval(frames: list, query: str) -> dict:
    import torch
    from PIL import Image

    cfg = get_config()
    model, processor = _load_clip(cfg)
    emb = _clip_video_embeddings(cfg)
    if not query.strip():
        query = "abnormal suspicious event"
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        t_emb = model.get_text_features(**inputs)
    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
    sims = (emb @ t_emb.numpy().astype(np.float32).T).ravel()
    top = np.argsort(-sims)[:5]
    lines = [f"frame {int(i)+1} ({os.path.basename(_CONTEXT.frame_paths[int(i)])}): "
             f"similarity {sims[int(i)]:.3f}" for i in top]
    return _ok(f"CLIP retrieval of '{query}' over {len(_CONTEXT.frame_paths)} "
               f"extracted frames, top-5:\n" + "\n".join(lines), None)


# ---------------------------------------------------------------------------
# 8. web_search — Tavily API (stub when TAVILY_API_KEY is unset)
# ---------------------------------------------------------------------------
@_tool
def web_search(frames: list, query: str) -> dict:
    key = os.environ.get("TAVILY_API_KEY", "")
    if not key:
        return {"ok": False, "text_info": "", "frames": None,
                "error": ("TAVILY_API_KEY is not set; web_search is unavailable "
                          "(stub). Re-run with the env var to enable it.")}
    from tavily import TavilyClient

    client = TavilyClient(api_key=key)
    resp = client.search(query=query, max_results=5)
    lines = []
    for r in resp.get("results", []):
        lines.append(f"- {r.get('title', '')}: {r.get('content', '')[:300]}")
    return _ok(f"Web search results for '{query}':\n" + "\n".join(lines), None)


# ---------------------------------------------------------------------------
# Registry + fuzzy name matching (MLLMs emit names like "Image Super-Resolution")
# ---------------------------------------------------------------------------
TOOLS: dict[str, ToolFn] = {
    "image_denoise": image_denoise,
    "image_deblur": image_deblur,
    "image_brightness": image_brightness,
    "image_zoom": image_zoom,
    "image_super_resolution": image_super_resolution,
    "object_detection": object_detection,
    "image_retrieval": image_retrieval,
    "web_search": web_search,
}

_ALIASES = {
    "image denoise": "image_denoise", "denoise": "image_denoise",
    "denoising": "image_denoise", "image_denoising": "image_denoise",
    "image deblur": "image_deblur", "deblur": "image_deblur",
    "deblurring": "image_deblur", "image deblurring": "image_deblur",
    "sharpen": "image_deblur", "unsharp": "image_deblur",
    "image brightness": "image_brightness", "brightness": "image_brightness",
    "brightness enhancement": "image_brightness",
    "brightness_enhancement": "image_brightness",
    "image brightness enhancement": "image_brightness", "clahe": "image_brightness",
    "image zoom": "image_zoom", "zoom": "image_zoom",
    "image super-resolution": "image_super_resolution",
    "image super resolution": "image_super_resolution",
    "super-resolution": "image_super_resolution",
    "super resolution": "image_super_resolution",
    "super_resolution": "image_super_resolution",
    "realesrgan": "image_super_resolution", "esrgan": "image_super_resolution",
    "object detection": "object_detection", "detection": "object_detection",
    "yolo": "object_detection", "yolo-world": "object_detection",
    "yoloworld": "object_detection",
    "image retrieval": "image_retrieval", "image retrieve": "image_retrieval",
    "image_retrieve": "image_retrieval", "retrieval": "image_retrieval",
    "retrieve": "image_retrieval", "clip retrieval": "image_retrieval",
    "web search": "web_search", "websearch": "web_search",
    "search": "web_search", "tavily": "web_search",
}


def resolve_tool(name: str) -> Optional[ToolFn]:
    """Map a (possibly free-form) MLLM tool name to a registered tool."""
    if not name:
        return None
    key = name.strip().lower().replace("-", "_")
    if key in TOOLS:
        return TOOLS[key]
    spaced = key.replace("_", " ")
    if spaced in _ALIASES:
        return TOOLS[_ALIASES[spaced]]
    if key in _ALIASES:
        return TOOLS[_ALIASES[key]]
    # substring match, e.g. "use Image Super-Resolution tool"
    for alias, canon in _ALIASES.items():
        if alias in spaced or spaced in alias:
            return TOOLS[canon]
    return None


_CONFIG: Optional[Config] = None


def init_tools(cfg: Config) -> None:
    global _CONFIG
    _CONFIG = cfg


def get_config() -> Config:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = Config()
    return _CONFIG
