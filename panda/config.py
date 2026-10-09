"""Central configuration for the PANDA reproduction.

All hyper-parameters follow the paper (arXiv:2509.26386v2) and docs/paper_prompts.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ---- video sampling (paper §4.1) ----
    fps: int = 1                    # input video sampled at 1 FPS
    clip_frames: int = 5            # clip length s = 5 frames per inference step

    # ---- self-adaptive environmental perception (式1, M=300 offline / M=10 online)
    percept_m_offline: int = 300
    percept_m_online: int = 10
    # Reproduction decision: the paper sends M frames to the VLM in one call,
    # but the local vLLM server caps prompts at `--limit-mm-per-prompt image=8`
    # and the Qwen2.5-VL context is 32k; we still SAMPLE M frames uniformly,
    # then send them in chunks of `vlm_max_images` and merge the per-chunk JSON.
    percept_max_send: int = 32
    vlm_max_images: int = 8

    # ---- anomaly knowledge base + RAG (式2/式3)
    rules_per_type: int = 20        # H = 20 rules per anomaly type
    rag_top_k: int = 5              # retrieve top-k = 5 rules with EnvInfo as query

    # ---- tool-augmented self-reflection (§3.3)
    max_reflect: int = 3            # max reflection rounds r = 3
    insufficient_default_score: float = 0.5  # paper does not specify; sensitivity check in spec §6

    # ---- chain-of-memory (§3.4)
    short_com_len: int = 5          # l = 5 textual + visual steps
    short_com_visual_per_step: int = 1  # representative frames sent per memory step (token control)

    # ---- evaluation ----
    smooth_window: int = 10         # mean filter window for offline temporal smoothing

    # ---- VLM (Qwen2.5-VL-7B via local vLLM) ----
    vlm_base_url: str = os.environ.get("VLM_BASE_URL", "http://localhost:8000/v1")
    vlm_api_key: str = os.environ.get("VLM_API_KEY", "EMPTY")
    vlm_model: str = os.environ.get("VLM_MODEL", "qwen2.5-vl-7b")
    vlm_temperature: float = 0.0
    vlm_max_retries: int = 1        # one retry on JSON parse failure
    vlm_long_side: int = 448        # resize long side to control token count
    vlm_jpeg_quality: int = 85

    # ---- MLLM (planning / reflection; paper uses Gemini 2.0 Flash, we use Kimi)
    mllm_base_url: str = os.environ.get("KIMI_BASE_URL", "https://api.kimi.com/coding/v1")
    mllm_api_key: str = os.environ.get("KIMI_API_KEY", "")
    mllm_model: str = os.environ.get("KIMI_MODEL", "kimi-for-coding")
    mllm_max_tokens: int = 8192     # forced-reasoning model: leave ample head-room
    mllm_temperature: Optional[float] = None  # kimi-for-coding only accepts temperature=1; omit by default
    mllm_think_effort: str = os.environ.get("KIMI_THINK_EFFORT", "low")
    mllm_max_retries: int = 1

    # ---- embedding model (all-MiniLM-L6-v2, loaded from modelscope)
    embed_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_modelscope_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_fallback_ids: tuple = ("AI-ModelScope/all-MiniLM-L6-v2",)

    # ---- tool models ----
    clip_modelscope_id: str = "openai/clip-vit-base-patch32"
    clip_fallback_ids: tuple = ("AI-ModelScope/clip-vit-base-patch32",)
    yolo_world_modelscope_id: str = "AI-ModelScope/yolov8s-worldv2"
    realesrgan_weights_id: str = "RealESRGAN/RealESRGAN_x4plus"

    # ---- paths (remote AutoDL layout; see spec data contract) ----
    data_root: str = os.environ.get("PANDA_DATA_ROOT", "/root/autodl-tmp/data/ucf_crime")
    cache_dir: str = os.environ.get("PANDA_CACHE_DIR", "cache")
    results_dir: str = os.environ.get("PANDA_RESULTS_DIR", "results")

    # UCF-Crime 13 anomaly classes (user query, paper Fig.2)
    ucf_event_types: tuple = (
        "Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion",
        "Fighting", "Road Accidents", "Robbery", "Shooting", "Shoplifting",
        "Stealing", "Vandalism",
    )

    def __post_init__(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)


UCF_USER_QUERY = (
    "Please help me detect the following types of abnormal events: "
    "Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, "
    "Road Accidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism."
)


@dataclass
class AblationConfig:
    """Module switches. M0 = all off (bare user-query VLM baseline);
    M6 = all on (full PANDA). Mapping to paper Table 4 ladder:

        M0: all off                      (75.25 AUC)
        M1: + planning                   (77.01)
        M2: + perception                 (78.92)
        M3: + rag (knowledge base)       (80.37)
        M4: + reflection                 (82.63)
        M5: + short_com                  (83.94)
        M6: + long_com (full PANDA)      (84.89)
    """
    perception: bool = True
    planning: bool = True
    rag: bool = True
    reflection: bool = True
    short_com: bool = True
    long_com: bool = True

    @classmethod
    def for_stage(cls, stage: str) -> "AblationConfig":
        stage = stage.lower()
        ladder = {
            "m0": dict(perception=False, planning=False, rag=False,
                       reflection=False, short_com=False, long_com=False),
            "m1": dict(perception=False, planning=True, rag=False,
                       reflection=False, short_com=False, long_com=False),
            "m2": dict(perception=True, planning=True, rag=False,
                       reflection=False, short_com=False, long_com=False),
            "m3": dict(perception=True, planning=True, rag=True,
                       reflection=False, short_com=False, long_com=False),
            "m4": dict(perception=True, planning=True, rag=True,
                       reflection=True, short_com=False, long_com=False),
            "m5": dict(perception=True, planning=True, rag=True,
                       reflection=True, short_com=True, long_com=False),
            "m6": dict(perception=True, planning=True, rag=True,
                       reflection=True, short_com=True, long_com=True),
        }
        if stage not in ladder:
            raise ValueError(f"unknown stage {stage!r}, expected one of {sorted(ladder)}")
        return cls(**ladder[stage])


def get_config() -> Config:
    return Config()
