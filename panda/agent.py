"""PANDA agent: LangGraph state machine implementing the paper pipeline.

Flow (paper §3, Fig.2):
    EnvInfo = VLM(F, Prompt_perception)                                    (式1)
    κa      = MLLM(User_query, Prompt_know)                                (式2)
    Rules_a = RetrieveTopK(EnvInfo, κa)                                    (式3)
    Plan    = MLLM(Prompt_plan)                                            (式4)
    per clip:
      V̂e_clip = Preprocessing(V_clip)                                      (式5)
      Result  = VLM(e_ct, Memory_visual, Prompt_reasoning)                 (式6)
      if insufficient:
        Experience  = RetrieveTop1(reason, LongCoM)                        (式7)
        Reflection  = MLLM(Prompt_reflection)                              (式8)
        Tool result = ToolInvoke(tool)                                     (式9)
        Prompt_refined = Prompt ∪ {Text Info, New Rule, New Prompt}        (式10)
        Re-reason  = VLM(...)                                              (式11)
        (loop, max r rounds, fallback score 0.5)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from .config import AblationConfig, Config
from .knowledge import KnowledgeBase, env_info_text
from .memory import LongCoM, ShortCoM
from .models import MLLMClient, VLMClient
from .prompts import (
    M0_BARE_PROMPT,
    PERCEPTION_PROMPT,
    PLANNING_PROMPT,
    REASONING_PROMPT,
    REFLECTION_PROMPT,
    REFLECTION_TOOL_DESCRIPTIONS,
    format_enhancement_prompt,
    format_env_info,
    format_plan_info,
    format_rules,
)
from .tools import resolve_tool, set_tool_context

logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    # video-level (set once)
    video_id: str
    frame_paths: list          # all extracted 1-FPS frame paths
    clips: list                # list of clips; each clip is a list of frame paths
    env_info: dict
    plan: dict
    rules: list
    # current-clip working state
    clip_idx: int
    current_frames: list       # BGR ndarrays fed to the VLM (clip or enhanced)
    status: str
    score: float
    reason: str
    reflection_round: int
    tools_used: list
    enhancement_text: str
    new_anomaly_rule: str
    new_heuristic_prompt: str
    last_reflection: dict
    # outputs
    clip_records: list
    frame_scores: list


class PANDAAgent:
    def __init__(self, cfg: Config, ablation: AblationConfig,
                 vlm: VLMClient, mllm: MLLMClient,
                 kb: Optional[KnowledgeBase] = None,
                 user_query: str = ""):
        self.cfg = cfg
        self.ablation = ablation
        self.vlm = vlm
        self.mllm = mllm
        self.kb = kb
        self.user_query = user_query
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ graph
    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("init", self._node_init)
        g.add_node("perceive", self._node_perceive)
        g.add_node("plan", self._node_plan)
        g.add_node("load_clip", self._node_load_clip)
        g.add_node("reason", self._node_reason)
        g.add_node("reflect", self._node_reflect)
        g.add_node("tool", self._node_tool)
        g.add_node("rereason", self._node_rereason)
        g.add_node("finalize", self._node_finalize)

        ab = self.ablation
        g.set_entry_point("init")
        g.add_edge("init", "perceive" if ab.perception else ("plan" if ab.planning else "load_clip"))
        g.add_edge("perceive", "plan" if ab.planning else "load_clip")
        g.add_edge("plan", "load_clip")
        g.add_edge("load_clip", "reason")
        g.add_conditional_edges("reason", self._after_reason,
                                {"reflect": "reflect", "finalize": "finalize"})
        g.add_edge("reflect", "tool")
        g.add_edge("tool", "rereason")
        g.add_conditional_edges("rereason", self._after_rereason,
                                {"reflect": "reflect", "finalize": "finalize"})
        g.add_conditional_edges("finalize", self._after_finalize,
                                {"load_clip": "load_clip", END: END})
        return g.compile()

    # ------------------------------------------------------------------ video
    def run_video(self, video_id: str, frame_paths: list) -> dict:
        """Process one video; returns per-frame scores + clip records."""
        self.short_com = ShortCoM(self.cfg)
        self.long_com = LongCoM(self.cfg)
        set_tool_context(video_id, frame_paths)
        clips = [frame_paths[i:i + self.cfg.clip_frames]
                 for i in range(0, len(frame_paths), self.cfg.clip_frames)]
        init_state: AgentState = {
            "video_id": video_id,
            "frame_paths": frame_paths,
            "clips": clips,
            "env_info": {},
            "plan": {},
            "rules": [],
            "clip_idx": 0,
            "clip_records": [],
            "frame_scores": [],
        }
        final_state = self.graph.invoke(init_state)
        return {
            "video_id": video_id,
            "frame_scores": final_state["frame_scores"],
            "clip_records": final_state["clip_records"],
            "env_info": final_state.get("env_info", {}),
            "plan": final_state.get("plan", {}),
        }

    # ------------------------------------------------------------------ nodes
    def _node_init(self, state: AgentState) -> dict:
        logger.info("=== video %s: %d frames, %d clips ===",
                    state["video_id"], len(state["frame_paths"]),
                    len(state["clips"]))
        return {}

    def _node_perceive(self, state: AgentState) -> dict:
        """Environmental perception (式1): uniformly sample M frames -> VLM."""
        cfg = self.cfg
        n = len(state["frame_paths"])
        m = min(cfg.percept_m_offline, n)
        idxs = np.linspace(0, n - 1, m).astype(int)
        sampled = [state["frame_paths"][i] for i in idxs]
        # token-budget guard: the local Qwen2.5-VL context cannot hold M=300
        # frames in one call, so cap the number actually sent.
        max_send = getattr(cfg, "percept_max_send", 32)
        if len(sampled) > max_send:
            stride = np.linspace(0, len(sampled) - 1, max_send).astype(int)
            sampled = [sampled[i] for i in stride]
            logger.info("perception: sampled %d frames, sending %d (token budget)",
                        m, len(sampled))
        prompt = PERCEPTION_PROMPT.substitute(user_query=self.user_query)
        env_info = self._perception_env_info(prompt, sampled)
        logger.info("perception done: %s", env_info)
        return {"env_info": env_info}

    def _perception_env_info(self, prompt: str, sampled: list) -> dict:
        """Ask the VLM for the perception JSON object (fields of 式1).

        The sampled frames are sent in chunks of `vlm_max_images` (the vLLM
        server caps prompts at 8 images); per-chunk JSON outputs are merged.
        """
        from .models import extract_json_object

        chunk = max(1, getattr(self.cfg, "vlm_max_images", 8))
        merged: dict = {}
        fields = ("Scene Overview", "Weather Condition", "Video Quality",
                  "Potential Anomalies")
        for start in range(0, len(sampled), chunk):
            part = sampled[start:start + chunk]
            for attempt in range(self.cfg.vlm_max_retries + 1):
                try:
                    text = self.vlm.chat(prompt, part)
                except Exception as exc:
                    logger.warning("perception request failed (chunk %d, attempt %d)"
                                   ": %s", start // chunk, attempt + 1, exc)
                    continue
                obj = extract_json_object(text)
                if isinstance(obj, dict):
                    for f in fields:
                        v = str(obj.get(f, "")).strip()
                        if v and v.lower() not in ("unknown", "none", "null"):
                            if f not in merged:
                                merged[f] = v
                            elif v not in merged[f].split("; "):
                                merged[f] = merged[f] + "; " + v
                    break
                logger.warning("perception output not parseable (chunk %d, "
                               "attempt %d): %.200s", start // chunk,
                               attempt + 1, text)
        return merged

    def _node_plan(self, state: AgentState) -> dict:
        """RAG retrieval (式3) + strategy planning (式4)."""
        env_info = state.get("env_info") or {}
        rules: list = []
        if self.ablation.rag and self.kb is not None:
            rules = self.kb.retrieve(env_info_text(env_info), self.cfg.rag_top_k)
            logger.info("planning: retrieved %d rules", len(rules))
        if not self.ablation.planning:
            return {"rules": rules}
        prompt = PLANNING_PROMPT.substitute(
            user_query=self.user_query,
            anomaly_rules=format_rules(rules),
            **format_env_info(env_info),
        )
        plan = self.mllm.generate_json(prompt)
        if not isinstance(plan, dict):
            logger.warning("planning output not parseable; using empty plan")
            plan = {}
        logger.info("planning done: preprocessing=%s, anomalies=%s",
                    plan.get("Preprocessing", ""), plan.get("Potential Anomalies", ""))
        return {"rules": rules, "plan": plan}

    def _node_load_clip(self, state: AgentState) -> dict:
        clip_idx = state["clip_idx"]
        clip_paths = state["clips"][clip_idx]
        frames = [self._read(p) for p in clip_paths]
        frames = [f for f in frames if f is not None]
        # Preprocessing (式5): apply the plan's first recognized preprocessing
        # step to the current clip (visual enhancement is clip-level in our impl).
        if self.ablation.planning and state.get("plan") and frames:
            pre = str(state["plan"].get("Preprocessing", "") or "")
            tool = self._preprocessing_tool(pre, state["plan"].get("_preproc_tool"))
            if tool is not None:
                res = tool(frames, "")
                if res.get("ok") and res.get("frames"):
                    logger.info("clip %d: preprocessing '%s' applied", clip_idx, pre)
                    frames = res["frames"]
        return {
            "clip_idx": clip_idx,
            "current_frames": frames,
            "status": "", "score": 0.5, "reason": "",
            "reflection_round": 0,
            "tools_used": [], "enhancement_text": "",
            "new_anomaly_rule": "", "new_heuristic_prompt": "",
            "last_reflection": {},
        }

    def _preprocessing_tool(self, preprocessing: str, cached):
        if cached is not None:
            return cached
        if not preprocessing or preprocessing.strip().lower() in ("none", "no", "n/a"):
            return None
        for step in preprocessing.replace("->", ",").split(","):
            tool = resolve_tool(step.strip())
            if tool is not None:
                return tool
        return None

    def _node_reason(self, state: AgentState) -> dict:
        prompt, images = self._reasoning_input(state, first=True)
        result = self.vlm.detect(prompt, images)
        logger.info("clip %d reason: score=%.3f status=%s",
                    state["clip_idx"], result["score"], result["status"])
        return {"status": result["status"], "score": result["score"],
                "reason": result["reason"]}

    def _reasoning_input(self, state: AgentState, first: bool):
        """Build the reasoning prompt (式6 / 式10) and image list."""
        clip_idx = state["clip_idx"]
        if not self.ablation.planning:
            prompt = M0_BARE_PROMPT.substitute(
                user_query=self.user_query, clip_index=clip_idx)
            return prompt, list(state["current_frames"])

        history = (self.short_com.text_prompt() if self.ablation.short_com
                   else "No reliable historical detection information available.")
        enhancement = "None." if first else format_enhancement_prompt(
            state.get("enhancement_text", ""),
            state.get("new_anomaly_rule", ""),
            state.get("new_heuristic_prompt", ""))
        plan_info = format_plan_info(state.get("plan"))
        prompt = REASONING_PROMPT.substitute(
            user_query=self.user_query,
            history_result_prompt=history,
            clip_index=clip_idx,
            potential_anomalies=plan_info["potential_anomalies"],
            anomaly_rules=format_rules(state.get("rules")),
            heuristic_prompts=plan_info["heuristic_prompts"],
            formatted_enhancement_prompt=enhancement,
        )
        images = list(state["current_frames"])
        if self.ablation.short_com:
            for p in self.short_com.visual_frames():
                img = self._read(p)
                if img is not None:
                    images.append(img)
        return prompt, images

    def _after_reason(self, state: AgentState) -> str:
        if (state["status"] == "insufficient" and self.ablation.reflection
                and state["reflection_round"] < self.cfg.max_reflect):
            return "reflect"
        return "finalize"

    def _node_reflect(self, state: AgentState) -> dict:
        """Experience-driven reflection (式7/式8)."""
        round_idx = state["reflection_round"] + 1
        env = format_env_info(state.get("env_info"))
        plan_info = format_plan_info(state.get("plan"))
        # 式7: retrieve the most similar past experience from Long CoM
        memory_context = "No relevant history experience."
        if self.ablation.long_com:
            unit = self.long_com.retrieve_top1(state["reason"])
            memory_context = self.long_com.format_experience(unit)
        history = (self.short_com.text_prompt() if self.ablation.short_com
                   else "No historical detection results.")
        prompt = REFLECTION_PROMPT.substitute(
            user_query=self.user_query,
            anomaly_rules=format_rules(state.get("rules")),
            potential_anomalies=plan_info["potential_anomalies"],
            historical_results=history,
            reason=state["reason"],
            tools_already_used=", ".join(state["tools_used"]) or "None",
            memory_context=memory_context,
            tool_description_text=REFLECTION_TOOL_DESCRIPTIONS,
            scene_overview=env["scene_overview"],
            weather_condition=env["weather_condition"],
            video_quality=env["video_quality"],
        )
        refl = self.mllm.generate_json(prompt)
        if not isinstance(refl, dict):
            logger.warning("reflection output not parseable; using empty reflection")
            refl = {"reason": "reflection parse failure", "tools_to_use": [],
                    "new_anomaly_rule": "", "new_heuristic_prompt": ""}
        logger.info("clip %d reflection round %d: tools=%s",
                    state["clip_idx"], round_idx, refl.get("tools_to_use"))
        # reflection-stage short CoM = set of past reflection outputs
        if self.ablation.short_com:
            self.short_com.add_reflection(round_idx, refl)
        return {"reflection_round": round_idx, "last_reflection": refl}

    def _node_tool(self, state: AgentState) -> dict:
        """Tool invocation (式9): one most-critical tool per round (paper note)."""
        refl = state.get("last_reflection") or {}
        tool_specs = refl.get("tools_to_use") or []
        if isinstance(tool_specs, dict):
            tool_specs = [tool_specs]
        if not isinstance(tool_specs, list):
            tool_specs = []
        tool_specs = [t for t in tool_specs if isinstance(t, dict)]
        if not tool_specs:
            logger.info("clip %d: reflection suggested no tools", state["clip_idx"])
            return {}
        spec = tool_specs[0]  # one most critical tool at a time (paper Fig.10 note)
        name = str(spec.get("tool_name", ""))
        query = str(spec.get("query", "") or "")
        tool = resolve_tool(name)
        if tool is None:
            logger.warning("clip %d: unknown tool '%s'", state["clip_idx"], name)
            return {}
        tools_used = list(state["tools_used"]) + [name]
        res = tool(list(state["current_frames"]), query)
        logger.info("clip %d tool '%s': ok=%s", state["clip_idx"], name, res.get("ok"))
        enhancement_text = state.get("enhancement_text", "")
        if res.get("ok"):
            info = f"[{name}] {res.get('text_info', '')}"
            enhancement_text = (enhancement_text + "\n" + info).strip()
        else:
            enhancement_text = (enhancement_text +
                                f"\n[{name}] tool failed: {res.get('error')}").strip()
        update: dict = {
            "tools_used": tools_used,
            "enhancement_text": enhancement_text,
            "new_anomaly_rule": str(refl.get("new_anomaly_rule", "") or ""),
            "new_heuristic_prompt": str(refl.get("new_heuristic_prompt", "") or ""),
        }
        if res.get("ok") and res.get("frames"):
            update["current_frames"] = res["frames"]  # Visual Enhancement Info ĉt
        return update

    def _node_rereason(self, state: AgentState) -> dict:
        """Refined reasoning (式10/式11)."""
        prompt, images = self._reasoning_input(state, first=False)
        result = self.vlm.detect(prompt, images)
        logger.info("clip %d re-reason (round %d): score=%.3f status=%s",
                    state["clip_idx"], state["reflection_round"],
                    result["score"], result["status"])
        return {"status": result["status"], "score": result["score"],
                "reason": result["reason"]}

    def _after_rereason(self, state: AgentState) -> str:
        if (state["status"] == "insufficient"
                and state["reflection_round"] < self.cfg.max_reflect):
            return "reflect"
        return "finalize"

    def _node_finalize(self, state: AgentState) -> dict:
        clip_idx = state["clip_idx"]
        clip_paths = state["clips"][clip_idx]
        score = float(state["score"])
        status = state["status"]
        if status == "insufficient":
            # paper §3.3: after r rounds still insufficient -> default score
            score = self.cfg.insufficient_default_score
            logger.info("clip %d: still insufficient after %d rounds; "
                        "fallback score %.2f", clip_idx, state["reflection_round"],
                        score)
        frame_scores = list(state["frame_scores"]) + [score] * len(clip_paths)
        record = {
            "clip_idx": clip_idx,
            "n_frames": len(clip_paths),
            "frame_start": clip_idx * self.cfg.clip_frames,
            "score": score,
            "status": "insufficient_fallback" if status == "insufficient" else status,
            "reason": state["reason"],
            "reflection_rounds": state["reflection_round"],
            "tools_used": state["tools_used"],
            "reflection": state.get("last_reflection", {}),
        }
        clip_records = list(state["clip_records"]) + [record]
        # chain-of-memory updates (§3.4)
        if self.ablation.short_com:
            self.short_com.add_reasoning(clip_idx, record, clip_paths)
        if self.ablation.long_com:
            self.long_com.add({
                "clip_idx": clip_idx,
                "reasoning": {"score": score, "status": status,
                              "reason": state["reason"]},
                "reflection": state.get("last_reflection", {}),
                "refined_reasoning": {"score": score, "status": status,
                                      "reason": state["reason"]}
                if state.get("reflection_round") else {},
            })
        return {"frame_scores": frame_scores, "clip_records": clip_records,
                "clip_idx": clip_idx + 1}

    def _after_finalize(self, state: AgentState) -> str:
        if state["clip_idx"] < len(state["clips"]):
            return "load_clip"
        logger.info("=== video %s done: %d clips ===",
                    state["video_id"], len(state["clips"]))
        return END

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _read(path: str) -> Optional[np.ndarray]:
        import cv2

        img = cv2.imread(path)
        if img is None:
            logger.warning("failed to read frame %s", path)
        return img
