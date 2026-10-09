"""System prompts for PANDA, copied VERBATIM from docs/paper_prompts.md
(arXiv:2509.26386v2 Appendix E, Fig.6-10).

Placeholders are written as ``$name`` (string.Template) because the verbatim
prompt text itself contains literal ``{...}`` JSON examples that would clash
with ``str.format``.
"""

from __future__ import annotations

import string

# ---------------------------------------------------------------------------
# 1. Anomaly Knowledge Base Construction Prompt (Fig.6)
# ---------------------------------------------------------------------------
KB_CONSTRUCTION_PROMPT = string.Template(
    """You are an expert in designing detection rules for video anomaly detection. Based
on the user's specified types of abnormal events, your task is to generate 20
comprehensive and diverse detection rules for each event type.
User Requirement:
$user_query
Each rule must include the following four fields:
1. Rule ID: A unique identifier for the rule (e.g., "FIG-X001").
2. Event Type: The corresponding abnormal event category (e.g., "Fighting").
3. Rule Description: A concise and clear description of the behavioral pattern
that defines this event.
4. Applicable Scenes: A list of real-world scenarios where this rule may be
applicable (e.g., "Street, Shopping mall, School playground").
The rules should cover a wide range of realistic situations for each event type.
Please output your response in the following structured JSON format:
[{
"Rule ID": "FIG-X001",
"Event Type": "Fighting",
"Rule Description": "Pulling hair or grabbing clothes during struggle",
"Applicable Scenes": "Shopping mall, Playground, Street corner"
},
...
]"""
)

# ---------------------------------------------------------------------------
# 2. Self-Adaption Environmental Perception Prompt (Fig.7)
# ---------------------------------------------------------------------------
PERCEPTION_PROMPT = string.Template(
    """You are an expert in video anomaly perception. Your task is to perform an initial
understanding and analysis of the provided video frames based on the user's
specified requirements.
User Requirement:
$user_query
Please respond by completing the following four aspects:
1. Scene Overview: Describe the environment shown in the video (e.g., shopping
mall, office, street, surveillance corridor) and briefly summarize the main
activities or events observed.
2. Weather Condition: Describe the visual lighting conditions (e.g.,
daytime/nighttime, sunny/overcast, bright/dim).
3. Video Quality: Comment on the overall quality of the video (e.g., clear,
blurry, noisy, low frame rate, low light.etc.).
4. Potential Anomalies: Based on the visual content, what types of abnormal events
are likely to occur in this video (e.g., Fighting, Stealing).
Please return your output strictly in the following JSON format:
{
"Scene Overview": "...",
"Weather Condition": "...",
"Video Quality": "...",
"Potential Anomalies": "..."
}"""
)

# ---------------------------------------------------------------------------
# 3. Anomaly Detection Strategy Planning Prompt (Fig.8)
# ---------------------------------------------------------------------------
PLANNING_PROMPT = string.Template(
    """You are a strategic expert in video anomaly detection, specializing in planning
effective detection strategies based on user-defined requirements, video
environment information, and scene-specific anomaly rules. Your role is to
generate an optimal plan that guides the analysis module in accurately detecting
anomalies in the current video scene.
User Requirement:
$user_query
Video Environment Information:
1. Scene Overview: $scene_overview
2. Weather Condition: $weather_condition
3. Video Quality: $video_quality
4. Potential Anomalies: $potential_anomalies
Anomaly Detection Rules:
$anomaly_rules
Based on the user's requirement, the preliminary video environment information,
and the provided anomaly rules, please design a strategy tailored for this video
scenario. Your response must include the following three components:
1. Preprocessing Recommendations and Pipeline
Suggest a sequence of preprocessing steps (e.g., Image Deblurring, Brightness
Enhancement, Image Denoising) that can help improve video quality and support
better anomaly detection, especially if the video is of poor quality.
2. Potential Anomaly Types
Based on the preliminary video environment information and the given scenerelated anomaly rules, further infer and list the most possible types of anomalies
in this scenario.
3. Heuristic Prompts for VLM
Using the anomaly rules as guidance, craft chain-of-thought–style heuristic
prompts for each potential anomaly type. These prompts are intended to assist
Visual Language Model in performing accurate anomaly judgments.
Please return your output strictly in the following JSON format:
{
"Preprocessing": "Step1 -> Step2 -> ...",
"Potential Anomalies": "Fighting, Stealing, ...",
"Heuristic Prompts": {
"Fighting": "Heuristic prompt with reasoning steps...",
"Stealing": "Heuristic prompt with reasoning steps...",
...
}
}
Example Output:
{
"Preprocessing": "Image Brightness Enhancement -> Image Denoising",
"Potential Anomalies": "Fighting, Stealing",
"Heuristic Prompts": {
"Fighting": "Observe the number of people, their movements, and interactions.
If two individuals are repeatedly making aggressive contact, consider it a
potential fight.",
"Stealing": "Identify solitary individuals interacting with objects,
especially if they conceal items or leave quickly without paying."
}
}"""
)

# ---------------------------------------------------------------------------
# 4. Goal-Driven Heuristic Reasoning Prompt (Fig.9)
# ---------------------------------------------------------------------------
REASONING_PROMPT = string.Template(
    """You are a highly skilled expert in video anomaly detection, specializing in
identifying abnormal events through temporal and spatial analysis of visual
evidence.
Given a sequence of video frames, user requirements, potential anomalies, anomaly
detection rules, heuristic prompts, and enhancement/reflection information, your
task is to assess the likelihood of abnormal events in the current video clip.
You must output:
- A soft anomaly score between **0.0 (clearly normal)** and **1.0 (clearly
abnormal)**.
- A status label from: **"normal"**, **"abnormal"**, or **"insufficient"**.
- A reason justifying your decision.
User Requirement:
$user_query
Historical Detection Info:
$history_result_prompt
Current Video Clip Index:
Clip $clip_index
Potential Anomalies:
$potential_anomalies
Anomaly Detection Rules:
$anomaly_rules
Heuristic Prompt:
$heuristic_prompts
Enhancement and Reflection Information:
$formatted_enhancement_prompt
Your analysis should follow three steps:
1. Describe the main visible actions and interactions between people or objects in
the scene.
2. Assess how strongly these actions match any known abnormal event patterns using
the provided rules and prompts. The provided anomaly rules may not be
comprehensive, so you also apply your own expert reasoning.
3. Based on your assessment, assign a score and label, and explain your reasoning
clearly.
4. If the [Enhancement and Reflection Information] section provides additional
information, you should refer to it emphatically.
Scoring Guidelines:
- A score close to **1.0** indicates clear and confident abnormal behavior.
- A score close to **0.0** indicates clearly normal behavior.
- A score near **0.5** means uncertain, ambiguous behavior or mixed signals.
Examples of valid reasons for "insufficient":
- "The entire scene is too blurry or dark, making it difficult to distinguish any
actions."
- "All persons are either occluded or out of frame."
- "Only partial limbs are visible and motion cues are unclear."
Please strictly output your response in the following JSON format:
{
"score": float, //anomaly score in [0.0, 1.0]
"status": "normal/abnormal/insufficient",
"reason": "A detailed explanation of your reasoning..."
}"""
)

# ---------------------------------------------------------------------------
# 5. Tool-Augmented Self-Reflection Prompt (Fig.10)
# ---------------------------------------------------------------------------
REFLECTION_PROMPT = string.Template(
    """You are a reflection assistant within a video anomaly detection system.
The current VLM analysis module has returned "insufficient statu" for determining
whether an abnormal event occurred in the given video clip.
Your task is to critically analyze the situation based on the provided context and
recommend solutions.
Here is the contextual information:
- User Requirement: $user_query
- Video Environment Information:
**Scene Overview: $scene_overview
**Weather Condition: $weather_condition
**Video Quality: $video_quality
- Anomaly Detection Rules:
$anomaly_rules
- Potential Anomalies: $potential_anomalies
- Historical Detection Results:
$historical_results
- Current VLM Output Reason: $reason
- Information Enhancement tools Already Used: $tools_already_used
- History Experience: $memory_context
Based on this information, your tasks are:
1. Analyze and determine the primary reasons for the insufficient information.
2. Recommend which tools from the available options should be used to enhance the
information for better anomaly detection.
$tool_description_text
3. For any selected tool that requires a 'query' input (e.g., image_retrieve,
web_search), generate an appropriate query based on the context; otherwise leave
the 'query' field empty.
4. Propose a new representative anomaly detection rule derived from the current
situation to better support future VLM analysis.
5. Propose an additional heuristic prompt based on the context and your analysis
to better guide the VLM toward an accurate judgment.
Please output your response in the following structured JSON format:
{
"reason": "...your analysis of why the information is insufficient...",
"tools_to_use": [
{
"tool_name": "One of the most critical tools.",
"query": "generated query if needed, otherwise leave empty"
}
],
},
"new_anomaly_rule": "...a new representative anomaly rule derived from your
analysis and context...",
"new_heuristic_prompt": "...additional guidance to help the VLM make a more
accurate judgment..."
}
Important Notes:
- When calling tools, make sure you don't duplicate any of the information
enhancement tools that have been already applied, and use only one of the most
critical tools at a time.
- If all available information enhancement tools have been exhausted, you should
directly suggest in the 'new_heuristic_prompt' how to guide the VLM analysis
module to make the most reasonable judgment based on incomplete evidence.
- If the current context provides enough information to make a clear judgment,
please directly guide the VLM analysis module in the 'new_heuristic_prompt' to
conclude whether the event is abnormal or normal."""
)

# ---------------------------------------------------------------------------
# M0 bare-baseline prompt.
# NOTE: the paper does NOT provide a prompt for the direct user-query baseline
# (Table 4 row 1). This minimal prompt was authored for the reproduction; it asks
# for the same {score, status, reason} JSON schema as the reasoning stage.
# ---------------------------------------------------------------------------
M0_BARE_PROMPT = string.Template(
    """You are an expert in video anomaly detection. Given a sequence of video frames,
assess the likelihood of abnormal events in the current video clip based solely
on the user's requirement.
User Requirement:
$user_query
Current Video Clip Index:
Clip $clip_index
Assign a soft anomaly score between **0.0 (clearly normal)** and **1.0 (clearly
abnormal)**, a status label from **"normal"**, **"abnormal"**, or **"insufficient"**,
and a reason justifying your decision.
Please strictly output your response in the following JSON format:
{
"score": float, //anomaly score in [0.0, 1.0]
"status": "normal/abnormal/insufficient",
"reason": "A detailed explanation of your reasoning..."
}"""
)

# ---------------------------------------------------------------------------
# Helpers to format structured fields used inside the templates.
# ---------------------------------------------------------------------------

def format_env_info(env_info: dict) -> dict:
    """Map EnvInfo dict (式1) onto the planning/reflection placeholder names,
    using the paper's own defaults ("Unknown") when perception is disabled."""
    env_info = env_info or {}
    return {
        "scene_overview": env_info.get("Scene Overview", "Unknown"),
        "weather_condition": env_info.get("Weather Condition", "Unknown"),
        "video_quality": env_info.get("Video Quality", "Unknown"),
        "potential_anomalies": env_info.get("Potential Anomalies", "Unknown"),
    }


def format_plan_info(plan: dict) -> dict:
    """Map Plan_strategy (式4) onto the reasoning/reflection placeholder names."""
    plan = plan or {}
    return {
        "potential_anomalies": plan.get("Potential Anomalies", ""),
        "heuristic_prompts": _format_heuristic_prompts(plan.get("Heuristic Prompts", {})),
    }


def _format_heuristic_prompts(hp: object) -> str:
    if isinstance(hp, dict):
        return "\n".join(f"- {k}: {v}" for k, v in hp.items())
    return str(hp or "")


def format_rules(rules: list) -> str:
    """Render retrieved knowledge-base rules (式3) as prompt text."""
    if not rules:
        return "None provided."
    lines = []
    for r in rules:
        if isinstance(r, dict):
            lines.append(
                f"- [{r.get('Rule ID', 'N/A')}] {r.get('Event Type', '')}: "
                f"{r.get('Rule Description', '')} "
                f"(Applicable Scenes: {r.get('Applicable Scenes', '')})"
            )
        else:
            lines.append(f"- {r}")
    return "\n".join(lines)


def format_enhancement_prompt(tool_text: str, new_rule: str, new_prompt: str) -> str:
    """Build the [Enhancement and Reflection Information] block (式10)."""
    parts = []
    if tool_text:
        parts.append(f"Tool Enhancement Information:\n{tool_text}")
    if new_rule:
        parts.append(f"New Anomaly Rule:\n{new_rule}")
    if new_prompt:
        parts.append(f"New Heuristic Prompt:\n{new_prompt}")
    if not parts:
        return "None."
    return "\n\n".join(parts)


REFLECTION_TOOL_DESCRIPTIONS = (
    "Available information enhancement tools:\n"
    "- image_denoise: reduce noise in the current clip frames (no query needed).\n"
    "- image_deblur: sharpen blurry frames via unsharp masking (no query needed).\n"
    "- image_brightness: enhance dark/low-contrast frames via CLAHE (no query needed).\n"
    "- image_zoom: magnify the central region of frames (no query needed).\n"
    "- image_super_resolution: 4x super-resolution of frames (no query needed).\n"
    "- object_detection: open-vocabulary detection; query = comma-separated object classes.\n"
    "- image_retrieval: retrieve similar historical keyframes of this video; query = text description.\n"
    "- web_search: search the web for contextual knowledge; query = search text."
)
