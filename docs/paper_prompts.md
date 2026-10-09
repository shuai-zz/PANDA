# PANDA 论文附录 System Prompts（arXiv:2509.26386v2 附录 E 逐字摘录）

> 实现时以本文件为准。`{placeholder}` 为模板变量。

## 1. Anomaly Knowledge Base Construction Prompt（Fig.6）

```
You are an expert in designing detection rules for video anomaly detection. Based
on the user's specified types of abnormal events, your task is to generate 20
comprehensive and diverse detection rules for each event type.
User Requirement:
{user_query}
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
]
```

## 2. Self-Adaption Environmental Perception Prompt（Fig.7）

```
You are an expert in video anomaly perception. Your task is to perform an initial
understanding and analysis of the provided video frames based on the user's
specified requirements.
User Requirement:
{user_query}
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
}
```

## 3. Anomaly Detection Strategy Planning Prompt（Fig.8）

```
You are a strategic expert in video anomaly detection, specializing in planning
effective detection strategies based on user-defined requirements, video
environment information, and scene-specific anomaly rules. Your role is to
generate an optimal plan that guides the analysis module in accurately detecting
anomalies in the current video scene.
User Requirement:
{user_query}
Video Environment Information:
1. Scene Overview: {env_info.get("Scene Overview", "Unknown")}
2. Weather Condition: {env_info.get("Weather Condition", "Unknown")}
3. Video Quality: {env_info.get("Video Quality", "Unknown")}
4. Potential Anomalies: {env_info.get("Potential Anomalies", "Unknown")}
Anomaly Detection Rules:
{anomaly_rules}
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
}
```

## 4. Goal-Driven Heuristic Reasoning Prompt（Fig.9）

```
You are a highly skilled expert in video anomaly detection, specializing in
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
{user_query}
Historical Detection Info:
{history_result_prompt if history_result_prompt.strip() else "No reliable
historical detection information available."}
Current Video Clip Index:
Clip {index}
Potential Anomalies:
{planning_info.get('potential_anomalies', '')}
Anomaly Detection Rules:
{anomaly_rules}
Heuristic Prompt:
{planning_info.get('heuristic_prompts', '')}
Enhancement and Reflection Information:
{formatted_enhancement_prompt}
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
}
```

## 5. Tool-Augmented Self-Reflection Prompt（Fig.10）

```
You are a reflection assistant within a video anomaly detection system.
The current VLM analysis module has returned "insufficient statu" for determining
whether an abnormal event occurred in the given video clip.
Your task is to critically analyze the situation based on the provided context and
recommend solutions.
Here is the contextual information:
- User Requirement: {user_query}
- Video Environment Information:
**Scene Overview: {env_info.get("Scene Overview", "Unknown")}
**Weather Condition: {env_info.get("Weather Condition", "Unknown")}
**Video Quality: {env_info.get("Video Quality", "Unknown")}
- Anomaly Detection Rules:
{anomaly_rules}
- Potential Anomalies: {planning_info.get('Potential Anomalies', '')}
- Historical Detection Results:
{historical_results}
- Current VLM Output Reason: {reason}
- Information Enhancement tools Already Used: {tools_already_used}
- History Experience: {memory_context}
Based on this information, your tasks are:
1. Analyze and determine the primary reasons for the insufficient information.
2. Recommend which tools from the available options should be used to enhance the
information for better anomaly detection.
{tool_description_text}
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
conclude whether the event is abnormal or normal.
```

## 关键超参数（论文正文 + 附录）

- 采样：1 FPS；clip 长度 s=5 帧；clip 分数广播到 clip 内所有帧
- 感知采样：离线 M=300 帧均匀采样；在线 M=10 帧
- 知识库：每类异常 H=20 条规则；RAG 检索 top-k=5；编码 all-MiniLM-L6-v2 + FAISS
- 反思：最多 r=3 轮；每轮只调用一个工具；r 轮后仍 insufficient 给兜底分（论文未给值，实现取 0.5）
- short CoM：l=5 步（文本 + 视觉记忆）；long CoM：全历史，按 insufficient reason 检索 top-1
- 离线评测：帧级 AUC（UCF-Crime/UBnormal/CSAD）、AP（XD-Violence）；mean filter 窗口=10 时序平滑
- UCF-Crime user query 13 类：Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, Road Accidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism
