# PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer

> **本仓库为 fork，包含独立的完整复现实现与复现报告，见 [REPRODUCTION.md](REPRODUCTION.md)。**
> 复现进展：M0 基线复现成功（77.41 vs 论文 75.25）；完整管线因时间/算力限制未完成全量评测，已完成部分显示正向增益（+2.0），代码与任务链可直接续跑。

<!-- <div align="center">
  <img src="assets/panda_logo.png" width="100"/>
</div> -->
**Zhiwei Yang, Chen Gao, Mike Zheng Shou**  

[**NeurIPS 2025**] Accepted Paper  

---

## 🔍 Overview

Video anomaly detection (VAD) is a critical yet challenging task due to the **complex and diverse nature of real-world scenarios**. Existing approaches usually require **domain-specific training data** and heavy manual tuning, making them costly and less generalizable to unseen anomaly types or environments.  

We present **PANDA**, an **agentic AI engineer** built on MLLMs (multimodal large language models), designed to achieve **generalist video anomaly detection**—handling **any scene and anomaly type automatically, without additional training or human intervention**.  

---

## 🚀 Key Contributions

PANDA introduces **four major capabilities** to realize training-free, generalist VAD:

1. **Self-adaptive scene-aware strategy planning**  
   - A RAG-based mechanism retrieves anomaly-specific knowledge for tailored detection strategies.

2. **Goal-driven heuristic reasoning**  
   - Latent anomaly-guided prompts refine reasoning precision.

3. **Tool-augmented self-reflection**  
   - Progressive reflection with tools (e.g., super-resolution, object detection, retrieval, web search) improves decision-making.

4. **Self-improving chain-of-memory (CoM)**  
   - Leverages past experiences for continual performance enhancement across diverse scenarios.

---

## 🧩 System Framework

<div align="center">
  <img src="assets/PANDA_Pipeline.png" width="800"/>
</div>

- **Perception**: VLM extracts environmental and video-specific cues.  
- **Planning**: MLLM designs anomaly detection strategies via rules + heuristic prompts.  
- **Reflection**: Iterative tool invocation (e.g., super-resolution, retrieval, web search) enhances analysis.  
- **Chain-of-Memory**: Past reasoning and decisions are recalled to guide future anomaly detection.  

---

## 🔥 Updates
- [09/2025] Repo initialized.  
- ✨ Code release coming soon!   

---

## 复现实现（Reproduction Implementation）

本仓库现包含论文的完整第三方复现代码（training-free，无任何训练/梯度更新）。
5 个 system prompt 逐字取自论文附录 E（`docs/paper_prompts.md`）。

### 环境

- 远程：AutoDL RTX 4090 24GB，conda 环境 `panda`（python 3.12, torch 2.13, vllm 0.31）
- VLM：本机 vLLM 提供 OpenAI 兼容 API（`http://localhost:8000/v1`，模型 `qwen2.5-vl-7b`）
- MLLM（规划/反思/知识库）：Kimi（`kimi-for-coding`，环境变量 `KIMI_BASE_URL` / `KIMI_API_KEY`）
- 依赖模型均从 modelscope 下载（HF 不可直连）：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 数据准备

```bash
# UCF-Crime 测试集 290 个视频放入 <data_root>/videos/<类别>/，标注放入
# <data_root>/annotations/Temporal_Anomaly_Annotation_for_Testing.txt
python scripts/extract_frames.py --data-root /root/autodl-tmp/data/ucf_crime   # ffmpeg 1 FPS
```

### 运行（消融阶梯对应论文 Table 4，UCF-Crime 帧级 AUC% 目标）

| 命令 | 模块 | 论文目标 |
|---|---|---|
| `python scripts/run_eval.py --stage m0 --split subset` | 裸 user query 直接问 VLM | 75.25 |
| `python scripts/run_eval.py --stage m1 --split subset` | + 策略规划 | 77.01 |
| `python scripts/run_eval.py --stage m2 --split subset` | + 场景感知 | 78.92 |
| `python scripts/run_eval.py --stage m3 --split subset` | + RAG 知识库 | 80.37 |
| `python scripts/run_eval.py --stage m4 --split subset` | + 工具反思 | 82.63 |
| `python scripts/run_eval.py --stage m5 --split subset` | + short CoM | 83.94 |
| `python scripts/run_eval.py --stage m6 --split full`  | + long CoM（完整 PANDA） | 84.89 |

- `--split subset`：normal/abnormal 各前 10 个视频（冒烟测试）；`--split full`：全量 290
- 细粒度消融开关：`--no-perception --no-planning --no-rag --no-reflection --no-short-com --no-long-com`（可叠加在任意 `--stage` 上强制关闭某模块）
- 结果：每视频逐帧分数/标签 + 每 clip 完整记录（reason、反思轨迹）写 JSONL 到 `results/`，并打印 pooled/macro 的 AUC（mean filter 窗口 10，原始+平滑）
- 调试：`--limit N`（只跑前 N 个视频）、`--max-clips N`（每视频只跑前 N 个 clip）

### 代码结构

```
panda/
  config.py      # 全部超参（FPS=1, CLIP_FRAMES=5, H=20, TOP_K=5, r=3, l=5, M=300/10, 平滑窗口 10, 兜底 0.5）
  prompts.py     # 5 个论文附录 prompt 逐字常量 + M0 裸基线 prompt
  models.py      # VLMClient（OpenAI 兼容、多图、t=0、JSON 解析正则兜底+重试）/ MLLMClient（Kimi）
  knowledge.py   # 知识库构建（MLLM 一次性生成，缓存 JSON）+ MiniLM/FAISS RAG
  tools.py       # 8 个工具：去噪/去模糊/CLAHE/放大/Real-ESRGAN/YOLO-World/CLIP 检索/Tavily
  memory.py      # ShortCoM（文本 l=5 + 视觉帧 l=5）、LongCoM（Mt 全历史 + FAISS top-1）
  agent.py       # LangGraph 状态机 perceive -> plan -> reason -> reflect -> tool -> re-reason -> next clip
  data.py        # UCF-Crime 加载、clip 迭代、帧级 GT（fps 从视频元信息读取）
  evaluate.py    # 帧级 AUC/AP + mean filter 平滑
scripts/
  extract_frames.py  # ffmpeg 1 FPS 抽帧
  run_eval.py        # 评测入口（--stage / --no-* / --split / --out）
```

---

## 📜 Citation

If you find PANDA useful for your research, please consider citing:

```bibtex
@inproceedings{yang2025panda,
  title={PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer},
  author={Yang, Zhiwei and Gao, Chen and Shou, Mike Zheng},
  booktitle={NeurIPS},
  year={2025}
}
```
