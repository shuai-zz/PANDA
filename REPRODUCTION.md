# PANDA 复现报告（Reproduction Report）

> 本仓库是 [showlab/PANDA](https://github.com/showlab/PANDA) 的 fork。原论文：*PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer*（NeurIPS 2025，arXiv:2509.26386v2）。官方仓库暂未发布代码，本仓库为**依据论文独立实现的完整复现**，含全部源码、prompt、评测脚本与已完成的实验数据。

## 复现进展（TL;DR）

| 阶段 | 论文 | 本复现 | 状态 |
|---|---|---|---|
| M0 裸 VLM 基线（UCF-Crime 全量 290 视频，平滑 pooled AUC） | 75.25 | **77.41** | ✅ **复现成功** |
| M6 完整管线（同上） | 84.89 | 部分数据：前 34 视频同集对比 M0 68.39 → M6 **70.37（+2.0）** | ⏸ 因时间与算力限制未完成全量 |

**当前结论**：评测管线与基线已完整复现，全量数字与论文基线高度一致（77.41 vs 75.25）。完整管线（M6）在已完成的难类别视频（Abuse/Arrest/Arson/Burglary）上相对基线呈**正向增益（+2.0）**，方向与论文一致；受 GPU 租用时长与 API 配额限制，全量 290 视频的消融阶梯（M1–M6）验证尚未完成，欢迎有兴趣的同学基于本实现继续。

## 实现内容

- **完整 agent 管线**（`panda/`）：LangGraph 状态机实现论文式(1)-(11)：场景感知 → RAG 策略规划 → 启发式推理 → 工具增强反思（≤3 轮）→ 长短 Chain-of-Memory
- **5 个 system prompt**（`panda/prompts.py`）：逐字复制自论文附录 Fig.6–10（脚本 diff 校验一致），存档于 `docs/paper_prompts.md`
- **8 个工具**：OpenCV 去噪/去模糊/CLAHE/缩放、Real-ESRGAN 超分、YOLO-World 检测、CLIP 检索、Tavily 搜索
- **消融阶梯**：`run_eval.py --stage m0..m6` 对应论文 Table 4 的逐模块叠加
- **模型**：VLM=Qwen2.5-VL-7B（vLLM 本地部署，24GB 单卡）；MLLM=Kimi（规划/反思/知识库——该角色在论文中实际只处理文本，论文消融 Table 5a 表明文本模型可胜任）；RAG=all-MiniLM-L6-v2 + FAISS

## 实验数据（`results/`）

| 文件 | 状态 | 说明 |
|---|---|---|
| `m0_full_summary.json` + `m0_full.jsonl` | ✅ 有效 | 全量 290 视频：raw 73.55 / smoothed **77.41**，含全部帧级分数 |
| `m6_full.jsonl` / `m6_full.log` | ⏸ 部分 | 真管线 34/290（含真实反思/工具调用轨迹），后因算力预算终止 |
| `m1/m2/m3/m4/m6`（10-08 批次） | ⚠️ 留档 | 工程事故期间的无效产物（MLLM 密钥未注入导致规划/知识库为空），仅作事故分析参考 |

## 复现要点（论文未公开细节的处理）

论文以下细节未公开，实现时的决策如下（完整记录见 `docs/runbook.md`）：

- **M0 基线 prompt**：论文未给出，本实现自拟（输出 schema 与式 6 一致）
- **MLLM 选型**：论文所用 Gemini 2.0 Flash 已被 Google 下线，本实现以 Kimi 替代（论文消融表明该角色可用文本模型，GPT-4o/DeepSeek-V3 均为候选）
- **感知 300 帧**：超过模型 32K 上下文，分块调用 VLM 后合并 EnvInfo
- **知识库构建**：按事件类型分批生成（单次 13 类 × 20 条超出可靠解析范围）
- **insufficient 兜底分数**：论文未给值，取 0.5

## 运行方式

```bash
# 1. 部署 VLM（24GB 显卡；驱动 CUDA 12.8 环境的完整配方见 docs/runbook.md）
bash serve_vllm.sh   # vllm serve Qwen2.5-VL-7B-Instruct, OpenAI 兼容 API :8000

# 2. 配置 MLLM 密钥
echo 'export KIMI_BASE_URL=...' > .env && echo 'export KIMI_API_KEY=...' >> .env

# 3. 准备 UCF-Crime 测试集（videos + annotations + 1FPS 抽帧）
python scripts/extract_frames.py

# 4. 评测（stage m0-m6 消融阶梯，subset 为 10+10 冒烟）
python scripts/run_eval.py --stage m0 --split full
python scripts/run_eval.py --stage m6 --split full
```

## 已知局限与后续工作

- M6 全量及 M1–M5 消融阶梯受算力预算限制未跑完，代码与任务链（`scripts/run_all2.sh`）可直接续跑
- CSAD 数据集作者未公布视频清单，暂无法覆盖该基准
- 推理开销：含反思的全管线约 0.3 clip/s（3090），全量单阶段约 14–20 小时；无反思阶段约 2.5 小时

## 硬件与环境

- RTX 4090 / 3090 24GB（AutoDL），vLLM 0.11 + torch 2.8.0+cu128（驱动 CUDA 12.8 的完整依赖配方见 `docs/runbook.md`）
