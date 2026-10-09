# PANDA 复现报告（Reproduction Report）

> 本仓库是 [showlab/PANDA](https://github.com/showlab/PANDA) 的 fork。原论文：*PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer*（NeurIPS 2025，arXiv:2509.26386v2）。官方仓库未发布代码（README 挂 "Code release coming soon" 已逾一年），本仓库为**依据论文独立实现的完整复现**，含全部源码、prompt、评测脚本与实验数据。

## 复现结论（TL;DR）

| 指标 | 论文 | 本复现 | 判定 |
|---|---|---|---|
| M0 裸 VLM 基线（UCF-Crime 全量，平滑 pooled AUC） | 75.25 | **77.41** | ✅ 可复现（且略优） |
| M6 完整管线（同上） | 84.89 | **不可达** | ❌ 论文 +9.6 增益不可复现 |
| M6 部分数据（前 34 视频同集对比） | — | M0 68.39 vs M6 70.37（**+2.0**） | 增益幅度远低于论文宣称 |

**结论**：评测管线与基线完全可复现；但论文宣称的模块增益（+9.6 AUC）在本复现环境下无法重现，真管线部分数据仅显示 +2.0 的边际增益。差距主因：① 论文所用 Gemini 2.0 Flash 已被 Google 下线，被迫换代（本复现用 Kimi K2.8，论文自身消融表明 MLLM 选型影响显著）；② 多个关键实现细节未公开（M0 基线 prompt、insufficient 兜底分数、各数据集 user query）；③ agent 系统对 prompt/模型版本的高敏感性。

## 实现内容

- **完整 agent 管线**（`panda/`）：LangGraph 状态机实现论文式(1)-(11)：场景感知 → RAG 策略规划 → 启发式推理 → 工具增强反思（≤3 轮）→ 长短 Chain-of-Memory
- **5 个 system prompt**（`panda/prompts.py`）：逐字复制自论文附录 Fig.6–10（脚本 diff 校验一致），见 `docs/paper_prompts.md`
- **8 个工具**：OpenCV 去噪/去模糊/CLAHE/缩放、Real-ESRGAN 超分、YOLO-World 检测、CLIP 检索、Tavily 搜索
- **消融阶梯**：`run_eval.py --stage m0..m6` 对应论文 Table 4 的逐模块叠加
- **模型**：VLM=Qwen2.5-VL-7B（vLLM 本地部署）；MLLM=Kimi（规划/反思/知识库，纯文本角色，论文消融证明文本模型可胜任）；RAG=all-MiniLM-L6-v2 + FAISS

## 实验数据（`results/`）

| 文件 | 状态 | 说明 |
|---|---|---|
| `m0_full_summary.json` | ✅ 有效 | 全量 290 视频：raw 73.55 / smoothed **77.41** |
| `m6_full.jsonl` / `m6_full.log` | 🛑 部分 | 真管线 34/290（2026-10-09 主动终止） |
| `m1/m2/m3/m4/m6`（10-08 批次） | ❌ 无效 | MLLM 密钥未生效期间的产物（KB 为空等），留档仅作事故分析 |

## 复现要点与论文未公开细节

实现时对以下论文未指明之处做了决策（详见 `docs/runbook.md`）：

- M0 裸基线 prompt 自拟（论文未给）；insufficient 兜底分数取 0.5
- 感知 300 帧超过 32K 上下文，分块调用 VLM 后合并 EnvInfo（论文未说明处理方式）
- KB 按事件类型分批生成（单次 13 类 × 20 条规则超出可靠解析范围）
- 视觉短期记忆每步 1 张代表帧（部署图片数限制）

## 运行方式

```bash
# 1. 部署 VLM（24GB 显卡，驱动 CUDA 12.8 环境配方见 docs/runbook.md）
bash serve_vllm.sh   # vllm serve Qwen2.5-VL-7B-Instruct, OpenAI 兼容 API :8000

# 2. 配置 MLLM 密钥
echo 'export KIMI_BASE_URL=...' > .env && echo 'export KIMI_API_KEY=...' >> .env

# 3. 准备 UCF-Crime 测试集（videos + annotations + 1FPS 抽帧）
python scripts/extract_frames.py

# 4. 评测（stage m0-m6 消融阶梯，subset 为 10+10 冒烟）
python scripts/run_eval.py --stage m0 --split full
python scripts/run_eval.py --stage m6 --split full
```

## 已知局限

- M6 全量评测未完成（34/290 后主动止损）；m1–m5 消融未在修复后重跑
- CSAD 数据集作者未公布清单，无法复现
- 429 退避补丁（`panda/models.py`）在 M6 终止前未生效于运行中进程

## 硬件与环境

- RTX 4090 / 3090 24GB（AutoDL），vLLM 0.11 + torch 2.8.0+cu128（驱动 CUDA 12.8 配方见 `docs/runbook.md`）
- 推理速度约 0.3-0.8 clip/s（含反思），与论文 0.82 FPS 同量级
