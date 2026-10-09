# PANDA 复现 Spec

> 基于论文 arXiv:2509.26386v2（NeurIPS 2025）。官方仓库未开源代码（仅 README），本文档定义复现范围、模块拆分、里程碑与验收标准。

## 1. 项目本质

纯 training-free 的 agent harness：无任何训练/梯度更新，全部组件为预训练模型推理 + prompt 编排。复现工作 = LangGraph 状态机 + 5 个 prompt（论文附录 Fig.6–10 全文给出）+ 8 个工具封装 + 长短记忆存取 + 评测管线。

## 2. 复现策略：以论文消融表为阶梯

论文附录 C.1（Table 4）给出了天然的渐进验收曲线（UCF-Crime AUC%），每个里程碑加一层模块，误差 ±1 AUC 内视为复现成功：

| 里程碑 | 叠加模块 | 论文目标 AUC |
|---|---|---|
| M0 | 裸基线：user query 直接问 VLM | 75.25 |
| M1 | + 检测策略规划 | 77.01 |
| M2 | + 自适应场景感知 | 78.92 |
| M3 | + RAG 异常知识库 | 80.37 |
| M4 | + 工具增强自我反思 | 82.63 |
| M5 | + short CoM | 83.94 |
| M6 | + long CoM（完整 PANDA） | 84.89 |

## 3. 最小复现范围（M0，第一阶段）

**目标**：先把"评测管线 + VLM 裸推理"跑通，拿到 ~75 AUC 的基线，证明数据、采样、打分、AUC 计算全链路正确。

包含内容：

1. **数据管线**：UCF-Crime 测试集（150 normal + 140 abnormal），帧级标注读取；1 FPS 采样；每 clip s=5 帧为一个推理单元；clip 分数广播给 clip 内所有帧
2. **VLM 推理**：Qwen2.5-VL-7B（本地，bf16），输出严格 JSON `{score, status, reason}`
3. **评测**：帧级 AUC，离线模式加 mean filter（window=10）时序平滑
4. **user query**：UCF-Crime 13 类异常（Abuse, Arrest, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism）
5. 先在一个小子集（如 10 normal + 10 abnormal）上 sanity check，再跑全量

不包含（后续里程碑再做）：规划、感知、RAG、反思、记忆、工具集、在线模式。

## 4. 完整模块清单（M1–M6 逐步叠加）

### M1 策略规划（Planning）
- MLLM 生成检测策略 `{Preprocessing, Potential Anomalies, Heuristic Prompts}`
- prompt：附录 Fig.8 全文
- 论文 MLLM = Gemini 2.0 Flash（已弃用，**替换决策见 §6**）

### M2 场景感知（Perception）
- 均匀采样 M=300 帧（离线）/ M=10 帧（在线）送 VLM
- 输出 `{Scene Overview, Weather Condition, Video Quality, Potential Anomalies}`
- prompt：附录 Fig.7 全文

### M3 RAG 异常知识库
- 知识库构建：MLLM 按 user query 每类异常生成 H=20 条规则（prompt 附录 Fig.6）
- 编码：all-MiniLM-L6-v2；索引：FAISS；以 EnvInfo 为 query 检索 top-k=5 条规则

### M4 工具增强自我反思
- 触发条件：status == "insufficient"；最多 r=3 轮；兜底分数（论文未给值，先取 0.5）
- 反思输出 `{reason, tools_to_use, new_anomaly_rule, new_heuristic_prompt}`（prompt 附录 Fig.10）
- 工具集（附录 D）：
  - 图像增强：OpenCV fastNLMeans 去噪 / unsharp masking 去模糊 / CLAHE 亮度 / bicubic 缩放
  - Real-ESRGAN 超分（预训练）
  - YOLO-World 目标检测（预训练，开放词表）
  - CLIP 图文检索（历史关键帧库）
  - Tavily web 搜索（需 API key；可先做 stub，仅联网类 case 受影响）
- 每轮反思后用增强信息重推理（prompt 附录 Fig.9）

### M5 short CoM
- 文本记忆：最近 l=5 步推理结果；视觉记忆：最近 l=5 步对应帧
- 反思阶段的 short CoM = 历史反思输出集合

### M6 long CoM
- 每时间步记忆单元 Mt = {推理结果, 反思结果, 重推理结果}
- 反思时用 insufficient reason 检索最相似历史经验（RetrieveTop1）
- 视频开始时为空，随处理逐步累积

## 5. 技术栈与环境

- Python + PyTorch + LangGraph（状态机编排）
- Qwen2.5-VL-7B：transformers 或 vLLM 本地推理，1×24GB GPU 可跑
- MLLM（规划/反思）：见 §6 决策
- 依赖模型全部 HF 可下载：Qwen2.5-VL-7B、all-MiniLM-L6-v2、YOLO-World、Real-ESRGAN、CLIP

## 6. 待决策项

1. **MLLM 选型（已定）**：使用 Kimi（`kimi-for-coding` / K2.8），走环境变量 `KIMI_BASE_URL` + `KIMI_API_KEY`（已配置在 ~/.bashrc）。
   - 依据：论文消融（Table 5a）证明 MLLM 角色只处理文本，DeepSeek-V3（纯文本）84.72 vs 原配置 84.89，换代影响可忽略；Kimi 支持文本/图像/视频输入、1M 上下文，能力覆盖有余
   - 注意：该模型为强制推理模型（reasoning 默认 max 档位），实现时 `max_tokens` 需留足余量，并评估调低 think effort 以控制反思环节的延迟与成本
   - 备选：Gemini 2.5 Flash / 本地 Qwen2.5-72B（如需对照实验）
2. **CSAD 数据集**：作者未公布 100 个视频清单，无法精确复现；计划按论文描述自行从 UCF/XD/UB 采样重建一个近似版本，数字仅供内部参考
3. **XD-Violence / UBnormal 的 user query**：论文只给了 UCF 13 类示例，其余按各数据集类别自行构造
4. **insufficient 兜底分数**：默认 0.5，跑通后做敏感性验证

## 7. 复现风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| Qwen2.5-VL-7B JSON 输出不稳定 | 解析失败、分数丢失 | 重试 + 正则兜底提取；温度设 0 |
| MLLM 换代（2.0→2.5 Flash）行为漂移 | M1–M6 数字偏移 | 消融表给了 4 个 MLLM 的对照，漂移可量化 |
| 全量 UCF 评测耗时（~0.8 FPS） | 单次全量数天 | 子集先行；vLLM 加速；多卡并行 |
| Tavily key 缺失 | web 搜索工具不可用 | stub 实现，反思时跳过该工具 |

## 8. 验收标准

- M0 通过：UCF-Crime 测试集（或标注明确的子集）帧级 AUC 落在 74–76 区间
- 最终通过：M6 完整管线 UCF AUC ≥ 83.5；XD-Violence AP、UBnormal AUC 与论文同量级（±1.5）
