# PANDA 复现 — 暂停/恢复手册

> 状态快照：**2026-10-09 22:30 用户决定止损，项目暂停**。恢复时按下方步骤即可。

## 当前进度

- ✅ M0 全量完成且有效：pooled AUC raw 73.55 / smoothed **77.41**（`results/m0_full_summary.json`）
- 🛑 M6 真管线终止于 34/290（部分结果在 `results/m6_full.jsonl`；前 32 视频同集对比 M0 68.39 vs M6 70.37）。注意该进程用的是无 429 退避的旧代码，models.py 已打补丁（磁盘版）
- ⏳ 未跑：m1–m5 全量
- ❌ 2026-10-08 的 m1/m2/m3/m4/m6 结果为 MLLM 失效期间的无效产物，留档勿用
- 核心结论：基线可复现（77.41 ≥ 论文 75.25）；论文 +9.6 的增益在我们的环境下不可复现（真管线部分数据仅 +2.0）；84.89 不可达

## 环境位置（AutoDL 实例，数据盘持久，关机不丢）

- 代码：`/root/autodl-tmp/PANDA/`（本地 `~/Documents/workspace/PANDA` 为源，注意 `.env` 只存在于远程，rsync --delete 会删掉它）
- 密钥：`/root/autodl-tmp/PANDA/.env`（KIMI_BASE_URL / KIMI_API_KEY）
- conda 环境：`panda`（vllm 0.31 + torch 2.13 + 全部依赖）
- 模型：`/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct`（16GB）
- 数据：`/root/autodl-tmp/data/ucf_crime/`（290 视频 + 标注 + 37,066 帧）
- 知识库缓存：`/root/autodl-tmp/PANDA/cache/knowledge_base_7a1382a565bb.json`（260 条规则，有效）

## 恢复步骤

1. AutoDL 控制台开机（有卡模式 4090），**SSH 地址/端口会变**，更新本地 `~/.ssh/config` 里 `autodl-panda` 的 HostName/Port
2. 启动 vLLM（约 1 分钟）：
   ```bash
   ssh autodl-panda 'nohup bash /root/autodl-tmp/serve_vllm.sh > /root/autodl-tmp/vllm.log 2>&1 &'
   # 等待就绪：curl localhost:8000/v1/models 返回 qwen2.5-vl-7b
   ```
3. 按需裁剪 `scripts/run_all2.sh` 的阶段列表（例如只跑剩余阶段），然后：
   ```bash
   ssh autodl-panda 'nohup bash /root/autodl-tmp/PANDA/scripts/run_all2.sh > /root/autodl-tmp/PANDA/results/run_all2.log 2>&1 &'
   ```
   脚本自带 MLLM 预检，缺 key 会立即失败不会空跑。预计时长：m6 ~20h（反思重），m1–m3 各 ~3-4h，m4–m5 各 ~6h。

## 已知坑（别再踩）

- nohup 非交互 shell 不读 ~/.bashrc → 密钥必须放项目 `.env` 并由脚本 source
- Kimi `kimi-for-coding` 只接受 temperature=1 → 代码默认不传 temperature
- pkill -f 的模式会匹配发起命令自身 → 用 `run_all[0-9]*[.]sh` 这类带括号的模式
- vLLM 参数 `--limit-mm-per-prompt` 要 JSON 且脚本里单引号包裹
- flashinfer JIT 编译失败 → serve 脚本里已 export VLLM_USE_FLASHINFER_SAMPLER=0
- AutoDL 克隆系统镜像**不含数据盘**（模型/数据集要重拷或重下）
- AutoDL 不同宿主机 NVIDIA 驱动版本不同：新 3090 实例驱动仅 CUDA 12.8，跑不动 torch cu130 → 需用 vllm 0.11.0 + torch 2.8.0+cu128 + torchvision 0.23.0 + triton 3.4.0 + xformers 0.0.32.post2 + transformers 4.56.2 + sentence-transformers 3.4.1，并卸载 torchaudio/flashinfer/cu13 系 nvidia 包（见 /root/autodl-tmp/fix_torch.sh、fix_nccl.sh）
- pip 大文件下载：USTC 镜像（mirrors.ustc.edu.cn/pypi）最快（16MB/s），tuna 会掉到 0.4MB/s，network_turbo 代理对 pypi 反而更慢（70KB/s），仅适合 GitHub/HF
