#!/usr/bin/env python3
"""Evaluation entry point for the PANDA reproduction.

Ablation ladder (paper Table 4, UCF-Crime AUC% targets):
    --stage m0   bare user-query VLM                (75.25)
    --stage m1   + planning                         (77.01)
    --stage m2   + perception                       (78.92)
    --stage m3   + RAG knowledge base               (80.37)
    --stage m4   + tool-augmented reflection        (82.63)
    --stage m5   + short CoM                        (83.94)
    --stage m6   + long CoM (full PANDA)            (84.89)

The --no-* flags force a module OFF regardless of the stage, e.g.
    python scripts/run_eval.py --stage m6 --no-reflection ...

Usage:
    python scripts/run_eval.py --stage m0 --split subset
    python scripts/run_eval.py --stage m6 --split full --out results/m6_full.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panda.agent import PANDAAgent
from panda.config import AblationConfig, Config, UCF_USER_QUERY
from panda.data import frame_labels, load_dataset, select_subset, with_fps
from panda.evaluate import aggregate, evaluate_video
from panda.knowledge import KnowledgeBase, build_knowledge_base
from panda.models import MLLMClient, VLMClient
from panda.tools import init_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("panda_eval.log", mode="a")],
)
logger = logging.getLogger("run_eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PANDA evaluation")
    p.add_argument("--stage", default="m6",
                   choices=["m0", "m1", "m2", "m3", "m4", "m5", "m6"])
    p.add_argument("--no-perception", action="store_true")
    p.add_argument("--no-planning", action="store_true")
    p.add_argument("--no-rag", action="store_true")
    p.add_argument("--no-reflection", action="store_true")
    p.add_argument("--no-short-com", action="store_true")
    p.add_argument("--no-long-com", action="store_true")
    p.add_argument("--split", default="subset", choices=["subset", "full"])
    p.add_argument("--metric", default="auc", choices=["auc", "ap"],
                   help="ap is for XD-Violence-style evaluation")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=None,
                   help="JSONL output path (default: results/<stage>_<split>.jsonl)")
    p.add_argument("--user-query", default=UCF_USER_QUERY)
    p.add_argument("--limit", type=int, default=0, help="debug: only first N videos")
    p.add_argument("--max-clips", type=int, default=0,
                   help="debug: cap clips per video")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config()
    if args.data_root:
        cfg.data_root = args.data_root
    init_tools(cfg)

    ablation = AblationConfig.for_stage(args.stage)
    for flag, name in [
        (args.no_perception, "perception"), (args.no_planning, "planning"),
        (args.no_rag, "rag"), (args.no_reflection, "reflection"),
        (args.no_short_com, "short_com"), (args.no_long_com, "long_com"),
    ]:
        if flag:
            setattr(ablation, name, False)
    logger.info("stage=%s ablation=%s", args.stage, ablation)

    needs_mllm = ablation.planning or ablation.rag or ablation.reflection or ablation.long_com
    if needs_mllm and not os.environ.get("KIMI_API_KEY"):
        logger.error("stage %s requires the MLLM but KIMI_API_KEY is not set; aborting", args.stage)
        return 2

    metas = load_dataset(cfg.data_root)
    metas = with_fps(metas)
    if args.split == "subset":
        metas = select_subset(metas, 10, 10)
    if args.limit:
        metas = metas[:args.limit]
    logger.info("evaluating %d videos from %s", len(metas), cfg.data_root)

    vlm = VLMClient(cfg)
    mllm = MLLMClient(cfg)

    kb = None
    if ablation.rag:
        rules = build_knowledge_base(cfg, mllm, args.user_query,
                                     cfg.ucf_event_types)
        kb = KnowledgeBase(cfg, rules)

    agent = PANDAAgent(cfg, ablation, vlm, mllm, kb=kb,
                       user_query=args.user_query)

    out_path = args.out or os.path.join(cfg.results_dir,
                                        f"{args.stage}_{args.split}.jsonl")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    all_labels: list = []
    all_scores: list = []
    all_smoothed: list = []
    video_metrics: list = []

    with open(out_path, "w") as fout:
        for meta in metas:
            t0 = time.time()
            frame_paths = meta.frame_paths
            if args.max_clips:
                frame_paths = frame_paths[:args.max_clips * cfg.clip_frames]
            logger.info("video %s (%s): %d frames, fps=%.2f",
                        meta.video_id, "normal" if meta.is_normal else "abnormal",
                        len(frame_paths), meta.fps)
            try:
                result = agent.run_video(meta.video_id, frame_paths)
            except Exception as exc:
                logger.exception("video %s failed: %s", meta.video_id, exc)
                continue
            scores = result["frame_scores"]
            labels = frame_labels(meta)[:len(scores)]
            metrics = evaluate_video(labels, scores, cfg.smooth_window)
            metrics["video_id"] = meta.video_id
            metrics["category"] = meta.category
            metrics["is_normal"] = meta.is_normal
            video_metrics.append(metrics)
            all_labels.extend(labels)
            all_scores.extend(scores)
            all_smoothed.extend(metrics["scores_smoothed"])
            record = {
                "video_id": meta.video_id,
                "category": meta.category,
                "is_normal": meta.is_normal,
                "fps": meta.fps,
                "stage": args.stage,
                "ablation": ablation.__dict__,
                "labels": labels,
                "frame_scores": scores,
                "metrics": {k: v for k, v in metrics.items()
                            if k != "scores_smoothed"},
                "clip_records": result["clip_records"],
                "env_info": result.get("env_info", {}),
                "plan": result.get("plan", {}),
                "elapsed_sec": round(time.time() - t0, 1),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            logger.info("video %s done in %.1fs: %s raw=%.2f smoothed=%.2f",
                        meta.video_id, time.time() - t0, args.metric,
                        metrics[f"{args.metric}_raw"],
                        metrics[f"{args.metric}_smoothed"])

    summary = aggregate(all_labels, all_scores, all_smoothed, video_metrics,
                        metric=args.metric)
    summary.update({"stage": args.stage, "split": args.split,
                    "ablation": ablation.__dict__, "out": out_path})
    summary_path = os.path.splitext(out_path)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    key = "ap" if args.metric == "ap" else "auc"
    print("\n===== RESULTS =====")
    print(f"stage={args.stage} split={args.split} metric={args.metric} "
          f"videos={summary['n_videos']}")
    print(f"pooled {key.upper()} raw:      {summary[f'pooled_{key}_raw']:.2f}")
    print(f"pooled {key.upper()} smoothed: {summary[f'pooled_{key}_smoothed']:.2f}")
    print(f"macro  {key.upper()} raw mean:  {summary[f'macro_{key}_raw_mean']:.2f}")
    print(f"macro  {key.upper()} smoothed:  {summary[f'macro_{key}_smoothed_mean']:.2f}")
    print(f"summary saved to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
