"""Anomaly knowledge base construction (式2) and FAISS RAG retrieval (式3).

KB is generated once per user query by the MLLM and cached to JSON. Rules and
EnvInfo are encoded with all-MiniLM-L6-v2 (loaded from modelscope) and indexed
with FAISS for top-k retrieval.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Optional

import numpy as np

from .config import Config
from .models import MLLMClient
from .prompts import KB_CONSTRUCTION_PROMPT

logger = logging.getLogger(__name__)


def _download_from_modelscope(model_id: str, fallbacks: tuple = ()) -> str:
    """snapshot_download from modelscope hub, trying fallback ids."""
    from modelscope import snapshot_download

    last_exc: Optional[Exception] = None
    for mid in (model_id, *fallbacks):
        try:
            path = snapshot_download(mid)
            logger.info("modelscope model %s -> %s", mid, path)
            return path
        except Exception as exc:
            logger.warning("modelscope download failed for %s: %s", mid, exc)
            last_exc = exc
    raise RuntimeError(f"could not download {model_id} from modelscope") from last_exc


class Embedder:
    """Lazy singleton wrapper around all-MiniLM-L6-v2 (CPU)."""

    _instance: Optional["Embedder"] = None

    def __init__(self, cfg: Config):
        from sentence_transformers import SentenceTransformer

        path = _download_from_modelscope(cfg.embed_modelscope_id, cfg.embed_fallback_ids)
        self.model = SentenceTransformer(path, device="cpu")

    @classmethod
    def get(cls, cfg: Config) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls(cfg)
        return cls._instance

    def encode(self, texts: list) -> np.ndarray:
        emb = self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)


def rule_text(rule: dict) -> str:
    """Canonical text form of a knowledge-base rule, used for embedding."""
    if isinstance(rule, str):
        return rule
    return (
        f"Event Type: {rule.get('Event Type', '')}. "
        f"Rule Description: {rule.get('Rule Description', '')}. "
        f"Applicable Scenes: {rule.get('Applicable Scenes', '')}."
    )


def env_info_text(env_info: dict) -> str:
    """Canonical text form of EnvInfo (式1), used as the RAG query."""
    if not env_info:
        return ""
    return (
        f"Scene Overview: {env_info.get('Scene Overview', '')}. "
        f"Weather Condition: {env_info.get('Weather Condition', '')}. "
        f"Video Quality: {env_info.get('Video Quality', '')}. "
        f"Potential Anomalies: {env_info.get('Potential Anomalies', '')}."
    )


class KnowledgeBase:
    """Anomaly rules (式2) + FAISS index for RetrieveTopK (式3)."""

    def __init__(self, cfg: Config, rules: list):
        self.cfg = cfg
        self.rules = rules
        self.embedder = Embedder.get(cfg)
        self.index = None
        if rules:
            import faiss

            emb = self.embedder.encode([rule_text(r) for r in rules])
            self.index = faiss.IndexFlatIP(emb.shape[1])
            self.index.add(emb)

    def retrieve(self, query_text: str, top_k: Optional[int] = None) -> list:
        """RetrieveTopK(EnvInfo, κa) — returns the top-k rules as dicts."""
        top_k = top_k or self.cfg.rag_top_k
        if self.index is None or not query_text.strip():
            return []
        q = self.embedder.encode([query_text])
        scores, idxs = self.index.search(q, min(top_k, len(self.rules)))
        return [self.rules[int(i)] for i in idxs[0] if int(i) >= 0]


def kb_cache_path(cfg: Config, user_query: str) -> str:
    digest = hashlib.md5(user_query.encode("utf-8")).hexdigest()[:12]
    return os.path.join(cfg.cache_dir, f"knowledge_base_{digest}.json")


def build_knowledge_base(cfg: Config, mllm: MLLMClient, user_query: str,
                         event_types: tuple) -> list:
    """Generate (or load from cache) H=20 rules per event type via the MLLM.

    Decision: the paper issues Prompt_know once per user query. We call the MLLM
    once *per event type* (with a single-type user requirement) because a single
    response covering 13 types x 20 rules is too large to parse reliably; the
    resulting KB is identical in structure.
    """
    cache = kb_cache_path(cfg, user_query)
    if os.path.exists(cache):
        logger.info("loading knowledge base from cache %s", cache)
        with open(cache) as f:
            cached_rules = json.load(f)
        if cached_rules:
            return cached_rules
        logger.warning("cached knowledge base %s is empty; rebuilding", cache)

    all_rules: list = []
    for event_type in event_types:
        sub_query = (
            f"Please help me detect the following types of abnormal events: {event_type}."
        )
        prompt = KB_CONSTRUCTION_PROMPT.substitute(user_query=sub_query)
        logger.info("building knowledge base for event type: %s", event_type)
        obj = mllm.generate_json(prompt)
        rules = _extract_rule_list(obj)
        if not rules:
            logger.warning("no rules parsed for %s; skipping", event_type)
            continue
        for r in rules:
            r.setdefault("Event Type", event_type)
        all_rules.extend(rules)
        logger.info("event type %s: %d rules", event_type, len(rules))

    if not all_rules:
        raise RuntimeError(
            "knowledge base construction produced 0 rules; check MLLM credentials "
            "and connectivity before rerunning (empty KB is never cached)")

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w") as f:
        json.dump(all_rules, f, ensure_ascii=False, indent=2)
    logger.info("knowledge base cached to %s (%d rules)", cache, len(all_rules))
    return all_rules


def _extract_rule_list(obj) -> list:
    """Accept a raw list of rules, or a dict wrapping such a list."""
    if isinstance(obj, list):
        return [r for r in obj if isinstance(r, dict)]
    if isinstance(obj, dict):
        for key in ("rules", "Rules", "knowledge_base", "Knowledge Base"):
            if isinstance(obj.get(key), list):
                return [r for r in obj[key] if isinstance(r, dict)]
        # single dict that looks like a rule
        if "Rule Description" in obj:
            return [obj]
    return []
