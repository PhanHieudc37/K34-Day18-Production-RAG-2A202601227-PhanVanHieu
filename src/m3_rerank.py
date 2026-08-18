from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class _LexicalFallbackCrossEncoder:
    """Offline fallback with the same ``predict(pairs)`` contract."""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for query, document in pairs:
            query_tokens = self._tokens(query)
            document_tokens = self._tokens(document)
            if not query_tokens or not document_tokens:
                scores.append(0.0)
                continue
            overlap = len(query_tokens & document_tokens)
            scores.append(overlap / (len(query_tokens) * len(document_tokens)) ** 0.5)
        return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Load once per reranker instance and reuse it for subsequent calls."""
        if self._model is None:
            try:
                # Use sentence-transformers, not FlagEmbedding: the latter is
                # incompatible with the Transformers version used by this lab.
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                print(f"  Cross-encoder unavailable; using lexical fallback: {exc}")
                self._model = _LexicalFallbackCrossEncoder()
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """Score all query-document pairs and keep the highest-scoring top-k."""
        # Avoid loading a large model when there is no work to perform.
        if not documents or top_k <= 0:
            return []

        import numpy as np

        model = self._load_model()
        pairs = [(query, document["text"]) for document in documents]
        raw_scores = model.predict(pairs)
        scores = np.asarray(raw_scores, dtype=float).reshape(-1)
        if len(scores) != len(documents):
            raise ValueError(
                f"Cross-encoder returned {len(scores)} scores "
                f"for {len(documents)} documents"
            )

        scored = sorted(
            zip(scores, documents),
            key=lambda item: float(item[0]),
            reverse=True,
        )
        return [
            RerankResult(
                text=document["text"],
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Optional lightweight extension point; not used by the main pipeline."""

    def __init__(self):
        self._model = None

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        return []


def benchmark_reranker(
    reranker,
    query: str,
    documents: list[dict],
    n_runs: int = 5,
) -> dict:
    """Benchmark reranking latency over ``n_runs`` calls."""
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for result in reranker.rerank(query, docs):
        print(f"[{result.rank}] {result.rerank_score:.4f} | {result.text}")
