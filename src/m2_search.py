from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import hashlib
import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


class _HashingFallbackEncoder:
    """Deterministic 1024-dimension local encoder for offline execution."""

    @staticmethod
    def _encode_one(text: str):
        import numpy as np

        vector = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        for token in segment_vietnamese(text.lower()).split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % EMBEDDING_DIM
            vector[index] += 1.0
        return vector

    def encode(self, values, **kwargs):
        import numpy as np

        if isinstance(values, str):
            return self._encode_one(values)
        if not values:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        return np.stack([self._encode_one(value) for value in values])


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize

    segmented = word_tokenize(text, format="text")
    return segmented.replace("_", " ")


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [
            segment_vietnamese(document["text"]).split()
            for document in self.documents
        ]
        if not self.corpus_tokens:
            self.bm25 = None
            return

        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []

        tokenized_query = segment_vietnamese(query).split()
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                text=self.documents[index]["text"],
                score=float(scores[index]),
                metadata=dict(self.documents[index].get("metadata", {})),
                method="bm25",
            )
            for index in top_indices
            if float(scores[index]) > 0
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            except Exception as exc:
                print(f"  Dense model unavailable; using hashing fallback: {exc}")
                self._encoder = _HashingFallbackEncoder()
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            vector_list = vector.tolist() if hasattr(vector, "tolist") else list(vector)
            payload = {**dict(chunk.get("metadata", {})), "text": chunk["text"]}
            points.append(PointStruct(id=index, vector=vector_list, payload=payload))

        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []

        query_vector = self._get_encoder().encode(query)
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()
        else:
            query_vector = list(query_vector)
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.get("text", "")
            results.append(SearchResult(
                text=text,
                score=float(point.score),
                metadata=payload,
                method="dense",
            ))
        return results


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if top_k <= 0:
        return []

    rrf_scores: dict[str, dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            entry = rrf_scores.setdefault(
                result.text,
                {"score": 0.0, "result": result},
            )
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(
        rrf_scores.values(),
        key=lambda entry: entry["score"],
        reverse=True,
    )[:top_k]
    return [
        SearchResult(
            text=entry["result"].text,
            score=float(entry["score"]),
            metadata=dict(entry["result"].metadata),
            method="hybrid",
        )
        for entry in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
