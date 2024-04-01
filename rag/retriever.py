"""
Hybrid retriever: BM25 (keyword) + MiniLM dense (semantic) + cross-encoder re-rank.

Architecture:
    1. BM25 retrieves top-k_bm25 candidates from the text corpus.
    2. Dense FAISS retrieves top-k_dense candidates from the embedding index.
    3. Results are merged (union, deduplicated by index).
    4. A cross-encoder re-ranks the merged set and returns top-k_final.

Ablation in the RAG eval notebook showed this hybrid beats either approach alone
by ~8 pp on RAGAS context-precision on the DSM evaluation set.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RetrievedChunk:
    text: str
    metadata: dict
    score: float
    source: str = "unknown"


@dataclass
class IndexStore:
    """Holds a FAISS index, BM25 index, and associated text/metadata."""
    chunks: list[str]
    metadata: list[dict]
    faiss_index: faiss.Index
    bm25: BM25Okapi = field(repr=False)

    @classmethod
    def load(cls, index_dir: Path) -> IndexStore:
        faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        with open(index_dir / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
        tokenized = [c.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)
        return cls(chunks=chunks, metadata=metadata, faiss_index=faiss_index, bm25=bm25)


class HybridRetriever:
    def __init__(
        self,
        stores: dict[str, IndexStore],
        k_bm25: int = 20,
        k_dense: int = 20,
        k_final: int = 5,
        rerank: bool = True,
    ):
        self.stores = stores
        self.k_bm25 = k_bm25
        self.k_dense = k_dense
        self.k_final = k_final
        self.rerank = rerank
        self._encoder: SentenceTransformer | None = None
        self._reranker: CrossEncoder | None = None

    @classmethod
    def from_index_dirs(
        cls,
        index_dirs: dict[str, Path],
        **kwargs,
    ) -> HybridRetriever:
        stores = {name: IndexStore.load(path) for name, path in index_dirs.items()}
        return cls(stores=stores, **kwargs)

    def _get_encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            self._encoder = SentenceTransformer(EMBED_MODEL)
        return self._encoder

    def _get_reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL)
        return self._reranker

    def retrieve(self, query: str, source_filter: list[str] | None = None) -> list[RetrievedChunk]:
        stores_to_search = {
            k: v for k, v in self.stores.items()
            if source_filter is None or k in source_filter
        }
        if not stores_to_search:
            return []

        candidates: list[tuple[str, dict, float]] = []

        for store_name, store in stores_to_search.items():
            bm25_hits = self._bm25_search(query, store)
            dense_hits = self._dense_search(query, store)
            seen = set()
            for text, meta, score in bm25_hits + dense_hits:
                key = text[:80]
                if key not in seen:
                    seen.add(key)
                    candidates.append((text, {**meta, "_source": store_name}, score))

        if not candidates:
            return []

        if self.rerank and len(candidates) > 1:
            return self._rerank(query, candidates)

        candidates.sort(key=lambda x: x[2], reverse=True)
        return [
            RetrievedChunk(text=t, metadata=m, score=s, source=m.get("_source", ""))
            for t, m, s in candidates[: self.k_final]
        ]

    def _bm25_search(
        self, query: str, store: IndexStore
    ) -> list[tuple[str, dict, float]]:
        tokens = query.lower().split()
        scores = store.bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][: self.k_bm25]
        return [
            (store.chunks[i], store.metadata[i], float(scores[i]))
            for i in top_idx
            if scores[i] > 0
        ]

    def _dense_search(
        self, query: str, store: IndexStore
    ) -> list[tuple[str, dict, float]]:
        enc = self._get_encoder()
        q_emb = enc.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = store.faiss_index.search(q_emb, self.k_dense)
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx >= 0:
                results.append((store.chunks[idx], store.metadata[idx], float(score)))
        return results

    def _rerank(
        self, query: str, candidates: list[tuple[str, dict, float]]
    ) -> list[RetrievedChunk]:
        reranker = self._get_reranker()
        pairs = [(query, text) for text, _, _ in candidates]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(
            zip(rerank_scores, candidates, strict=False), key=lambda x: x[0], reverse=True
        )
        return [
            RetrievedChunk(
                text=text,
                metadata={**meta},
                score=float(rs),
                source=meta.get("_source", ""),
            )
            for rs, (text, meta, _) in ranked[: self.k_final]
        ]

    def retrieve_with_sources(self, query: str) -> tuple[str, list[dict]]:
        """Convenience method returning formatted context + citation list."""
        chunks = self.retrieve(query)
        if not chunks:
            return "", []
        context = "\n\n---\n\n".join(c.text for c in chunks)
        citations = [
            {
                "source": c.source,
                "pmid": c.metadata.get("pmid"),
                "disorder": c.metadata.get("disorder"),
                "title": c.metadata.get("title"),
                "score": round(c.score, 4),
            }
            for c in chunks
        ]
        return context, citations
