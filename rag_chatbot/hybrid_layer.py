from __future__ import annotations

from dataclasses import dataclass

from rag_chatbot.embedding_layer import IndexedChunk, SearchResult
from rag_chatbot.keyword_layer import KeywordSearchResult


@dataclass(frozen=True)
class HybridSearchResult:
    """A candidate produced by fusing semantic and keyword rankings."""

    score: float
    vector_score: float
    keyword_score: float
    chunk: IndexedChunk


def fuse_search_results(
    semantic_results: list[SearchResult],
    keyword_results: list[KeywordSearchResult],
    *,
    top_k: int = 10,
    semantic_weight: float = 1.0,
    keyword_weight: float = 1.0,
    rrf_k: int = 60,
) -> list[HybridSearchResult]:
    """Fuse semantic and BM25 rankings with reciprocal-rank fusion."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if semantic_weight < 0 or keyword_weight < 0:
        raise ValueError("fusion weights cannot be negative")
    if semantic_weight == 0 and keyword_weight == 0:
        raise ValueError("at least one fusion weight must be positive")

    chunks: dict[str, IndexedChunk] = {}
    fused_scores: dict[str, float] = {}
    vector_scores: dict[str, float] = {}
    keyword_scores: dict[str, float] = {}

    for rank, result in enumerate(semantic_results, start=1):
        chunk_id = result.chunk.chunk_id
        chunks[chunk_id] = result.chunk
        vector_scores[chunk_id] = result.score
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
            semantic_weight / (rrf_k + rank)
        )

    for rank, result in enumerate(keyword_results, start=1):
        chunk_id = result.chunk.chunk_id
        chunks[chunk_id] = result.chunk
        keyword_scores[chunk_id] = result.score
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
            keyword_weight / (rrf_k + rank)
        )

    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True,
    )

    return [
        HybridSearchResult(
            score=fused_scores[chunk_id],
            vector_score=vector_scores.get(chunk_id, 0.0),
            keyword_score=keyword_scores.get(chunk_id, 0.0),
            chunk=chunks[chunk_id],
        )
        for chunk_id in ranked_ids[:top_k]
    ]
