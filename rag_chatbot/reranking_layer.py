from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rag_chatbot.embedding_layer import IndexedChunk, SearchResult, chunk_to_embedding_text
from rag_chatbot.hybrid_layer import HybridSearchResult
from rag_chatbot.keyword_layer import KeywordSearchResult


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass(frozen=True)
class RerankedSearchResult:
    """A vector-search candidate scored again by a cross-encoder reranker."""

    vector_score: float
    keyword_score: float
    hybrid_score: float
    rerank_score: float
    chunk: IndexedChunk


class PassageScorer(Protocol):
    """Interface used by the reranking pipeline."""

    def score(
        self,
        query: str,
        passages: list[str],
        *,
        batch_size: int = 2,
    ) -> list[float]: ...


class BGEReranker:
    """Local BGE reranker that scores query and passage pairs together."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        max_length: int = 512,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing reranking dependencies. Install project dependencies with "
                "`python -m pip install -e .` before using reranking."
            ) from exc

        self.torch = torch
        self.model_name = model_name
        self.max_length = max_length
        self.device = select_device(torch)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score(
        self,
        query: str,
        passages: list[str],
        *,
        batch_size: int = 2,
    ) -> list[float]:
        """Return normalized relevance scores for query-passage pairs."""
        scores: list[float] = []
        for start in range(0, len(passages), batch_size):
            passage_batch = passages[start : start + batch_size]
            pairs = [[query, passage] for passage in passage_batch]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            ).to(self.device)

            with self.torch.no_grad():
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                normalized = self.torch.sigmoid(logits)
            scores.extend(normalized.cpu().tolist())
        return scores


def rerank_search_results(
    query: str,
    candidates: list[SearchResult | KeywordSearchResult | HybridSearchResult],
    *,
    top_k: int = 5,
    model_name: str = DEFAULT_RERANKER_MODEL,
    batch_size: int = 2,
    max_length: int = 512,
    reranker: PassageScorer | None = None,
) -> list[RerankedSearchResult]:
    """Rerank vector-search candidates and return the most relevant chunks."""
    if not query.strip():
        raise ValueError("query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not candidates:
        return []

    scorer = reranker or BGEReranker(model_name, max_length=max_length)
    passages = [chunk_to_embedding_text(candidate.chunk) for candidate in candidates]
    rerank_scores = scorer.score(query, passages, batch_size=batch_size)
    results: list[RerankedSearchResult] = []
    for candidate, rerank_score in zip(candidates, rerank_scores, strict=True):
        vector_score, keyword_score = candidate_source_scores(candidate)
        results.append(
            RerankedSearchResult(
                vector_score=vector_score,
                keyword_score=keyword_score,
                hybrid_score=candidate.score,
                rerank_score=rerank_score,
                chunk=candidate.chunk,
            )
        )
    return sorted(results, key=lambda result: result.rerank_score, reverse=True)[:top_k]


def candidate_source_scores(
    candidate: SearchResult | KeywordSearchResult | HybridSearchResult,
) -> tuple[float, float]:
    """Return semantic and keyword scores without conflating search modes."""
    if isinstance(candidate, HybridSearchResult):
        return candidate.vector_score, candidate.keyword_score
    if isinstance(candidate, KeywordSearchResult):
        return 0.0, candidate.score
    return candidate.score, 0.0


def select_device(torch_module: object) -> str:
    """Select an available accelerator without requiring one."""
    if torch_module.cuda.is_available():
        return "cuda"
    if torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"
