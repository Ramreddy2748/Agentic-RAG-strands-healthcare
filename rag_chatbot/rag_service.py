from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from threading import Lock
from typing import TypeAlias

from rag_chatbot.answer_layer import (
    DEFAULT_ANSWER_MODEL,
    AnswerGenerator,
    ClinicalAnswer,
    GeneratedAnswer,
    generate_grounded_answer,
)
from rag_chatbot.embedding_layer import (
    BGEEmbedder,
    DEFAULT_INDEX_DIR,
    SearchResult,
    TextEmbedder,
    VectorIndex,
    load_vector_index,
    search_vector_index,
)
from rag_chatbot.hybrid_layer import HybridSearchResult, fuse_search_results
from rag_chatbot.keyword_layer import BM25Index, KeywordSearchResult
from rag_chatbot.observability import (
    PipelineTimings,
    StageTimer,
    log_event,
    new_request_id,
)
from rag_chatbot.reranking_layer import (
    BGEReranker,
    DEFAULT_RERANKER_MODEL,
    PassageScorer,
    RerankedSearchResult,
    rerank_search_results,
)
from rag_chatbot.routing_layer import (
    DEFAULT_ROUTER_MODEL,
    QueryRouter,
    RoutingDecision,
    route_query,
)


RetrievalResult: TypeAlias = SearchResult | KeywordSearchResult | HybridSearchResult
RankedResult: TypeAlias = RetrievalResult | RerankedSearchResult

SEARCH_MODE_CANDIDATE_COUNTS = {
    "keyword": 5,
    "semantic": 8,
    "hybrid": 12,
}


@dataclass(frozen=True)
class RetrievalStats:
    """Candidate counts produced by each retrieval stage."""

    semantic_candidates: int
    keyword_candidates: int
    fused_candidates: int
    final_results: int


@dataclass(frozen=True)
class RAGResponse:
    """Structured result returned by the complete RAG pipeline."""

    request_id: str
    question: str
    search_mode: str
    routing_reason: str
    results: list[RankedResult]
    answer: ClinicalAnswer | None
    stats: RetrievalStats
    timings: PipelineTimings


class RAGService:
    """Coordinate routing, retrieval, fusion, reranking, and answering."""

    def __init__(
        self,
        index: VectorIndex,
        *,
        embedder: TextEmbedder | None = None,
        router: QueryRouter | None = None,
        reranker: PassageScorer | None = None,
        answer_generator: AnswerGenerator | None = None,
        router_model: str = DEFAULT_ROUTER_MODEL,
        reranker_model: str = DEFAULT_RERANKER_MODEL,
        answer_model: str = DEFAULT_ANSWER_MODEL,
    ) -> None:
        self.index = index
        self.keyword_index = BM25Index(index.chunks)
        self.embedder = embedder
        self.router = router
        self.reranker = reranker
        self.answer_generator = answer_generator
        self.router_model = router_model
        self.reranker_model = reranker_model
        self.answer_model = answer_model
        self._model_init_lock = Lock()
        self._warmup_lock = Lock()
        self._models_warmed = False

    @classmethod
    def from_index_dir(
        cls,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        **kwargs: object,
    ) -> RAGService:
        """Create a service from a persisted local vector index."""
        return cls(load_vector_index(index_dir), **kwargs)

    @property
    def models_ready(self) -> bool:
        """Return whether both local query-time models are loaded."""
        return self.embedder is not None and self.reranker is not None

    def load_models(self, *, warm_up: bool = True) -> None:
        """Load reusable local models once and optionally run warmup inference."""
        timer = StageTimer()
        with self._model_init_lock:
            if self.embedder is None:
                self.embedder = BGEEmbedder(self.index.model_name)
            if self.reranker is None:
                self.reranker = BGEReranker(self.reranker_model)

        if warm_up:
            self.warm_up_models()
        log_event(
            "rag_models_loaded",
            embedding_model=self.index.model_name,
            reranker_model=self.reranker_model,
            elapsed_ms=timer.elapsed_ms(),
        )

    def warm_up_models(self) -> None:
        """Warm already loaded models so the first request is predictable."""
        with self._warmup_lock:
            if self._models_warmed:
                return
            if self.embedder is None or self.reranker is None:
                self.load_models(warm_up=False)

            timer = StageTimer()
            embedder_warm_up = getattr(self.embedder, "warm_up", None)
            if embedder_warm_up:
                embedder_warm_up()
            reranker_warm_up = getattr(self.reranker, "warm_up", None)
            if reranker_warm_up:
                reranker_warm_up()
            self._models_warmed = True
            log_event("rag_models_warmed", elapsed_ms=timer.elapsed_ms())

    def get_embedder(self) -> TextEmbedder:
        """Return the shared embedder, loading it once when necessary."""
        if self.embedder is None:
            with self._model_init_lock:
                if self.embedder is None:
                    self.embedder = BGEEmbedder(self.index.model_name)
        return self.embedder

    def get_reranker(self) -> PassageScorer:
        """Return the shared reranker, loading it once when necessary."""
        if self.reranker is None:
            with self._model_init_lock:
                if self.reranker is None:
                    self.reranker = BGEReranker(self.reranker_model)
        return self.reranker

    def ask(
        self,
        question: str,
        *,
        search_mode: str = "auto",
        router_fallback: str = "hybrid",
        top_k: int = 3,
        embedding_batch_size: int = 8,
        semantic_weight: float = 1.0,
        keyword_weight: float = 1.0,
        rerank: bool = True,
        reranker_batch_size: int = 2,
        reranker_max_length: int = 512,
        generate_answer: bool = True,
        answer_top_k: int = 3,
        request_id: str | None = None,
    ) -> RAGResponse:
        """Run one question through the complete configured RAG pipeline."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        if search_mode not in {"auto", "semantic", "keyword", "hybrid"}:
            raise ValueError(f"Invalid search mode: {search_mode}")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        active_request_id = request_id or new_request_id()
        total_timer = StageTimer()
        log_event(
            "rag_request_started",
            request_id=active_request_id,
            search_mode_requested=search_mode,
            top_k=top_k,
            rerank=rerank,
            generate_answer=generate_answer,
        )

        try:
            stage_timer = StageTimer()
            routing = self.select_search_mode(
                question,
                search_mode=search_mode,
                fallback_mode=router_fallback,
            )
            routing_ms = stage_timer.elapsed_ms()
            candidate_k = SEARCH_MODE_CANDIDATE_COUNTS[routing.mode]
            if top_k > candidate_k:
                raise ValueError(
                    f"top_k cannot exceed {candidate_k} for {routing.mode} search"
                )

            stage_timer = StageTimer()
            semantic_results, keyword_results = self.retrieve(
                question,
                search_mode=routing.mode,
                candidate_k=candidate_k,
                embedding_batch_size=embedding_batch_size,
            )
            retrieval_ms = stage_timer.elapsed_ms()

            stage_timer = StageTimer()
            candidates = self.combine_candidates(
                routing.mode,
                semantic_results,
                keyword_results,
                candidate_k=candidate_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )
            fusion_ms = stage_timer.elapsed_ms()

            stage_timer = StageTimer()
            results = self.rank_candidates(
                question,
                candidates,
                top_k=top_k,
                rerank=rerank,
                batch_size=reranker_batch_size,
                max_length=reranker_max_length,
            )
            reranking_ms = stage_timer.elapsed_ms()

            stage_timer = StageTimer()
            generated_answer = self.answer(
                question,
                results,
                enabled=generate_answer,
                top_k=answer_top_k,
            )
            answer_generation_ms = stage_timer.elapsed_ms()

            timings = PipelineTimings(
                routing_ms=routing_ms,
                retrieval_ms=retrieval_ms,
                fusion_ms=fusion_ms,
                reranking_ms=reranking_ms,
                answer_generation_ms=answer_generation_ms,
                total_ms=total_timer.elapsed_ms(),
            )
            stats = RetrievalStats(
                semantic_candidates=len(semantic_results),
                keyword_candidates=len(keyword_results),
                fused_candidates=len(candidates),
                final_results=len(results),
            )
            log_event(
                "rag_request_completed",
                request_id=active_request_id,
                search_mode=routing.mode,
                candidate_k=candidate_k,
                routing_reason=routing.reason,
                stats=stats,
                timings=timings,
                sections=[result.chunk.section_title for result in results],
            )
            return RAGResponse(
                request_id=active_request_id,
                question=question,
                search_mode=routing.mode,
                routing_reason=routing.reason,
                results=results,
                answer=generated_answer.content if generated_answer else None,
                stats=stats,
                timings=timings,
            )
        except Exception as exc:
            log_event(
                "rag_request_failed",
                level=logging.ERROR,
                request_id=active_request_id,
                error_type=type(exc).__name__,
                error=str(exc),
                total_ms=total_timer.elapsed_ms(),
            )
            raise

    def select_search_mode(
        self,
        question: str,
        *,
        search_mode: str,
        fallback_mode: str,
    ) -> RoutingDecision:
        """Resolve automatic routing or preserve a manual search mode."""
        if search_mode != "auto":
            return RoutingDecision(
                mode=search_mode,
                reason="Search mode selected manually.",
            )
        return route_query(
            question,
            router=self.router,
            model_name=self.router_model,
            fallback_mode=fallback_mode,
        )

    def retrieve(
        self,
        question: str,
        *,
        search_mode: str,
        candidate_k: int,
        embedding_batch_size: int,
    ) -> tuple[list[SearchResult], list[KeywordSearchResult]]:
        """Run only the retrievers required by the selected search mode."""
        semantic_results: list[SearchResult] = []
        if search_mode in {"semantic", "hybrid"}:
            semantic_results = search_vector_index(
                self.index,
                question,
                top_k=candidate_k,
                batch_size=embedding_batch_size,
                embedder=self.get_embedder(),
            )

        keyword_results: list[KeywordSearchResult] = []
        if search_mode in {"keyword", "hybrid"}:
            keyword_results = self.keyword_index.search(
                question,
                top_k=candidate_k,
            )
        return semantic_results, keyword_results

    def combine_candidates(
        self,
        search_mode: str,
        semantic_results: list[SearchResult],
        keyword_results: list[KeywordSearchResult],
        *,
        candidate_k: int,
        semantic_weight: float,
        keyword_weight: float,
    ) -> list[RetrievalResult]:
        """Return one candidate list for the selected retrieval strategy."""
        if search_mode == "hybrid":
            return fuse_search_results(
                semantic_results,
                keyword_results,
                top_k=candidate_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight,
            )
        if search_mode == "semantic":
            return semantic_results
        return keyword_results

    def rank_candidates(
        self,
        question: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int,
        rerank: bool,
        batch_size: int,
        max_length: int,
    ) -> list[RankedResult]:
        """Optionally rerank candidates and select the final result count."""
        if not rerank:
            return candidates[:top_k]
        return rerank_search_results(
            question,
            candidates,
            top_k=top_k,
            model_name=self.reranker_model,
            batch_size=batch_size,
            max_length=max_length,
            reranker=self.get_reranker(),
        )

    def answer(
        self,
        question: str,
        results: list[RankedResult],
        *,
        enabled: bool,
        top_k: int,
    ) -> GeneratedAnswer | None:
        """Generate a grounded answer from the final ranked chunks."""
        if not enabled:
            return None
        return generate_grounded_answer(
            question,
            results,
            top_k=top_k,
            model_name=self.answer_model,
            generator=self.answer_generator,
        )
