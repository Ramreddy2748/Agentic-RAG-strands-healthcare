from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from rag_chatbot.answer_layer import DEFAULT_ANSWER_MODEL
from rag_chatbot.embedding_layer import DEFAULT_INDEX_DIR
from rag_chatbot.rag_service import RAGResponse, RAGService, RankedResult
from rag_chatbot.reranking_layer import DEFAULT_RERANKER_MODEL
from rag_chatbot.routing_layer import DEFAULT_ROUTER_MODEL


SearchMode = Literal["auto", "semantic", "keyword", "hybrid"]
FallbackMode = Literal["semantic", "keyword", "hybrid"]


class AskRequest(BaseModel):
    """Request controls for one RAG question."""

    question: str = Field(min_length=1, max_length=2000)
    search_mode: SearchMode = "auto"
    router_fallback: FallbackMode = "hybrid"
    candidate_k: int = Field(default=10, ge=1, le=50)
    top_k: int = Field(default=3, ge=1, le=10)
    rerank: bool = True
    generate_answer: bool = True


class SourceResponse(BaseModel):
    """One ranked source returned by the RAG pipeline."""

    rank: int
    chunk_id: str
    source_id: str
    source_path: str
    chapter_title: str
    section_title: str
    page_number: int
    end_page_number: int
    text: str
    vector_score: float | None = None
    keyword_score: float | None = None
    retrieval_score: float
    rerank_score: float | None = None


class RetrievalStatsResponse(BaseModel):
    """Candidate counts from each retrieval stage."""

    semantic_candidates: int
    keyword_candidates: int
    combined_candidates: int
    final_results: int


class AskResponse(BaseModel):
    """Structured API response for a grounded RAG question."""

    question: str
    search_mode: str
    routing_reason: str
    answer: str | None
    sources: list[SourceResponse]
    stats: RetrievalStatsResponse


class HealthResponse(BaseModel):
    """Basic process and index availability status."""

    status: str
    index_dir: str
    index_available: bool


app = FastAPI(
    title="Agentic Healthcare RAG API",
    description="Grounded search over DNV NIAHO hospital accreditation requirements.",
    version="0.1.0",
)


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Load and cache the vector index and reusable query-time services."""
    load_dotenv()
    return RAGService.from_index_dir(
        os.getenv("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)),
        router_model=os.getenv("ROUTER_MODEL", DEFAULT_ROUTER_MODEL),
        reranker_model=os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        answer_model=os.getenv("ANSWER_MODEL", DEFAULT_ANSWER_MODEL),
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the API process can see the persisted index."""
    index_dir = Path(os.getenv("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
    index_available = (
        (index_dir / "metadata.json").exists()
        and (index_dir / "embeddings.npz").exists()
    )
    return HealthResponse(
        status="ok" if index_available else "degraded",
        index_dir=str(index_dir),
        index_available=index_available,
    )


@app.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    """Run one question through routing, retrieval, reranking, and answering."""
    try:
        response = await run_in_threadpool(
            service.ask,
            request.question,
            search_mode=request.search_mode,
            router_fallback=request.router_fallback,
            candidate_k=request.candidate_k,
            top_k=request.top_k,
            rerank=request.rerank,
            generate_answer=request.generate_answer,
            answer_top_k=min(3, request.top_k),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return response_to_api(response)


def response_to_api(response: RAGResponse) -> AskResponse:
    """Convert internal dataclasses into stable API response models."""
    return AskResponse(
        question=response.question,
        search_mode=response.search_mode,
        routing_reason=response.routing_reason,
        answer=response.answer,
        sources=[
            result_to_source(rank, result)
            for rank, result in enumerate(response.results, start=1)
        ],
        stats=RetrievalStatsResponse(
            semantic_candidates=response.stats.semantic_candidates,
            keyword_candidates=response.stats.keyword_candidates,
            combined_candidates=response.stats.fused_candidates,
            final_results=response.stats.final_results,
        ),
    )


def result_to_source(rank: int, result: RankedResult) -> SourceResponse:
    """Normalize semantic, keyword, hybrid, and reranked result scores."""
    chunk = result.chunk
    rerank_score = getattr(result, "rerank_score", None)
    vector_score = getattr(result, "vector_score", None)
    keyword_score = getattr(result, "keyword_score", None)
    retrieval_score = getattr(result, "hybrid_score", None)
    if retrieval_score is None:
        retrieval_score = result.score

    if vector_score is None and result.__class__.__name__ == "SearchResult":
        vector_score = result.score
    if keyword_score is None and result.__class__.__name__ == "KeywordSearchResult":
        keyword_score = result.score

    return SourceResponse(
        rank=rank,
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        source_path=chunk.source_path,
        chapter_title=chunk.chapter_title,
        section_title=chunk.section_title,
        page_number=chunk.page_number,
        end_page_number=chunk.end_page_number,
        text=chunk.text,
        vector_score=vector_score,
        keyword_score=keyword_score,
        retrieval_score=retrieval_score,
        rerank_score=rerank_score,
    )
