from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import lru_cache
import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field

from rag_chatbot.answer_layer import DEFAULT_ANSWER_MODEL, ClinicalAnswer
from rag_chatbot.document_layer import UploadedDocument, save_uploaded_document
from rag_chatbot.embedding_layer import DEFAULT_INDEX_DIR
from rag_chatbot.ingestion_layer import (
    DocumentElement,
    IngestionResult,
    ingest_uploaded_document,
)
from rag_chatbot.indexing_layer import (
    DocumentIndexingResult,
    index_uploaded_document,
)
from rag_chatbot.index_storage import ensure_index_available, index_is_available
from rag_chatbot.observability import PipelineTimings, new_request_id
from rag_chatbot.rag_service import RAGResponse, RAGService, RankedResult
from rag_chatbot.reranking_layer import DEFAULT_RERANKER_MODEL
from rag_chatbot.routing_layer import DEFAULT_ROUTER_MODEL
from rag_chatbot.verification_layer import DEFAULT_VERIFICATION_MODEL
from rag_chatbot.security_layer import (
    AuthenticatedPrincipal,
    SlidingWindowRateLimiter,
    authenticate_principal,
    configured_cors_origins,
    configured_api_keys,
    detect_prompt_injection,
)


SearchMode = Literal["auto", "semantic", "keyword", "hybrid"]
FallbackMode = Literal["semantic", "keyword", "hybrid"]
QualityMode = Literal["fast", "balanced", "strict"]
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)
PROXY_SECRET_HEADER = APIKeyHeader(name="X-Proxy-Secret", auto_error=False)
PROXY_USER_HEADER = APIKeyHeader(name="X-Authenticated-User", auto_error=False)

load_dotenv()
STATIC_DIR = Path(__file__).resolve().parent / "static"


class AskRequest(BaseModel):
    """Request controls for one RAG question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    search_mode: SearchMode = "auto"
    quality_mode: QualityMode = "balanced"
    router_fallback: FallbackMode = "hybrid"
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


class PipelineTimingsResponse(BaseModel):
    """Elapsed milliseconds for each query-time stage."""

    routing_ms: float
    retrieval_ms: float
    fusion_ms: float
    reranking_ms: float
    answer_generation_ms: float
    verification_ms: float
    total_ms: float


class VerificationResponse(BaseModel):
    """Claim-grounding verification metadata for the generated answer."""

    enabled: bool
    verified: bool
    confidence: float
    checked_claims: int
    supported_claims: int
    removed_claims: int
    unclear_claims: int
    reason: str
    unsupported_claims: list[str]


class AskResponse(BaseModel):
    """Structured API response for a grounded RAG question."""

    request_id: str
    question: str
    quality_mode: str
    search_mode: str
    routing_reason: str
    answer: ClinicalAnswer | None
    verification: VerificationResponse
    sources: list[SourceResponse]
    evidence_sufficient: bool
    evidence_score: float
    evidence_threshold: float
    evidence_reason: str
    stats: RetrievalStatsResponse
    timings: PipelineTimingsResponse


class HealthResponse(BaseModel):
    """Basic process and index availability status."""

    status: str
    vector_backend: str
    index_dir: str
    index_available: bool
    preload_models: bool
    max_concurrent_requests: int


class DocumentUploadResponse(BaseModel):
    """Response returned after storing an uploaded source document."""

    document_id: str
    filename: str
    stored_filename: str
    content_type: str
    file_extension: str
    size_bytes: int
    status: str
    created_at: str


class DocumentElementResponse(BaseModel):
    """One normalized element extracted from an uploaded document."""

    content_type: str
    text: str
    page_number: int | None = None
    row_number: int | None = None
    json_path: str | None = None
    metadata: dict[str, str] | None = None


class DocumentIngestionResponse(BaseModel):
    """Preview response for extracting an uploaded document."""

    document_id: str
    filename: str
    file_extension: str
    element_count: int
    elements: list[DocumentElementResponse]


class DocumentIndexingResponse(BaseModel):
    """Response returned after embedding and upserting an uploaded document."""

    document_id: str
    filename: str
    file_extension: str
    element_count: int
    chunk_count: int
    upserted_count: int
    model_name: str


class DocumentUploadAndIndexResponse(BaseModel):
    """Response returned after uploading and indexing one source document."""

    document: DocumentUploadResponse
    indexing: DocumentIndexingResponse
    status: str


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    """Load and cache the vector index and reusable query-time services."""
    load_dotenv()
    vector_backend = os.getenv("VECTOR_BACKEND", "local").strip().lower()
    service_kwargs = {
        "router_model": os.getenv("ROUTER_MODEL", DEFAULT_ROUTER_MODEL),
        "reranker_model": os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL),
        "answer_model": os.getenv("ANSWER_MODEL", DEFAULT_ANSWER_MODEL),
        "verification_model": os.getenv(
            "VERIFICATION_MODEL",
            DEFAULT_VERIFICATION_MODEL,
        ),
    }
    if vector_backend == "mongodb":
        return RAGService.from_mongodb(**service_kwargs)
    if vector_backend != "local":
        raise RuntimeError(f"Unsupported VECTOR_BACKEND: {vector_backend}")
    return RAGService.from_index_dir(
        os.getenv("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)),
        **service_kwargs,
    )


def env_flag(name: str, default: bool) -> bool:
    """Read a conventional boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def require_principal(
    request: Request,
    provided_key: str | None = Depends(API_KEY_HEADER),
    proxy_secret: str | None = Depends(PROXY_SECRET_HEADER),
    proxy_user: str | None = Depends(PROXY_USER_HEADER),
) -> AuthenticatedPrincipal:
    """Authenticate the caller and apply an identity-scoped rate limit."""
    auth_mode = os.getenv("AUTH_MODE", "api_key").strip().lower()
    if auth_mode == "api_key" and not configured_api_keys():
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured.",
        )
    if auth_mode == "trusted_proxy" and not os.getenv("TRUSTED_PROXY_SECRET"):
        raise HTTPException(
            status_code=503,
            detail="Trusted proxy authentication is not configured.",
        )
    try:
        principal = authenticate_principal(
            api_key=provided_key,
            proxy_secret=proxy_secret,
            proxy_user=proxy_user,
            auth_mode=auth_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if principal is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing authentication credentials.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    limiter = getattr(request.app.state, "security_rate_limiter", None)
    if limiter is None:
        limiter = build_security_rate_limiter()
        request.app.state.security_rate_limiter = limiter
    allowed, retry_after = limiter.check(principal.identifier)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Request rate limit exceeded.",
            headers={"Retry-After": str(retry_after)},
        )
    return principal


def build_security_rate_limiter() -> SlidingWindowRateLimiter:
    """Create the configured in-process per-identity limiter."""
    return SlidingWindowRateLimiter(
        requests=max(1, int(os.getenv("RATE_LIMIT_REQUESTS", "5"))),
        window_seconds=max(1.0, float(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))),
    )


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Load and warm local models once before accepting API traffic."""
    load_dotenv()
    max_concurrent_requests = max(
        1,
        int(os.getenv("MAX_CONCURRENT_REQUESTS", "1")),
    )
    app_instance.state.request_limiter = asyncio.Semaphore(
        max_concurrent_requests
    )
    app_instance.state.security_rate_limiter = build_security_rate_limiter()

    vector_backend = os.getenv("VECTOR_BACKEND", "local").strip().lower()
    index_dir = Path(os.getenv("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
    if get_rag_service not in app_instance.dependency_overrides:
        if vector_backend == "local":
            await run_in_threadpool(
                ensure_index_available,
                index_dir,
                region=os.getenv("AWS_REGION"),
            )

    should_preload = env_flag("PRELOAD_MODELS", True)
    if should_preload and get_rag_service not in app_instance.dependency_overrides:
        service = get_rag_service()
        await run_in_threadpool(service.load_models, warm_up=True)
    yield


app = FastAPI(
    title="Agentic Healthcare RAG API",
    description="Grounded search over DNV NIAHO hospital accreditation requirements.",
    version="0.1.0",
    lifespan=lifespan,
)

cors_origins = configured_cors_origins()
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the API process can see the persisted index."""
    vector_backend = os.getenv("VECTOR_BACKEND", "local").strip().lower()
    index_dir = Path(os.getenv("RAG_INDEX_DIR", str(DEFAULT_INDEX_DIR)))
    index_available = True if vector_backend == "mongodb" else index_is_available(index_dir)
    return HealthResponse(
        status="ok" if index_available else "degraded",
        vector_backend=vector_backend,
        index_dir=str(index_dir),
        index_available=index_available,
        preload_models=env_flag("PRELOAD_MODELS", True),
        max_concurrent_requests=max(
            1,
            int(os.getenv("MAX_CONCURRENT_REQUESTS", "1")),
        ),
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index_page() -> HTMLResponse:
    """Serve the small browser UI for asking RAG questions."""
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    _: AuthenticatedPrincipal = Depends(require_principal),
) -> DocumentUploadResponse:
    """Store one source document for a later ingestion/indexing step."""
    try:
        content = await file.read()
        document = await run_in_threadpool(
            save_uploaded_document,
            filename=file.filename or "",
            content_type=file.content_type,
            content=content,
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return document_to_api(document)


@app.post(
    "/documents/upload-and-index",
    response_model=DocumentUploadAndIndexResponse,
)
async def upload_and_index_document(
    file: UploadFile = File(...),
    _: AuthenticatedPrincipal = Depends(require_principal),
) -> DocumentUploadAndIndexResponse:
    """Store one source document and immediately index it into the vector store."""
    try:
        content = await file.read()
        document = await run_in_threadpool(
            save_uploaded_document,
            filename=file.filename or "",
            content_type=file.content_type,
            content=content,
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
        )
        indexing = await run_in_threadpool(
            index_uploaded_document,
            document.document_id,
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DocumentUploadAndIndexResponse(
        document=document_to_api(document),
        indexing=indexing_to_api(indexing),
        status="ready",
    )


@app.post(
    "/documents/{document_id}/ingest",
    response_model=DocumentIngestionResponse,
)
async def ingest_uploaded_document_endpoint(
    document_id: str,
    show: int = 5,
    _: AuthenticatedPrincipal = Depends(require_principal),
) -> DocumentIngestionResponse:
    """Preview extraction for one uploaded document without embedding it."""
    try:
        result = await run_in_threadpool(
            ingest_uploaded_document,
            document_id,
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ingestion_to_api(result, show=max(1, min(show, 50)))


@app.post(
    "/documents/{document_id}/index",
    response_model=DocumentIndexingResponse,
)
async def index_uploaded_document_endpoint(
    document_id: str,
    _: AuthenticatedPrincipal = Depends(require_principal),
) -> DocumentIndexingResponse:
    """Embed and upsert one uploaded document into the vector store."""
    try:
        result = await run_in_threadpool(
            index_uploaded_document,
            document_id,
            upload_dir=os.getenv("UPLOAD_DIR", "uploads"),
            model_name=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return indexing_to_api(result)


@app.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    http_request: Request,
    http_response: Response,
    _: AuthenticatedPrincipal = Depends(require_principal),
    service: RAGService = Depends(get_rag_service),
) -> AskResponse:
    """Run one question through routing, retrieval, reranking, and answering."""
    request_id = http_request.headers.get("X-Request-ID") or new_request_id()
    http_response.headers["X-Request-ID"] = request_id
    injection_findings = detect_prompt_injection(request.question)
    if injection_findings:
        raise HTTPException(
            status_code=400,
            detail={
                "request_id": request_id,
                "error": (
                    "The question contains instructions that cannot be processed."
                ),
                "code": "prompt_injection_detected",
            },
        )
    request_limiter = getattr(http_request.app.state, "request_limiter", None)
    if request_limiter is None:
        request_limiter = asyncio.Semaphore(
            max(1, int(os.getenv("MAX_CONCURRENT_REQUESTS", "1")))
        )
        http_request.app.state.request_limiter = request_limiter
    try:
        async with request_limiter:
            response = await run_in_threadpool(
                service.ask,
                request.question,
                search_mode=request.search_mode,
                quality_mode=request.quality_mode,
                router_fallback=request.router_fallback,
                top_k=request.top_k,
                rerank=request.rerank,
                generate_answer=request.generate_answer,
                answer_top_k=min(3, request.top_k),
                request_id=request_id,
            )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"request_id": request_id, "error": str(exc)},
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={"request_id": request_id, "error": str(exc)},
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail={"request_id": request_id, "error": str(exc)},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "request_id": request_id,
                "error": "Unexpected pipeline error.",
            },
        ) from exc

    return response_to_api(response)


def document_to_api(document: UploadedDocument) -> DocumentUploadResponse:
    """Convert uploaded document metadata into the public API response."""
    return DocumentUploadResponse(
        document_id=document.document_id,
        filename=document.original_filename,
        stored_filename=document.stored_filename,
        content_type=document.content_type,
        file_extension=document.file_extension,
        size_bytes=document.size_bytes,
        status=document.status,
        created_at=document.created_at,
    )


def ingestion_to_api(
    result: IngestionResult,
    *,
    show: int,
) -> DocumentIngestionResponse:
    """Convert an ingestion preview into the public API response."""
    return DocumentIngestionResponse(
        document_id=result.document_id,
        filename=result.filename,
        file_extension=result.file_extension,
        element_count=result.element_count,
        elements=[element_to_api(element) for element in result.elements[:show]],
    )


def indexing_to_api(result: DocumentIndexingResult) -> DocumentIndexingResponse:
    """Convert indexing output into the public API response."""
    return DocumentIndexingResponse(
        document_id=result.document_id,
        filename=result.filename,
        file_extension=result.file_extension,
        element_count=result.element_count,
        chunk_count=result.chunk_count,
        upserted_count=result.upserted_count,
        model_name=result.model_name,
    )


def element_to_api(element: DocumentElement) -> DocumentElementResponse:
    """Convert one normalized extracted element into the API schema."""
    return DocumentElementResponse(
        content_type=element.content_type,
        text=element.text,
        page_number=element.page_number,
        row_number=element.row_number,
        json_path=element.json_path,
        metadata=element.metadata,
    )


def response_to_api(response: RAGResponse) -> AskResponse:
    """Convert internal dataclasses into stable API response models."""
    return AskResponse(
        request_id=response.request_id,
        question=response.question,
        quality_mode=response.quality_mode,
        search_mode=response.search_mode,
        routing_reason=response.routing_reason,
        answer=response.answer,
        verification=VerificationResponse(
            enabled=response.verification.enabled,
            verified=response.verification.verified,
            confidence=response.verification.confidence,
            checked_claims=response.verification.checked_claims,
            supported_claims=response.verification.supported_claims,
            removed_claims=response.verification.removed_claims,
            unclear_claims=response.verification.unclear_claims,
            reason=response.verification.reason,
            unsupported_claims=response.verification.unsupported_claims,
        ),
        sources=[
            result_to_source(rank, result)
            for rank, result in enumerate(response.results, start=1)
        ],
        evidence_sufficient=response.evidence.sufficient,
        evidence_score=response.evidence.score,
        evidence_threshold=response.evidence.threshold,
        evidence_reason=response.evidence.reason,
        stats=RetrievalStatsResponse(
            semantic_candidates=response.stats.semantic_candidates,
            keyword_candidates=response.stats.keyword_candidates,
            combined_candidates=response.stats.fused_candidates,
            final_results=response.stats.final_results,
        ),
        timings=timings_to_api(response.timings),
    )


def timings_to_api(timings: PipelineTimings) -> PipelineTimingsResponse:
    """Convert internal timing measurements to the API schema."""
    return PipelineTimingsResponse(
        routing_ms=timings.routing_ms,
        retrieval_ms=timings.retrieval_ms,
        fusion_ms=timings.fusion_ms,
        reranking_ms=timings.reranking_ms,
        answer_generation_ms=timings.answer_generation_ms,
        verification_ms=timings.verification_ms,
        total_ms=timings.total_ms,
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
