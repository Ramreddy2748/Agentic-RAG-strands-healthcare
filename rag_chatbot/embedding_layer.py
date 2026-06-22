from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from threading import Lock
from typing import Iterable, Protocol

import numpy as np

from rag_chatbot.data_layer import DocumentChunk


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_INDEX_DIR = Path(".rag_index")
METADATA_FILE = "metadata.json"
EMBEDDINGS_FILE = "embeddings.npz"


@dataclass(frozen=True)
class IndexedChunk:
    """A chunk and its metadata as stored in the vector index."""

    chunk_id: str
    source_id: str
    source_path: str
    page_number: int
    end_page_number: int
    chapter_title: str
    section_title: str
    text: str
    word_count: int


@dataclass(frozen=True)
class SearchResult:
    """A retrieved chunk with similarity score."""

    score: float
    chunk: IndexedChunk


@dataclass(frozen=True)
class VectorIndex:
    """Embeddings and chunk metadata loaded from disk."""

    model_name: str
    chunks: list[IndexedChunk]
    embeddings: np.ndarray


class TextEmbedder(Protocol):
    """Interface for reusable text embedding models."""

    def encode(self, texts: list[str], *, batch_size: int = 8) -> np.ndarray: ...


class BGEEmbedder:
    """Local BGE-M3 embedder using sentence-transformers."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing embedding dependency. Install project dependencies with "
                "`python3 -m pip install -e .` before building embeddings."
            ) from exc

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self._inference_lock = Lock()

    def encode(self, texts: list[str], *, batch_size: int = 8) -> np.ndarray:
        with self._inference_lock:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 1,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def warm_up(self) -> None:
        """Run one tiny inference so the first request avoids setup overhead."""
        self.encode(["warmup"], batch_size=1)


def build_vector_index(
    chunks: Iterable[DocumentChunk],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 8,
) -> VectorIndex:
    """Embed chunks and return an in-memory vector index."""
    chunk_list = list(chunks)
    if not chunk_list:
        raise ValueError("Cannot build a vector index with zero chunks")

    embedder = BGEEmbedder(model_name)
    texts = [chunk_to_embedding_text(chunk) for chunk in chunk_list]
    embeddings = embedder.encode(texts, batch_size=batch_size)
    return VectorIndex(
        model_name=model_name,
        chunks=[indexed_chunk_from_document_chunk(chunk) for chunk in chunk_list],
        embeddings=embeddings,
    )


def save_vector_index(index: VectorIndex, index_dir: str | Path = DEFAULT_INDEX_DIR) -> None:
    """Persist vector embeddings and chunk metadata."""
    root = Path(index_dir)
    root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": index.model_name,
        "chunks": [asdict(chunk) for chunk in index.chunks],
    }
    (root / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    np.savez_compressed(root / EMBEDDINGS_FILE, embeddings=index.embeddings)


def load_vector_index(index_dir: str | Path = DEFAULT_INDEX_DIR) -> VectorIndex:
    """Load vector embeddings and chunk metadata from disk."""
    root = Path(index_dir)
    metadata_path = root / METADATA_FILE
    embeddings_path = root / EMBEDDINGS_FILE
    if not metadata_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            f"Vector index not found in {root}. Run scripts/build_vector_index.py first."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with np.load(embeddings_path) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)

    chunks = [IndexedChunk(**chunk) for chunk in metadata["chunks"]]
    if len(chunks) != len(embeddings):
        raise ValueError("Vector index metadata and embeddings have different lengths")

    return VectorIndex(
        model_name=metadata["model_name"],
        chunks=chunks,
        embeddings=embeddings,
    )


def search_vector_index(
    index: VectorIndex,
    query: str,
    *,
    top_k: int = 5,
    batch_size: int = 8,
    embedder: TextEmbedder | None = None,
) -> list[SearchResult]:
    """Embed a query and return the closest chunks by cosine similarity."""
    if not query.strip():
        raise ValueError("query cannot be empty")

    active_embedder = embedder or BGEEmbedder(index.model_name)
    query_embedding = active_embedder.encode([query], batch_size=batch_size)[0]
    scores = index.embeddings @ query_embedding
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        SearchResult(score=float(scores[index_number]), chunk=index.chunks[index_number])
        for index_number in top_indices
    ]


def chunk_to_embedding_text(chunk: DocumentChunk | IndexedChunk) -> str:
    """Add source metadata to the text sent to the embedding model."""
    page_range = str(chunk.page_number)
    if chunk.end_page_number != chunk.page_number:
        page_range = f"{chunk.page_number}-{chunk.end_page_number}"

    return "\n".join(
        [
            f"Chapter: {chunk.chapter_title}",
            f"Section: {chunk.section_title}",
            f"Pages: {page_range}",
            "",
            chunk.text,
        ]
    )


def indexed_chunk_from_document_chunk(chunk: DocumentChunk) -> IndexedChunk:
    """Convert a Data Layer chunk to serializable index metadata."""
    return IndexedChunk(
        chunk_id=chunk.chunk_id,
        source_id=chunk.source_id,
        source_path=str(chunk.source_path),
        page_number=chunk.page_number,
        end_page_number=chunk.end_page_number,
        chapter_title=chunk.chapter_title,
        section_title=chunk.section_title,
        text=chunk.text,
        word_count=chunk.word_count,
    )
