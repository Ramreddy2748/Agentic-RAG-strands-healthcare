from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol

import numpy as np

from rag_chatbot.data_layer import DocumentChunk, slugify, split_words
from rag_chatbot.embedding_layer import (
    BGEEmbedder,
    DEFAULT_MODEL_NAME,
    IndexedChunk,
    TextEmbedder,
    VectorIndex,
    chunk_to_embedding_text,
    indexed_chunk_from_document_chunk,
)
from rag_chatbot.ingestion_layer import (
    DocumentElement,
    IngestionResult,
    ingest_uploaded_document,
)
from rag_chatbot.mongo_vector_store import MongoVectorStore


class VectorStoreWriter(Protocol):
    """Storage backend that can persist embedded chunks."""

    def upsert_vector_index(
        self,
        index: VectorIndex,
        *,
        batch_size: int = 100,
    ) -> int: ...


@dataclass(frozen=True)
class DocumentIndexingResult:
    """Result of indexing one uploaded document into a vector store."""

    document_id: str
    filename: str
    file_extension: str
    element_count: int
    chunk_count: int
    upserted_count: int
    model_name: str


def index_uploaded_document(
    document_id: str,
    *,
    upload_dir: str | Path = "uploads",
    chunk_words: int = 900,
    overlap_words: int = 150,
    model_name: str = DEFAULT_MODEL_NAME,
    embedding_batch_size: int = 8,
    mongo_batch_size: int = 100,
    embedder: TextEmbedder | None = None,
    vector_store: VectorStoreWriter | None = None,
) -> DocumentIndexingResult:
    """Ingest, chunk, embed, and persist one uploaded document."""
    ingestion = ingest_uploaded_document(document_id, upload_dir=upload_dir)
    chunks = chunks_from_ingestion_result(
        ingestion,
        chunk_words=chunk_words,
        overlap_words=overlap_words,
    )
    index = build_index_from_document_chunks(
        chunks,
        model_name=model_name,
        batch_size=embedding_batch_size,
        embedder=embedder,
    )
    store = vector_store or MongoVectorStore.from_env()
    upserted_count = store.upsert_vector_index(index, batch_size=mongo_batch_size)
    return DocumentIndexingResult(
        document_id=ingestion.document_id,
        filename=ingestion.filename,
        file_extension=ingestion.file_extension,
        element_count=ingestion.element_count,
        chunk_count=len(chunks),
        upserted_count=upserted_count,
        model_name=model_name,
    )


def chunks_from_ingestion_result(
    ingestion: IngestionResult,
    *,
    chunk_words: int = 900,
    overlap_words: int = 150,
) -> list[DocumentChunk]:
    """Convert normalized extracted elements into searchable chunks."""
    if chunk_words < 100:
        raise ValueError("chunk_words must be at least 100 words")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks: list[DocumentChunk] = []
    for element_index, element in enumerate(ingestion.elements, start=1):
        location = element_location(element, element_index)
        section_title = element_section_title(element, location)
        page_number = element.page_number or element.row_number or 1
        parts = split_words(
            element.text,
            chunk_words=chunk_words,
            overlap_words=overlap_words,
        )
        for chunk_index, text in enumerate(parts, start=1):
            chunks.append(
                DocumentChunk(
                    chunk_id=(
                        f"{ingestion.document_id}:"
                        f"{slugify(section_title)}:c{chunk_index}"
                    ),
                    source_id=ingestion.document_id,
                    source_path=Path(element.source_path),
                    page_number=page_number,
                    end_page_number=page_number,
                    chapter_title=f"Uploaded {ingestion.file_extension.upper()} Document",
                    section_title=section_title,
                    text=text,
                    word_count=len(text.split()),
                )
            )
    return chunks


def build_index_from_document_chunks(
    chunks: list[DocumentChunk],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 8,
    embedder: TextEmbedder | None = None,
) -> VectorIndex:
    """Embed uploaded document chunks and return a vector index batch."""
    if not chunks:
        raise ValueError("Cannot index an uploaded document with zero chunks")

    active_embedder = embedder or BGEEmbedder(model_name)
    texts = [chunk_to_embedding_text(chunk) for chunk in chunks]
    embeddings = active_embedder.encode(texts, batch_size=batch_size)
    return VectorIndex(
        model_name=model_name,
        chunks=[indexed_chunk_from_document_chunk(chunk) for chunk in chunks],
        embeddings=np.asarray(embeddings, dtype=np.float32),
    )


def element_location(element: DocumentElement, element_index: int) -> str:
    """Return a stable human-readable location for an extracted element."""
    if element.page_number is not None:
        return f"page {element.page_number}"
    if element.row_number is not None:
        return f"row {element.row_number}"
    if element.json_path:
        return element.json_path
    return f"element {element_index}"


def element_section_title(element: DocumentElement, location: str) -> str:
    """Build the section title used in chunk metadata."""
    if element.content_type == "pdf_page":
        return f"PDF {location}"
    if element.content_type == "csv_row":
        return f"CSV {location}"
    if element.content_type == "json_value":
        return f"JSON {location}"
    return f"{element.content_type} {location}"


def default_embedding_model() -> str:
    """Return the configured upload indexing embedding model."""
    return os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME)
