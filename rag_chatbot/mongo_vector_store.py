from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any

import numpy as np

from rag_chatbot.embedding_layer import (
    DEFAULT_MODEL_NAME,
    IndexedChunk,
    SearchResult,
    TextEmbedder,
    VectorIndex,
)


DEFAULT_MONGO_DATABASE = "healthcare_rag"
DEFAULT_MONGO_COLLECTION = "chunks"
DEFAULT_MONGO_VECTOR_INDEX = "chunk_embedding_vector_index"


@dataclass(frozen=True)
class MongoVectorConfig:
    """Connection settings for MongoDB Atlas Vector Search."""

    uri: str
    database: str = DEFAULT_MONGO_DATABASE
    collection: str = DEFAULT_MONGO_COLLECTION
    vector_index: str = DEFAULT_MONGO_VECTOR_INDEX
    model_name: str = DEFAULT_MODEL_NAME
    num_candidates: int = 100


class MongoVectorStore:
    """Persist and search RAG chunks in MongoDB Atlas Vector Search."""

    def __init__(self, config: MongoVectorConfig) -> None:
        self.config = config
        self._client: Any | None = None

    @classmethod
    def from_env(cls) -> MongoVectorStore:
        """Build a Mongo vector store from environment variables."""
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise RuntimeError("MONGODB_URI is required when VECTOR_BACKEND=mongodb.")
        return cls(
            MongoVectorConfig(
                uri=uri,
                database=os.getenv("MONGODB_DATABASE", DEFAULT_MONGO_DATABASE),
                collection=os.getenv("MONGODB_COLLECTION", DEFAULT_MONGO_COLLECTION),
                vector_index=os.getenv(
                    "MONGODB_VECTOR_INDEX",
                    DEFAULT_MONGO_VECTOR_INDEX,
                ),
                model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL_NAME),
                num_candidates=max(
                    1,
                    int(os.getenv("MONGODB_VECTOR_NUM_CANDIDATES", "100")),
                ),
            )
        )

    @property
    def client(self) -> Any:
        """Create the Mongo client only when MongoDB is actually used."""
        if self._client is None:
            try:
                from pymongo import MongoClient, UpdateOne
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Missing MongoDB dependency. Install it with "
                    "`python -m pip install -e '.[mongodb]'`."
                ) from exc

            self._update_one = UpdateOne
            self._client = MongoClient(self.config.uri)
        return self._client

    @property
    def collection(self) -> Any:
        """Return the configured chunk collection."""
        return self.client[self.config.database][self.config.collection]

    def upsert_vector_index(
        self,
        index: VectorIndex,
        *,
        batch_size: int = 100,
    ) -> int:
        """Upload a local vector index into MongoDB."""
        if len(index.chunks) != len(index.embeddings):
            raise ValueError("Vector index metadata and embeddings have different lengths")

        _ = self.client
        operations = []
        written = 0
        for chunk, embedding in zip(index.chunks, index.embeddings, strict=True):
            document = {
                "_id": chunk.chunk_id,
                **asdict(chunk),
                "model_name": index.model_name,
                "embedding": np.asarray(embedding, dtype=np.float32).tolist(),
            }
            operations.append(
                self._update_one(
                    {"_id": chunk.chunk_id},
                    {"$set": document},
                    upsert=True,
                )
            )
            if len(operations) >= batch_size:
                result = self.collection.bulk_write(operations, ordered=False)
                written += result.upserted_count + result.modified_count
                operations = []

        if operations:
            result = self.collection.bulk_write(operations, ordered=False)
            written += result.upserted_count + result.modified_count
        return written

    def load_vector_index(self) -> VectorIndex:
        """Load chunk metadata from MongoDB for keyword search and responses."""
        cursor = self.collection.find(
            {"model_name": self.config.model_name},
            {
                "embedding": 0,
            },
        ).sort([("source_id", 1), ("page_number", 1), ("chunk_id", 1)])
        chunks = [chunk_from_mongo_document(document) for document in cursor]
        if not chunks:
            raise FileNotFoundError(
                "No chunks were found in MongoDB. Upload the local index first with "
                "`python scripts/upload_vector_index_to_mongodb.py`."
            )
        return VectorIndex(
            model_name=self.config.model_name,
            chunks=chunks,
            embeddings=np.empty((len(chunks), 0), dtype=np.float32),
        )

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        *,
        top_k: int,
    ) -> list[SearchResult]:
        """Run Atlas Vector Search and return matching chunks."""
        query_vector = np.asarray(embedding, dtype=np.float32).tolist()
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.config.vector_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": max(self.config.num_candidates, top_k * 10),
                    "limit": top_k,
                    "filter": {"model_name": self.config.model_name},
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "chunk_id": 1,
                    "source_id": 1,
                    "source_path": 1,
                    "page_number": 1,
                    "end_page_number": 1,
                    "chapter_title": 1,
                    "section_title": 1,
                    "text": 1,
                    "word_count": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        return [
            SearchResult(
                score=float(document["score"]),
                chunk=chunk_from_mongo_document(document),
            )
            for document in self.collection.aggregate(pipeline)
        ]


class MongoSemanticSearchBackend:
    """Semantic retriever that delegates vector search to MongoDB."""

    def __init__(self, store: MongoVectorStore) -> None:
        self.store = store

    def search(
        self,
        question: str,
        *,
        top_k: int,
        batch_size: int,
        embedder: TextEmbedder,
    ) -> list[SearchResult]:
        """Embed the query locally, then retrieve nearest chunks from MongoDB."""
        query_embedding = embedder.encode([question], batch_size=batch_size)[0]
        return self.store.search_by_embedding(query_embedding, top_k=top_k)


def chunk_from_mongo_document(document: dict[str, Any]) -> IndexedChunk:
    """Convert a MongoDB document into the common indexed chunk model."""
    return IndexedChunk(
        chunk_id=str(document["chunk_id"]),
        source_id=str(document["source_id"]),
        source_path=str(document["source_path"]),
        page_number=int(document["page_number"]),
        end_page_number=int(document["end_page_number"]),
        chapter_title=str(document["chapter_title"]),
        section_title=str(document["section_title"]),
        text=str(document["text"]),
        word_count=int(document["word_count"]),
    )
