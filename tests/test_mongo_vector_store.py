from __future__ import annotations

import unittest

import numpy as np

from rag_chatbot.mongo_vector_store import (
    MongoSemanticSearchBackend,
    chunk_from_mongo_document,
)


class FakeEmbedder:
    def encode(self, texts: list[str], *, batch_size: int = 8) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


class FakeMongoStore:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int]] = []

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        *,
        top_k: int,
    ) -> list[object]:
        self.calls.append((embedding, top_k))
        return []


class MongoVectorStoreTests(unittest.TestCase):
    def test_chunk_from_mongo_document_maps_metadata(self) -> None:
        chunk = chunk_from_mongo_document(
            {
                "chunk_id": "pdf:p1:c1",
                "source_id": "pdf",
                "source_path": "data/source.pdf",
                "page_number": 1,
                "end_page_number": 2,
                "chapter_title": "Quality",
                "section_title": "QM.1",
                "text": "Hospital quality text.",
                "word_count": 3,
            }
        )

        self.assertEqual(chunk.chunk_id, "pdf:p1:c1")
        self.assertEqual(chunk.source_path, "data/source.pdf")
        self.assertEqual(chunk.page_number, 1)
        self.assertEqual(chunk.end_page_number, 2)
        self.assertEqual(chunk.section_title, "QM.1")

    def test_semantic_backend_embeds_query_before_mongo_search(self) -> None:
        store = FakeMongoStore()
        backend = MongoSemanticSearchBackend(store)  # type: ignore[arg-type]

        results = backend.search(
            "What is QM.1?",
            top_k=5,
            batch_size=1,
            embedder=FakeEmbedder(),
        )

        self.assertEqual(results, [])
        self.assertEqual(len(store.calls), 1)
        embedding, top_k = store.calls[0]
        self.assertEqual(top_k, 5)
        np.testing.assert_array_equal(embedding, np.ones(2, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
