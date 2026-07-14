from __future__ import annotations

import tempfile
import unittest

import numpy as np

from rag_chatbot.document_layer import save_uploaded_document
from rag_chatbot.indexing_layer import (
    build_index_from_document_chunks,
    chunks_from_ingestion_result,
    index_uploaded_document,
)
from rag_chatbot.ingestion_layer import ingest_uploaded_document


class FakeEmbedder:
    def encode(self, texts: list[str], *, batch_size: int = 8) -> np.ndarray:
        return np.ones((len(texts), 3), dtype=np.float32)


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunk_count = 0
        self.batch_size = 0

    def upsert_vector_index(self, index, *, batch_size: int = 100) -> int:
        self.chunk_count = len(index.chunks)
        self.batch_size = batch_size
        return len(index.chunks)


class IndexingLayerTests(unittest.TestCase):
    def test_chunks_from_ingestion_result_creates_searchable_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            document = save_uploaded_document(
                filename="policies.csv",
                content_type="text/csv",
                content=b"section,requirement\nIC.1,Maintain IPCP\n",
                upload_dir=temp_dir,
            )
            ingestion = ingest_uploaded_document(document.document_id, upload_dir=temp_dir)

            chunks = chunks_from_ingestion_result(
                ingestion,
                chunk_words=100,
                overlap_words=10,
            )

            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].source_id, document.document_id)
            self.assertEqual(chunks[0].chapter_title, "Uploaded CSV Document")
            self.assertEqual(chunks[0].section_title, "CSV row 1")
            self.assertIn("section: IC.1", chunks[0].text)

    def test_build_index_from_document_chunks_uses_injected_embedder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            document = save_uploaded_document(
                filename="policies.csv",
                content_type="text/csv",
                content=b"section,requirement\nIC.1,Maintain IPCP\n",
                upload_dir=temp_dir,
            )
            ingestion = ingest_uploaded_document(document.document_id, upload_dir=temp_dir)
            chunks = chunks_from_ingestion_result(
                ingestion,
                chunk_words=100,
                overlap_words=10,
            )

            index = build_index_from_document_chunks(
                chunks,
                model_name="test-model",
                embedder=FakeEmbedder(),
            )

            self.assertEqual(index.model_name, "test-model")
            self.assertEqual(len(index.chunks), 1)
            self.assertEqual(index.embeddings.shape, (1, 3))

    def test_index_uploaded_document_upserts_embedded_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            document = save_uploaded_document(
                filename="policies.csv",
                content_type="text/csv",
                content=b"section,requirement\nIC.1,Maintain IPCP\n",
                upload_dir=temp_dir,
            )
            vector_store = FakeVectorStore()

            result = index_uploaded_document(
                document.document_id,
                upload_dir=temp_dir,
                chunk_words=100,
                overlap_words=10,
                model_name="test-model",
                embedder=FakeEmbedder(),
                vector_store=vector_store,
                mongo_batch_size=7,
            )

            self.assertEqual(result.document_id, document.document_id)
            self.assertEqual(result.element_count, 1)
            self.assertEqual(result.chunk_count, 1)
            self.assertEqual(result.upserted_count, 1)
            self.assertEqual(vector_store.chunk_count, 1)
            self.assertEqual(vector_store.batch_size, 7)


if __name__ == "__main__":
    unittest.main()
