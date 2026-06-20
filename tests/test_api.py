from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from rag_chatbot.api import app, get_rag_service
from rag_chatbot.embedding_layer import SearchResult
from rag_chatbot.rag_service import RAGResponse, RetrievalStats

from test_reranking_layer import make_chunk


class FakeRAGService:
    def ask(self, question: str, **kwargs: object) -> RAGResponse:
        return RAGResponse(
            question=question,
            search_mode="keyword",
            routing_reason="Exact requirement code.",
            results=[SearchResult(score=0.8, chunk=make_chunk(0))],
            answer="Grounded API answer.",
            stats=RetrievalStats(
                semantic_candidates=0,
                keyword_candidates=1,
                fused_candidates=1,
                final_results=1,
            ),
        )


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        app.dependency_overrides[get_rag_service] = lambda: FakeRAGService()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_health_reports_index_status(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertIn("index_available", response.json())

    def test_ask_returns_structured_answer_and_sources(self) -> None:
        response = self.client.post(
            "/ask",
            json={
                "question": "QM.1",
                "search_mode": "auto",
                "top_k": 3,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["search_mode"], "keyword")
        self.assertEqual(payload["answer"], "Grounded API answer.")
        self.assertEqual(payload["sources"][0]["section_title"], "QM.1 TEST")
        self.assertEqual(payload["sources"][0]["vector_score"], 0.8)

    def test_ask_rejects_empty_question(self) -> None:
        response = self.client.post("/ask", json={"question": ""})

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
