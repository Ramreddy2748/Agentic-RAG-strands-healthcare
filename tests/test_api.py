from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from rag_chatbot.answer_layer import CitedStatement, ClinicalAnswer
from rag_chatbot.api import app, get_rag_service
from rag_chatbot.embedding_layer import SearchResult
from rag_chatbot.observability import PipelineTimings
from rag_chatbot.rag_service import RAGResponse, RetrievalStats

from test_reranking_layer import make_chunk


class FakeRAGService:
    def ask(self, question: str, **kwargs: object) -> RAGResponse:
        return RAGResponse(
            request_id=str(kwargs.get("request_id", "fake-request")),
            question=question,
            search_mode="keyword",
            routing_reason="Exact requirement code.",
            results=[SearchResult(score=0.8, chunk=make_chunk(0))],
            answer=ClinicalAnswer(
                summary=CitedStatement(
                    text="Grounded API answer.",
                    citations=[1],
                ),
                key_requirements=[
                    CitedStatement(text="Maintain the QMS.", citations=[1]),
                ],
            ),
            stats=RetrievalStats(
                semantic_candidates=0,
                keyword_candidates=1,
                fused_candidates=1,
                final_results=1,
            ),
            timings=PipelineTimings(
                routing_ms=1.0,
                retrieval_ms=2.0,
                fusion_ms=0.1,
                reranking_ms=3.0,
                answer_generation_ms=4.0,
                total_ms=10.1,
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
        self.assertEqual(payload["answer"]["summary"]["text"], "Grounded API answer.")
        self.assertEqual(payload["answer"]["summary"]["citations"], [1])
        self.assertEqual(payload["sources"][0]["section_title"], "QM.1 TEST")
        self.assertEqual(payload["sources"][0]["vector_score"], 0.8)
        self.assertEqual(payload["timings"]["total_ms"], 10.1)
        self.assertEqual(response.headers["x-request-id"], payload["request_id"])

    def test_ask_preserves_caller_request_id(self) -> None:
        response = self.client.post(
            "/ask",
            headers={"X-Request-ID": "client-request-123"},
            json={"question": "QM.1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-request-id"], "client-request-123")
        self.assertEqual(response.json()["request_id"], "client-request-123")

    def test_ask_rejects_empty_question(self) -> None:
        response = self.client.post("/ask", json={"question": ""})

        self.assertEqual(response.status_code, 422)

    def test_ask_rejects_removed_candidate_override(self) -> None:
        response = self.client.post(
            "/ask",
            json={"question": "QM.1", "candidate_k": 20},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
