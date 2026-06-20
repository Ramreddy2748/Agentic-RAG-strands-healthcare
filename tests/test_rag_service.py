from __future__ import annotations

import unittest

import numpy as np

from rag_chatbot.answer_layer import AnswerGenerator
from rag_chatbot.embedding_layer import VectorIndex
from rag_chatbot.rag_service import RAGService
from rag_chatbot.reranking_layer import PassageScorer
from rag_chatbot.routing_layer import RoutingDecision

from test_reranking_layer import make_chunk


class KeywordRouter:
    def route(self, query: str) -> RoutingDecision:
        return RoutingDecision(mode="keyword", reason="Exact requirement code.")


class FakeReranker(PassageScorer):
    def score(
        self,
        query: str,
        passages: list[str],
        *,
        batch_size: int = 2,
    ) -> list[float]:
        return [1.0 - index * 0.1 for index in range(len(passages))]


class FakeAnswerGenerator(AnswerGenerator):
    def generate(self, prompt: str) -> str:
        return "Grounded service answer. [Source 1: QM.1 TEST, pages 1]"


class RAGServiceTests(unittest.TestCase):
    def test_ask_coordinates_keyword_reranking_and_answering(self) -> None:
        index = VectorIndex(
            model_name="test-model",
            chunks=[make_chunk(index) for index in range(3)],
            embeddings=np.zeros((3, 2), dtype=np.float32),
        )
        service = RAGService(
            index,
            router=KeywordRouter(),
            reranker=FakeReranker(),
            answer_generator=FakeAnswerGenerator(),
        )

        response = service.ask(
            "Candidate",
            candidate_k=3,
            top_k=2,
            answer_top_k=1,
        )

        self.assertEqual(response.search_mode, "keyword")
        self.assertEqual(response.stats.semantic_candidates, 0)
        self.assertEqual(response.stats.keyword_candidates, 3)
        self.assertEqual(response.stats.final_results, 2)
        self.assertIn("Grounded service answer", response.answer or "")


if __name__ == "__main__":
    unittest.main()
