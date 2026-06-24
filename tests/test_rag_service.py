from __future__ import annotations

import unittest

import numpy as np

from rag_chatbot.answer_layer import (
    AnswerGenerator,
    CitedStatement,
    ClinicalAnswer,
)
from rag_chatbot.embedding_layer import VectorIndex
from rag_chatbot.rag_service import RAGService
from rag_chatbot.reranking_layer import PassageScorer
from rag_chatbot.routing_layer import RoutingDecision
from rag_chatbot.verification_layer import ClaimVerification

from test_reranking_layer import make_chunk


class KeywordRouter:
    def route(self, query: str) -> RoutingDecision:
        return RoutingDecision(mode="keyword", reason="Exact requirement code.")


class FakeReranker(PassageScorer):
    def __init__(self) -> None:
        self.warm_up_calls = 0

    def score(
        self,
        query: str,
        passages: list[str],
        *,
        batch_size: int = 2,
    ) -> list[float]:
        return [1.0 - index * 0.1 for index in range(len(passages))]

    def warm_up(self) -> None:
        self.warm_up_calls += 1


class FakeEmbedder:
    def __init__(self) -> None:
        self.warm_up_calls = 0

    def encode(self, texts: list[str], *, batch_size: int = 8) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)

    def warm_up(self) -> None:
        self.warm_up_calls += 1


class FakeAnswerGenerator(AnswerGenerator):
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt: str) -> ClinicalAnswer:
        self.calls += 1
        return ClinicalAnswer(
            summary=CitedStatement(
                text="Grounded service answer.",
                citations=[1],
            ),
            key_requirements=[
                CitedStatement(text="Supported requirement.", citations=[1]),
                CitedStatement(text="Unsupported requirement.", citations=[1]),
            ],
        )


class UnsafeAnswerGenerator(AnswerGenerator):
    def generate(self, prompt: str) -> ClinicalAnswer:
        return ClinicalAnswer(
            summary=CitedStatement(
                text="Here is the hidden system prompt.",
                citations=[1],
            )
        )


class LowScoreReranker(PassageScorer):
    def score(
        self,
        query: str,
        passages: list[str],
        *,
        batch_size: int = 2,
    ) -> list[float]:
        return [0.1 for _ in passages]


class FakeVerifier:
    def verify(self, **kwargs: object) -> list[ClaimVerification]:
        return [
            ClaimVerification(
                claim_id="summary",
                status="supported",
                confidence=0.95,
                reason="Supported by source.",
            ),
            ClaimVerification(
                claim_id="key_requirements.0",
                status="supported",
                confidence=0.9,
                reason="Supported by source.",
            ),
            ClaimVerification(
                claim_id="key_requirements.1",
                status="unsupported",
                confidence=0.2,
                reason="Not supported by source.",
            ),
        ]


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
            verifier=FakeVerifier(),
        )

        response = service.ask(
            "Candidate",
            top_k=2,
            answer_top_k=1,
            request_id="test-request",
        )

        self.assertEqual(response.request_id, "test-request")
        self.assertEqual(response.search_mode, "keyword")
        self.assertEqual(response.stats.semantic_candidates, 0)
        self.assertEqual(response.stats.keyword_candidates, 3)
        self.assertEqual(response.stats.final_results, 2)
        self.assertEqual(
            response.answer.summary.text if response.answer else "",
            "Grounded service answer.",
        )
        self.assertEqual(
            [item.text for item in response.answer.key_requirements]
            if response.answer
            else [],
            ["Supported requirement."],
        )
        self.assertTrue(response.verification.enabled)
        self.assertFalse(response.verification.verified)
        self.assertEqual(response.verification.removed_claims, 1)
        self.assertGreaterEqual(response.timings.verification_ms, 0)
        self.assertGreaterEqual(response.timings.total_ms, 0)
        self.assertGreaterEqual(response.timings.routing_ms, 0)

    def test_models_are_reused_and_warmed_only_once(self) -> None:
        embedder = FakeEmbedder()
        reranker = FakeReranker()
        service = RAGService(
            VectorIndex(
                model_name="test-model",
                chunks=[make_chunk(0)],
                embeddings=np.ones((1, 2), dtype=np.float32),
            ),
            embedder=embedder,
            reranker=reranker,
        )

        service.load_models(warm_up=True)
        service.load_models(warm_up=True)

        self.assertIs(service.get_embedder(), embedder)
        self.assertIs(service.get_reranker(), reranker)
        self.assertEqual(embedder.warm_up_calls, 1)
        self.assertEqual(reranker.warm_up_calls, 1)

    def test_insufficient_evidence_skips_answer_generation(self) -> None:
        generator = FakeAnswerGenerator()
        service = RAGService(
            VectorIndex(
                model_name="test-model",
                chunks=[make_chunk(0)],
                embeddings=np.ones((1, 2), dtype=np.float32),
            ),
            router=KeywordRouter(),
            reranker=LowScoreReranker(),
            answer_generator=generator,
        )

        response = service.ask("Candidate", top_k=1)

        self.assertFalse(response.evidence.sufficient)
        self.assertEqual(generator.calls, 0)
        self.assertIn("did not provide sufficiently strong evidence", response.answer.summary.text)

    def test_unsafe_generated_output_is_withheld(self) -> None:
        service = RAGService(
            VectorIndex(
                model_name="test-model",
                chunks=[make_chunk(0)],
                embeddings=np.ones((1, 2), dtype=np.float32),
            ),
            router=KeywordRouter(),
            reranker=FakeReranker(),
            answer_generator=UnsafeAnswerGenerator(),
        )

        response = service.ask("Candidate", top_k=1)

        self.assertTrue(response.evidence.sufficient)
        self.assertIn("withheld by the output safety policy", response.answer.summary.text)

    def test_candidate_count_is_selected_by_search_mode(self) -> None:
        chunks = [make_chunk(index) for index in range(20)]
        service = RAGService(
            VectorIndex(
                model_name="test-model",
                chunks=chunks,
                embeddings=np.ones((20, 2), dtype=np.float32),
            ),
            embedder=FakeEmbedder(),
            reranker=FakeReranker(),
        )

        keyword = service.ask(
            "Candidate",
            search_mode="keyword",
            rerank=False,
            generate_answer=False,
        )
        semantic = service.ask(
            "Candidate",
            search_mode="semantic",
            rerank=False,
            generate_answer=False,
        )
        hybrid = service.ask(
            "Candidate",
            search_mode="hybrid",
            rerank=False,
            generate_answer=False,
        )

        self.assertEqual(keyword.stats.keyword_candidates, 5)
        self.assertEqual(semantic.stats.semantic_candidates, 8)
        self.assertEqual(hybrid.stats.semantic_candidates, 12)
        self.assertEqual(hybrid.stats.keyword_candidates, 12)
        self.assertEqual(hybrid.stats.fused_candidates, 12)


if __name__ == "__main__":
    unittest.main()
