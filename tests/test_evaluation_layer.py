from __future__ import annotations

import unittest

import numpy as np

from rag_chatbot.answer_layer import CitedStatement, ClinicalAnswer
from rag_chatbot.evaluation_layer import (
    ClaimJudgmentSchema,
    EvaluationCase,
    FaithfulnessResponseSchema,
    RequestRateLimiter,
    build_report,
    evaluate_faithfulness,
    evaluate_response,
)
from rag_chatbot.observability import PipelineTimings
from rag_chatbot.rag_service import RAGResponse, RetrievalStats
from rag_chatbot.reranking_layer import RerankedSearchResult

from test_reranking_layer import make_chunk


class EvaluationLayerTests(unittest.TestCase):
    def test_scores_retrieval_terms_citations_and_latency(self) -> None:
        relevant_chunk = make_chunk(0)
        relevant_chunk = type(relevant_chunk)(
            **{
                **relevant_chunk.__dict__,
                "section_title": "QM.1 RESPONSIBILITY AND ACCOUNTABILITY",
                "text": (
                    "The governing body is responsible and accountable for the "
                    "quality management system."
                ),
            }
        )
        response = make_response(
            results=[
                make_result(relevant_chunk, 0.9),
                make_result(make_chunk(1), 0.5),
            ],
            answer=ClinicalAnswer(
                summary=CitedStatement(
                    text="Leadership is responsible and accountable.",
                    citations=[1],
                )
            ),
        )
        case = EvaluationCase(
            case_id="qm-1",
            question="Who is accountable for quality?",
            expected_sections=("QM.1",),
            expected_terms=("quality management system", "responsible and accountable"),
        )

        evaluation = evaluate_response(case, response)

        self.assertTrue(evaluation.metrics.section_hit)
        self.assertEqual(evaluation.metrics.section_recall, 1.0)
        self.assertEqual(evaluation.metrics.first_relevant_rank, 1)
        self.assertEqual(evaluation.metrics.reciprocal_rank, 1.0)
        self.assertEqual(evaluation.metrics.evidence_term_recall, 1.0)
        self.assertEqual(evaluation.metrics.answer_term_recall, 0.5)
        self.assertTrue(evaluation.metrics.citations_valid)
        self.assertEqual(evaluation.metrics.total_ms, 125.0)

    def test_report_averages_only_measured_metrics(self) -> None:
        first = evaluate_response(
            EvaluationCase(
                case_id="one",
                question="Question one",
                expected_sections=("QM.1",),
            ),
            make_response(results=[make_result(make_chunk(0), 0.9)]),
        )
        second = evaluate_response(
            EvaluationCase(
                case_id="two",
                question="Question two",
            ),
            make_response(results=[make_result(make_chunk(1), 0.8)]),
        )

        report = build_report([first, second])

        self.assertEqual(report.summary.total_cases, 2)
        self.assertEqual(report.summary.section_hit_rate, 1.0)
        self.assertEqual(report.summary.mean_reciprocal_rank, 1.0)
        self.assertEqual(report.summary.mean_total_ms, 125.0)

    def test_faithfulness_uses_one_judgment_for_each_claim(self) -> None:
        response = make_response(
            results=[
                make_result(make_chunk(0), 0.9),
                make_result(make_chunk(1), 0.8),
            ],
            answer=ClinicalAnswer(
                summary=CitedStatement(
                    text="The first claim is supported.",
                    citations=[1],
                ),
                key_requirements=[
                    CitedStatement(
                        text="The second claim is only partly supported.",
                        citations=[2],
                    )
                ],
            ),
        )
        judge = FakeFaithfulnessJudge()

        faithfulness = evaluate_faithfulness(response, judge=judge)

        self.assertEqual(judge.calls, 1)
        self.assertIn("Claim 1:", judge.prompt)
        self.assertIn("Source 2", judge.prompt)
        self.assertEqual(faithfulness.total_claims, 2)
        self.assertEqual(faithfulness.supported_claim_rate, 0.5)
        self.assertEqual(faithfulness.grounded_claim_rate, 1.0)

    def test_request_rate_limiter_waits_between_api_call_starts(self) -> None:
        clock = FakeClock()
        limiter = RequestRateLimiter(
            15.0,
            clock=clock.now,
            sleeper=clock.sleep,
        )

        limiter.wait()
        clock.value = 4.0
        limiter.wait()
        clock.value = 20.0
        limiter.wait()

        self.assertEqual(clock.sleeps, [11.0, 10.0])


class FakeFaithfulnessJudge:
    def __init__(self) -> None:
        self.calls = 0
        self.prompt = ""

    def judge(self, prompt: str) -> FaithfulnessResponseSchema:
        self.calls += 1
        self.prompt = prompt
        return FaithfulnessResponseSchema(
            judgments=[
                ClaimJudgmentSchema(
                    claim_number=1,
                    verdict="supported",
                    reason="Source 1 directly supports the claim.",
                ),
                ClaimJudgmentSchema(
                    claim_number=2,
                    verdict="partially_supported",
                    reason="Source 2 supports only part of the claim.",
                ),
            ]
        )


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.value += seconds


def make_result(chunk, score: float) -> RerankedSearchResult:
    return RerankedSearchResult(
        rerank_score=score,
        hybrid_score=score,
        vector_score=score,
        keyword_score=0.0,
        chunk=chunk,
    )


def make_response(
    *,
    results: list[RerankedSearchResult],
    answer: ClinicalAnswer | None = None,
) -> RAGResponse:
    return RAGResponse(
        request_id="evaluation-test",
        question="Test question",
        search_mode="hybrid",
        routing_reason="Test route",
        results=results,
        answer=answer,
        stats=RetrievalStats(
            semantic_candidates=len(results),
            keyword_candidates=len(results),
            fused_candidates=len(results),
            final_results=len(results),
        ),
        timings=PipelineTimings(
            routing_ms=5.0,
            retrieval_ms=20.0,
            fusion_ms=1.0,
            reranking_ms=90.0,
            answer_generation_ms=9.0,
            total_ms=125.0,
        ),
    )


if __name__ == "__main__":
    unittest.main()
