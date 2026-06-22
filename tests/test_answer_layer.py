from __future__ import annotations

import unittest

from rag_chatbot.answer_layer import (
    CitedStatement,
    ClinicalAnswer,
    build_answer_prompt,
    generate_grounded_answer,
    normalize_clinical_answer,
    validate_citations,
)
from rag_chatbot.embedding_layer import SearchResult

from test_reranking_layer import make_chunk


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt: str) -> ClinicalAnswer:
        self.prompt = prompt
        return ClinicalAnswer(
            summary=CitedStatement(
                text="Grounded summary.",
                citations=[1],
            ),
            key_requirements=[
                CitedStatement(text="Maintain the QMS.", citations=[1]),
            ],
        )


class AnswerLayerTests(unittest.TestCase):
    def test_prompt_contains_question_and_source_metadata(self) -> None:
        prompt = build_answer_prompt("What is QMS?", [make_chunk(0)])

        self.assertIn("Question:\nWhat is QMS?", prompt)
        self.assertIn("[Source 1]", prompt)
        self.assertIn("Section: QM.1 TEST", prompt)
        self.assertIn("Candidate passage 0", prompt)

    def test_generation_uses_only_requested_top_chunks(self) -> None:
        results = [
            SearchResult(score=1.0 - index * 0.1, chunk=make_chunk(index))
            for index in range(3)
        ]
        generator = FakeAnswerGenerator()

        answer = generate_grounded_answer(
            "What is QMS?",
            results,
            top_k=2,
            generator=generator,
        )

        self.assertEqual(len(answer.sources), 2)
        self.assertEqual(answer.content.summary.text, "Grounded summary.")
        self.assertIn("Candidate passage 1", generator.prompt)
        self.assertNotIn("Candidate passage 2", generator.prompt)

    def test_rejects_citations_outside_supplied_sources(self) -> None:
        answer = ClinicalAnswer(
            summary=CitedStatement(text="Claim", citations=[4]),
        )

        with self.assertRaises(ValueError):
            validate_citations(answer, source_count=3)

    def test_removes_redundant_inline_citation_numbers(self) -> None:
        answer = ClinicalAnswer(
            summary=CitedStatement(
                text="Maintain the QMS (1, 2).",
                citations=[1, 2],
            )
        )

        normalized = normalize_clinical_answer(answer)

        self.assertEqual(normalized.summary.text, "Maintain the QMS.")
        self.assertEqual(normalized.summary.citations, [1, 2])


if __name__ == "__main__":
    unittest.main()
