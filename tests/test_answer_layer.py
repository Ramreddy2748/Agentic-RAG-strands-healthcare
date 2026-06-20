from __future__ import annotations

import unittest

from rag_chatbot.answer_layer import build_answer_prompt, generate_grounded_answer
from rag_chatbot.embedding_layer import SearchResult

from test_reranking_layer import make_chunk


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return "Grounded summary. [Source 1: QM.1 TEST, pages 1]"


class AnswerLayerTests(unittest.TestCase):
    def test_prompt_contains_question_and_source_metadata(self) -> None:
        prompt = build_answer_prompt("What is QMS?", [make_chunk(0)])

        self.assertIn("Question:\nWhat is QMS?", prompt)
        self.assertIn("[Source 1: QM.1 TEST, pages 1]", prompt)
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
        self.assertIn("Candidate passage 1", generator.prompt)
        self.assertNotIn("Candidate passage 2", generator.prompt)


if __name__ == "__main__":
    unittest.main()
