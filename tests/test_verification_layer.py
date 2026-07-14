from __future__ import annotations

import unittest

from rag_chatbot.answer_layer import CitedStatement, ClinicalAnswer
from rag_chatbot.verification_layer import (
    ClaimVerification,
    apply_verification,
    build_verification_prompt,
    flatten_answer_claims,
    parse_agent_json,
    verify_clinical_answer,
)
from rag_chatbot.validation_layer import validate_clinical_answer

from test_reranking_layer import make_chunk


class FakeVerifier:
    def __init__(self, judgments: list[ClaimVerification]) -> None:
        self.judgments = judgments

    def verify(self, **kwargs: object) -> list[ClaimVerification]:
        return self.judgments


class VerificationLayerTests(unittest.TestCase):
    def test_flatten_answer_claims_creates_stable_ids(self) -> None:
        answer = ClinicalAnswer(
            summary=CitedStatement(text="Summary.", citations=[1]),
            key_requirements=[
                CitedStatement(text="Requirement.", citations=[1]),
            ],
            clinical_actions=[
                CitedStatement(text="Action.", citations=[1]),
            ],
        )

        claims = flatten_answer_claims(answer)

        self.assertEqual(
            [claim.claim_id for claim in claims],
            ["summary", "key_requirements.0", "clinical_actions.0"],
        )

    def test_prompt_contains_claims_and_cited_sources(self) -> None:
        answer = ClinicalAnswer(
            summary=CitedStatement(text="Summary.", citations=[1]),
        )
        prompt = build_verification_prompt(
            question="What is QM.1?",
            claims=flatten_answer_claims(answer),
            sources=[make_chunk(0)],
        )

        self.assertIn("Question:\nWhat is QM.1?", prompt)
        self.assertIn("[Source 1]", prompt)
        self.assertIn('"claim_id": "summary"', prompt)

    def test_apply_verification_removes_unsupported_claims(self) -> None:
        answer = ClinicalAnswer(
            summary=CitedStatement(text="Summary.", citations=[1]),
            key_requirements=[
                CitedStatement(text="Supported.", citations=[1]),
                CitedStatement(text="Unsupported.", citations=[1]),
            ],
        )
        verified = apply_verification(
            answer,
            flatten_answer_claims(answer),
            [
                ClaimVerification("summary", "supported", 0.9, "ok"),
                ClaimVerification("key_requirements.0", "supported", 0.8, "ok"),
                ClaimVerification("key_requirements.1", "unsupported", 0.2, "bad"),
            ],
        )

        self.assertEqual(
            [item.text for item in verified.answer.key_requirements],
            ["Supported."],
        )
        self.assertFalse(verified.verification.verified)
        self.assertEqual(verified.verification.removed_claims, 1)
        self.assertEqual(verified.verification.checked_claims, 3)

    def test_verify_can_be_disabled_without_calling_agent(self) -> None:
        answer = ClinicalAnswer(
            summary=CitedStatement(text="Summary.", citations=[1]),
        )

        verified = verify_clinical_answer(
            question="Question",
            answer=answer,
            sources=[make_chunk(0)],
            verifier=FakeVerifier([]),
            enabled=False,
        )

        self.assertIs(verified.answer, answer)
        self.assertFalse(verified.verification.enabled)
        self.assertTrue(verified.verification.verified)

    def test_validation_layer_alias_uses_strands_verification_pipeline(self) -> None:
        answer = ClinicalAnswer(
            summary=CitedStatement(text="Summary.", citations=[1]),
        )

        validated = validate_clinical_answer(
            question="Question",
            answer=answer,
            sources=[make_chunk(0)],
            verifier=FakeVerifier([]),
            enabled=False,
        )

        self.assertIs(validated.answer, answer)
        self.assertFalse(validated.verification.enabled)

    def test_parse_agent_json_accepts_markdown_fence(self) -> None:
        payload = parse_agent_json(
            '```json\n{"claims":[{"claim_id":"summary"}]}\n```'
        )

        self.assertEqual(payload["claims"][0]["claim_id"], "summary")


if __name__ == "__main__":
    unittest.main()
