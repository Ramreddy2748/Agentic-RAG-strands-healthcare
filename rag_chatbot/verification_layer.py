from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import re
from typing import Protocol, Sequence

from rag_chatbot.answer_layer import CitedStatement, ClinicalAnswer
from rag_chatbot.embedding_layer import IndexedChunk


DEFAULT_VERIFICATION_MODEL = "default"

VERIFICATION_SYSTEM_PROMPT = """You are a claim verification agent for hospital-accreditation RAG.

Rules:
- Use only the supplied cited source text.
- Verify whether each answer claim is directly supported by its cited sources.
- Mark a claim unsupported if it adds requirements, timelines, roles, or actions
  that are not present in the cited source text.
- Mark a claim unclear if the source is related but not strong enough.
- Return JSON only.
"""


@dataclass(frozen=True)
class ClaimForVerification:
    """One generated answer claim and its cited source numbers."""

    claim_id: str
    text: str
    citations: list[int]


@dataclass(frozen=True)
class ClaimVerification:
    """Agent judgment for one generated claim."""

    claim_id: str
    status: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class VerificationMetadata:
    """Summary metadata returned with the final verified response."""

    enabled: bool
    verified: bool
    confidence: float
    checked_claims: int
    supported_claims: int
    removed_claims: int
    unclear_claims: int = 0
    reason: str = ""
    unsupported_claims: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VerifiedAnswer:
    """Answer content after verification and the verification audit metadata."""

    answer: ClinicalAnswer
    verification: VerificationMetadata


class AnswerVerifier(Protocol):
    """Interface for claim-grounding verifiers."""

    def verify(
        self,
        *,
        question: str,
        answer: ClinicalAnswer,
        sources: Sequence[IndexedChunk],
    ) -> list[ClaimVerification]: ...


class StrandsVerificationAgent:
    """Use a Strands agent to verify answer claims against cited source text."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_VERIFICATION_MODEL,
    ) -> None:
        try:
            from strands import Agent
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing Strands dependency. Install it with "
                "`python -m pip install -e '.[strands]'`."
            ) from exc

        self.model_name = model_name
        agent_kwargs: dict[str, object] = {
            "system_prompt": VERIFICATION_SYSTEM_PROMPT,
        }
        if model_name and model_name != DEFAULT_VERIFICATION_MODEL:
            agent_kwargs["model"] = model_name
        self.agent = Agent(**agent_kwargs)

    def verify(
        self,
        *,
        question: str,
        answer: ClinicalAnswer,
        sources: Sequence[IndexedChunk],
    ) -> list[ClaimVerification]:
        """Ask Strands to return claim-level grounding judgments."""
        claims = flatten_answer_claims(answer)
        if not claims:
            return []

        prompt = build_verification_prompt(
            question=question,
            claims=claims,
            sources=sources,
        )
        response = self.agent(prompt)
        payload = parse_agent_json(str(response))
        judgments = payload.get("claims", payload)
        if not isinstance(judgments, list):
            raise RuntimeError("Verification agent returned invalid JSON.")

        return [
            ClaimVerification(
                claim_id=str(item.get("claim_id", "")),
                status=normalize_status(str(item.get("status", ""))),
                confidence=clamp_confidence(item.get("confidence", 0.0)),
                reason=str(item.get("reason", "")),
            )
            for item in judgments
            if isinstance(item, dict)
        ]


def verify_clinical_answer(
    *,
    question: str,
    answer: ClinicalAnswer,
    sources: Sequence[IndexedChunk],
    verifier: AnswerVerifier | None = None,
    enabled: bool = True,
    model_name: str = DEFAULT_VERIFICATION_MODEL,
) -> VerifiedAnswer:
    """Verify generated answer claims and remove unsupported statements."""
    if not enabled:
        return VerifiedAnswer(
            answer=answer,
            verification=VerificationMetadata(
                enabled=False,
                verified=True,
                confidence=1.0,
                checked_claims=0,
                supported_claims=0,
                removed_claims=0,
                reason="Verification disabled.",
            ),
        )

    claims = flatten_answer_claims(answer)
    if not claims:
        return VerifiedAnswer(
            answer=answer,
            verification=VerificationMetadata(
                enabled=True,
                verified=True,
                confidence=1.0,
                checked_claims=0,
                supported_claims=0,
                removed_claims=0,
                reason="No generated claims required verification.",
            ),
        )

    active_verifier = verifier or StrandsVerificationAgent(model_name=model_name)
    judgments = active_verifier.verify(
        question=question,
        answer=answer,
        sources=sources,
    )
    return apply_verification(answer, claims, judgments)


def flatten_answer_claims(answer: ClinicalAnswer) -> list[ClaimForVerification]:
    """Create stable claim IDs for summary, requirements, and actions."""
    claims = [
        ClaimForVerification(
            claim_id="summary",
            text=answer.summary.text,
            citations=answer.summary.citations,
        )
    ]
    claims.extend(
        ClaimForVerification(
            claim_id=f"key_requirements.{index}",
            text=statement.text,
            citations=statement.citations,
        )
        for index, statement in enumerate(answer.key_requirements)
    )
    claims.extend(
        ClaimForVerification(
            claim_id=f"clinical_actions.{index}",
            text=statement.text,
            citations=statement.citations,
        )
        for index, statement in enumerate(answer.clinical_actions)
    )
    return claims


def build_verification_prompt(
    *,
    question: str,
    claims: Sequence[ClaimForVerification],
    sources: Sequence[IndexedChunk],
) -> str:
    """Build the Strands verification task from claims and cited sources."""
    source_blocks = []
    for number, source in enumerate(sources, start=1):
        source_blocks.append(
            "\n".join(
                [
                    f"[Source {number}]",
                    f"Chapter: {source.chapter_title}",
                    f"Section: {source.section_title}",
                    f"Pages: {source.page_number}-{source.end_page_number}",
                    source.text,
                ]
            )
        )

    claim_payload = [
        {
            "claim_id": claim.claim_id,
            "text": claim.text,
            "citations": claim.citations,
        }
        for claim in claims
    ]

    return "\n\n".join(
        [
            f"Question:\n{question.strip()}",
            "Cited sources:",
            *source_blocks,
            "Claims to verify:",
            json.dumps(claim_payload, indent=2),
            (
                "Return JSON with this shape: "
                '{"claims":[{"claim_id":"summary","status":"supported|unsupported|unclear",'
                '"confidence":0.0,"reason":"short reason"}]}'
            ),
        ]
    )


def apply_verification(
    answer: ClinicalAnswer,
    claims: Sequence[ClaimForVerification],
    judgments: Sequence[ClaimVerification],
) -> VerifiedAnswer:
    """Remove unsupported/unclear claims and summarize verification results."""
    claim_ids = {claim.claim_id for claim in claims}
    judgment_by_id = {
        judgment.claim_id: judgment
        for judgment in judgments
        if judgment.claim_id in claim_ids
    }
    supported_ids = {
        claim_id
        for claim_id, judgment in judgment_by_id.items()
        if judgment.status == "supported"
    }
    unclear_ids = {
        claim_id
        for claim_id, judgment in judgment_by_id.items()
        if judgment.status == "unclear"
    }
    removed_ids = claim_ids - supported_ids

    summary = answer.summary
    if "summary" in removed_ids:
        summary = CitedStatement(
            text=(
                "The generated summary was removed because the verification "
                "agent could not confirm it from the cited source text."
            ),
            citations=[],
        )

    verified_answer = ClinicalAnswer(
        summary=summary,
        key_requirements=[
            statement
            for index, statement in enumerate(answer.key_requirements)
            if f"key_requirements.{index}" in supported_ids
        ],
        clinical_actions=[
            statement
            for index, statement in enumerate(answer.clinical_actions)
            if f"clinical_actions.{index}" in supported_ids
        ],
        limitations=answer.limitations,
    )

    unsupported_claims = [
        claim.text
        for claim in claims
        if claim.claim_id in removed_ids
    ]
    confidences = [
        judgment.confidence
        for judgment in judgment_by_id.values()
    ]
    confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
    metadata = VerificationMetadata(
        enabled=True,
        verified=not removed_ids,
        confidence=confidence,
        checked_claims=len(claims),
        supported_claims=len(supported_ids),
        removed_claims=len(removed_ids),
        unclear_claims=len(unclear_ids),
        unsupported_claims=unsupported_claims,
        reason=verification_reason(len(removed_ids), len(unclear_ids)),
    )
    return VerifiedAnswer(answer=verified_answer, verification=metadata)


def verification_is_enabled() -> bool:
    """Read the runtime verification flag."""
    return os.getenv("ENABLE_VERIFICATION", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def parse_agent_json(text: str) -> dict[str, object]:
    """Parse raw Strands output that may include markdown fences."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


def normalize_status(status: str) -> str:
    """Collapse agent status variants to supported, unsupported, or unclear."""
    normalized = status.strip().lower()
    if normalized in {"supported", "support", "yes"}:
        return "supported"
    if normalized in {"unsupported", "not_supported", "not supported", "no"}:
        return "unsupported"
    return "unclear"


def clamp_confidence(value: object) -> float:
    """Return confidence in the inclusive 0.0-1.0 range."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))


def verification_reason(removed_claims: int, unclear_claims: int) -> str:
    """Human-readable reason for verification metadata."""
    if removed_claims == 0:
        return "All generated claims were supported by their cited sources."
    if unclear_claims:
        return (
            f"Removed {removed_claims} claim(s); {unclear_claims} were unclear "
            "against the cited source text."
        )
    return f"Removed {removed_claims} unsupported claim(s)."
